"""Evaluate released VGAS weights on pinned LIBERO initial states."""
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def configure(directory):
    import yaml
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ["LIBERO_CONFIG_PATH"] = directory
    libero = ROOT / "vendor/LIBERO"
    sys.path[:0] = list(map(str, [ROOT, ROOT / "vendor/VGAS", libero]))
    base = libero / "libero/libero"
    paths = dict(benchmark_root=base, assets=base/"assets", init_states=base/"init_files",
                 bddl_files=base/"bddl_files", datasets=libero)
    Path(directory, "config.yaml").write_text(yaml.safe_dump({k:str(v) for k,v in paths.items()}))


def load_actor(assets, fp32=False):
    import torch
    from safetensors.torch import load_model
    from models.smolvla.modeling_smolvla import SmolVLAPolicy
    from smolvla_qchunk.eval.bestofn_eval import _load_policy_config
    from lerobot.policies.factory import make_pre_post_processors
    path = assets / "author_assets/smolvla/5_SHOT/pretrained_model"
    cfg = _load_policy_config(path)
    cfg.vlm_model_name = str(assets / "backbone_metadata")
    cfg.load_vlm_weights = False
    cfg.device = "cpu"
    cfg.n_action_steps = 20
    policy = SmolVLAPolicy(cfg)
    policy.model.vlm_with_expert.vlm.to(dtype=torch.bfloat16)
    load_model(policy, str(path/"model.safetensors"), strict=True)
    if fp32:
        policy.float()
    cfg.device = "cuda"
    policy.cuda().eval().requires_grad_(False)
    pre, post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=str(path),
        preprocessor_overrides={"device_processor":{"device":"cuda"},
                                "tokenizer_processor":{"tokenizer_name":cfg.vlm_model_name}})
    return policy, pre, post


def encode(policy, batch):
    from smolvla_qchunk.eval.bestofn_eval import encode_policy_observations
    return encode_policy_observations(policy, batch, use_vlm_backbone_encode=True)


def critic(policy, batch, assets, suite, width):
    import torch
    from smolvla_qchunk.eval.bestofn_eval import _load_critic_checkpoint, BestOfNCriticTrainer
    ckpt = _load_critic_checkpoint(assets / "author_assets" / suite / "vgas_critic/last.ckpt")
    ckpt.config.action_samples = width
    builder = dict(batch, action=torch.zeros(1, ckpt.config.q_chunk_len, 7, device="cuda"))
    trainer = BestOfNCriticTrainer.build(policy, builder, ckpt.config, torch.device("cuda"), encode)
    trainer.load_state_dict(ckpt.state_dict)
    trainer.target_critic = trainer.critic.eval().requires_grad_(False)
    return trainer


def original(policy, trainer, batch, bank):
    import torch
    predict = policy._get_action_chunk
    heads = []
    hook = trainer.target_critic.register_forward_hook(lambda module, inputs, outputs: heads.extend(outputs))
    try:
        with patch.object(policy, "_get_action_chunk", lambda expanded, noise=None: predict(expanded, noise=bank)):
            chosen, _, candidates = trainer._best_of_n_actions_soon(policy, batch, encode(policy, dict(batch)),
                                                                    return_candidates=True)
    finally:
        hook.remove()
    return candidates[0], torch.stack(heads).amin(0)[:, 0], chosen


def prepare(pre, env, obs):
    import numpy as np
    from lerobot.envs.utils import preprocess_observation
    batch = preprocess_observation(dict(pixels={k: np.asarray(v)[None].copy() for k,v in obs["pixels"].items()},
                                        agent_pos=np.asarray(obs["agent_pos"])[None].copy()))
    return pre(dict(batch, task=[env.task_description]))


def initial(suite_name, task, index, seed):
    from libero.libero import benchmark
    from lerobot.envs.libero import LiberoEnv, get_libero_dummy_action
    name = "libero_object" if suite_name == "object" else "libero_10"
    suite = benchmark.get_benchmark_dict()[name]()
    env = LiberoEnv(suite, task, name, obs_type="pixels_agent_pos", episode_index=index, num_steps_wait=10)
    try:
        env._env.seed(seed)
        env._env.reset()
        raw = env._env.set_init_state(env._init_states[index])
        for _ in range(10):
            raw, _, _, _ = env._env.step(get_libero_dummy_action())
        return env, env._format_raw_obs(raw)
    except BaseException:
        env.close()
        raise


def run(args):
    import numpy as np
    import torch
    from method.shared_prefix_selection import select, selected_action
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    policy, pre, post = load_actor(args.assets)
    trainer = None
    budget = 280 if args.suite == "object" else 520
    for episode in range(args.episodes):
        index = args.init_id + episode
        seed = args.seed + episode
        np.random.seed(seed)
        env, obs = initial(args.suite, args.task_id, index, seed)
        steps = queries = 0
        success = False
        try:
            policy.reset()
            while steps < budget:
                batch = prepare(pre, env, obs)
                if args.candidates > 1 and trainer is None:
                    trainer = critic(policy, batch, args.assets, args.suite, args.candidates)
                query_seed = seed + steps if args.suite == "object" else (seed << 12) + queries + 1
                rng = torch.Generator(device="cuda").manual_seed(query_seed)
                bank = torch.randn(16, policy.config.chunk_size, policy.config.max_action_dim,
                                   device="cuda", generator=rng)[:args.candidates]
                with torch.inference_mode():
                    if args.candidates == 1:
                        chosen = policy._get_action_chunk(batch, noise=bank)
                    elif args.impl == "original":
                        _, _, chosen = original(policy, trainer, batch, bank)
                    else:
                        pool, _, best, _ = select(policy, trainer, batch, bank)
                        chosen = selected_action(pool, best)
                    actions = post(chosen)[0].cpu().numpy()
                queries += 1
                for action in actions[:min(20, budget-steps)]:
                    obs, _, terminated, truncated, info = env.step(action)
                    steps += 1
                    success = bool(info["is_success"])
                    if terminated or truncated:
                        break
                if terminated or truncated:
                    break
        finally:
            env.close()
        print(json.dumps(dict(suite=args.suite, task=args.task_id, init=index, seed=seed,
                              candidates=args.candidates, impl=args.impl, steps=steps,
                              queries=queries, success=success)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["object","long"], default="object")
    parser.add_argument("--impl", choices=["original","shared"], default="shared")
    parser.add_argument("--candidates", type=int, choices=[1,2,4,8,16], default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--init-id", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--assets", type=Path, default=ROOT/"assets")
    args = parser.parse_args()
    if args.task_id is None:
        args.task_id = 0 if args.suite == "object" else 2
    if not 0 <= args.task_id < 10 or args.episodes < 1 or args.init_id < 0:
        parser.error("Check task index, initial state index, and episode count.")
    with tempfile.TemporaryDirectory() as directory:
        configure(directory)
        run(args)
