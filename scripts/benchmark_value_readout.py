"""Paired A100 replay on archived Object camera observations. Run in smolvla-plus."""

import argparse
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("workspace", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    ROOT = args.workspace.resolve()
    sys.path[:0] = [
        str(Path(__file__).resolve().parents[1]),
        str(ROOT / "upstream/Jyugo-15-VGAS-ef656cb"),
    ]
    os.environ.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false"
    )
    from method.value_token_readout import value_token_decoders
    from method.shared_prefix_selection import select
    import numpy as np
    import torch
    from safetensors.torch import load_model
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.envs.utils import preprocess_observation
    from smolvla_qchunk.eval.bestofn_eval import (
        _load_policy_config,
        _load_critic_checkpoint,
        BestOfNCriticTrainer,
        encode_policy_observations,
    )
    from models.smolvla.modeling_smolvla import SmolVLAPolicy

    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    if torch.cuda.mem_get_info()[0] < 4752 * 1024**2:
        raise RuntimeError("Insufficient shared GPU headroom")
    info = json.loads(
        (ROOT / "outputs/p0_object_plus_20261112/actors.json").read_text()
    )["B"]
    path = Path(info["path"])
    cfg = _load_policy_config(path)
    cfg.vlm_model_name = str(
        ROOT / "research/reuse_candidates_20261030/backbone_metadata"
    )
    cfg.load_vlm_weights = False
    cfg.device = "cpu"
    cfg.n_action_steps = 20
    policy = SmolVLAPolicy(cfg)
    policy.model.vlm_with_expert.vlm.to(dtype=torch.bfloat16)
    load_model(policy, str(path / "model.safetensors"), strict=True)
    cfg.device = "cuda"
    policy.to("cuda").eval().requires_grad_(False)
    pre, post = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=str(path),
        preprocessor_overrides={
            "device_processor": {"device": "cuda"},
            "tokenizer_processor": {"tokenizer_name": cfg.vlm_model_name},
        },
    )
    case = (
        ROOT
        / "outputs/object_vgas_confirmation_20261113/formal/B-BC_000/object_t0_init20_policy0"
    )
    obs = np.load(case / "query000/observation.npz")
    result = json.loads((case / "result.json").read_text())
    batch = preprocess_observation(
        {
            "pixels": {k: obs[k][None].copy() for k in ["image", "image2"]},
            "agent_pos": obs["state"][None].copy(),
        }
    )
    batch["task"] = [result["task_instruction"]]
    batch = pre(batch)
    ckpt = _load_critic_checkpoint(
        ROOT
        / "research/reuse_candidates_20261030/author_assets/object/vgas_critic/last.ckpt"
    )
    encoder = lambda p, b: encode_policy_observations(
        p, b, use_vlm_backbone_encode=True
    )
    trainer = BestOfNCriticTrainer.build(
        policy,
        dict(batch, action=torch.zeros(1, 32, 7, device="cuda")),
        ckpt.config,
        torch.device("cuda"),
        encoder,
    )
    trainer.load_state_dict(ckpt.state_dict)
    trainer.target_critic = trainer.critic
    trainer.critic.eval().requires_grad_(False)
    heads = [branch.head for branch in trainer.critic.heads]
    reference = [head.decoder for head in heads]
    compact = value_token_decoders(trainer.critic)
    rows = []
    sequence = []

    def shape_hook(module, inputs):
        sequence.append(list(inputs[0].shape))

    def timed(fn):
        torch.cuda.synchronize()
        start = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        return out, (time.perf_counter() - start) * 1000

    with torch.inference_mode():
        for task in range(10):
            case = next(
                (ROOT / "outputs/object_vgas_confirmation_20261113/formal").glob(
                    f"B-BC_*/object_t{task}_init20_policy0"
                )
            )
            frames = sorted(case.glob("query*/observation.npz"))
            for phase, frame in enumerate([frames[0], frames[len(frames) // 2]]):
                with np.load(frame) as obs:
                    batch = preprocess_observation(
                        {
                            "pixels": {
                                k: obs[k][None].copy() for k in ["image", "image2"]
                            },
                            "agent_pos": obs["state"][None].copy(),
                        }
                    )
                batch["task"] = [
                    json.loads((case / "result.json").read_text())["task_instruction"]
                ]
                batch = pre(batch)
                generator = torch.Generator(device="cuda").manual_seed(
                    20261120 + 2 * task + phase
                )
                noise = torch.randn(
                    (4, 50, cfg.max_action_dim), device="cuda", generator=generator
                )
                # Generate the pool once for isolated critic timing. Also measure the complete selector.
                pool, _, _, _ = select(policy, trainer, batch, noise, score_final=False)
                actions = pool[:, :32, :7]
                enc = encoder(policy, batch).repeat(4)

                def score():
                    return (
                        torch.stack(
                            trainer.critic(
                                enc,
                                actions,
                                action_mask=torch.zeros(
                                    (4, 32), device="cuda", dtype=torch.bool
                                ),
                                raw_state=batch["observation.state"].expand(4, -1),
                            )
                        )
                        .min(0)
                        .values[:, 0]
                    )

                row = {"task": task, "frame": frame.parent.name}
                for rep in range(4):
                    for mode in (
                        ["full", "value"]
                        if (task + phase + rep) % 2 == 0
                        else ["value", "full"]
                    ):
                        for head, decoder in zip(
                            heads, reference if mode == "full" else compact
                        ):
                            head.decoder = decoder
                        if task == 0 and phase == 0 and rep == 0:
                            hook = (
                                reference[0]
                                .layers[-1]
                                .mlp.register_forward_pre_hook(shape_hook)
                            )
                            score()
                            hook.remove()
                            row[mode + "_mlp_input"] = sequence[-1]
                        q, ms = timed(score)
                        (_, end_q, winner, _), total = timed(
                            lambda: select(policy, trainer, batch, noise)
                        )
                        row[mode + "_scores"] = end_q.tolist()
                        row[mode + "_winner"] = winner
                        if rep:
                            row.setdefault(mode + "_critic_ms", []).append(ms)
                            row.setdefault(mode + "_selector_ms", []).append(total)
                rows.append(row)
    payload = {
        "gpu": torch.cuda.get_device_name(),
        "torch": str(torch.__version__),
        "rows": rows,
    }
    args.output.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
