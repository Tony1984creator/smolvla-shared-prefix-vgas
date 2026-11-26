"""Paired CUDA Graph replay on the frozen LIBERO-Object fresh-layout cohort."""
import argparse
import hashlib
from itertools import permutations
import json
import os
from pathlib import Path
import random
import tempfile
from time import perf_counter

import numpy as np
import torch

from eval_libero import configure, critic, load_actor, prepare

ARMS = ("single", "dense", "front5")
ORDERS = tuple(permutations(ARMS))


def warm(policy, trainer, batch, experts, noise):
    from method.shared_prefix_selection import select

    startup = {}
    for width, expert in experts.items():
        z = noise[:width]
        ref, values, best, _ = select(policy, trainer, batch, z, score_final=width > 1)
        torch.cuda.synchronize()
        start = perf_counter()
        with expert.patch:
            out, scores, winner, _ = select(policy, trainer, batch, z,
                                             score_final=width > 1, denoise_loop=expert.replay)
        torch.cuda.synchronize()
        startup[width] = (perf_counter() - start) * 1000
        torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)
        if width > 1:
            torch.testing.assert_close(scores, values, atol=1e-4, rtol=1e-4)
        if winner != best:
            raise RuntimeError("Graph and eager selected different candidates during warm-up")
    return startup


def noise_bank(policy, seed):
    rng = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(16, policy.config.chunk_size, policy.config.max_action_dim,
                       device="cuda", generator=rng)


def episode(policy, trainer, pre, post, env, obs, case, arm, experts):
    from method.shared_prefix_selection import select, selected_action

    steps = 0
    success = done = False
    widths, winners, times = [], [], []
    while steps < 280 and not done:
        batch = prepare(pre, env, obs)
        width = 4 if arm == "dense" or (arm == "front5" and len(times) < 5) else 1
        bank = noise_bank(policy, case["policy_seed"] + steps)
        expert = experts[width]
        torch.cuda.synchronize()
        start = perf_counter()
        pool, _, winner, _ = select(policy, trainer, batch, bank[:width],
                                     score_final=width > 1, denoise_loop=expert.replay)
        normalized = selected_action(pool, winner)
        torch.cuda.synchronize()
        times.append((perf_counter() - start) * 1000)
        winners.append(winner)
        widths.append(width)
        actions = post(normalized)[0].cpu().numpy()
        for action in actions[:min(20, 280 - steps)]:
            obs, _, terminated, truncated, info = env.step(action)
            steps += 1
            success = bool(info["is_success"])
            done = success or terminated or truncated
            if done:
                break
    return dict(steps=steps, success=success, widths=widths, winners=winners, decision_ms=times)


@torch.inference_mode()
def run(args):
    from libero.libero import benchmark
    from lerobot.envs.libero import LiberoEnv
    from benchmark_cuda_graph import Expert, graph_safe_suffix

    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    if torch.cuda.mem_get_info()[0] < 4752 * 1024**2:
        raise RuntimeError("Not enough free GPU memory for an evaluator")
    policy, pre, post = load_actor(args.assets)
    suite = benchmark.get_benchmark_dict()["libero_object"]()
    task_ids = {task.name: i for i, task in enumerate(suite.tasks)}
    cases = json.loads(args.manifest.read_text(encoding="utf-8"))[args.shard::args.shards]
    trainer = None
    experts = {width: Expert(policy.model, width) for width in (1, 4)}
    for expert in experts.values():
        expert.patch = graph_safe_suffix(policy.model)
    with args.output.open("x", encoding="utf-8") as output:
        for index, case in enumerate(cases):
            if torch.cuda.mem_get_info()[0] < 1024**3:
                raise RuntimeError("Shared GPU headroom lost")
            order = ORDERS[(index + args.shard) % len(ORDERS)]
            for arm in order:
                scene = case["scene_seed"]
                random.seed(scene)
                np.random.seed(scene)
                task = task_ids[case["task_name"]]
                env = LiberoEnv(suite, task, "libero_object", obs_type="pixels_agent_pos",
                                episode_index=0, num_steps_wait=10, init_states=False)
                try:
                    obs, _ = env.reset(seed=scene)
                    policy.reset()
                    physics = np.asarray(env._env.sim.get_state().flatten())
                    identity = hashlib.sha256(physics.tobytes()).hexdigest()
                    if identity != case["placement_identity"]["physical_sha"]:
                        raise RuntimeError(f"Frozen physical reset differs for {case['case']}: {identity}")
                    if trainer is None:
                        batch = prepare(pre, env, obs)
                        trainer = critic(policy, batch, args.assets, "object", 4)
                        startup = warm(policy, trainer, batch, experts, noise_bank(policy, case["policy_seed"]))
                        print(json.dumps(dict(warm_capture_ms=startup, device=torch.cuda.get_device_name(),
                                              torch_version=str(torch.__version__))), flush=True)
                    row = episode(policy, trainer, pre, post, env, obs, case, arm, experts)
                finally:
                    env.close()
                row.update(case=case["case"], task=task, arm=arm, physical_sha=identity,
                           scene_seed=scene, policy_seed=case["policy_seed"], shard=args.shard)
                output.write(json.dumps(row) + "\n")
                output.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--upstream", action="store_true", help="Use upstream paths and config from the environment.")
    args = parser.parse_args()
    if not 0 <= args.shard < args.shards:
        parser.error("Shard index must be within the requested shard count.")
    if args.upstream:
        if not os.environ.get("LIBERO_CONFIG_PATH"):
            parser.error("--upstream requires LIBERO_CONFIG_PATH.")
        run(args)
    else:
        with tempfile.TemporaryDirectory() as directory:
            configure(directory)
            run(args)
