"""Benchmark the warm ten-step graph with fresh prefix KV and noise every decision."""
import argparse
import json
import os
import sys
import time
from types import FunctionType, MethodType
from unittest.mock import patch
from pathlib import Path
from statistics import median

import torch
from torch.utils._pytree import tree_map

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from method.shared_prefix_selection import select
from profile_loader import load


def graph_safe_suffix(model):
    """Localize the constant suffix-mask factory to this bound method."""
    original = model.embed_suffix.__func__

    class MaskFactory:
        def __getattr__(self, name):
            return getattr(torch, name)

        def tensor(self, data, **kwargs):
            if data != [1] * model.config.chunk_size:
                raise ValueError("Unexpected constant in embed_suffix")
            return torch.ones(len(data), **kwargs)

    function = FunctionType(original.__code__, dict(original.__globals__, torch=MaskFactory()),
                            original.__name__, original.__defaults__, original.__closure__)
    return patch.object(model, "embed_suffix", MethodType(function, model))


class Expert:
    def __init__(self, model, n):
        self.model = model
        self.n = n
        self.graph = None
        self.start, self.end = (torch.cuda.Event(enable_timing=True) for _ in range(2))

    def loop(self, pad, cache, x, t, dt):
        for _ in range(self.model.config.num_steps):
            x += dt * self.model.denoise_step(pad, cache, x, t.expand(self.n))
            t += dt
        return x

    def eager(self, *args):
        self.start.record()
        out = self.loop(*args)
        self.end.record()
        return out

    def replay(self, pad, cache, x, t, dt):
        if self.graph is None:
            self.pad = pad[:1].clone()
            self.cache = tree_map(lambda v: v[:1].clone(), cache)
            self.noise = x.clone()
            self.t, self.dt = t.clone(), dt.clone()
            args = (self.pad.expand_as(pad), tree_map(lambda v: v.expand(self.n, *v.shape[1:]), self.cache),
                    self.noise, self.t, self.dt)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.loop(*args)
            torch.cuda.current_stream().wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.loop(*args)
        self.start.record()
        self.pad.copy_(pad[:1])
        tree_map(lambda dst, src: dst.copy_(src[:1]), self.cache, cache)
        self.noise.copy_(x)
        self.t.copy_(t)
        self.dt.copy_(dt)
        self.graph.replay()
        self.end.record()
        return self.noise


def bench(policy, trainer, batch, n, reps, fp32):
    expert = Expert(policy.model, n)
    assert policy.model.config.num_steps == 10
    gen = torch.Generator(device="cuda").manual_seed(20261121)
    def noise():
        return torch.randn(n, policy.config.chunk_size, policy.config.max_action_dim, device="cuda", generator=gen)
    def run(b, z, loop=None):
        if loop is None:
            return select(policy, trainer, b, z)[:3]
        with graph_safe_suffix(policy.model):
            return select(policy, trainer, b, z, denoise_loop=loop)[:3]
    checks = []
    for i in range(3):
        b = dict(batch)
        if i == 2:
            key = "observation.state"
            b[key] = b[key] + 0.25
        z = noise()
        ref, q, best = run(b, z)
        pool, values, winner = run(b, z, expert.replay)
        torch.testing.assert_close(ref, pool, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(q, values, atol=1e-4, rtol=1e-4)
        assert winner == best
        checks.append(dict(pool_max_abs=(pool-ref).abs().max().item(), q_max_abs=(q-values).abs().max().item(),
                           eager_winner=best, graph_winner=winner))
    row = dict(n=n, parity=checks)
    if fp32:
        return row
    samples = {k: [] for k in ("eager", "graph")}
    for i in range(reps + 6):
        z = noise()
        for key in (("eager", "graph") if i % 2 == 0 else ("graph", "eager")):
            loop = expert.eager if key == "eager" else expert.replay
            torch.cuda.synchronize()
            start = time.perf_counter()
            run(batch, z, loop)
            torch.cuda.synchronize()
            total = (time.perf_counter() - start) * 1000
            if i >= 6:
                samples[key].append(dict(full_ms=total, expert_ms=expert.start.elapsed_time(expert.end)))
    row["samples"] = samples
    row["median_ms"] = {k: {stage: median(s[stage] for s in vals) for stage in vals[0]} for k, vals in samples.items()}
    return row


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reps", type=int, default=30)
    args = p.parse_args()
    policy, trainer, batch = load()
    result = dict(torch=str(torch.__version__), device=torch.cuda.get_device_name(), observation=Path(os.environ["OBS_NPZ"]).name,
                  seed=20261121, warmup_pairs=6, repetitions=args.reps, p5_precision=[], fp32=[])
    with torch.no_grad():
        for fp32 in (False, True):
            key = "fp32" if fp32 else "p5_precision"
            if fp32:
                policy.float()
                trainer.target_critic.float()
                torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
                assert all(v.dtype == torch.float32 for m in (policy, trainer.target_critic) for v in m.parameters() if v.is_floating_point())
            for n in (1, 4, 16):
                row = bench(policy, trainer, batch, n, args.reps, fp32)
                result[key].append(row)
                print(key, n, row.get("median_ms"), row["parity"], flush=True)
                args.output.write_text(json.dumps(result, indent=2))
