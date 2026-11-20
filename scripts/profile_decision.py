"""Measure shared selection on a resident observation; loader returns policy, trainer, batch."""
import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
from importlib import import_module
import json
from pathlib import Path
import statistics
import sys
from unittest.mock import patch

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from method.shared_prefix_selection import select


class Traffic(TorchDispatchMode):
    def __init__(self, weights, stage):
        super().__init__()
        self.weights = weights
        self.stage = stage
        self.traffic = defaultdict(Counter)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        outputs = func(*args, **kwargs)
        # Metadata-only view ops do not transfer data.
        schema = func._schema
        if any(r.alias_info for r in schema.returns) and not schema.is_mutable:
            return outputs
        totals = self.traffic[self.stage()]
        for direction, values in [('read', (args, kwargs)), ('write', outputs)]:
            for t in tree_leaves(values):
                if isinstance(t, torch.Tensor) and t.is_cuda:
                    kind = 'weight' if t.untyped_storage().data_ptr() in self.weights else 'activation'
                    totals[kind + '_' + direction] += t.numel() * t.element_size()
        return outputs


def measure(policy, trainer, batch, n, reps):
    model = policy.model
    noise = torch.randn(n, 50, policy.config.max_action_dim, device='cuda',
                        generator=torch.Generator(device='cuda').manual_seed(20261120))
    weights = {p.untyped_storage().data_ptr() for m in (policy, trainer.target_critic) for p in m.parameters()}
    records = []
    work = Counter()
    counter = None
    active = 'prefix'
    last_flops = 0
    events = []

    def boundary(name):
        nonlocal active, last_flops
        if counter is None:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            events.append((name, event))
        else:
            current = counter.get_total_flops()
            work[active] += current - last_flops
            last_flops = current
        active = name

    def wrapped(fn, group, leave=None):
        def call(*args, **kwargs):
            if active != group:
                boundary(group)
            out = fn(*args, **kwargs)
            if leave:
                boundary(leave)
            return out
        return call

    with ExitStack() as stack:
        for obj, attr, group, leave in [
                (model.vlm_with_expert, 'embed_image', 'vision', 'prefix'),
                (model, 'denoise_step', 'expert', None),
                (trainer.target_critic, 'forward', 'critic', None)]:
            stack.enter_context(patch.object(obj, attr, wrapped(getattr(obj, attr), group, leave)))
        for i in range(reps + 3):
            events.clear()
            boundary('prefix')
            select(policy, trainer, batch, noise)
            boundary('end')
            torch.cuda.synchronize()
            totals = Counter()
            for (key, begin), (_, end) in zip(events, events[1:]):
                totals[key] += begin.elapsed_time(end)
            if i >= 3:
                records.append(dict(totals))
        traffic = Traffic(weights, lambda: active)
        counter = FlopCounterMode(display=False)
        with traffic, counter:
            boundary('prefix')
            select(policy, trainer, batch, noise)
            boundary('end')
    return dict(n=n, raw_ms=records, stages={k:dict(ms=statistics.median(r[k] for r in records),
                flops=work[k], bytes=dict(traffic.traffic[k])) for k in records[0]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--loader', required=True, help='module:function')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reps', type=int, default=20)
    args = parser.parse_args()
    module, fn = args.loader.split(':')
    policy, trainer, batch = getattr(import_module(module), fn)()
    rows = []
    with torch.no_grad():
        for n in (1, 4, 8, 16):
            row = measure(policy, trainer, batch, n, args.reps)
            rows.append(row)
            print(n, row['stages'], flush=True)
            args.output.write_text(json.dumps(dict(device=torch.cuda.get_device_name(), torch=torch.__version__,
                                                  rows=rows), indent=2))


if __name__ == '__main__':
    main()
