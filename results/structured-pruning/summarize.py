"""Summarize matched timing blocks and paired layout-bootstrap intervals."""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


def summarize(data):
    blocks = []
    rng = np.random.default_rng(20261127)
    for block in sorted({row["block"] for row in data["blocks"]}):
        by_task_layout = defaultdict(lambda: defaultdict(list))
        rows = [row for row in data["blocks"] if row["block"] == block]
        arms = ("dense", "prune", "cache", "prune_cache")
        times = np.asarray([[row["ms"][arm] for arm in arms] for row in rows])
        for index, row in enumerate(rows):
            task, _ = row["case"].split("_init", 1)
            by_task_layout[task][row["case"]].append(index)
        deltas = []
        for _ in range(20000):
            matched = []
            for layouts in by_task_layout.values():
                groups = list(layouts.values())
                for group in rng.integers(len(groups), size=len(groups)):
                    matched.extend(groups[group])
            sample = np.median(times[matched], axis=0)
            deltas.append(sample - sample[0])
        limits = np.percentile(deltas, [2.5, 97.5], axis=0)
        p50, p90 = np.percentile(times, [50, 90], axis=0)
        blocks.append({"block": block, "queries": len(rows), "arms": {
            arm: {
                "p50_ms": float(p50[i]), "p90_ms": float(p90[i]),
                "p50_speedup_pct": float((1 - p50[i] / p50[0]) * 100),
                "p50_delta_95ci_ms": limits[:, i].tolist(),
            } for i, arm in enumerate(arms)
        }})
    return blocks


if __name__ == "__main__":
    path = Path(__file__).with_name("timing.json")
    print(json.dumps(summarize(json.loads(path.read_text())), indent=2))
