"""Summarize the paired two-by-two lighting screen and selected branch outcomes."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


def load(directory, pattern):
    return [json.loads(line) for path in sorted(directory.glob(pattern))
            for line in path.read_text().splitlines()]


def paired_interval(rows, values):
    by_task = defaultdict(lambda: defaultdict(list))
    for row, value in zip(rows, values, strict=True):
        by_task[row["task"]][row["case"]].append(value)
    rng = np.random.default_rng(20261127)
    estimate = np.zeros((20000, 2))
    for layouts in by_task.values():
        weights = np.array([[np.sum(v), len(v)] for v in layouts.values()])
        draws = rng.integers(len(weights), size=(len(estimate), len(weights)))
        estimate += weights[draws].sum(1)
    return np.percentile(estimate[:, 0] / estimate[:, 1], [2.5, 97.5]).tolist()


def screen(rows):
    assert len(rows) == 150 and len({(x["case"], x["query"]) for x in rows}) == 150
    variants = [(a, b) for a, b in (("cc", "cl"), ("lc", "ll"), ("cc", "lc"), ("cl", "ll"))]
    comparisons = {}
    for left, right in variants:
        flags = [int(row["winner"][left] != row["winner"][right]) for row in rows]
        margins = []
        for row in rows:
            winner = row["winner"][left]
            delta = np.asarray(row["scores"][right]) - np.asarray(row["scores"][left])
            margins.extend(abs(delta[winner] - delta[i]) for i in range(4) if i != winner)
        comparisons[f"{left}_vs_{right}"] = {
            "flips": sum(flags), "rate": float(np.mean(flags)),
            "paired_95ci": paired_interval(rows, flags),
            "winner_margin_abs_change_median": float(np.median(margins)),
        }
    shifts = [x for row in rows for x in row["matched_action_l1"]]
    shift_ratios = [np.mean(row["matched_action_l1"]) / max(row["clean_pool_l1"], 1e-8)
                    for row in rows]
    critic_rate = comparisons["lc_vs_ll"]["rate"]
    clean_critic_rate = comparisons["cc_vs_cl"]["rate"]
    if max(critic_rate, clean_critic_rate) >= 0.15:
        kind = "critic"
        actor_pool = "light" if critic_rate >= 0.15 else "clean"
        left, right = ("lc", "ll") if actor_pool == "light" else ("cc", "cl")

        def priority(row):
            old, new = (row["winner"][k] for k in (left, right))
            scores = row["scores"][right]
            return (old != new, scores[new] - scores[old],
                    max(abs(x - y) for x, y in zip(row["scores"][left], scores)))
    elif np.median(shift_ratios) >= 0.5:
        kind, actor_pool = "actor", None

        def priority(row):
            return np.mean(row["matched_action_l1"]) / max(row["clean_pool_l1"], 1e-8)
    else:
        kind, actor_pool = None, None

        def priority(row):
            return 0

    ranked = [{k: row[k] for k in ("case", "task", "query", "winner")}
              for row in sorted(rows, key=priority, reverse=True)[:40]] if kind else []
    report = {
        "anchors": len(rows), "layouts": len({row["case"] for row in rows}),
        "matched_action_l1_p10_p50_p90": np.percentile(shifts, [10, 50, 90]).tolist(),
        "within_clean_pool_l1_median": float(np.median([row["clean_pool_l1"] for row in rows])),
        "mean_matched_l1_over_pool_l1_median": float(np.median(shift_ratios)),
        "image_mae_median": {k: float(np.median([r["image_mae"][k] for r in rows]))
                             for k in ("image", "image2")},
        "physics_max_error": max(r["physics_max_error"] for r in rows),
        "comparisons": comparisons, "branch_type": kind, "branch_actor_pool": actor_pool,
    }
    return report, {"type": kind, "actor_pool": actor_pool, "anchors": ranked}


def branches(rows, plan):
    intended = {(row["case"], row["query"]): row for row in plan["anchors"]}
    observed = {(row["case"], row["query"]): row for row in rows}
    assert len(intended) == 40 and len(observed) == len(rows)
    assert observed.keys() <= intended.keys()
    missing = sorted(intended.keys() - observed.keys())
    kind = plan["type"]
    metrics = []
    for row in rows:
        outcomes = {(b["actor"], b["candidate"]): int(b["success"]) for b in row["outcomes"]}
        if kind == "critic":
            actor = plan["actor_pool"]
            left, right = ("lc", "ll") if actor == "light" else ("cc", "cl")
            old, new = (row["winner"][k] for k in (left, right))
            a, b = [outcomes[actor, idx] for idx in (old, new)]
            metrics.append((row, a, b, old != new))
        else:
            a, b = [np.mean([outcomes[label, i] for i in (0, 1)]) for label in ("clean", "light")]
            metrics.append((row, a, b, True))

    def summarize(group):
        if not group:
            return None
        cases, clean, light, _ = zip(*group)
        delta = np.array(light) - np.array(clean)
        return {
            "anchors": len(group),
            "clean": float(np.sum(clean)), "light": float(np.sum(light)),
            "rescues": int(np.sum(delta > 0)), "harms": int(np.sum(delta < 0)),
            "light_minus_clean": float(np.mean(delta)),
            "paired_95ci": paired_interval(cases, delta),
        }

    return {"type": kind, "actor_pool": plan["actor_pool"],
            "intended_anchors": len(intended), "missing": [intended[key] for key in missing],
            "all": summarize(metrics), "affected": summarize([row for row in metrics if row[3]])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("screen", "branches"))
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    plan_path = args.directory / "branch_cases.json"
    if args.phase == "screen":
        report, plan = screen(load(args.directory, "screen_[01].jsonl"))
        plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    else:
        report = branches(load(args.directory, "branches_[0-3].jsonl")
                          + load(args.directory, "branch_*.jsonl"),
                          json.loads(plan_path.read_text()))
    print(json.dumps(report, indent=2))
