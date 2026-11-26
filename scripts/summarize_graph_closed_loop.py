"""Validate paired Graph rollouts and regenerate their statistics and table."""
import argparse
import json
from pathlib import Path

import numpy as np

from make_paper_tables import ROOT, table

ARMS = ("single", "dense", "front5")
LABELS = ("Single", "Full Bo4", "EWS-5")
DATA = ROOT / "results/graph-closed-loop"


def validate(rows, manifest):
    if len(manifest) != 200 or len({case["physical_id"] for case in manifest}) != 200:
        raise ValueError("Expected 200 distinct physical layouts")
    indexed = {(row["case"], row["arm"]): row for row in rows}
    expected = {(case["case"], arm) for case in manifest for arm in ARMS}
    if len(indexed) != len(rows) or indexed.keys() != expected:
        raise ValueError("Incomplete or duplicated paired results")
    grouped = {}
    for case in manifest:
        pair = [indexed[case["case"], arm] for arm in ARMS]
        for row in pair:
            for field in ("scene_seed", "policy_seed"):
                if row[field] != case[field]:
                    raise ValueError(f"Case {row['case']} has different {field}")
            if row["physical_sha"] != case["placement_identity"]["physical_sha"]:
                raise ValueError(f"Case {row['case']} has different starting physics")
            n = len(row["decision_ms"])
            widths = {"single": [1] * n, "dense": [4] * n,
                      "front5": [4] * min(n, 5) + [1] * max(0, n - 5)}
            if (type(row["success"]) is not bool or not 0 < n <= 14
                    or not 20 * (n - 1) < row["steps"] <= 20 * n
                    or row["widths"] != widths[row["arm"]] or len(row["winners"]) != n
                    or not all(0 <= win < width for win, width in zip(row["winners"], row["widths"]))
                    or not np.isfinite(row["decision_ms"]).all() or min(row["decision_ms"]) <= 0):
                raise ValueError(f"Invalid schedule or decision log for {row['case']}")
        task = int(case["case"].split("_")[1][1:])
        if {row["task"] for row in pair} != {task} or len({row["shard"] for row in pair}) != 1:
            raise ValueError(f"Case {case['case']} lost task or worker pairing")
        grouped.setdefault(task, []).append(pair)
    if sorted(len(pairs) for pairs in grouped.values()) != [20] * 10:
        raise ValueError("Expected 20 layouts in each of ten Object tasks")
    return [grouped[task] for task in sorted(grouped)]


def summarize(groups):
    observations = np.array([[[[row["success"], sum(row["decision_ms"])]
                               for row in pair] for pair in group] for group in groups])
    rng = np.random.default_rng(20261126)
    draws = rng.integers(20, size=(20000, 10, 20))
    means = observations[np.arange(10)[None, :, None], draws].mean(axis=(1, 2))
    averages = observations.mean(axis=(0, 1))
    stats = {}
    for i, arm in enumerate(ARMS):
        records = [pair[i] for group in groups for pair in group]
        times = np.array([ms for row in records for ms in row["decision_ms"]])
        widths = [width for row in records for width in row["widths"]]
        stats[arm] = dict(successes=sum(row["success"] for row in records), episodes=len(records),
                          queries=len(widths), four_way_calls=widths.count(4), candidates=sum(widths),
                          mean_episode_ms=round(float(averages[i, 1]), 6),
                          decision_p50_ms=round(float(np.percentile(times, 50)), 6),
                          decision_p90_ms=round(float(np.percentile(times, 90)), 6))
    contrasts = {}
    for method, baseline in ((2, 1), (2, 0), (1, 0)):
        difference = (averages[method] - averages[baseline]) * [100, 1]
        distribution = (means[:, method] - means[:, baseline]) * [100, 1]
        lo, hi = np.percentile(distribution, [2.5, 97.5], axis=0)
        contrast = observations[:, :, method, 0] - observations[:, :, baseline, 0]
        contrasts[f"{ARMS[method]}_minus_{ARMS[baseline]}"] = dict(
            success_difference_pp=round(float(difference[0]), 6),
            success_ci95_pp=[round(float(bound), 6) for bound in (lo[0], hi[0])],
            episode_ms_difference=round(float(difference[1]), 6),
            episode_ms_ci95=[round(float(bound), 6) for bound in (lo[1], hi[1])],
            rescues=int((contrast == 1).sum()), harms=int((contrast == -1).sum()))
    return dict(arms=stats, paired=contrasts, bootstrap=dict(seed=20261126, draws=20000,
                                                            unit="physical layout within task"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((args.data / "object-layouts.json").read_bytes())
    rows = [json.loads(line) for line in (args.data / "episodes.jsonl").read_bytes().splitlines()]
    result = summarize(validate(rows, manifest))
    summary = json.dumps(result, indent=2) + "\n"
    summary_path = args.data / "summary.json"
    if args.verify:
        if summary_path.read_text(encoding="utf-8") != summary:
            raise ValueError("Published summary differs from the paired ledger")
    else:
        summary_path.write_text(summary, encoding="utf-8")
        table("graph-rollout", "Warm CUDA Graph replay on 200 paired LIBERO-Object layouts. "
              "Core timing includes encoding, graph input refresh, denoising and any critic pass. "
              "Startup, observation preprocessing, action postprocessing and simulation are excluded.",
              "Schedule & Success & Queries & Bo4 calls & Core ms/episode & Decision p50/p90 (ms)",
              "lrrrrr", [[name, f'{data["successes"]}/200', f'{data["queries"]:,}',
                          f'{data["four_way_calls"]:,}', f'{data["mean_episode_ms"]:,.1f}',
                          f'{data["decision_p50_ms"]:.1f}/{data["decision_p90_ms"]:.1f}']
                         for name, data in zip(LABELS, result["arms"].values())])
    print(summary, end="")


if __name__ == "__main__":
    main()
