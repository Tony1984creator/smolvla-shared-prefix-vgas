"""Rebuild the scheduled-selection table, paired statistics and dose ledger."""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/scheduled_selection"
PAPER = ROOT / "paper"
SWEEP = {0: "B-BC", 3: "front3", 5: "front5", 7: "early", 10: "front10", 14: "B-Bo4"}


def read_rows(name):
    with (DATA / f"{name}.csv").open(newline="") as stream:
        yield from csv.DictReader(stream)
    if name == "object-sweep":
        with (DATA / "late-calibration.csv").open(newline="") as stream:
            yield from csv.DictReader(stream)
    if name == "object-fresh":
        with (DATA / "late-object.csv").open(newline="") as stream:
            for row in csv.DictReader(stream):
                if row["arm"] == "late":
                    yield dict(row, layout=row["case"], repeat="0")


def read_cohort(name, widths, tasks, layouts, repeats, horizon):
    records = defaultdict(dict)
    episode_times = defaultdict(list)
    for row in read_rows(name):
        task, layout, arm = (row[k] for k in ("task", "layout", "arm"))
        key = task, int(layout) if repeats > 1 else layout, int(row["repeat"])
        assert arm in widths and arm not in records[key]
        win, queries, calls, chunks = (int(row[k]) for k in ("success", "queries", "calls", "candidates"))
        limit = min(widths[arm], queries)
        assert win in (0, 1) and 1 <= queries <= horizon and chunks == queries + 3 * calls
        if arm == "random":
            assert 0 <= calls <= limit
        elif arm == "late":
            assert calls == max(queries - 5, 0)
        else:
            assert calls == limit
        records[key][arm] = (win, queries, calls, chunks)
        if "inference_ms" in row:
            episode_times[arm].append(float(row["inference_ms"]))
    grouped = defaultdict(list)
    keys = sorted(records)
    for task, layout, repeat in keys:
        grouped[task, layout].append(repeat)
    assert len({key[0] for key in grouped}) == tasks and len(grouped) == tasks * layouts
    assert all(sum(key[0] == task for key in grouped) == layouts for task, _, _ in keys)
    assert all(ids == list(range(repeats)) for ids in grouped.values())
    assert all(set(row) == set(widths) for row in records.values())
    arrays = {arm: np.array([records[k][arm] for k in keys]) for arm in widths}
    outcomes = {arm: array[:, 0] for arm, array in arrays.items()}
    totals = {arm: dict(zip(("wins", "queries", "calls", "chunks"), map(int, array.sum(axis=0))))
              for arm, array in arrays.items()}
    for arm, times in episode_times.items():
        assert len(times) == len(keys) and np.isfinite(times).all() and min(times) > 0
        totals[arm]["mean_inference_ms"] = float(np.mean(times))
    return outcomes, totals


def paired(delta, tasks, layouts, repeats, seed, draws):
    rng = np.random.default_rng(seed)
    groups = delta.reshape(tasks, layouts, repeats).mean(axis=2)
    boot = [group[rng.integers(layouts, size=(draws, layouts))].mean(axis=1) for group in groups]
    return {"pp": float(100 * delta.mean()),
            "ci": (100 * np.quantile(np.mean(boot, axis=0), [.025, .975])).tolist(),
            "rescues": int((delta == 1).sum()), "harms": int((delta == -1).sum())}


def pair_text(pair):
    lo, hi = (round(value, 2) or 0.0 for value in pair["ci"])
    return "$" + f'{pair["pp"]:+.2f}' + f"$ $[{lo:.2f}, {hi:.2f}]$"


def write_curve(comparisons):
    curve = [(arm, k, comparisons[arm]["single"]) for k, arm in SWEEP.items()]
    random = comparisons["random"]["single"]
    late = comparisons["late"]["single"]
    dense = comparisons["B-Bo4"]["single"]["pp"]
    with (DATA / "curve.csv").open("w", newline="", encoding="ascii") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("schedule", "planned_slots", "gain_pp", "ci_low_pp", "ci_high_pp", "linear_gain_pp"))
        for arm, k, effect in curve + [("random", 7, random), ("late", 9, late)]:
            writer.writerow([arm, k, f'{effect["pp"]:.2f}', *(f"{v:.2f}" for v in effect["ci"]),
                             "" if arm in ("random", "late") else f"{k * dense / 14:.2f}"])


def main():
    configs = [
        ("object-sweep", "Object sweep", {arm: k for k, arm in SWEEP.items()} | {"random": 7, "late": 9},
         10, 30, 1, 14, 20261122, 10000, "B-BC", "B-Bo4", {arm: f"First {k}" for k, arm in SWEEP.items()}
         | {"B-BC": "Single", "B-Bo4": "Full Bo4", "random": "Random seven", "late": "Late"}),
        ("object-fresh", "Object fresh", {"single": 0, "dense": 14, "front": 5, "late": 9},
         10, 20, 1, 14, 20261123, 10000, "single", "dense",
         {"single": "Single", "dense": "Full Bo4", "front": "First five", "late": "Late"}),
        ("long-task2", "Long Task2", {"Z": 0, "SharedBo4": 26, "front8": 8},
         1, 128, 2, 26, 42, 20000, "Z", "SharedBo4",
         {"Z": "Single", "SharedBo4": "Full Bo4", "front8": "First eight"}),
        ("spatial-window", "Spatial", {"single": 0, "dense": 14, "front": 5, "late": 9},
         10, 30, 1, 14, 20261124, 20000, "single", "dense",
         {"single": "Single", "dense": "Full Bo4", "front": "First five", "late": "Late"}),
        ("plus-light-window", "Plus lighting", {"single": 0, "dense": 14, "front": 5},
         10, 30, 1, 14, 20261124, 20000, "single", "dense",
         {"single": "Single", "dense": "Full Bo4", "front": "First five"}),
    ]
    lines = [r"\begin{table*}[t]", r"\centering\small\setlength{\tabcolsep}{3.5pt}",
             r"\caption{Episode-level selection. Calls count four-way scoring batches. Chunks include single proposals. "
             r"Paired 95\% percentile intervals resample within task; Long resamples physical layouts "
             r"with both noise repeats together. The combined Object contrast resamples within cohort and task. "
             r"Times shown for the two new cohorts are mean A100 core ms per episode. See Table~\ref{tab:episode-cost} for the full timing comparison.}",
             r"\label{tab:scheduled}",
             r"\begin{tabular}{@{}llrrrrll@{}}\toprule",
             r"Cohort & Schedule & Wins & Calls & Chunks & ms/episode & vs.\ single (pp) & vs.\ full Bo4 (pp) \\ \midrule"]
    summaries = {}
    timing_deltas = {}
    for name, label, widths, tasks, layouts, repeats, horizon, seed, draws, single, dense, labels in configs:
        outcomes, totals = read_cohort(name, widths, tasks, layouts, repeats, horizon)
        comparisons = {}
        for index, arm in enumerate(widths):
            vs_single, vs_dense = (paired(outcomes[arm]-outcomes[ref], tasks, layouts, repeats, seed, draws)
                                   for ref in (single, dense))
            comparisons[arm] = {"single": vs_single, "dense": vs_dense}
            row = totals[arm]
            cohort = f"{label} ({tasks * layouts * repeats})" if index == 0 else ""
            bc_text = "--" if arm == single else pair_text(vs_single)
            dense_text = "--" if arm in (dense, "random") else pair_text(vs_dense)
            time = f'{row["mean_inference_ms"]:,.2f}' if "mean_inference_ms" in row else "--"
            lines.append(f'{cohort} & {labels[arm]} & {row["wins"]} & {row["calls"]:,} & {row["chunks"]:,} & '
                         f"{time} & {bc_text} & {dense_text} " + r"\\")
        lines.append(r"\addlinespace")
        summaries[name] = {"totals": totals, "paired": comparisons}
        if name in ("object-sweep", "object-fresh"):
            early = "front5" if name == "object-sweep" else "front"
            delta = outcomes[early] - outcomes["late"]
            timing_deltas[name] = delta.reshape(tasks, layouts)
            summaries[name]["first_five_vs_late"] = paired(delta, tasks, layouts, repeats, seed, draws)
        if name == "object-sweep":
            summaries[name]["first_seven_vs_random"] = paired(
                outcomes["early"] - outcomes["random"], tasks, layouts, repeats, seed, draws)
            summaries[name]["first_ten_dense_agreement"] = int(np.sum(outcomes["front10"] == outcomes[dense]))
            summaries[name]["minimum_cutoff"] = min(k for k, arm in SWEEP.items()
                                                     if abs(comparisons[arm]["dense"]["pp"]) <= 3)
            write_curve(comparisons)
    # One episode per physical layout, resampled within each task and cohort.
    rng = np.random.default_rng(20261124)
    strata = [row for matrix in timing_deltas.values() for row in matrix]
    delta = np.concatenate(strata)
    boot = sum(row[rng.integers(len(row), size=(20000, len(row)))].sum(axis=1) for row in strata)
    combined = {"pp": float(100 * delta.mean()),
                "ci": (100 * np.quantile(boot / len(delta), [.025, .975])).tolist(),
                "rescues": int((delta == 1).sum()), "harms": int((delta == -1).sum())}
    summaries["object-combined"] = {"first_five_vs_late": combined}
    comparisons = [(label, summaries[name]["first_five_vs_late"]) for label, name in
                   (("Object sweep (300)", "object-sweep"), ("Object fresh (200)", "object-fresh"),
                    ("Combined Object (500)", "object-combined"))]
    lines[-1] = r"\midrule"
    for label, outcome in comparisons:
        lines.append(label + r" & \multicolumn{5}{l}{First five vs. Late (pp)} & \multicolumn{2}{l}{"
                     + pair_text(outcome) + r"} \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (PAPER / "tables/scheduled-selection.tex").write_text("\n".join(lines)+"\n", encoding="ascii")
    (DATA / "summary.json").write_text(json.dumps(summaries, indent=2)+"\n", encoding="ascii")


if __name__ == "__main__":
    main()
