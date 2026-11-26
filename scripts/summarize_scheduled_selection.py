"""Rebuild the scheduled-selection table, paired statistics and dose figure."""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from reportlab.lib.colors import HexColor
from reportlab.pdfgen.canvas import Canvas

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


def draw_curve(comparisons):
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
    # Draw directly to vector PDF so text and error bars stay sharp in print.
    pdf = Canvas(str(PAPER / "figures/scheduled-dose.pdf"), pagesize=(256, 172), invariant=True)
    left, bottom, width, height = 39, 30, 208, 132

    def x(value):
        return left + width * (value + 0.5) / 15

    lower = min(-2, int(np.floor(late["ci"][0])))
    upper = max(16, int(np.ceil(late["ci"][1])))

    def y(value):
        return bottom + height * (value - lower) / (upper - lower)

    def segment(x1, y1, x2, y2):
        pdf.line(x(x1), y(y1), x(x2), y(y2))

    def stroke(color, weight=0.8):
        pdf.setStrokeColor(HexColor(color))
        pdf.setLineWidth(weight)

    def marker(slot, value, square=False):
        if square:
            pdf.rect(x(slot) - 2, y(value) - 2, 4, 4, fill=1)
        else:
            pdf.circle(x(slot), y(value), 2, fill=1)

    def interval(slot, low, high):
        segment(slot, low, slot, high)
        for end in (low, high):
            pdf.line(x(slot) - 2.2, y(end), x(slot) + 2.2, y(end))

    pdf.setFont("Helvetica", 7)
    for tick in (lower, 0, 4, 8, 12, 16):
        stroke("#bfc4c9" if tick else "#858d94", 0.4)
        segment(-0.5, tick, 14.5, tick)
        pdf.drawRightString(left - 4, y(tick) - 2.3, str(tick))
    stroke("#172531")
    segment(-0.5, lower, 14.5, lower)
    segment(-0.5, lower, -0.5, upper)
    for slot in sorted(set(SWEEP) | {9}):
        pdf.line(x(slot), bottom, x(slot), bottom - 2.5)
        pdf.drawCentredString(x(slot), bottom - 10, str(slot))
    pdf.drawCentredString(left + width / 2, 6, "Planned four-way queries per episode")
    pdf.saveState()
    pdf.translate(9, bottom + height / 2)
    pdf.rotate(90)
    pdf.drawCentredString(0, 0, "Gain vs. single proposal (pp)")
    pdf.restoreState()

    stroke("#69737a")
    pdf.setDash(3, 2)
    segment(0, 0, 14, dense)
    pdf.setDash()
    stroke("#22577a")
    pdf.setFillColor(HexColor("#22577a"))
    for (_, a, start), (_, b, end) in zip(curve, curve[1:]):
        segment(a, start["pp"], b, end["pp"])
    for _, slot, effect in curve:
        interval(slot, *effect["ci"])
        marker(slot, effect["pp"])
    stroke("#b24c30")
    pdf.setFillColor(HexColor("#ffffff"))
    interval(7, *random["ci"])
    marker(7, random["pp"], square=True)

    stroke("#56713b")
    pdf.setFillColor(HexColor("#56713b"))
    interval(9, *late["ci"])
    marker(9, late["pp"], square=True)

    legend = (("First K", "#22577a"), ("Random seven", "#b24c30"),
              ("Late (from query 6)", "#56713b"), ("Linear reference", "#69737a"))
    for i, (label, color) in enumerate(legend):
        row_y = 157 - i * 9
        stroke(color)
        pdf.setFillColor(HexColor("#ffffff" if i == 1 else color))
        if i in (1, 2):
            pdf.rect(left + 7, row_y - 1, 4, 4, fill=1)
        else:
            if i == 3:
                pdf.setDash(3, 2)
            pdf.line(left + 3, row_y + 1, left + 15, row_y + 1)
            pdf.setDash()
            if i == 0:
                pdf.circle(left + 9, row_y + 1, 1.8, fill=1)
        pdf.setFillColor(HexColor("#172531"))
        pdf.drawString(left + 18, row_y - 1.5, label)
    pdf.save()


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
            draw_curve(comparisons)
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
