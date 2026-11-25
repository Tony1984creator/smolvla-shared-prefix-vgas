"""Summarize paired increments between frozen Object prefix schedules."""

import csv
import json
from itertools import pairwise

import numpy as np
from reportlab.lib.colors import HexColor
from reportlab.pdfgen.canvas import Canvas

from summarize_scheduled_selection import DATA, PAPER, SWEEP, paired, read_cohort


def draw(segments):
    pdf = Canvas(str(PAPER / "figures/marginal-value.pdf"), pagesize=(500, 166), invariant=True)
    blue = HexColor("#22577a")
    grid = HexColor("#d8dde0")
    ink = HexColor("#172531")
    pdf.setFont("Helvetica", 9)
    labels = [f'{r["first_query"]}–{r["last_query"]}' for r in segments]
    for panel, left, right, low, high, ticks in (
        ("A   Increment per block (pp)", 27, 237, -5, 12, (-4, 0, 4, 8, 12)),
        ("B   Cumulative gain captured (%)", 270, 480, 0, 115, (0, 25, 50, 75, 100)),
    ):
        bottom, top = 35, 130
        pdf.setFillColor(ink)
        pdf.drawString(left, 148, panel)

        def y(v):
            return bottom + (top - bottom) * (v - low) / (high - low)

        for tick in ticks:
            pdf.setStrokeColor(ink if tick in (0, 100) else grid)
            pdf.setLineWidth(0.55)
            pdf.line(left, y(tick), right, y(tick))
            pdf.drawRightString(left - 4, y(tick) - 3, str(tick))
        xs = [left + (right - left) * (i + 0.5) / len(segments) for i in range(len(segments))]
        for x, label in zip(xs, labels):
            pdf.drawCentredString(x, 20, label)
        pdf.drawCentredString((left + right) / 2, 6, "Query block")
        pdf.setStrokeColor(blue)
        pdf.setFillColor(blue)
        if left == 27:
            for x, r in zip(xs, segments):
                height = y(r["delta_pp"]) - y(0)
                if height:
                    pdf.rect(x - 9, min(y(0), y(r["delta_pp"])), 18, abs(height), fill=1, stroke=0)
                lo, hi = map(y, (r["ci95_low_pp"], r["ci95_high_pp"]))
                pdf.setStrokeColor(ink)
                pdf.setLineWidth(1)
                pdf.line(x, lo, x, hi)
                for end in (lo, hi):
                    pdf.line(x - 3, end, x + 3, end)
                if lo == hi:
                    pdf.circle(x, lo, 1.5, fill=1, stroke=0)
        else:
            points = [(x, y(r["capture_pct"])) for x, r in zip(xs, segments)]
            pdf.setLineWidth(1.4)
            for (x1, y1), (x2, y2) in pairwise(points):
                pdf.line(x1, y1, x2, y2)
            for x, yy in points:
                pdf.circle(x, yy, 2.6, fill=1, stroke=0)
    pdf.save()


def main():
    widths = {arm: k for k, arm in SWEEP.items()} | {"random": 7, "late": 9}
    outcomes, counts = read_cohort("object-sweep", widths, 10, 30, 1, 14)
    full_gain = counts["B-Bo4"]["wins"] - counts["B-BC"]["wins"]
    segments = []
    for (lo, prior), (hi, arm) in pairwise(SWEEP.items()):
        effect = paired(outcomes[arm] - outcomes[prior], 10, 30, 1, 20261122, 10000)
        delta = counts[arm]["wins"] - counts[prior]["wins"]
        segments.append(dict(
            first_query=lo+1, last_query=hi, prior_cutoff=lo, cutoff=hi,
            delta_wins=delta, delta_pp=effect["pp"], pp_per_planned_slot=effect["pp"]/(hi-lo),
            ci95_low_pp=effect["ci"][0], ci95_high_pp=effect["ci"][1],
            rescues=effect["rescues"], harms=effect["harms"],
            cumulative_gain_pp=100*(counts[arm]["wins"]-counts["B-BC"]["wins"])/300,
            capture_pct=100*(counts[arm]["wins"]-counts["B-BC"]["wins"])/full_gain,
            four_way_calls=counts[arm]["calls"],
            full_call_share_pct=100*counts[arm]["calls"]/counts["B-Bo4"]["calls"],
        ))
    summary = json.loads((DATA / "summary.json").read_bytes())
    goal = json.loads((DATA / "goal-window.json").read_bytes())
    controls = {
        "object_calibration_late_vs_single": summary["object-sweep"]["paired"]["late"]["single"],
        "object_fresh_late_vs_single": summary["object-fresh"]["paired"]["late"]["single"],
        "goal_late_vs_single": goal["paired"]["late_vs_single"],
        "spatial_late_vs_single": summary["spatial-window"]["paired"]["late"]["single"],
        "object_random_seven_vs_single": summary["object-sweep"]["paired"]["random"]["single"],
        "object_first_seven_vs_random_seven": summary["object-sweep"]["first_seven_vs_random"],
    }
    controls["object_random_seven_gain_pp_per_planned_slot"] = controls["object_random_seven_vs_single"]["pp"]/7
    result = dict(episodes=300, bootstrap_draws=10000, bootstrap_seed=20261122,
                  full_gain_wins=full_gain, full_four_way_calls=counts["B-Bo4"]["calls"],
                  block_point_estimates_nonincreasing=all(
                      a["delta_pp"] >= b["delta_pp"] for a,b in pairwise(segments)),
                  slot_normalized_point_estimates_nonincreasing=all(
                      a["pp_per_planned_slot"] >= b["pp_per_planned_slot"] for a,b in pairwise(segments)),
                  segments=segments, controls=controls)
    (DATA / "marginal-value.json").write_text(json.dumps(result, indent=2) + "\n", encoding="ascii")
    with (DATA / "marginal-value.csv").open("w", encoding="ascii", newline="") as stream:
        writer = csv.DictWriter(stream, segments[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(segments)
    draw(segments)


if __name__ == "__main__":
    main()
