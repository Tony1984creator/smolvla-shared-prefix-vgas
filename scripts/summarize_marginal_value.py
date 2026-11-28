"""Summarize paired increments between frozen Object prefix schedules."""

import csv
import json
from itertools import pairwise
from pathlib import Path
import sys

import numpy as np

from summarize_scheduled_selection import DATA, SWEEP, paired, read_cohort


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
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paper"))
    from make_submission import marginal

    marginal()


if __name__ == "__main__":
    main()
