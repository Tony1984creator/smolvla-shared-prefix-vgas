"""Verify summary arithmetic from the exported paired outcomes using Python 3."""
import csv
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1] / "results"


def load(name):
    return json.loads((ROOT / (name + ".json")).read_bytes())


def paired(left, right, expected):
    a, b = ({(r["layout"], r["noise"]): r["success"] for r in rows}
            for rows in (left, right))
    if len(a) != len(left) or len(b) != len(right) or a.keys() != b.keys():
        raise ValueError("Incomplete or duplicate paired keys")
    observed = (sum(a.values()), sum(b.values()), len(a))
    if observed != expected:
        raise ValueError(f"Observed {observed}, expected {expected}")
    rescues = sum(not a[key] and b[key] for key in a)
    harms = sum(a[key] and not b[key] for key in a)
    difference = (observed[1] - observed[0]) * 100 / len(a)
    print(f"{observed[0]}/{len(a)} -> {observed[1]}/{len(a)} | {difference:+.2f} pp | rescue/harm {rescues}/{harms}")


def verify_width_time():
    data = load("width-time/summary")["schedules"]
    layouts = {row["case"]: row for row in load("width-time/layouts")}
    with (ROOT / "scheduled_selection/object-sweep.csv").open(newline="") as stream:
        older = {r["layout"]: int(r["success"]) for r in csv.DictReader(stream) if r["arm"] == "front5"}
    if layouts.keys() != older.keys() or len(layouts) != 300:
        raise ValueError("Width/time comparison uses different calibration layouts")
    groups = {arm: {} for arm in ("bo2-full", "bo2-front10", "bo8-front3")}
    with (ROOT / "width-time/episodes.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            case = row["case"]
            metadata = ("task", "policy_seed", "scene_seed", "physical_sha")
            group = groups[row["arm"]]
            if case in group or any(row[key] != layouts[case][key] for key in metadata):
                raise ValueError(f"Unpaired or duplicate width/time outcome for {case}")
            group[case] = row
    for name, group in groups.items():
        rows = list(group.values())
        if group.keys() != layouts.keys():
            raise ValueError(f"Incomplete width/time arm {name}")
        new, old = sum(r["success"] for r in rows), sum(older.values())
        extra = mean(sum(r["widths"]) - len(r["widths"]) for r in rows)
        expected = data[name]
        delta = expected["paired_vs_ews"]
        if (new != expected["wins"] or old != data["front5"]["wins"]
                or abs(extra - expected["mean_extra_candidates"]) > 1e-9
                or abs(100 * (new - old) / len(rows) - delta["difference_pp"]) > 1e-9):
            raise ValueError(f"Incorrect width/time summary for {name}")
        print(f"{name}: {new}/{len(rows)} | {extra:.2f} extra proposals per episode")


def main():
    obj = load("object-paired")
    long = load("long-task2-paired")
    light = load("plus-light-paired")
    print("Object single vs shared Bo4")
    paired(obj["B-BC"], obj["B-Bo4"], (144, 169, 300))
    print("Object original Bo4 vs shared Bo4")
    paired(obj["B-OrigBo4"], obj["B-Bo4"], (170, 169, 300))
    print("Long Task2 single vs shared Bo4")
    paired(long["Z"], long["SharedBo4"], (93, 124, 256))
    print("Plus lighting single vs shared Bo4")
    paired(light["Plus_light_BC"], light["Plus_light_Bo4"], (100, 123, 300))
    ms = [mean(t for row in obj[a] for t in row["decision_seconds"]) * 1000
          for a in ("B-OrigBo4", "B-Bo4")]
    if [round(t, 2) for t in ms] != [271.64, 216.78]:
        raise ValueError(f"Unexpected A100 core decision latency: {ms}")
    print(f"Object A100 decision mean {ms[0]:.2f} -> {ms[1]:.2f} ms")
    verify_width_time()


if __name__ == "__main__":
    main()
