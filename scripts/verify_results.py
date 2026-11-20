"""Verify summary arithmetic from the exported paired outcomes using Python 3."""
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


if __name__ == "__main__":
    main()
