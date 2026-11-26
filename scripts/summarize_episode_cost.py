"""Rebuild the A100 inference-per-episode table from archived query-time sums."""
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/scheduled_selection"
GENERALIZATION = {"LIBERO-Spatial": "spatial-window.csv", "Plus lighting": "plus-light-window.csv"}
groups = defaultdict(dict)
with (DATA / "episode-cost.csv").open(newline="") as stream:
    for row in csv.DictReader(stream):
        group = groups[row["cohort"], row["arm"]]
        assert row["case"] not in group
        group[row["case"]] = row
for cohort, filename in GENERALIZATION.items():
    with (DATA / filename).open(newline="") as stream:
        for row in csv.DictReader(stream):
            group = groups[cohort, row["arm"]]
            assert row["layout"] not in group
            group[row["layout"]] = row

labels = {"single":"Single", "original":"Original Bo4", "dense":"Shared Bo4",
          "front5":"EWS-5", "front8":"EWS-8", "front":"EWS-5", "late":"Late-5"}
lines = [r"\begin{table*}[t]", r"\centering\small",
         r"\caption{Measured inference work per episode on A100. We sum the core times of all queries in an episode and average over episodes. The timer covers encoding, generation and scoring where applicable. Environment execution and action postprocessing are outside the timer. Calls are cohort totals.}",
         r"\label{tab:episode-cost}",
         r"\begin{tabular}{@{}llrrr@{}}\toprule",
         r"Cohort & Configuration & Inference ms/episode & Successes & Four-way calls \\ \midrule"]
for cohort, count, arms in [("Object calibration", 300, ("single","original","dense","front5")),
                            ("Object fresh", 200, ("single","dense","front5")),
                            ("Long Task2", 256, ("single","dense","front8")),
                            ("LIBERO-Goal", 300, ("single","dense","front","late")),
                            ("LIBERO-Spatial", 300, ("single","dense","front","late")),
                            ("Plus lighting", 300, ("single","dense","front"))]:
    for i, arm in enumerate(arms):
        records = groups[cohort,arm]
        assert len(records) == count and records.keys() == groups[cohort,"single"].keys()
        wins = sum(int(row["success"]) for row in records.values())
        calls = sum(int(row["calls"]) for row in records.values())
        elapsed = mean(float(row["inference_ms"]) for row in records.values())
        lines.append(f"{cohort if i==0 else ''} & {labels[arm]} & {elapsed:,.2f} & {wins}/{count} & {calls:,} " + r"\\")
    lines.append(r"\addlinespace")
lines[-1:] = [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
(ROOT / "paper/tables/episode-cost.tex").write_text("\n".join(lines)+"\n",encoding="ascii")
