"""Rebuild the README figure from the released rows (NumPy and Matplotlib)."""
from collections import defaultdict
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results"


def marginal_rates(rows, arms, rng):
    layouts = defaultdict(dict)
    for arm in arms:
        for row in rows[arm]:
            layouts[row["layout"]].setdefault(arm, []).append(row["success"])
    matrix = np.array([[np.mean(group[arm]) for arm in arms] for group in layouts.values()])
    draw = rng.integers(len(matrix), size=(10000, len(matrix)))
    return matrix.mean(0) * 100, np.percentile(matrix[draw].mean(1) * 100, [2.5, 97.5], axis=0)


def main():
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.8, 3.8), layout="constrained")
    colors = ["#50627c", "#08766c"]
    rng = np.random.default_rng(20261120)
    for i, (filename, arms) in enumerate([
        ("object-paired.json", ("B-BC", "B-Bo4")),
        ("long-task2-paired.json", ("Z", "SharedBo4")),
    ]):
        mean, interval = marginal_rates(json.loads((DATA / filename).read_bytes()), arms, rng)
        for k, label in enumerate(["One proposal", "Shared Bo4"]):
            x = i + (k - .5) * .32
            left.bar(x, mean[k], width=.29, color=colors[k], label=label if i == 0 else None)
            left.errorbar(x, mean[k], yerr=[[mean[k]-interval[0,k]], [interval[1,k]-mean[k]]],
                          color="#23313c", capsize=3, lw=1.4)
            left.text(x, 3, f"{mean[k]:.1f}%", ha="center", color="white", fontsize=9)
    left.set(xticks=[0, 1], xticklabels=["Object · 10 tasks", "Long · Task2"], ylim=(0, 77),
             ylabel="Episode success (%)", title="Closed-loop task outcomes")
    with (DATA / "fixed-observation-latency.csv").open() as handle:
        records = list(csv.DictReader(handle))
    for key, label, color in zip(["original", "shared"], ["Original VGAS", "Shared prefix"], colors):
        rows = [r for r in records if r["implementation"] == key]
        right.plot(range(len(rows)), [float(r["median_ms"]) for r in rows], "o-",
                   color=color, lw=2, label=label)
    right.set(xticks=range(5), xticklabels=["1","2","4","8","16"], ylim=(195, 356),
              xlabel="Candidates per decision", ylabel="Median decision time (ms)",
              title="Fixed observation · A100")
    for ax in [left, right]:
        ax.legend(frameon=False, ncol=2, loc="upper left", fontsize=9)
        ax.spines[["right", "top"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=.15)
    fig.savefig(ROOT / "docs/media/decision-cost.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
