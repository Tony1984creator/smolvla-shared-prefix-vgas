"""Draw the measured success and realized extra-proposal trade-off."""
import json
from pathlib import Path

from figstyle import COLORS, WIDTH, grid, plt, save

ROOT = Path(__file__).resolve().parents[1]
CURVE = ("B-BC", "front3", "front5", "early", "front10", "B-Bo4")
ANNOTATIONS = {
    "B-BC": (5, 10, "left"), "front3": (-6, 7, "right"), "front5": (-4, 12, "right"),
    "early": (2, 12, "left"), "front10": (1, -14, "center"), "B-Bo4": (-1, 13, "center"),
    "late": (6, -7, "left"), "random": (6, 6, "left"), "bo2-full": (6, -5, "left"),
    "bo8-front3": (6, 5, "left"), "bo2-front10": (-6, -8, "right"),
}
LABELS = {
    "B-BC": "Single", "front3": "K=3", "front5": "EWS-5",
    "early": "K=7", "front10": "K=10", "B-Bo4": "Full Bo4",
    "late": "Late", "random": "Random 7", "bo2-full": "Full Bo2",
    "bo8-front3": "Bo8 × 3", "bo2-front10": "Bo2 × 10",
}


def draw():
    data = json.loads((ROOT / "results/width-time/summary.json").read_bytes())
    rows = data["schedules"]
    ews = 100 * rows["front5"]["wins"] / rows["front5"]["episodes"]

    def anchor(row):
        return (row["mean_extra_candidates"], 100 * row["wins"] / row["episodes"])

    fig, ax = plt.subplots(figsize=(WIDTH, 2.43))
    fig.subplots_adjust(left=.115, right=.973, top=.80, bottom=.20)
    ax.plot(*zip(*(anchor(rows[arm]) for arm in CURVE)), color=COLORS["dense"], alpha=.55,
            linewidth=.9, label="Bo4 prefix sweep", zorder=1)
    for arm, row in rows.items():
        x, y = anchor(row)
        interval = row["paired_vs_ews"]["ci95_pp"]
        color = COLORS["dense"] if arm in CURVE else COLORS["green"]
        marker = "o"
        if arm in ("bo2-front10", "bo2-full", "bo8-front3"):
            marker = {"bo2-front10": "v", "bo2-full": "^", "bo8-front3": "P"}[arm]
        if arm in ("late", "random"):
            marker = {"late": "s", "random": "x"}[arm]
            color = COLORS["late"]
        if arm in ("front5", "B-BC"):
            marker, color = (("D", COLORS["front"]) if arm == "front5" else ("o", COLORS["single"]))
        ax.errorbar(x, y, yerr=[[max(0, y - (ews + interval[0]))],
                               [max(0, ews + interval[1] - y)]],
                    color=color, fmt=marker, capsize=2.2, elinewidth=.72, ms=4,
                    alpha=.85, zorder=3)
        dx, dy, horizontal_alignment = ANNOTATIONS[arm]
        ax.annotate(LABELS[arm], (x, y), xytext=(dx, dy), textcoords="offset points",
                    ha=horizontal_alignment, va="center", color=color, fontsize=7)
    ax.axhline(ews, color=COLORS["front"], ls=":", lw=.7, alpha=.65)
    ax.set(xlabel="Realized extra candidates per episode", ylabel="Task success (%)")
    ax.set_xlim(-2.3, max(anchor(row)[0] for row in rows.values()) + 4)
    lows = [ews + row["paired_vs_ews"]["ci95_pp"][0] for row in rows.values()]
    highs = [ews + row["paired_vs_ews"]["ci95_pp"][1] for row in rows.values()]
    ax.set_ylim(min(lows) - 2.5, max(highs) + 3)
    grid(ax)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), borderaxespad=0)
    save(fig, "width-time")


if __name__ == "__main__":
    draw()
