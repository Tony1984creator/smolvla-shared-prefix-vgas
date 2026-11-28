"""Publication size and visual vocabulary for the NeurIPS preprint."""
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

WIDTH = 5.5
FIGURES = Path(__file__).resolve().parent / "figures"
COLORS = {
    "single": "#626A72", "dense": "#0072B2", "front": "#D55E00",
    "late": "#8055A2", "oracle": "#171A1D", "critic": "#0072B2",
    "uniform": "#626A72", "green": "#00856A", "light": "#E8EDF1",
    "ink": "#202A33", "grid": "#DDE3E8", "pale_blue": "#E7F2F8",
    "pale_orange": "#FCEEE6",
}
LABELS = {"single": "Single", "dense": "Full Bo4", "front": "EWS", "late": "Late"}
MARKERS = {"single": "o", "dense": "s", "front": "D", "late": "^"}

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.55, "axes.edgecolor": COLORS["single"],
    "text.color": COLORS["ink"], "axes.labelcolor": COLORS["ink"],
    "xtick.color": COLORS["ink"], "ytick.color": COLORS["ink"],
    "lines.linewidth": 1.1, "errorbar.capsize": 2.2,
    "pdf.fonttype": 42, "ps.fonttype": 42, "figure.facecolor": "white",
    "savefig.dpi": 300, "legend.frameon": False, "axes.axisbelow": True,
    "mathtext.fontset": "stix",
})


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=COLORS["grid"], linewidth=0.45)
    ax.tick_params(length=2, pad=2, width=0.5)


def panel(ax, letter, x=-0.13, y=1.025, *, transform=None, va="bottom"):
    ax.text(x, y, f"({letter})", transform=ax.transAxes if transform is None else transform,
            va=va, ha="left", fontweight="bold", fontsize=8,
            clip_on=False, gid=f"panel-label-{letter}")


def save(fig, name):
    def check_panel_margins(event):
        margin = event.renderer.points_to_pixels(2)
        labels = fig.findobj(lambda artist: (artist.get_gid() or "").startswith("panel-label-"))
        for label in labels:
            bounds = label.get_window_extent(event.renderer)
            if (bounds.x0 < margin or bounds.y0 < margin
                    or bounds.x1 > fig.bbox.width - margin or bounds.y1 > fig.bbox.height - margin):
                raise ValueError(f"{name}: panel label {label.get_text()} lacks an export margin")

    FIGURES.mkdir(exist_ok=True)
    connection = fig.canvas.mpl_connect("draw_event", check_panel_margins)
    try:
        fig.savefig(FIGURES / f"{name}.pdf", metadata={"CreationDate": None})
        fig.savefig(FIGURES / f"{name}.png", dpi=300)
    finally:
        fig.canvas.mpl_disconnect(connection)
        plt.close(fig)
