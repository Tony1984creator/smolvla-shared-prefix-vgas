"""Plot paired success and inference-time differences from the released Graph study."""
import json
from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.pdfgen.canvas import Canvas

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/graph-closed-loop"
INK = HexColor("#22262b")
GRAY = HexColor("#626a74")
RULE = HexColor("#a8afb6")
BLUE = HexColor("#234d73")


def panel(c, x, title, xlabel, values, values_ci, is_time):
    c.setFillColor(INK)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(x, 119, title)
    limit = max(abs(bound) for ci in values_ci for bound in ci)
    if limit <= 0:
        limit = 1
    tick = next(step for step in ([25, 50, 100, 200, 400, 800, 1600, 3200, 6400] if is_time
                                  else [2, 5, 10, 20, 40, 80, 100]) if step * 2 >= limit)
    low, high = -2.25 * tick, 2.25 * tick
    origin = x + 69
    width = 190

    def coord(value):
        return origin + width * (value - low) / (high - low)

    c.setLineWidth(.65)
    c.setStrokeColor(RULE)
    c.line(origin, 37, origin + width, 37)
    c.setDash(2, 2)
    c.line(coord(0), 37, coord(0), 104)
    c.setDash()
    c.setFont("Helvetica", 8)
    c.setFillColor(GRAY)
    for value in range(-2 * tick, 2 * tick + 1, tick):
        pos = coord(value)
        c.line(pos, 34, pos, 37)
        c.drawCentredString(pos, 23, f"{value:+d}" if value else "0")
    c.drawCentredString(origin + width / 2, 9, xlabel)
    for (value, (lo, hi)), label, y in zip(zip(values, values_ci), ("Single", "EWS-5"), (85, 57)):
        color = BLUE if label == "EWS-5" else GRAY
        c.setFillColor(INK)
        c.drawString(x, y - 2.5, label)
        c.setFillColor(color)
        c.setStrokeColor(color)
        c.setLineWidth(1.1)
        c.line(coord(lo), y, coord(hi), y)
        for bound in (lo, hi):
            c.line(coord(bound), y - 3, coord(bound), y + 3)
        c.circle(coord(value), y, 2.9, stroke=0, fill=1)


def main():
    data = json.loads((DATA / "summary.json").read_bytes())["paired"]
    d, f = data["dense_minus_single"], data["front5_minus_dense"]
    c = Canvas(str(ROOT / "paper/figures/graph-closed-loop.pdf"), pagesize=(576, 137), invariant=True)
    c.setTitle("Paired differences with warm CUDA Graph replay")
    for x, title, xlabel, estimate, interval, is_time in [
        (6, "(a)  Task success", "Difference from full Bo4 (pp)", "success_difference_pp", "success_ci95_pp", False),
        (303, "(b)  Core inference per episode", "Difference from full Bo4 (ms)", "episode_ms_difference",
         "episode_ms_ci95", True),
    ]:
        panel(c, x, title, xlabel, [-d[estimate], f[estimate]],
              [[-d[interval][1], -d[interval][0]], f[interval]], is_time)
    c.showPage()
    c.save()


if __name__ == "__main__":
    main()
