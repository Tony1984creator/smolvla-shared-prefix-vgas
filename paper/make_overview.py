"""Draw the inference overview with the archived Object query and vector geometry."""
import json
from pathlib import Path

import numpy as np
from PIL import Image
from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen.canvas import Canvas

PAPER = Path(__file__).resolve().parent
INK, GRAY, RULE = "#172b39", "#495e69", "#c7d1d4"
TEAL, BLUE, GOLD = "#006e65", "#315a80", "#96581b"


def draw():
    c = Canvas(str(PAPER / "figures/method-overview.pdf"),
               pagesize=(704, 236), invariant=1)
    c.setTitle("Shared-prefix inference and action horizons")

    def text(x, y, label, size=10, color=INK, bold=False):
        c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
        c.setFillColor(HexColor(color))
        c.drawString(x, y, label)

    def line(x, y, end, height, color=RULE, width=1):
        c.setStrokeColor(HexColor(color))
        c.setLineWidth(width)
        c.line(x, y, end, height)

    def box(x, y, w, h, fill, edge):
        c.setStrokeColor(HexColor(edge))
        c.setFillColor(HexColor(fill))
        c.setLineWidth(.8)
        c.roundRect(x, y, w, h, 5, stroke=1, fill=1)

    def arrow(x, y, end, color=GRAY):
        line(x, y, end, y, color)
        line(end-4, y+2.6, end, y, color)
        line(end-4, y-2.6, end, y, color)

    text(4, 221, "a", 15, bold=True)
    text(24, 221, "One decision from the current observation", 14, bold=True)
    text(4, 194, "Observation", 12, bold=True)
    with np.load(PAPER.parent / "examples/object_query/observation.npz") as obs:
        for x, key in [(4, "image"), (56, "image2")]:
            c.drawImage(ImageReader(Image.fromarray(obs[key])), x, 134, 50, 50)
    text(4, 119, "Soup can into basket", 9)
    text(4, 106, "+ robot state", 9, GRAY)
    arrow(110, 157, 124)
    box(127, 124, 114, 62, "#e0f1ec", TEAL)
    text(137, 168, "Encode once", 14, TEAL, True)
    text(137, 151, "Shared observation", 10)
    text(137, 137, "prefix", 10)
    arrow(241, 157, 285, TEAL)
    text(243, 166, "KV cache", 8, TEAL)
    # The critic consumes features from the same pass, alongside each chunk.
    line(185, 124, 185, 102, TEAL)
    line(185, 102, 526, 102, TEAL)
    line(526, 102, 526, 124, TEAL)
    line(523, 119, 526, 124, TEAL)
    line(529, 119, 526, 124, TEAL)
    text(287, 106, "Shared observation features", 9, TEAL)
    text(289, 195, "Frozen SmolVLA actor", 12, BLUE, True)
    text(289, 181, "Four independent noise draws", 9, GRAY)
    replay = json.loads((PAPER.parent / "results/value-token-replay.json").read_bytes())
    query = next(r for r in replay["rows"] if r["task"] == 0 and r["frame"] == "query000")
    winner = query["full_winner"]
    for i in range(4):
        y = 168 - 12*i
        text(289, y-3, f"{i}", 8, BLUE)
        for j in range(10):
            c.setFillColor(HexColor(GOLD if i == winner else "#abc0d2"))
            c.roundRect(303+j*12, y-3.5, 10, 7, 1.5, stroke=0, fill=1)
    arrow(428, 151, 455)
    box(459, 124, 137, 62, "#edf2f7", BLUE)
    text(459, 195, "Frozen VGAS critic", 12, BLUE, True)
    for i, score in enumerate(query["full_scores"]):
        y = 168 - 12*i
        text(469, y-3, f"Q({i})", 9, GRAY)
        text(501, y-3, f"{score:.3f}", 10, GOLD if i == winner else INK, i == winner)
    arrow(598, 145, 616, GOLD)
    text(620, 158, "Select", 12, bold=True)
    text(620, 142, f"chunk {winner}", 12, GOLD, True)
    line(4, 91, 700, 91)
    text(4, 72, "b", 15, bold=True)
    text(24, 72, "Action horizon", 14, bold=True)
    start, span = 176, 310
    for y, label, n, color in [(54,"Generate",50,BLUE),(39,"Score",32,BLUE),(24,"Score",50,BLUE),(9,"Execute",20,GOLD)]:
        text(24,y-3,label,10,color,True)
        text(80,y-3, {54:"Both suites",39:"Object",24:"Long Task2",9:"Both suites"}[y],9,GRAY)
        line(start,y,start+span,y,"#e1e6e9",4)
        line(start,y,start+span*n/50,y,color,4)
        text(start+span+12,y-3,"up to 20 steps" if y==9 else f"{n} steps",9,color)
    text(585, 28, "Next observation", 11, TEAL, True)
    text(585, 13, "if still running", 10, GRAY)
    arrow(563, 11, 579, TEAL)
    c.showPage()
    c.save()


if __name__ == "__main__":
    draw()
