"""Render the architecture and early-window schedule as a vector figure."""
from pathlib import Path

import numpy as np
from PIL import Image
from reportlab.lib.colors import HexColor, white
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen.canvas import Canvas

PAPER = Path(__file__).resolve().parent
INK = HexColor("#22262b")
GRAY = HexColor("#505964")
RULE = HexColor("#a8afb6")
BLUE = HexColor("#234d73")
LIGHT_BLUE = HexColor("#eaf0f6")


def draw():
    c = Canvas(str(PAPER / "figures/method-overview.pdf"),
               pagesize=(576, 234), invariant=True)
    c.setTitle("Shared-prefix inference and early-window scheduling")

    def text(x, y, label, size=9, *, bold=False, color=INK, center=False):
        c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
        c.setFillColor(color)
        (c.drawCentredString if center else c.drawString)(x, y, label)

    def line(x, y, end, height, color=RULE, width=.7):
        c.setStrokeColor(color)
        c.setLineWidth(width)
        c.line(x, y, end, height)

    def arrow(x, y, end, height=None, color=GRAY):
        height = y if height is None else height
        line(x, y, end, height, color)
        if x < end:
            line(end-4, height+2.4, end, height, color)
            line(end-4, height-2.4, end, height, color)
        else:
            line(end-2.4, height-4, end, height, color)
            line(end+2.4, height-4, end, height, color)

    def block(x, w, title, subtitle, fill=white, edge=RULE):
        c.setStrokeColor(edge)
        c.setFillColor(fill)
        c.setLineWidth(.8)
        c.rect(x, 144, w, 40, stroke=1, fill=1)
        text(x + w / 2, 168, title, 9.5, bold=True, center=True)
        text(x + w / 2, 153, subtitle, 8, color=GRAY, center=True)

    text(6, 218, "(a)  Inference at a four-proposal query", 10, bold=True)
    with np.load(PAPER.parent / "examples/object_query/observation.npz") as observation:
        for x, key in [(7, "image"), (47, "image2")]:
            camera = Image.fromarray(observation[key])
            c.drawImage(ImageReader(camera), x, 148, 38, 38)
    text(7, 193, "LIBERO observation", 8.5, bold=True)
    text(7, 137, "Instruction + proprioception", 7.5, color=GRAY)
    block(123, 100, "Encode once", "Shared prefix", LIGHT_BLUE, BLUE)
    block(240, 112, "Frozen SmolVLA expert", "10 flow steps per sample")
    block(370, 106, "Released VGAS critic", "Rank by minimum twin Q")
    block(494, 76, "Execute", "Up to 20 steps")
    for start, end in [(99, 123), (223, 240), (352, 370), (476, 494)]:
        arrow(start, 164, end)
    text(223, 190, "KV cache", 8, color=BLUE, center=True)
    text(296, 200, "4 independent noise draws", 8, color=GRAY, center=True)
    text(359, 190, "4 chunks", 8, color=GRAY, center=True)
    text(489, 190, "Winner", 8, color=GRAY, center=True)
    line(172, 144, 172, 125, BLUE)
    line(172, 125, 423, 125, BLUE)
    arrow(423, 125, 423, 143, BLUE)
    text(207, 114, "Shared observation tokens", 8, color=BLUE)

    line(6, 104, 570, 104)
    text(6, 90, "(b)  Allocation across a LIBERO-Object episode", 10, bold=True)
    text(6, 74, "Query", 8, color=GRAY)
    for query in range(1, 15):
        x = 84 + (query - 1) * 34
        text(x + 14.5, 74, str(query), 8, color=GRAY, center=True)
        for y, active in [(56, False), (39, True), (22, query <= 5)]:
            c.setStrokeColor(BLUE if active else RULE)
            c.setFillColor(LIGHT_BLUE if active else white)
            c.rect(x, y, 29, 12, stroke=1, fill=1)
            text(x + 14.5, y + 3, "4" if active else "1", 7.5, center=True)
    for y, label in [(59, "Single"), (42, "Full Bo4"), (25, "EWS-5")]:
        text(6, y, label, 8.5, bold=(label == "EWS-5"))
    text(84, 7, "1: one proposal   4: four proposals and critic scoring", 8, color=GRAY)
    c.showPage()
    c.save()


if __name__ == "__main__":
    draw()
