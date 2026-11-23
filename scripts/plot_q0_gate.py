"""Render candidate-zero score drift and simulated gate rates from archived aggregates."""
import csv
from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.pdfgen.canvas import Canvas

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/scheduled_selection/q0-by-query.csv"
with DATA.open(newline="") as stream:
    rows = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)]
assert [row["query"] for row in rows] == list(range(1,15))
assert sum(row["n"] for row in rows) == 3099
pdf = Canvas(str(ROOT / "paper/figures/q0-gate.pdf"), pagesize=(256,188), invariant=True)
colors = {"median":"#22577a","gate25":"#22577a","gate50":"#a44926","gate75":"#56713b"}


def panel(bottom, top, lo, hi, ticks):
    def position(query, value):
        return 39 + (query-1)*15, bottom + (top-bottom)*(value-lo)/(hi-lo)
    pdf.setFont("Helvetica",7)
    for tick in ticks:
        x,y = position(1,tick)
        pdf.setStrokeColor(HexColor("#bec5ca"))
        pdf.setLineWidth(.35)
        pdf.line(x,y,234,y)
        pdf.drawRightString(x-5,y-2,str(tick))
    return position


def line(column, pos):
    pdf.setStrokeColor(HexColor(colors[column]))
    pdf.setLineWidth(1)
    points = [pos(row["query"],row[column]) for row in rows]
    for a,b in zip(points,points[1:]):
        pdf.line(*a,*b)


score = panel(108,165,-65,-20,[-60,-45,-30])
for row in rows:
    x,y=score(row["query"],row["q25"])
    _,high=score(row["query"],row["q75"])
    pdf.setStrokeColor(HexColor("#8fb7cb"))
    pdf.setLineWidth(1.4)
    pdf.line(x,y,x,high)
line("median",score)
pdf.drawString(39,174,"Candidate-zero score (median and interquartile range)")
gate = panel(26,90,0,1,[0,.5,1])
for idx,col in enumerate(("gate25","gate50","gate75")):
    line(col,gate)
    pdf.setFillColor(HexColor(colors[col]))
    pdf.drawString(120+idx*39,98,col[-2:]+"th pct.")
pdf.setFillColor(HexColor("#000000"))
pdf.drawString(39,98,"Gate fraction")
for row in rows:
    x,_=gate(row["query"],0)
    pdf.drawCentredString(x,15,str(int(row["query"])))
pdf.drawCentredString(137,4,"Query index")
pdf.save()
