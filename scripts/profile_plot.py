"""Render the stage decomposition and logical-traffic roofline."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('data', type=Path)
parser.add_argument('figure', type=Path)
args = parser.parse_args()
rows = json.loads(args.data.read_text())['rows']
fig, (bars, roof) = plt.subplots(1, 2, figsize=(10.3, 4.2), layout='constrained')
colors = ['#12747a', '#75b9ba', '#364985', '#dd8938']
indices = np.arange(len(rows))
base = np.zeros(len(rows))
for stage, label, color in zip(('vision', 'prefix', 'expert', 'critic'),
        ('Vision', 'Language / prefix', 'Action expert', 'Critic'), colors):
    points = [r['stages'][stage] for r in rows]
    ms = np.array([p['ms'] for p in points])
    bars.bar(indices, ms, bottom=base, color=color, label=label, width=.64)
    base += ms
    intensity = [p['flops'] / sum(p['bytes'].values()) for p in points]
    perf = [p['flops'] / (p['ms'] * 1e9) for p in points]
    roof.plot(intensity, perf, 'o-', ms=4, color=color, label=label)
    if stage in ('expert', 'critic'):
        for i in (0, -1):
            roof.annotate(str(rows[i]['n']), (intensity[i], perf[i]),
                          xytext=(2, 5), textcoords='offset points', fontsize=8)
bars.set(xticks=indices, xticklabels=[str(r['n']) for r in rows],
         xlabel='Candidates', ylabel='Stage median (ms)', title='Shared decision latency')
bars.legend(fontsize=8, ncol=2, loc='lower left')
x = np.geomspace(.5, 1000)
for peak, style in [(156, '--'), (312, ':')]:
    roof.plot(x, np.minimum(1.935*x, peak), color='0.46', linestyle=style, lw=1)
roof.set(xscale='log', yscale='log', xlabel='Estimated FLOP / logical byte',
         ylabel='Estimated TFLOP/s', title='A100 PCIe roofline reference')
roof.text(.02,.98,'Dense BF16 (312) / TF32 (156) TFLOP/s', ha='left', va='top',
          fontsize=8, transform=roof.transAxes)
for ax in (bars,roof):
    ax.spines[['top','right']].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis='y', alpha=.2)
fig.savefig(args.figure, dpi=190)
