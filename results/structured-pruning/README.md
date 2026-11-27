# Structured FFN pruning under CUDA Graph replay

Each of two interleaved blocks used 60 archived LIBERO-Object queries sampled
from the first three queries of the reference policy across ten tasks. Every
query used the same observation and noise across four arms. We used one
shared A100 80 GB PCIe, four candidates and ten Euler steps. The timer
includes prefix encoding, warm Graph input refresh and replay, scoring and
selection. Observation preprocessing, noise generation and first capture
are outside the timer.

For each of the 16 action-expert FFNs, we split the 2,048 channels into four
contiguous groups. We removed the group with the lowest squared weight norm
in the down projection and narrowed the gate, up and down matrices
accordingly. The cache uses the original batched K/V projection shape,
computing it once per cross-attention layer and reusing it across denoising
steps. Cached and uncached variants agreed on all actions, scores and winner
indices within their corresponding model arms.

| Arm | Block 1 p50 / p90 (ms) | Block 2 p50 / p90 (ms) |
| --- | ---: | ---: |
| Dense | 82.92 / 84.36 | 82.76 / 85.35 |
| 25% FFN pruning | 81.92 / 83.22 | 82.05 / 84.75 |
| Cache only | 81.54 / 82.92 | 82.00 / 84.75 |
| Pruning and cache | 80.98 / 82.46 | 81.30 / 84.70 |

Pruning reduced full-decision medians by 1.20% and 0.87%. Pruning with caching
reduced them by 2.33% and 1.77%. These reductions fell short of the 5% screen.
The experiment did not proceed to ranking or closed-loop evaluation. It
measures the sensitivity of full-decision latency to these two changes.
Kernel-level attribution to attention, memory movement or scheduling remains open.

Run `python results/structured-pruning/summarize.py` with NumPy to recompute
the block percentiles and task-stratified paired layout-bootstrap intervals
from [the matched timing rows](timing.json). [Summary](summary.json)
