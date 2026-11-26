# CUDA Graph replay in closed loop

This experiment repeats the 200 previously frozen LIBERO-Object layouts under
three schedules using the same captured action-expert path. Each of the ten
tasks contributes 20 layouts. Each layout shares physical reset, actor, policy
noise seed, and the 20-step execution horizon across all arms. The rollout cap
is 280 environment steps. Layouts overlap the earlier eager Object study by
design, so they are a systems replication on the same evaluation cohort.

| Arm | Candidate width by query | Critic use |
|---|---|---|
| Single | One at every query | Skipped |
| Full Bo4 | Four at every query | Released Object critic at every query |
| EWS-5 | Four at queries 1–5, then one | Only at four-proposal queries |

One width-specific CUDA Graph per worker captures the ten denoising steps.
Replay refreshes the current observation's cache and attention mask and the
candidate noise and time scalars. Observation encoding and scoring execute
eagerly. Before recording outcomes, each worker compares eager and replay
actions and any scores and winner indices at both widths. Initialization and
the first graph call (including warm-up and capture) are outside the timer.

Each worker runs all three arms for every assigned layout and cycles through
their six orders. The A100 nodes are shared, with two experiment workers on
each of two 80 GB PCIe devices. Reported timing includes encoding, input
refresh, generation and scoring when active. It excludes input and action
processing on either side of the model path, environment steps and startup.
Times are deployment observations for this shared runtime. They do not
measure full robot control latency.

The [episode ledger](episodes.jsonl) includes the initial state digest, scene and policy seeds,
decision latencies, widths, selected indices, and terminal outcome. The
recomputation checks the complete layout-by-arm grid and resamples whole
layouts within task for paired intervals on success and mean inference time.
The [startup ledger](startup.json) records each worker's first Graph decision
at each width including warm-up and capture.

From the repository root, with the pinned environment and assets from
[installation](../../docs/INSTALL.md), use:

    python scripts/eval_graph_object.py --assets assets --manifest results/graph-closed-loop/object-layouts.json --output graph-object.jsonl
    python scripts/summarize_graph_closed_loop.py --verify

For a distributed reproduction, launch all distinct shard indices using
`--shard` and `--shards` and concatenate the JSONL files. The archived manifest
enforces the physical reset digest, which requires the pinned simulator and
dependencies used in the original LIBERO-Object fresh-layout run.
