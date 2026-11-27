# Paired lighting diagnostic

The screen used 50 archived LIBERO-Object layouts, five per task, and the
first three reference queries of each layout. We replayed the archived
selected actions in the original environment and in the matching official
LIBERO-Plus lighting variants at revision `4976dc3`. At each retained query, both environments
agreed exactly in joint positions and velocities, and clean inference agreed
with the archived candidate chunks, scores and selected index within the
numeric replay tolerances. Both inputs to the model share proprioception,
instruction and noise. The critic's observation changes independently of
the actor's observation.

The first letter in a condition identifies the actor input. The second
identifies the critic input. The letters `c` and `l` mean clean and lighting
variant, respectively.

| Change | Winner index changes | 95% layout-bootstrap interval | Median absolute change in winner-to-other score gaps |
| --- | ---: | ---: | ---: |
| Critic image on the clean-generated pool (`cc` to `cl`) | 36/150 (24.0%) | [18.7%, 29.3%] | 0.428 |
| Critic image on the lighting-generated pool (`lc` to `ll`) | 26/150 (17.3%) | [11.3%, 23.3%] | 0.541 |
| Actor image with clean scoring (`cc` to `lc`) | 52/150 (34.7%) | [28.7%, 41.3%] | 0.818 |
| Actor image with lighting scoring (`cl` to `ll`) | 42/150 (28.0%) | [22.7%, 33.3%] | 0.806 |

The actor's same-noise normalized action difference over the first 20
steps had p10/p50/p90 of 0.060/0.107/0.205 mean absolute units. The median
per-query change, divided by the mean pairwise distance within its clean
pool, was 0.535. For actor swaps a winner index denotes a shared noise
slot. For critic swaps it denotes the same executable candidate. These
diagnostics measure distinct perturbations, with task outcome only tested
by the branch experiment.

## Branch protocol

We selected 40 anchors from the screen, prioritizing the 26 winner changes
on the lighting-generated pool. For each branchable anchor, the simulator
restores the same perturbed state before executing each of its four
candidates. The candidate executes for at most 20 steps. Subsequently,
the same one-proposal policy receives the perturbed observation and
continues with matched noise until success, termination or 280 steps.
The contrast compares outcomes of the clean-scored and lighting-scored
winners on that same lighting-generated pool. Intervals resample whole
layouts within tasks. The selected cohort is enriched for winner changes.

One anchor changed its lighting-scored winner on rerender at the same
recorded physical state. The remaining 39 anchors contributed 156
branches. Clean scoring selected 14 successful branches and lighting
scoring selected 10. Light minus clean was −10.26 pp (95% descriptive
interval [−22.86, 0.00], two rescues and six harms). On the 25 verified
winner changes, the comparison was 11 versus 7 (−16.0 pp
[−32.0, 0.0]). The branch sample provides evidence about the local
consequences of scorer sensitivity under the tested continuation.
The actor outcome effect remains unmeasured.

[Screen summary](screen-summary.json) · [Screen shard 0](screen_0.jsonl) ·
[Screen shard 1](screen_1.jsonl) · [Selected anchors](branch_cases.json) ·
[Branch summary](branch-summary.json)

The branch ledger is split across `branches_0.jsonl` through
`branches_3.jsonl` and `branch_tomato.jsonl`. From the repository root,
recompute both summaries with NumPy:

    python results/plus-light-decomposition/summarize.py screen results/plus-light-decomposition
    python results/plus-light-decomposition/summarize.py branches results/plus-light-decomposition
