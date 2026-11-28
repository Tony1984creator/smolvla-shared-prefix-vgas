# Static selection width and timing on Object

The comparison uses 300 calibration layouts, with the same actor and critic,
layout, and policy seed across schedules. The additional arms are full Bo2,
Bo2 for the first ten queries, and Bo8 for the first three queries. The
archived single, full Bo4, EWS-5, Late, random-seven and Bo4 prefix sweep are
joined by their original outcome keys. Each query reaches at most 20 actions
and each episode has a 280-step cap.

The planned cap on extra proposals is `sum(Nq - 1)` over permitted queries.
The realized cost uses the same sum over reached queries, including the
one-proposal tail. A Bo2 query contributes one extra proposal, while a Bo8
query contributes seven. Schedules have different realized costs because
widths, windows and termination all affect the count.

At a matched state and query, each width draws the prefix of a single
Gaussian noise bank. The GPU executes the scheduled width with eager
shared-prefix inference. The fixed-observation checks in
[parity.json](parity.json) record numerical differences across actual batch
shapes, including one restricted winner change. Equal noise slots do not
guarantee identical estimated chunks or scores across batch shapes. After
trajectory divergence the same noise can be used at different observations.

The [layout manifest](layouts.json) records reset digests and seeds.
The [new episode ledger](episodes.jsonl) records success, steps, candidate
width, winner and core inference time for each query. Core time covers
encoding, denoising and scoring where applicable. It excludes simulation
and action postprocessing. Historical and new eager timings come from
separate rollout batches. Episode success is the primary outcome.

The [summary](summary.json) compares each schedule against EWS-5 with
descriptive paired 95% percentile intervals from 20,000 task-stratified
layout resamples. It also gives a secondary exploratory comparison of
Bo4 for the first three queries with Bo2 for the first ten and of Bo4 for
the first seven with Bo8 for the first three. These pairs spent nearly the
same mean budget. Recompute from the repository root:

```sh
python scripts/verify_results.py
python scripts/summarize_width_time.py --check
python paper/make_submission.py
```
