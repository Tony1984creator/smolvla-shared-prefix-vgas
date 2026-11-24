# Early-window scheduled selection

Run `python scripts/summarize_scheduled_selection.py` from the repository root.
NumPy and ReportLab regenerate `summary.json`, `curve.csv`, the LaTeX table,
and the vector dose figure. Each ledger row represents one episode and
records its task, layout label, policy-noise repeat, arm, terminal success,
total queries, four-way calls, and generated candidates. The arm labels
preserve their source spelling. Long layout labels index the archived
scene seeds in ascending order.

| Ledger | Archive source | Protocol |
| --- | --- | --- |
| `object-sweep.csv` | `outputs/p4_front_sweep_20261123/paired.csv` plus the first-seven and random-seven rows of [the previously released ledger](../object-query-budget.csv) | 30 paired layouts per task, 10 Object tasks and one noise repeat per layout. Fixed cutoffs K=0, 3, 5, 7, 10, 14 and a random seven-slot schedule. The sweep overlaps the 50-layout pilot. The smallest tested cutoff within 3 pp of the dense point estimate was five. |
| `object-fresh.csv` | `outputs/p4_front_fresh_finish_20261123/paired.csv` and its frozen `paired_manifest.json` | 20 fresh BDDL starts per Object task after ten settling steps. The 200 layouts are disjoint from the pilot and the 300-layout sweep. Single, full Bo4 and first-five share the physical initial state and policy seed. |
| `long-task2.csv` | `outputs/long_front8_finish_20261123/result.json` plus `deliverables/hardware_paper_algo_20261114/long_shared_bo4_20261118/rows.json` | 128 historical Task2 layouts and two shared noise repeats. The single and full Bo4 results are archived comparators. The eight-query cutoff was set separately using the longer task horizon. |
| [spatial-window.csv](spatial-window.csv) | `outputs/ews_generalize_20261124/spatial-paired.csv` and its preregistered manifest | Ten Spatial tasks, 30 official BDDL starts per task (indices 20–49), and released Spatial critic. Four arms share layout and seeds. The 280-step horizon gave 14 queries and K=round(14/3)=5. Late runs Bo4 at queries 6–14. |
| [plus-light-window.csv](plus-light-window.csv) | `outputs/ews_generalize_20261124/plus-paired.csv` | Ten fixed LIBERO-Plus lighting variants on the 300 paired Object starts, with the archived single and dense controls and newly evaluated EWS-5. Four-way calls fell from 3,376 to 1,500. |
| `pruning-summary.json` | `outputs/p4_prune_bo4_20261123/result.json` | Offline ranking of four proposals on 150 archived Object queries, 15 per task. Step-ten scores determine the reference winner. |

Each query commits at most 20 environment steps. Maximum episode budgets are
14 queries on Object, Spatial and Plus lighting, 15 on Goal and 26 on Long Task2. A four-proposal query counts as one
four-way scoring call. Stopping on termination can reduce realized calls.
The paired percentile bootstrap resamples within each Object task (10,000
draws, seed 20261122 for the sweep and 20261123 for the fresh cohort). Long
resamples physical layouts and keeps both noise repeats together (20,000
draws, seed 42). The 3 pp rule applies to point estimates.

The four-arm Late control reuses all three archived fresh Object arms and adds Bo4 from query 6 onward on the same 200 starts. The episode ledger and task-stratified comparisons are in late-object.csv and late-object.json, exported from outputs/p8_late_object_20261123/. The q0-by-query.csv and q0-gates.json files aggregate the repaired shared full-Bo4 Object archive in outputs/p4_fixed_closed_loop_20261118/. The q0 diagnostic is retrospective and conditions on queries reached by full Bo4.

The calibration Late arm is exported from outputs/p9_late_calibration_20261124/. It shares all 300 physical starts and strategy-noise seeds with the original Object sweep. It runs one proposal through query 5 and Bo4 thereafter. The combined first-five minus Late estimate pools 500 disjoint physical layouts. Its percentile interval resamples whole layouts within each task and cohort (20,000 draws, seed 20261124). The sweep was also used to choose K; the 200 fresh layouts provide a separate comparison.

The episode-cost.csv ledger stores directly summed core inference time per episode. Object calibration uses the repaired P4 single, original and shared arms plus front-5 from p4_front_sweep_20261123. Fresh Object uses p4_front_fresh_finish_20261123 and its archived single-proposal records. Long Task2 single and dense use long_task2_shared_bo4_20261118; front-8 uses recorded core_seconds events from long_front8_finish_20261123 and the four accepted workers in long_front8_resume_20261123. The clocks exclude simulator execution and action postprocessing. The Spatial and Plus window ledgers carry their own core timing sums in inference_ms. Plus controls and EWS ran in separate batches. Run python scripts/summarize_episode_cost.py to generate the manuscript table from these sources.

Goal uses the shared actor with the released Goal critic at the same pinned VGAS revision. The 300 official BDDL starts (indices 20–49 per task) and K=5/15 were fixed before acting. goal-window.csv and goal-window.json come from outputs/goal_window_20261124. The Goal, Spatial and Plus paired intervals resample layouts within each task
(20,000 draws, seed 20261124). Spatial preregistered point-estimate tolerances
of 3 pp for both EWS versus dense and Late versus single. Plus evaluated
retention against dense on its frozen lighting cohort.

The [retrospective self-assessment results](self-assessment.json) summarize
1,056 archived full-Bo4 episodes. A 60/40 split within each task and cohort
kept repeated Long layouts together. The training split selected the best
score feature and the failure-precision thresholds. On the pooled holdout
(422 episodes), the best q0 feature had AUC 0.876 and no training threshold
with at least 90% failure precision. Query count alone had AUC 0.994. The
source is outputs/ews_generalize_20261124/self-assessment.json, with feature
definitions in its companion SELF_ASSESSMENT.md.
