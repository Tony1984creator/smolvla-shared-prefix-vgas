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
| `pruning-summary.json` | `outputs/p4_prune_bo4_20261123/result.json` | Offline ranking of four proposals on 150 archived Object queries, 15 per task. Step-ten scores determine the reference winner. |

Each query commits at most 20 environment steps. Maximum episode budgets are
14 queries on Object and 26 on Long. A four-proposal query counts as one
four-way scoring call. Stopping on termination can reduce realized calls.
The paired percentile bootstrap resamples within each Object task (10,000
draws, seed 20261122 for the sweep and 20261123 for the fresh cohort). Long
resamples physical layouts and keeps both noise repeats together (20,000
draws, seed 42). The 3 pp rule applies to point estimates.
