# Experiment archive

The local and server research trees retain their original relative paths. Use this map to find the corresponding executable experiments, records and presentation material.

| Stage | Local entry | Server entry relative to the home directory | Retained evidence |
|---|---|---|---|
| Early candidate, causal and trigger exploration | `work/causal-foundation-v1` and the CIAT, AFCR, CST, SPARC and PACE folders under `work` | `workspace/VLA-Corrector/experiments/causal-foundation-v1` | Paired replay, critic experiments, recoveries and harms |
| Official Corrector and QGF reproductions | `work/official-reproductions-20261022` | `workspace/reproductions/official-baselines-20261022` | Frozen upstream sources and public checkpoint references, official-task QGF reproduction, Corrector public-data adaptation |
| Combination and negative ablations | Same project's `deliverables/third_discussion_20261105` and `deliverables/stage_validation_20261105` | Same relative paths under the official reproduction root | Trigger, replanning, gradient, ZeroQ, candidate coverage and execution controls |
| Final inference and evidence | Same project's `deliverables/hardware_paper_algo_20261114` | Same relative path under the official reproduction root | Final results, reporting figures, raw-output pointers and demos |

## Final server outputs

- `outputs/p4_fixed_closed_loop_20261118` contains the corrected 1,500-episode Object experiment.
- `outputs/long_task2_shared_bo4_20261118` contains the 512-episode Long Task2 replay.
- `outputs/object_decision_ablation_20261118` and `outputs/object_decision_environment_20261118` contain 2,100 supplementary episodes.
- `outputs/p4_closed_loop_20261117` is a historical invalid run. The wrapper executed candidate zero regardless of the scorer. Use the corrected run above for conclusions.

Corrector public-data self-training is a method reproduction. The matched shorter-horizon control did not establish a separate trigger benefit. QGF reproduced positive results in selected official configurations. The visual VLA transfer did not establish stable gain. The final shared-prefix selector is the paper artifact.

## Workspace layout after 2026-11-20

- Local loose historical method implementations and tests are under `work/archive-20261120/early-method-staging`; generated report previews are under `work/archive-20261120/document-previews`. The relocation map is `work/archive-20261120/RELOCATION_MAP.json`.
- Research notes relocated from the official reproduction root are grouped in `archives/research-notes` in both copies. Each copy has its own relocation manifest because the notes and references differ. Canonical run outputs and models remain at their original locations.
