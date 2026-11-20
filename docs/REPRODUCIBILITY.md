# Reproducibility

## Environments and released assets

- VGAS source [ef656cb](https://github.com/Jyugo-15/VGAS/tree/ef656cbed9b740e035e5bd2981578531e4076e85).
- Author checkpoints [VGAS-5-shot f23261d](https://huggingface.co/SemyonXu616/VGAS-5-shot/tree/f23261d42938662851e438cee6e25b2ceaa05ceb). Use the shared 5-shot actor and separate Object/Long critics from the corresponding suite.
- Object and Plus used the server's smolvla-plus environment, Long used vgas-author. Object PyTorch was 2.7.1+cu126 on A100 80 GB PCIe. Checkout, compatible LIBERO/PyTorch stack and asset layout are dependencies of the full evaluator.
- Candidate generation emits 50×7 actions in 10 flow steps. Object critic consumes the first 32, Long critic 50. Both execute at most 20 before observing again. Episode budgets are 280 Object steps and 520 Long steps. Score with the smaller of the two critic outputs and execute the maximum-scoring candidate.

## Existing server evaluator entry points

| Evidence | Server command from the official-reproductions root | Frozen inputs |
|---|---|---|
| Object five-arm experiment | `envs/smolvla-plus/bin/python -m analysis.p4_fixed_closed_loop_20261118` | 300 Object confirmation layouts with third noise repeat |
| Long Task2 shared Bo4 replay | `envs/vgas-author/bin/python -m analysis.long_shared_bo4_replay` | Sealed E2 layouts, two noise banks |
| Object decision ablations | `envs/smolvla-plus/bin/python -m analysis.object_decision_ablation` | P4 manifest and rows |
| Plus lighting and generation horizon | `envs/smolvla-plus/bin/python -m analysis.object_decision_environment` | Completed Object ablations and fixed Plus lighting variants |
| Cross-arm report | `envs/smolvla-plus/bin/python -m analysis.object_decision_delivery` | Completed ablation and environment runs |

These are server entry points in the archived full workspace, with fixed output names. Use an isolated copy for reruns, preserve the input artifacts and keep new output paths separate. The exported JSON reproduces result arithmetic on any machine with Python 3. The full CUDA and simulator environment is a separate handoff prerequisite.

## Pairing and inference

Pair on physical initial state and policy noise. On Long, cluster the two repeats within each layout when estimating uncertainty. Bo4 versus single sample measures the effect of selecting among four proposals. Shared-Bo4 versus original-Bo4 isolates the effect of sharing under the same selection method. Report A100 timing as measured GPU decision latency. Peak reserved memory and timings are hardware and software dependent.
