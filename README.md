# Shared-prefix inference for VGAS on SmolVLA

This research artifact evaluates the released VGAS action selector and implements shared observation encoding for batched inference. The SmolVLA actor samples four action chunks from independent noise. The released critic ranks them. Our implementation computes the vision-language prefix once per query and reuses its attention cache and critic features across candidates.

## Main results

| Evaluation | Paired comparison | Success | Change |
|---|---|---:|---:|
| LIBERO-Object, all 10 tasks | Single sample → shared-prefix Bo4 | 144/300 → 169/300 | +8.33 pp [1.67, 15.00] |
| LIBERO-Long, Task2 | Single sample → shared-prefix Bo4 | 93/256 → 124/256 | +12.11 pp [5.86, 18.75] |
| LIBERO-Plus, fixed Object lighting variants | Single sample → shared-prefix Bo4 | 100/300 → 123/300 | +7.67 pp, adjusted interval [−1.67, 17.00] |

On Object, the authors’ original Bo4 implementation took **271.64 ms** per decision on A100. Our shared-prefix Bo4 took **216.78 ms** (−20.2%). The same-run single-sample baseline took 210.57 ms. Original and shared Bo4 both use four candidates and had 170/300 and 169/300 successes. The paired shared-minus-original interval was [−5.00, 4.33] pp. The comparison is descriptive. Timing covers encoding, generation and scoring, excluding action postprocessing and simulation.

Task2 was selected using development data. The replay uses 128 historical physical layouts with two noise repeats per layout. All settings use LIBERO simulation. The lighting result is a perturbation check. Target-chip latency, memory placement and power remain to be measured.

## Verify and reproduce

```bash
python scripts/verify_results.py
```

The command verifies the released paired outcomes and recomputes headline success counts and same-run A100 means without simulator or weights. The inference implementation is in [method/shared_prefix_selection.py](method/shared_prefix_selection.py). [Reproduction notes](docs/REPRODUCIBILITY.md) map each result to its evaluator, weights, server environment and trial configuration. A fresh-machine end-to-end run remains to be validated.

- [Experiment archive](docs/EXPERIMENT_ARCHIVE.md)
- [Briefing figure](docs/briefing.png)
- [Third-party attribution](docs/ATTRIBUTION.md)
