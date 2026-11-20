# Critic weight precision

The offline comparison uses 500 archived queries from the shared-Bo4 arm of the third Object repeat. The sampler sorts the original paths and draws 50 queries without replacement from each of ten tasks using Python seed 20261120. It writes the chosen paths before scoring. It restores each recorded observation and instruction, regenerates the original 16-sample CUDA noise bank from the recorded seed, and uses its first four samples. The largest observed absolute difference across the 2,000 replayed FP32 critic scores and their archived values was zero.

All formats score the same prefix embeddings and candidate action tensors. The reported Spearman statistic averages four-candidate rank correlations over queries with a defined correlation. Agreement uses the first maximum, matching the online selector. The INT8 criterion is at least 475 matching selections out of 500.

The experiment varies **weight storage**. INT8 uses symmetric quantization along axis zero of each parameter with at least two dimensions. Each axis-zero slice has one FP32 scale. Scalar and vector parameters, including bias and normalization vectors, retain their source precision. FP16 and BF16 round the same multidimensional weights. All variants restore a weight to its original arithmetic dtype on access. Storage counts include persistent tensor payloads, including scales, and exclude serialization metadata and temporary arithmetic tensors. This procedure measures neither native INT8 kernel latency nor peak runtime memory savings.

From the repository root, supply the original Object archives and checkpoints to reproduce the comparison. Install the pinned VGAS dependencies and add the VGAS checkout to PYTHONPATH.

```sh
python scripts/precision_offline.py --archive "$P4_ARCHIVE" --actor "$OBJECT_ACTOR" \
  --critic "$OBJECT_CRITIC" --backbone "$VLM_BACKBONE" --output offline
```

The optional archive replay helper takes the original P4 workspace as an explicit dependency. It runs the INT8 arm after the offline criterion is met and reuses the recorded BC and shared-Bo4 controls. Its fresh output directory holds the original trial and formal batches. Each case keeps its original initial state, scene seed and policy seed. Paired percentile intervals resample the same 300 case-level differences 20,000 times with seed 42.

```sh
CUDA_VISIBLE_DEVICES=0,1 python scripts/precision_closed_loop.py \
  --root "$P4_WORKSPACE" --offline offline/summary.json --output int8_object
```

## Offline results

| Stored weights | Same selection | Mean query Spearman | Persistent tensor bytes |
|---|---:|---:|---:|
| Original | Reference | Reference | 175,026,696 |
| INT8 | 495/500 (99.0%) | 0.9956 | 44,023,952 |
| FP16 | 500/500 (100.0%) | 1.0000 | 87,577,864 |
| BF16 | 498/500 (99.6%) | 0.9972 | 87,577,864 |

All 500 correlations were defined for every format. INT8 meets the closed-loop criterion and uses 25.15% of the released critic's original persistent tensor bytes. The frozen query list, candidate-level scores, and aggregate results are in [results/precision](../results/precision/).


## Object closed-loop results

| Policy | Successes | Success rate | INT8 minus policy, pp (paired 95% interval) |
|---|---:|---:|---:|
| BC | 144/300 | 48.00% | +8.67 [2.00, 15.33] |
| Shared Bo4, original critic | 169/300 | 56.33% | +0.33 [0.00, 1.00] |
| Shared Bo4, INT8 critic | 170/300 | 56.67% | Reference |

Against the original shared-Bo4 policy, one initial state changed from failure to success and the other 299 outcomes agreed. The paired interval includes zero at its lower endpoint, meeting the prespecified descriptive criterion. Formal equivalence would require an equivalence margin and a corresponding test. Against BC, INT8 succeeded on 66 states where BC failed and failed on 40 states where BC succeeded. [Case-level outcomes](../results/precision/closed_loop.json)

During this separate multi-worker A100 run the INT8-storage arm averaged 10.31 policy queries and 41.24 generated proposals per episode, with 222.16 ms per core decision. Per-access weight unpacking and shared-server scheduling both contribute to this timing. [Timing samples](../results/precision/closed_loop_costs.json).
