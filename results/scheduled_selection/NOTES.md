# Related explorations

| Exploration | Scoped result | Archive pointer |
| --- | --- | --- |
| Adaptive pool gate | Offline estimate was −0.92 pp relative to random allocation at the matching quota (p=1.0). | `outputs/adaptive_probe_gate_offline_20261121.json` |
| Antithetic noise | Both first-query diagnostic contrasts fell by 2 pp and the batch stopped. | `outputs/antithetic_object_20261122/` |
| Continuation-conditioned ranking | First-query winner was 2.5 pp below the uniform continuation reference. | `outputs/continuation_object_20261122/` |
| Half-episode schedule | Early/late interaction on terminal outcome was zero on the 50-layout pilot. | `outputs/half_schedule_object_20261122/` |
| Twin-head bound | Winner and action agreed on 100/100 queries. Head evaluations fell from 800 to 606, while mean full decision time rose by 1.7 ms. | `outputs/twin_bound_object_20261122/` |
| Denoising pruning | Step-four winner agreement was 78/150 (52.0%), below the 80% screen. | `outputs/p4_prune_bo4_20261123/` |
| S1 failure-triggered early exit | At most 21.4% of held-out failures exited early, with some successes also stopped. Scores from the first five queries did not separate outcomes sufficiently at the tested thresholds. | `outputs/algo_screen_20261124/screening.md` |
| S2 adaptive window | Against dense Bo4, Object was within the 3 pp point-estimate tolerance (−1.33 pp), while Goal missed it (−4.33 pp). The tested gate conserved calls on both cohorts, but its success retention varied. | `outputs/algo_screen_20261124/screening.md`, `outputs/algo_closed_20261124/RESULT.md` |
| S3 twin-head disagreement penalty | The tested penalty was −2.67 pp versus EWS-5 on Object (95% paired interval [−7.00, 1.67]) at equal call count. Head disagreement changed rankings, with no observed benefit from penalizing it. | `outputs/algo_screen_20261124/screening.md`, `outputs/algo_closed_20261124/RESULT.md` |
| S4 early acceptance | Across four archived cohorts, the best of the first two proposals was also best among four only 47.5–50.9% of the time in the early window. Half-pool ranking rarely recovered the full-pool winner reliably. | `outputs/algo_screen_20261124/screening.md` |
| Structured FFN pruning | A 25% width reduction lowered the Bo4 Graph decision median by 0.87–1.20%. Adding K/V reuse lowered it by 1.77–2.33%. Both variants missed the 5% continuation criterion, leaving ranking and task success unmeasured. | [A100 timing ablation](../structured-pruning/README.md) |
| Paired lighting diagnostic | Both stages were sensitive. On a fixed lighting-generated pool, the critic winner changed on 26/150 queries. Among 39 selected replayable anchors, clean and lighting scoring chose 14 and 10 successful branches, respectively (−10.26 pp [−22.86, 0.00]). | [Lighting diagnostic](../plus-light-decomposition/README.md) |

If the study resumes, prioritize one independent lighting experiment.
Persist both observation pairs and their candidate pools at collection time,
then branch both pools on independently selected layouts with multiple
matched continuation seeds. The crossed outcome comparison can estimate
the generation and scoring effects under the same distribution of states.
