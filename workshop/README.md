# When Does Test-Time Intervention Help a Small VLA?

**Workshop-style technical report · Research draft**

## Abstract

We studied three ways to improve a frozen vision-language-action policy at inference time: visual detection followed by replanning, action updates guided by a learned value gradient, and value-based selection among policy proposals. We distinguish reproductions on prior authors’ settings from transfer to SmolVLA. On our target tasks, matched controls did not establish a reliable incremental benefit from visual triggering or gradient guidance. Selection among four native proposals improved success in the evaluated settings. Reusing the observation prefix across candidates and the critic reduced same-run A100 decision latency by 20.2% relative to the original four-proposal evaluator.

## 1 · Background and research questions

VLA-Corrector [1] uses action-conditioned visual prediction to detect divergence, then truncates the current execution queue and replans. Its visual correction objective and a task-return critic provide different supervision. Q-Guided Flow (QGF) [2] uses the gradient of a pretrained critic inside action generation. VGAS [3] generates actions with a frozen policy and selects a completed proposal by its estimated task value.

We asked three separable questions. Does an intervention improve on replanning with the same execution interval? Can the value gradient improve execution beyond the matched zero-gradient control? Can selection realize useful proposals while keeping the inference cost small?

## 2 · Method and experimental controls

The early exploration compared relative action value, directional perturbations, action-sequence supervision and triggers. It retained useful experimental controls: pair physical initial states, bind the policy-noise schedule, restore simulator state for branch comparisons, and report both recovery and damage. Longer action sequences improved predictive supervision in development, while observed closed-loop gains remained dependent on the actor, action source and continuation policy.

For the final implementation we froze the released VGAS SmolVLA and suite-specific critics. At each query, four noise samples generate four action chunks. The smaller of two critic outputs scores each chunk. The executor takes the highest-scoring chunk and commits up to its first 20 actions. Our implementation reuses the vision-language prefix and attention cache while preserving candidate-specific denoising. See [reproduction details](../docs/REPRODUCIBILITY.md).

## 3 · Results

| Question | Controlled evidence | Interpretation |
|---|---|---|
| Can Corrector run with public training data? | Natural execution 113/120, detection plus replanning 115/120, with visual gradient 118/120 | Method reproduction in a development cohort. The original authors’ exact training split and checkpoints were unavailable to this run. Strict-prefix subgroups were smaller. |
| Does the detector help beyond frequent replanning? | Object, detector with matched 10-step reference: 260/300 vs 258/300, +0.67 pp [−1.00, 2.33] | Additional benefit remains uncertain. Earlier comparisons against a long fixed execution interval confound detection with more frequent observation. |
| Can the QGF result be recovered with released code and data? | Cube-triple, three fixed task/seed settings: 0→30, 2→27, 0→29 successes per 30 evaluations, zero vs 0.06 guidance | Positive reproduction in two task goals with three task/seed settings at 500k offline updates. Evaluation is setting-matched, with separate trials. |
| Does Q-gradient transfer improve SmolVLA? | Long Task2, QGF minus matched ZeroQ +7.03 pp, adjusted 98.75% interval [−1.56, 16.41] | Positive point estimate with substantial uncertainty. A useful ranking function can still provide unreliable local gradients for action generation. |
| Does visual scheduling improve on random scheduling? | 97/256 vs 103/256, −2.34 pp [−9.77, 5.47] at 97.5%. Actual per-episode fractions averaged 37.6% vs 47.7%. | Timing advantage was not established. Realized invocation counts also differ. |
| Does final shared-prefix VGAS improve task outcomes? | Object 144→169/300, +8.33 pp [1.67, 15.00]. Long Task2 93→124/256, +12.11 pp [5.86, 18.75]. | Positive paired closed-loop evidence against a single proposal. Task2 was selected from development results. |
| Does prefix reuse reduce cost? | On Object, original vs shared Bo4: 271.64→216.78 ms, 170→169 successes out of 300 | GPU latency improved in the matched four-candidate setting. Success rates were close, with a paired difference interval [−5.00, 4.33] pp. |

## 4 · Takeaways and limitations

- The original Corrector and QGF experiments answer different questions. Their successful settings do not establish compositional gain. Matched execution and zero-gradient controls are needed to credit a trigger or gradient.
- In our target setting the final evidence supports released VGAS selection plus our inference optimization. The project does not claim ownership of VGAS, a new critic-training algorithm, or theoretical novelty of cache reuse.
- The fixed lighting change in LIBERO-Plus produced a +7.67 pp point estimate, with a multiplicity-adjusted interval crossing zero. Both task suites and the lighting variants run in LIBERO. Hardware results on this page are A100 measurements.
- Reported intervals and trial units are heterogeneous across rows. The table compares evidence against within-study controls. These numbers do not rank algorithms across actors or datasets.

## 5 · Artifact and references

The [main artifact](../README.md) contains per-rollout records for headline VGAS results, verification code and the source archive map. The legacy full reports remain available through the [experiment archive](../docs/EXPERIMENT_ARCHIVE.md). This document is a technical draft for discussion and workshop preparation.

1. *VLA-Corrector: Lightweight Detect-and-Correct Inference for Adaptive Action Horizon.* [Paper](https://arxiv.org/abs/2607.01804) · [Code](https://github.com/ZJU-OmniAI/vla-corrector)
2. Zhou et al. *Test-Time Gradient Guidance of Flow Policies in Reinforcement Learning.* [arXiv:2606.11087](https://arxiv.org/abs/2606.11087) · [Code](https://github.com/zhouzypaul/qgf)
3. Xu et al. *VGAS: Value-Guided Action-Chunk Selection for Few-Shot Vision-Language-Action Adaptation.* [arXiv:2602.07399](https://arxiv.org/abs/2602.07399) · [Code](https://github.com/Jyugo-15/VGAS)
