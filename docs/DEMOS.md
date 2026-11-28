# Paired simulation rollouts
Two archived LIBERO-Object rollouts show why we count both recoveries and harms. Within each row the actor starts from the same physical layout with the same policy-noise index. The clips show recorded camera observations at policy queries; their durations differ because episodes can finish early.

| Paired Object case | SmolVLA · execute one proposal | VGAS · score four and execute the winner |
|---|---|---|
| **Recovery** · task 0, layout 22, noise 1 | ![Single proposal, unsuccessful](media/object_rescues_0_BC.gif) · [MP4](media/object_rescues_0_BC.mp4) | ![Four proposals, successful](media/object_rescues_0_Bo4.gif) · [MP4](media/object_rescues_0_Bo4.mp4) |
| **Harm** · task 1, layout 22, noise 0 | ![Single proposal, successful](media/object_damages_0_BC.gif) · [MP4](media/object_damages_0_BC.mp4) | ![Four proposals, unsuccessful](media/object_damages_0_Bo4.gif) · [MP4](media/object_damages_0_Bo4.mp4) |

These clips come from the earlier **native VGAS** comparison. They illustrate the selection mechanism. The [main benchmark](../README.md#closed-loop-results) measures the shared-prefix implementation.

## Paired Single and EWS-5

The side-by-side GIF uses recorded observations from the held-out
LIBERO-Object rollout `object_t0_fresh14`. Both arms share the physical
layout and policy seed. The single-proposal rollout reaches the
280-step cap and EWS-5 completes the task at step 175. The animation
advances by control query and holds a terminal frame once its arm ends.
The paired outcomes are listed in the [new-layout ledger](../results/scheduled_selection/object-fresh.csv).
