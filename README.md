# Spend Selection Early

Early-window selection (EWS) uses four SmolVLA proposals near the start of an episode and one thereafter, concentrating VGAS scoring where it yielded the most benefit.

![Panel a: snapshots from one EWS episode and its query schedule. Panel b: success and mean core inference time for Single, Full Bo4, and EWS under warm CUDA Graph replay.](paper/figures/teaser.png)

**(a)** One recorded LIBERO-Object episode. Frames mark queries 1, 3, 5, 7, and 9, and the strip shows all reached queries. **(b)** Success and mean core inference time on the same 200 layouts under warm CUDA Graph replay.

- **When to select.** On paired Object layouts, the early window beat the tested Late schedule by **7.60 percentage points**, with a 95% paired interval of [2.60, 12.60].
- **How much it saves.** Across the tested cohorts, EWS reduced realized four-way calls by about half while remaining within the specified 3 pp point-estimate tolerance of full selection.
- **Measured A100 cost.** Under warm CUDA Graph replay, EWS and full Bo4 each succeeded in **110/200** episodes. EWS reduced core inference time per episode by **9.0%**.

[Paper PDF](paper/main.pdf) · [中文一页总结](docs/SUMMARY_CN.md) · [Evidence audit](paper/CLAIMS_AUDIT.md) · [Extended experiment guide](docs/RESULTS_GUIDE.md)

## Closed-loop results

<!-- cohort-table:start -->
| Cohort | Single | Full Bo4 | EWS | Late | Four-way calls saved |
|---|---:|---:|---:|---:|---:|
| Object calibration | 144/300 | 169/300 | 167/300 | 141/300 | 51.6% |
| Object new layouts | 98/200 | 110/200 | 110/200 | 98/200 | 51.5% |
| Long Task2 | 93/256 | 124/256 | 127/256 | — | 58.6% |
| Goal | 145/300 | 165/300 | 159/300 | 138/300 | 50.5% |
| Spatial | 131/300 | 142/300 | 141/300 | 131/300 | 51.1% |
| Plus lighting | 100/300 | 123/300 | 123/300 | — | 55.6% |
<!-- cohort-table:end -->

Each cell reports successful episodes over the cohort. The manuscript gives paired EWS-minus-full intervals. Goal and Spatial were evaluated under preregistered point-estimate criteria. Long is the development-selected Task2. [Paired data and protocols](results/scheduled_selection/README.md)

## Reproduce

```sh
python scripts/verify_results.py
python -m pip install numpy matplotlib pillow
python paper/make_submission.py
tectonic --outdir paper paper/main.tex
```

[Install and run the policy](docs/INSTALL.md) · [Pinned checkpoints and settings](docs/REPRODUCIBILITY.md) · [Paper build](paper/README.md) · [License and upstream notices](THIRD_PARTY_NOTICES.md)
