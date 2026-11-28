# Manuscript build

[Read the PDF](main.pdf) · [Claim-to-evidence audit](CLAIMS_AUDIT.md)

The draft uses the official NeurIPS 2025 single-column preprint style and
the author line “Anonymous Authors.” This is a layout choice for author
review. Use the venue's current template when the authors select a venue.

From the repository root, regenerate assets and compile:

```sh
python -m pip install numpy matplotlib pillow
python scripts/verify_results.py
python scripts/make_paper_tables.py
python scripts/summarize_scheduled_selection.py
python scripts/summarize_episode_cost.py
python scripts/summarize_graph_closed_loop.py
python paper/make_figures.py --numbers-only
python paper/make_submission.py
tectonic --outdir paper paper/main.tex
```

The submission generator reads archived paired ledgers and summaries. It
writes the cross-cohort table, headline macros, README counts, nine
diagnostic or main plots plus the per-query Graph plot, and matching
300 dpi PNGs. The figures share `figstyle.py` and use the template's
5.5-inch text width. Camera frames are the recorded inputs documented in
[assets](assets/README.md).

For NeurIPS review format, remove the `preprint` option from the package
line and follow the selected track's instructions. For an IEEE conference,
create an `IEEEtran` conference front matter, adapt the author and
bibliography blocks, set the figure width in `figstyle.py` to the measured
column or full-page text width, and regenerate figures. This checkout
builds the single-column preprint by default.

[Official NeurIPS style source](https://media.neurips.cc/Conferences/NeurIPS2025/Styles.zip)
