# Hardware manuscript draft

Read the [LaTeX draft](main.tex) and [claims audit](CLAIMS_AUDIT.md).
The draft uses a venue-neutral two-column format and retains placeholders for author information.

![One shared observation, four independent proposals and their action horizons](figures/method-overview.png)

From the repository root, regenerate the tables and figures with NumPy, Matplotlib, ReportLab and Pillow, then compile using Tectonic:

    python scripts/make_paper_tables.py
    python scripts/summarize_scheduled_selection.py
    python paper/make_overview.py
    python paper/make_figures.py
    tectonic --outdir paper paper/main.tex

The overview uses the archived camera pair in `examples/object_query` and Task 0 query000 scores in `results/value-token-replay.json`. The dose plot uses the released scheduled-selection ledgers. The two timing plots use matched snapshot timing and stage profiling. They draw from cuda-graph.json, profile.json and the repository's released results. The fixed-observation eager VGAS scaling sweep has its own table because it uses a separate protocol from the matched Graph experiment.

Before submission, finalize authors, affiliations, funding, venue and anonymity policy. Confirm the arXiv account, posting date and release timing. Hardware measurements in this draft are scoped to A100.
