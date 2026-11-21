# Hardware manuscript draft

Read the [LaTeX draft](main.tex) and [claims audit](CLAIMS_AUDIT.md).
The draft uses a venue-neutral two-column format and retains placeholders for author information.

From the repository root, regenerate the tables and figures with NumPy and Matplotlib, then compile using Tectonic:

    python scripts/make_paper_tables.py
    python paper/make_figures.py
    tectonic --outdir paper paper/main.tex

The figures use matched snapshot timing and stage profiling. They draw from cuda-graph.json, profile.json and the repository's released results. The fixed-observation eager VGAS scaling sweep has its own table because it uses a separate protocol from the matched Graph experiment.
