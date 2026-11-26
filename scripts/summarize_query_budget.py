"""Rebuild the episode-budget table and paired intervals from released Object outcomes."""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARMS = {"B-BC": "One proposal", "B-Bo4": "Four at every query",
        "early": "Four at first seven", "random": "Four at random seven"}


def main():
    cases = defaultdict(dict)
    with (ROOT / "results/object-query-budget.csv").open() as f:
        for r in csv.DictReader(f):
            k, arm = r["case"], r["arm"]
            assert arm not in cases[k] and arm in ARMS
            q, calls, candidates = map(int, (r["queries"], r["bo4_calls"], r["candidate_samples"]))
            assert candidates == q + 3*calls
            if arm == "early":
                assert calls == min(q, 7)
            cases[k][arm] = r
    assert len(cases) == 300 and all(set(v)==set(ARMS) for v in cases.values())
    for v in cases.values():
        assert len({(r["task_name"], r["scene_seed"], r["policy_seed"]) for r in v.values()}) == 1
    ordered = sorted(cases, key=lambda k: (cases[k]["early"]["task_name"], k))
    tasks = [cases[k]["early"]["task_name"] for k in ordered]
    assert len(set(tasks)) == 10 and all(tasks.count(t)==30 for t in set(tasks))

    def values(arm, field):
        return np.array([int(cases[k][arm][field]) for k in ordered])

    lines = [r"\begin{tabular}{@{}lrrr@{}}\toprule",
             r"Schedule & Wins & Calls & Chunks \\ \midrule"]
    for arm, label in ARMS.items():
        wins, calls, candidates = (values(arm, key).sum() for key in ("success", "bo4_calls", "candidate_samples"))
        lines.append(rf"{label} & {wins} & {calls:,} & {candidates:,} \\")
    lines.append(r"\bottomrule\end{tabular}")
    (ROOT / "paper/tables/query-budget.tex").write_text("\n".join(lines)+"\n", encoding="ascii")
    for ref in ("B-BC", "B-Bo4", "random"):
        delta = (values("early","success")-values(ref,"success")).reshape(10,30)
        rng = np.random.default_rng(20261122)
        draws = [row[rng.integers(0,30,(10000,30))].mean(axis=1) for row in delta]
        ci = np.quantile(np.mean(draws,axis=0), [.025,.975])*100
        print(f"early vs {ref}: {100*delta.mean():+.2f} pp [{ci[0]:.2f}, {ci[1]:.2f}], "
              f"{(delta==1).sum()} rescues, {(delta==-1).sum()} harms")


if __name__ == "__main__":
    main()
