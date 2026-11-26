"""Compare selectors at archived Object anchors under one-proposal continuation."""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[1] / "results/scheduled_selection"
POLICIES = ("critic", "uniform", "oracle")


def summarize(values):
    return dict(zip(POLICIES, map(float, values.mean(axis=0))))


def main():
    by_task = defaultdict(lambda: defaultdict(dict))
    mixed = np.zeros(3, dtype=int)
    mixed_aucs = []
    with (DATA/"critic-oracle-branches.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            q = int(row["query"])
            wins = np.array([int(row[f"success_{i}"]) for i in range(4)], dtype=bool)
            scores = np.array([float(row[f"score_{i}"]) for i in range(4)])
            group = by_task[row["task"]][row["layout"]]
            winner = int(row["winner"])
            assert q in (1,2,3) and q not in group and winner == scores.argmax()
            assert np.isfinite(scores).all()
            group[q] = [wins[winner], wins.mean(), wins.any()]
            if wins.any() and not wins.all():
                diff = scores[wins,None]-scores[None,~wins]
                mixed_aucs.append(((diff>0)+.5*(diff==0)).mean())
                mixed[q-1] += 1
    assert len(by_task)==10 and all(len(layouts)==30 for layouts in by_task.values())
    assert all(set(row)=={1,2,3} for layouts in by_task.values() for row in layouts.values())
    anchors = np.array([[list(q[k] for k in (1,2,3)) for _,q in sorted(layouts.items())]
                        for _,layouts in sorted(by_task.items())])
    rng = np.random.default_rng(20261126)
    draws = 20000
    # Sample intact layouts within task, keeping all three anchors and policy outcomes paired.
    indices = rng.integers(30, size=(10,draws,30))
    boot = anchors[np.arange(10)[:,None,None],indices].mean(axis=(0,2,3))
    rates = anchors.mean(axis=(0,1,2))
    contrasts={}
    for left,right in (("oracle","critic"),("oracle","uniform"),("critic","uniform")):
        a,b = (POLICIES.index(k) for k in (left,right))
        contrasts[f"{left}_minus_{right}"] = dict(
            difference_pp=float(100*(rates[a]-rates[b])),
            ci95_pp=(100*np.quantile(boot[:,a]-boot[:,b],[.025,.975])).tolist())
    result=dict(layouts=300,queries=900,branches=3600,mixed_queries=int(mixed.sum()),
                mixed_auc=float(np.mean(mixed_aucs)),rates=summarize(anchors.reshape(-1,3)),
                by_query={str(q):dict(n=300,mixed=int(mixed[q-1]),
                                     rates=summarize(anchors[:,:,q-1,:].reshape(-1,3)))
                          for q in (1,2,3)},
                contrasts=contrasts,bootstrap=dict(draws=draws,seed=20261126,unit="layout within task"))
    (DATA/"critic-oracle.json").write_text(json.dumps(result,indent=2)+"\n", encoding="ascii")


if __name__=="__main__":
    main()
