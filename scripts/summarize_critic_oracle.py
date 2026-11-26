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



def decompose(anchors):
    critic, uniform, oracle = anchors.T
    total_gap = oracle - uniform
    gain = critic - uniform
    mixed = total_gap > 0
    assert np.all(gain[~mixed] == 0)
    n_mixed = int(mixed.sum())
    capture = gain.sum() / total_gap.sum() if n_mixed else None
    conditional = total_gap[mixed].mean() if n_mixed else None
    reconstruction = float(mixed.mean() * capture * conditional) if n_mixed else 0.
    assert np.isclose(reconstruction, gain.mean())
    return dict(anchors=len(anchors), mixed=n_mixed, mixed_rate=float(mixed.mean()),
                conditional_oracle_gap_pp=100*float(conditional) if n_mixed else None,
                unconditional_oracle_gap_pp=100*float(total_gap.mean()),
                capture=float(capture) if n_mixed else None,
                critic_gain_pp=100*float(gain.mean()), reconstructed_pp=100*reconstruction)


def write_decomposition(anchors, tasks, layout_draws):
    by_query = {str(q+1):decompose(anchors[:,:,q,:].reshape(-1,3)) for q in range(3)}
    by_task = {task:{str(q+1):decompose(anchors[i,:,q,:]) for q in range(3)}
               for i,task in enumerate(tasks)}
    per_task = list(by_task.values())
    consistency = dict(
        q3_mixed_lower_than_q1=sum(t["3"]["mixed"]<t["1"]["mixed"] for t in per_task),
        q3_mixed_lower_than_q2=sum(t["3"]["mixed"]<t["2"]["mixed"] for t in per_task),
        q3_mixed_equal_to_q1=sum(t["3"]["mixed"]==t["1"]["mixed"] for t in per_task),
        q3_mixed_equal_to_q2=sum(t["3"]["mixed"]==t["2"]["mixed"] for t in per_task),
        positive_gain_by_query={str(q):sum(t[str(q)]["critic_gain_pp"]>0 for t in per_task)
                                for q in (1,2,3)})
    gain = layout_draws[...,0]-layout_draws[...,1]
    boot = gain.mean(axis=(0,2))
    for q,row in enumerate(by_query.values()):
        row["critic_gain_ci95_pp"]=(100*np.quantile(boot[:,q],[.025,.975])).tolist()
    with (DATA/"marginal-value.csv").open(newline="") as f:
        dose = next(r for r in csv.DictReader(f) if r["first_query"]=="1" and r["last_query"]=="3")
    pooled = decompose(anchors.reshape(-1,3))
    branch_sum = sum(row["critic_gain_pp"] for row in by_query.values())
    assert np.isclose(branch_sum/3,pooled["critic_gain_pp"])
    comparison = dict(branch_sum_pp=branch_sum, branch_mean_pp=branch_sum/3,
                      branch_sum_ci95_pp=(100*np.quantile(boot.sum(axis=1),[.025,.975])).tolist(),
                      dose_prefix_delta_pp=float(dose["delta_pp"]),
                      dose_prefix_ci95_pp=[float(dose[k]) for k in ("ci95_low_pp","ci95_high_pp")],
                      dose_mean_per_planned_query_pp=float(dose["pp_per_planned_slot"]))
    payload=dict(pooled=pooled, by_query=by_query, by_task=by_task, task_consistency=consistency,
                 prefix_comparison=comparison)
    (DATA/"critic-oracle-decomposition.json").write_text(json.dumps(payload,indent=2)+"\n", encoding="ascii")


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
    layout_draws = anchors[np.arange(10)[:,None,None],indices]
    boot = layout_draws.mean(axis=(0,2,3))
    write_decomposition(anchors, sorted(by_task), layout_draws)
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
