"""Rebuild the static width/time comparison from frozen Object outcomes."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/width-time"
ARCHIVE = ROOT / "results/scheduled_selection"
NEW = {"bo2-full": (2, 14), "bo8-front3": (8, 3), "bo2-front10": (2, 10)}
OLD = {
    "B-BC": (1, 14), "front3": (4, 3), "front5": (4, 5), "early": (4, 7),
    "front10": (4, 10), "B-Bo4": (4, 14), "random": (4, 7), "late": (4, 9),
}
LABELS = {
    "B-BC": "Single", "front3": "Bo4, q1--3", "front5": "EWS-5",
    "early": "Bo4, q1--7", "front10": "Bo4, q1--10", "B-Bo4": "Full Bo4",
    "random": "Random-seven", "late": "Late", "bo2-full": "Full Bo2",
    "bo8-front3": "Bo8, q1--3", "bo2-front10": "Bo2, q1--10",
}
DRAWS, SEED = 20_000, 20261129
MAIN_ARMS = ("B-BC", "bo2-front10", "bo2-full", "front5", "bo8-front3", "B-Bo4")


def ledger(path):
    with path.open(newline="", encoding="utf-8") as stream:
        yield from csv.DictReader(stream)


def read_metrics():
    rows = json.loads((DATA / "layouts.json").read_bytes())
    reference = {row["case"]: row for row in rows}
    tasks = defaultdict(list)
    for case, row in reference.items():
        tasks[row["task"]].append(case)
    if len(reference) != len(rows) or len(tasks) != 10 or set(map(len, tasks.values())) != {30}:
        raise ValueError("Calibration layouts must provide 30 unique cases per task")

    timing_arms = {"single": "B-BC", "dense": "B-Bo4", "front5": "front5"}
    existing = {}
    for row in ledger(ARCHIVE / "episode-cost.csv"):
        if row["cohort"] == "Object calibration" and row["arm"] in timing_arms:
            existing[timing_arms[row["arm"]], row["case"]] = float(row["inference_ms"])
    arms = defaultdict(dict)
    for path in ("object-sweep.csv", "late-calibration.csv"):
        for row in ledger(ARCHIVE / path):
            arm, case = row["arm"], row["layout"]
            if case in arms[arm] or row["task"] != reference[case]["task"] or row["repeat"] != "0":
                raise ValueError(f"Invalid archived pairing at {arm}/{case}")
            queries, calls, candidates = (int(row[k]) for k in ("queries", "calls", "candidates"))
            if candidates != queries + 3 * calls:
                raise ValueError(f"Candidate accounting differs at {arm}/{case}")
            arms[arm][case] = dict(success=int(row["success"]), queries=queries, calls=calls,
                                   candidates=candidates, core_ms=existing.get((arm, case)))

    for line in (DATA / "episodes.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        case, arm = row["case"], row["arm"]
        ref = reference[case]
        if case in arms[arm] or any(row[k] != ref[k] for k in
                                   ("task", "scene_seed", "policy_seed", "physical_sha")):
            raise ValueError(f"Invalid new pairing at {arm}/{case}")
        width, window = NEW[arm]
        steps, widths = row["steps"], row["widths"]
        times, winners = row["decision_ms"], row["winners"]
        queries = len(widths)
        if (not 0 < steps <= 280 or queries != (steps + 19) // 20
                or widths != [width if q < window else 1 for q in range(queries)]
                or len(times) != queries or not np.isfinite(times).all() or min(times) <= 0
                or len(winners) != queries or any(not 0 <= w < n for w, n in zip(winners, widths))
                or type(row["success"]) is not bool):
            raise ValueError(f"Invalid new episode trace at {arm}/{case}")
        arms[arm][case] = dict(success=int(row["success"]), queries=queries,
                               calls=sum(n > 1 for n in widths), candidates=sum(widths), core_ms=sum(times))

    if set(arms) != set(OLD) | set(NEW) or any(set(rows) != set(reference) for rows in arms.values()):
        raise ValueError("Incomplete schedule cohort")
    ordered = [case for task in sorted(tasks) for case in sorted(tasks[task])]
    return ordered, arms


def interval(values, indices):
    samples = np.take_along_axis(values.reshape(10, 1, 30), indices, axis=2).mean(axis=(0, 2))
    return [round(float(x), 12) for x in np.quantile(samples, [0.025, 0.975])]


def paired(delta, indices):
    return {
        "difference_pp": 100 * float(delta.mean()),
        "ci95_pp": [100 * v for v in interval(delta, indices)],
        "rescues": int((delta > 0).sum()), "harms": int((delta < 0).sum()),
    }


def summarize():
    ordered, arms = read_metrics()
    rng = np.random.default_rng(SEED)
    indices = rng.integers(0, 30, size=(10, DRAWS, 30))
    reference = np.asarray([arms["front5"][case]["success"] for case in ordered])
    table = {}
    for arm in (*OLD, *NEW):
        episodes = [arms[arm][case] for case in ordered]
        wins = np.asarray([row["success"] for row in episodes])
        width, slots = (NEW | OLD)[arm]
        plan = dict(width=width, scoring_slots=slots if width > 1 else 0,
                    maximum_scored_candidates=width * slots if width > 1 else 0,
                    maximum_extra_candidates=(width - 1) * slots)
        extra = np.asarray([row["candidates"] - row["queries"] for row in episodes])
        times = [row["core_ms"] for row in episodes if row["core_ms"] is not None]
        table[arm] = dict(label=LABELS[arm], wins=int(wins.sum()), episodes=len(wins), plan=plan,
                          mean_queries=mean(row["queries"] for row in episodes),
                          mean_scoring_calls=mean(row["calls"] for row in episodes),
                          mean_candidates=mean(row["candidates"] for row in episodes),
                          mean_extra_candidates=float(extra.mean()),
                          mean_extra_candidates_ci95=interval(extra, indices),
                          mean_episode_ms=mean(times) if times else None,
                          paired_vs_ews=paired(wins-reference, indices))
        if len(times) not in (0, len(wins)):
            raise ValueError(f"Partial timing ledger for {arm}")
    near_budget = {}
    for left, right in (("front3", "bo2-front10"), ("early", "bo8-front3")):
        delta = np.asarray([arms[left][case]["success"] - arms[right][case]["success"]
                            for case in ordered])
        near_budget[f"{left}_vs_{right}"] = paired(delta, indices)
    return {"layouts": len(ordered), "bootstrap_draws": DRAWS, "bootstrap_seed": SEED,
            "reference_arm": "front5", "schedules": table,
            "exploratory_near_budget": near_budget}


def formats(data):
    schedules = data["schedules"]
    lines = [
        r"% Generated by scripts/summarize_width_time.py",
        r"\begin{table}[t]\centering\small",
        r"\caption{Width and timing on 300 paired Object layouts. Extra candidates are counted above one per reached query. Differences from EWS-5 have paired 95\% intervals.}",
        r"\label{tab:width-time}",
        r"\begin{tabular}{@{}lrrrl@{}}\toprule",
        r"Schedule & Wins & Extra cap & Mean extra & vs. EWS-5 (pp) \\",
        r"\midrule",
    ]
    for arm in MAIN_ARMS:
        row = schedules[arm]
        delta = row["paired_vs_ews"]
        lo, hi = delta["ci95_pp"]
        comparison = "--" if arm == "front5" else rf'${delta["difference_pp"]:+.2f}\;[{lo:.2f}, {hi:.2f}]$'
        lines.append(" & ".join((row["label"], str(row["wins"]),
                     str(row["plan"]["maximum_extra_candidates"]), f'{row["mean_extra_candidates"]:.2f}',
                     comparison)) + r" \\")
    lines.extend((r"\bottomrule\end{tabular}", r"\end{table}"))
    tags = {"front5": "Ews", "bo2-full": "TwoFull", "bo2-front10": "TwoFront", "bo8-front3": "EightFront"}
    values = {}
    for arm, tag in tags.items():
        row = schedules[arm]
        values[f"K{tag}Extra"] = f'{row["mean_extra_candidates"]:.2f}'
        if arm != "front5":
            values[f"K{tag}Delta"] = f'{row["paired_vs_ews"]["difference_pp"]:+.2f}'
    win_tags = {"B-BC": "Single", "B-Bo4": "BoFour",
                "bo2-full": "TwoFull", "bo2-front10": "TwoFront"}
    for arm, tag in win_tags.items():
        values[f"K{tag}Wins"] = str(schedules[arm]["wins"])
    near = data["exploratory_near_budget"]["early_vs_bo8-front3"]
    values["KSevenExtra"] = f'{schedules["early"]["mean_extra_candidates"]:.2f}'
    values["KNearDelta"] = f'{near["difference_pp"]:+.2f}'
    values["KNearCI"] = ", ".join(f"{bound:.2f}" for bound in near["ci95_pp"])
    macros = "% Generated by scripts/summarize_width_time.py\n"
    for name, value in values.items():
        macros += f"\\newcommand{{\\{name}}}{{{value}}}\n"
    return {
        DATA / "summary.json": json.dumps(data, indent=2) + "\n",
        ROOT / "paper/tables/width-time.tex": "\n".join(lines) + "\n",
        ROOT / "paper/width-time-metrics.tex": macros,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    options = parser.parse_args()
    for target, contents in formats(summarize()).items():
        if options.check:
            if contents != target.read_text(encoding="utf-8"):
                raise ValueError(f"{target.name} differs from the episode ledgers")
        else:
            target.write_text(contents, encoding="utf-8")


if __name__ == "__main__":
    main()
