"""Inspect one recorded four-candidate decision using only the Python standard library."""
import argparse
import json
from pathlib import Path

RESULT = Path(__file__).resolve().parents[1] / "results" / "value-token-replay.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=int, choices=range(10), default=0)
    parser.add_argument("--query", choices=("first", "middle"), default="first")
    args = parser.parse_args()
    rows = json.loads(RESULT.read_text(encoding="utf-8"))["rows"]
    cases = sorted((r for r in rows if r["task"] == args.task), key=lambda r: r["frame"])
    if len(cases) != 2:
        raise ValueError("Expected first and middle queries for this task")
    case = cases[0 if args.query == "first" else 1]
    scores = case["full_scores"]
    winner = max(range(len(scores)), key=scores.__getitem__)
    if len(scores) != 4 or winner != case["full_winner"]:
        raise ValueError("Candidate pool or selected index disagrees with recorded decision")
    print(f"Object Task {args.task}, {case['frame']} (zero-based candidate indices)")
    for index, score in enumerate(scores):
        print(f"  {index}   {score:8.3f}{'   selected' if index == winner else ''}")


if __name__ == "__main__":
    main()
