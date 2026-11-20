"""Independent checks of the paired counts, layout units and latency series."""
import csv
import json
from pathlib import Path
from statistics import mean
import unittest

RESULTS = Path(__file__).resolve().parents[1] / "results"


class ResultTests(unittest.TestCase):
    def test_pairing_and_effect(self):
        for filename, baseline, method, layouts, repeats, rescue, harm in [
            ("object-paired", "B-BC", "B-Bo4", 300, 1, 65, 40),
            ("long-task2-paired", "Z", "SharedBo4", 128, 2, 56, 25),
            ("plus-light-paired", "Plus_light_BC", "Plus_light_Bo4", 300, 1, 62, 39),
        ]:
            with self.subTest(filename=filename):
                data = json.loads((RESULTS / f"{filename}.json").read_bytes())
                groups = []
                for arm in (baseline, method):
                    rows = data[arm]
                    by_key = {(r["layout"], r["noise"]): r["success"] for r in rows}
                    self.assertEqual(len(rows), len(by_key))
                    self.assertEqual(len(rows), layouts * repeats)
                    self.assertEqual(len({key[0] for key in by_key}), layouts)
                    groups.append(by_key)
                a, b = groups
                self.assertEqual(a.keys(), b.keys())
                self.assertEqual(sum(b[k] and not a[k] for k in a), rescue)
                self.assertEqual(sum(a[k] and not b[k] for k in a), harm)

    def test_latency_scopes(self):
        obj = json.loads((RESULTS / "object-paired.json").read_bytes())
        for arm, expected in [("B-BC", 210.57), ("B-OrigBo4", 271.64), ("B-Bo4", 216.78)]:
            self.assertEqual(round(1000 * mean(t for r in obj[arm] for t in r["decision_seconds"]), 2), expected)
        with (RESULTS / "fixed-observation-latency.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 10)
        points = {(int(r["N"]), r["implementation"]): round(float(r["median_ms"]), 2) for r in rows}
        self.assertEqual(set(points), {(n, a) for n in (1, 2, 4, 8, 16) for a in ("original", "shared")})
        self.assertEqual((points[1, "shared"], points[16, "shared"]), (216.97, 219.66))

    def test_precision_replay(self):
        rows = json.loads((RESULTS / "precision/scores.json").read_bytes())
        manifest = json.loads((RESULTS / "precision/queries.json").read_bytes())
        self.assertEqual([row["query"] for row in rows], manifest)
        self.assertEqual(len(set(manifest)), 500)
        for row in rows:
            self.assertEqual(len(row["original"]), 4)
            self.assertEqual(row["original"], row["archived"])
        for arm, count in [("float16", 500), ("bfloat16", 498), ("int8", 495)]:
            def winner(scores):
                return max(range(4), key=scores.__getitem__)
            self.assertEqual(sum(winner(r["original"]) == winner(r[arm]) for r in rows), count)
