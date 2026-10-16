#!/usr/bin/env python3
"""Exercise the public benchmark output and sample admission interface."""

import csv
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class BenchmarkSampleContract(unittest.TestCase):
    def test_cpu_simulation_reports_separate_durations(self):
        for binary in sys.argv[1:]:
            with self.subTest(binary=binary), tempfile.TemporaryDirectory() as directory:
                csv_path = Path(directory) / "steps.csv"
                inventory_path = Path(directory) / "inventory.json"
                result = subprocess.run(
                    ["mpirun", "--oversubscribe", "-n", "1", binary,
                     "--n", "32", "--iterations", "2", "--output-csv", str(csv_path),
                     "--quiet-runtime-metrics", "--device-inventory-json", str(inventory_path)],
                    cwd=ROOT, text=True, capture_output=True, timeout=60,
                    env={**os.environ, "OMPI_ALLOW_RUN_AS_ROOT": "1", "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM": "1"},
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("mode=cpu_simulation", result.stdout)
                self.assertIn("validation=PASS", result.stdout)
                inventory = json.loads(inventory_path.read_text())
                self.assertEqual(inventory["schema_version"], 1)
                self.assertEqual(len(inventory["ranks"]), 1)
                self.assertEqual(inventory["ranks"][0]["rank"], 0)
                self.assertEqual(inventory["ranks"][0]["gpu_count"], 0)
                durations = dict(re.findall(r"(loop_time_s|validation_time_s|total_time_s)=([0-9.eE+-]+)", result.stdout))
                self.assertEqual(set(durations), {"loop_time_s", "validation_time_s", "total_time_s"})
                values = {key: float(value) for key, value in durations.items()}
                self.assertTrue(all(math.isfinite(value) and value >= 0 for value in values.values()))
                self.assertGreaterEqual(values["total_time_s"], values["loop_time_s"])
                self.assertGreaterEqual(values["total_time_s"], values["validation_time_s"])
                with csv_path.open(newline="") as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(len(rows), 2)
                self.assertTrue(all(row["execution_mode"] == "cpu_simulation" for row in rows))

    def test_strict_mode_rejects_one_rank_before_creating_sample(self):
        for binary in sys.argv[1:]:
            with self.subTest(binary=binary), tempfile.TemporaryDirectory() as directory:
                csv_path = Path(directory) / "steps.csv"
                inventory_path = Path(directory) / "inventory.json"
                result = subprocess.run(
                    ["mpirun", "--oversubscribe", "-n", "1", binary,
                     "--n", "32", "--iterations", "2", "--output-csv", str(csv_path),
                     "--require-two-rank-gpu", "--device-inventory-json", str(inventory_path)],
                    cwd=ROOT, text=True, capture_output=True, timeout=60,
                    env={**os.environ, "OMPI_ALLOW_RUN_AS_ROOT": "1", "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM": "1"},
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("two MPI ranks", result.stderr)
                self.assertFalse(csv_path.exists())

    def test_strict_mode_rejects_two_ranks_without_gpus_collectively(self):
        for binary in sys.argv[1:]:
            with self.subTest(binary=binary), tempfile.TemporaryDirectory() as directory:
                csv_path = Path(directory) / "steps.csv"
                inventory_path = Path(directory) / "inventory.json"
                result = subprocess.run(
                    ["mpirun", "--oversubscribe", "-n", "2", binary,
                     "--n", "32", "--iterations", "2", "--output-csv", str(csv_path),
                     "--require-two-rank-gpu", "--device-inventory-json", str(inventory_path)],
                    cwd=ROOT, text=True, capture_output=True, timeout=60,
                    env={**os.environ, "OMPI_ALLOW_RUN_AS_ROOT": "1", "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM": "1"},
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("one OpenCL GPU per rank", result.stderr)
                self.assertFalse(csv_path.exists())
                inventory = json.loads(inventory_path.read_text())
                self.assertEqual([item["rank"] for item in inventory["ranks"]], [0, 1])


if __name__ == "__main__":
    binaries = sys.argv[1:]
    if len(binaries) != 2:
        raise SystemExit("usage: test_benchmark_sample_contract.py KNEIGHBOR LEANMD")
    sys.argv = [sys.argv[0], *binaries]
    unittest.main(argv=[sys.argv[0]])
