"""Command-level checks for the historical benchmark analyzer."""

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


COMMAND = Path(__file__).resolve().parents[1] / "benchmarks" / "analyze_variability.py"


class AnalyzerCommandTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run = self.root / "run"
        self.run.mkdir()
        self.output = self.root / "analysis"
        self.csv_path = self.run / "kneighbor_erad_42.csv"
        self.log_path = self.run / "kneighbor_erad_42.log"
        self.write_csv([
            [0, "1", "compute", 0, 0, "gpu_opencl"],
            [1, "2", "rebalance", 1, 0, "gpu_opencl"],
            [2, "3", "compute", 0, 0, "gpu_opencl"],
        ])
        self.log_path.write_text("[kneighbor] algo=erad mode=gpu_opencl total_time_s=20 num_rebalances=1\n")

    def write_csv(self, rows):
        with self.csv_path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["iteration", "step_time_s", "action", "rebalanced", "rebalance_gain", "execution_mode"])
            writer.writerows(rows)

    def invoke(self):
        return subprocess.run([sys.executable, str(COMMAND), "--run-dir", str(self.run),
                               "--output-dir", str(self.output), "--workloads", "kneighbor",
                               "--algorithms", "erad", "--seeds", "42", "--iterations", "3"],
                              text=True, capture_output=True)

    def test_complete_run_keeps_large_remainder_unaccounted(self):
        result = self.invoke()
        self.assertEqual(result.returncode, 0, result.stderr)
        with (self.output / "per_condition.csv").open() as stream:
            condition = next(csv.DictReader(stream))
        self.assertEqual(condition["wall_time_s"], "20")
        self.assertEqual(condition["ordinary_step_sum_s"], "4")
        self.assertEqual(condition["rebalance_step_sum_s"], "2")
        self.assertEqual(condition["unaccounted_s"], "14")
        with (self.output / "per_iteration.csv").open() as stream:
            self.assertEqual([int(row["iteration"]) for row in csv.DictReader(stream)], [0, 1, 2])

    def test_missing_log_is_rejected(self):
        self.log_path.unlink()
        self.assertIn("missing", self.invoke().stderr.lower())
        self.assertFalse(self.output.exists())

    def test_truncated_csv_is_rejected(self):
        self.write_csv([[0, 1, "compute", 0, 0, "gpu_opencl"]])
        result = self.invoke()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("expected 3 iteration", result.stderr)

    def test_out_of_order_and_duplicate_iterations_are_rejected(self):
        for rows, expected in [
            ([[0, 1, "compute", 0, 0, "gpu_opencl"], [2, 2, "compute", 0, 0, "gpu_opencl"], [1, 3, "compute", 0, 0, "gpu_opencl"]], "iteration"),
            ([[0, 1, "compute", 0, 0, "gpu_opencl"], [1, 2, "compute", 0, 0, "gpu_opencl"], [1, 3, "compute", 0, 0, "gpu_opencl"]], "iteration"),
        ]:
            with self.subTest(rows=rows):
                self.write_csv(rows)
                result = self.invoke()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(expected, result.stderr.lower())

    def test_cpu_fallback_is_rejected(self):
        self.write_csv([[0, 1, "compute", 0, 0, "gpu_opencl"], [1, 2, "compute", 0, 0, "cpu"], [2, 3, "compute", 0, 0, "gpu_opencl"]])
        self.assertIn("gpu_opencl", self.invoke().stderr)

    def test_invalid_timings_are_rejected(self):
        for value in ["nan", "inf", "-1", "oops"]:
            with self.subTest(value=value):
                self.write_csv([[0, 1, "compute", 0, 0, "gpu_opencl"], [1, value, "compute", 0, 0, "gpu_opencl"], [2, 3, "compute", 0, 0, "gpu_opencl"]])
                result = self.invoke()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("step_time_s", result.stderr)

    def test_missing_or_duplicate_wall_time_is_rejected(self):
        for log in ["unfinished\n", "[kneighbor] algo=erad mode=gpu_opencl total_time_s=20\n" * 2]:
            with self.subTest(log=log):
                self.log_path.write_text(log)
                result = self.invoke()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("total_time_s", result.stderr)

    def test_invalid_wall_time_and_wrong_log_mode_are_rejected(self):
        for value in ["nan", "-1", "oops"]:
            with self.subTest(value=value):
                self.log_path.write_text(f"[kneighbor] algo=erad mode=gpu_opencl total_time_s={value}\n")
                result = self.invoke()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("total_time_s", result.stderr)
        self.log_path.write_text("[kneighbor] algo=erad mode=cpu total_time_s=20\n")
        result = self.invoke()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("gpu_opencl", result.stderr)

    def test_existing_output_is_not_overwritten(self):
        self.output.mkdir()
        sentinel = self.output / "per_condition.csv"
        sentinel.write_text("original")
        self.assertNotEqual(self.invoke().returncode, 0)
        self.assertEqual(sentinel.read_text(), "original")


if __name__ == "__main__":
    unittest.main()
