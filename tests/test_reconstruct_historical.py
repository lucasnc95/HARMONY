"""Command-level checks for joining historical CSV and runtime log records."""

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


COMMAND = Path(__file__).resolve().parents[1] / "benchmarks" / "reconstruct_historical.py"


class ReconstructionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run = self.root / "run"
        self.run.mkdir()
        self.out = self.root / "out"
        with (self.run / "kneighbor_erad_42.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["iteration", "step_time_s", "action", "rebalanced", "rebalance_gain", "execution_mode"])
            writer.writerows([
                [0, "0.020", "compute", 0, 0, "gpu_opencl"],
                [1, "0.100", "rebalance", 1, 0, "gpu_opencl"],
                [2, "0.020", "compute", 0, 0, "gpu_opencl"],
            ])
        self.log = self.run / "kneighbor_erad_42.log"
        self.write_log()

    def metrics(self, number, first, second):
        return (f"[DCL][balance-metrics] iteration={number}\n"
                "  device_times_s:\n"
                "    part[0]=0.00001\n"
                "    part[1]=0.00002\n"
                "  OLD loads:\n"
                f"    part[0] cum=50% share=50% device_global=0 rank=0 local_index=0 offset=0 count={first}\n"
                f"    part[1] cum=100% share=50% device_global=1 rank=1 local_index=-1 offset={first} count={second}\n"
                "    total=100%\n"
                "  CURRENT loads:\n"
                f"    part[0] cum=50% share=50% device_global=0 rank=0 local_index=0 offset=0 count={first}\n"
                f"    part[1] cum=100% share=50% device_global=1 rank=1 local_index=-1 offset={first} count={second}\n"
                "    total=100%\n")

    def write_log(self, text=None):
        if text is None:
            text = ("[device] rank=0 global_index=0 name=GPU A\n"
                    "[device] rank=1 global_index=1 name=GPU A\n"
                    + self.metrics(1, 5, 5)
                    + "DCL loads by partition:\n"
                    "  part[0] device_global=0 rank=0 local_index=0 offset=0 count=4 cum=40% share=40%\n"
                    "  part[1] device_global=1 rank=1 local_index=-1 offset=4 count=6 cum=100% share=60%\n"
                    "  total_share=100%\n"
                    + self.metrics(2, 4, 6) + self.metrics(3, 4, 6)
                    + "[kneighbor] algo=erad mode=gpu_opencl total_time_s=0.2\n")
        self.log.write_text(text)
        return text

    def invoke(self):
        return subprocess.run([sys.executable, str(COMMAND), "--run-dir", str(self.run),
                               "--output-dir", str(self.out), "--workloads", "kneighbor",
                               "--algorithms", "erad", "--seeds", "42", "--iterations", "3",
                               "--global-count", "10"], text=True, capture_output=True)

    def test_reconstructs_host_and_device_timelines_with_boundary(self):
        result = self.invoke()
        self.assertEqual(result.returncode, 0, result.stderr)
        with (self.out / "timeline.csv").open() as stream:
            timeline = list(csv.DictReader(stream))
        self.assertEqual([(r["iteration"], r["step_time_s"], r["device_event_max_s"])
                          for r in timeline], [("0", "0.020", "0.00002"), ("1", "0.100", "0.00002"), ("2", "0.020", "0.00002")])
        with (self.out / "partitions.csv").open() as stream:
            partitions = list(csv.DictReader(stream))
        self.assertIn(("1", "rebalance_result", "0", "4"),
                      [(r["iteration"], r["phase"], r["offset"], r["count"]) for r in partitions])
        self.assertIn(("1", "runtime_current", "4", "6"),
                      [(r["iteration"], r["phase"], r["offset"], r["count"]) for r in partitions])

    def test_missing_or_duplicate_runtime_record_is_rejected(self):
        base = self.log.read_text()
        for text in [base.replace(self.metrics(2, 4, 6), ""), base.replace(self.metrics(2, 4, 6), self.metrics(2, 4, 6) * 2)]:
            with self.subTest(text_len=len(text)):
                self.write_log(text)
                result = self.invoke()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("iteration", result.stderr)
                self.assertFalse(self.out.exists())

    def test_gap_and_ambiguous_boundary_are_rejected(self):
        base = self.log.read_text()
        for text in [base.replace("offset=4 count=6", "offset=5 count=5", 1),
                     base.replace("DCL loads by partition:\n", "DCL loads by partition:\n" * 2)]:
            with self.subTest(text_len=len(text)):
                self.write_log(text)
                result = self.invoke()
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(self.out.exists())

    def test_missing_event_and_boundary_are_rejected(self):
        base = self.log.read_text()
        for text in [base.replace("    part[1]=0.00002\n", "", 1),
                     base.replace("DCL loads by partition:\n", "NO BOUNDARY:\n")]:
            with self.subTest(text_len=len(text)):
                self.write_log(text)
                result = self.invoke()
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(self.out.exists())

    def test_orphan_records_are_rejected(self):
        base = self.log.read_text()
        for text in ["  device_times_s:\n" + base,
                     base + "    part[0]=0.00001\n"]:
            with self.subTest(text=text[:30]):
                self.write_log(text)
                result = self.invoke()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("unjoinable", result.stderr)
                self.assertFalse(self.out.exists())

    def test_wrong_global_count_is_rejected(self):
        result = subprocess.run([sys.executable, str(COMMAND), "--run-dir", str(self.run),
                                 "--output-dir", str(self.out), "--workloads", "kneighbor",
                                 "--algorithms", "erad", "--seeds", "42", "--iterations", "3",
                                 "--global-count", "11"], text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("covers 10, expected 11", result.stderr)
        self.assertFalse(self.out.exists())

    def test_zero_count_partition_is_preserved(self):
        base = self.log.read_text().replace("count=4", "count=0").replace("offset=4 count=6", "offset=0 count=10")
        self.write_log(base)
        result = self.invoke()
        self.assertEqual(result.returncode, 0, result.stderr)
        with (self.out / "partitions.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertTrue(any(row["phase"] == "rebalance_result" and row["count"] == "0" for row in rows))


if __name__ == "__main__":
    unittest.main()
