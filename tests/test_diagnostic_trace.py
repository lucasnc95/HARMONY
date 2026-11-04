#!/usr/bin/env python3
"""Exercise diagnostic traces through benchmark commands and the validator CLI."""

import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "benchmarks" / "validate_trace.py"
ENV = {**os.environ, "OMPI_ALLOW_RUN_AS_ROOT": "1", "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM": "1"}


class DiagnosticTraceContract(unittest.TestCase):
    def run_case(self, binary, directory, ranks=2, prefix=None):
        prefix = prefix or Path(directory) / "trace"
        return subprocess.run(
            ["mpirun", "--oversubscribe", "-n", str(ranks), binary,
             "--n", "16", "--iterations", "3", "--balance-interval", "2",
             "--output-csv", str(Path(directory) / "steps.csv"),
             "--quiet-runtime-metrics", "--diagnostics-prefix", str(prefix)],
            cwd=ROOT, text=True, capture_output=True, timeout=60, env=ENV,
        )

    def test_two_rank_cpu_trace_validates(self):
        for binary in sys.argv[1:]:
            with self.subTest(binary=binary), tempfile.TemporaryDirectory() as directory:
                result = self.run_case(binary, directory)
                self.assertEqual(result.returncode, 0, result.stderr)
                files = [Path(directory) / f"trace.rank{rank}.csv" for rank in (0, 1)]
                self.assertTrue(all(path.is_file() for path in files))
                with files[0].open(newline="") as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual([int(row["seq"]) for row in rows], list(range(len(rows))))
                self.assertIn("loop_total", {row["phase"] for row in rows})
                self.assertIn("cpu_simulation", {row["phase"] for row in rows})
                check = subprocess.run(
                    [sys.executable, str(VALIDATOR), str(Path(directory) / "trace"),
                     "--ranks", "2", "--iterations", "3", "--elements", "16",
                     "--balance-interval", "2", "--summary-json", str(Path(directory) / "summary.json")],
                    text=True, capture_output=True,
                )
                self.assertEqual(check.returncode, 0, check.stderr)
                summary = json.loads((Path(directory) / "summary.json").read_text())
                self.assertEqual([entry["rank"] for entry in summary["ranks"]], [0, 1])
                self.assertTrue(all("loop_total" in entry["host_phase_totals_s"]
                                    for entry in summary["ranks"]))
                repeated = subprocess.run(
                    [sys.executable, str(VALIDATOR), str(Path(directory) / "trace"),
                     "--ranks", "2", "--iterations", "3", "--elements", "16",
                     "--balance-interval", "2", "--summary-json", str(Path(directory) / "summary.json")],
                    text=True, capture_output=True,
                )
                self.assertNotEqual(repeated.returncode, 0)

    def test_validator_accepts_zero_count_partition_and_rejects_gap(self):
        fields = ["schema_version", "rank", "seq", "iteration", "seed", "algorithm",
                  "record_type", "phase", "part", "device_global", "owner_rank", "local_index",
                  "offset", "count", "duration_s", "device_event_s",
                  "boundary_shift_elements_estimate", "transfer_bytes_measured", "device_name",
                  "device_kind", "device_identity"]
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / "fixture"
            for rank in (0, 1):
                rows = []
                def add(iteration, kind, phase, **values):
                    row = dict.fromkeys(fields, "")
                    row.update(schema_version="1", rank=str(rank), seq=str(len(rows)),
                               iteration=str(iteration), seed="42", algorithm="erad",
                               record_type=kind, phase=phase)
                    row.update({key: str(value) for key, value in values.items()})
                    rows.append(row)
                for state in ("old", "new"):
                    add(0, "partition", state, part=0, device_global=0, owner_rank=0,
                        local_index=0, offset=0, count=4)
                    add(0, "partition", state, part=1, device_global=1, owner_rank=1,
                        local_index=0, offset=4, count=0)
                for phase in ("runtime_execute", "mpi_step_max", "mpi_device_times"):
                    add(0, "phase", phase, duration_s=0.001)
                for phase in ("loop_total", "gather", "reference", "mpi_validation"):
                    add(1, "phase", phase, duration_s=0.002)
                with Path(f"{prefix}.rank{rank}.csv").open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(rows)
            command = [sys.executable, str(VALIDATOR), str(prefix), "--ranks", "2",
                       "--iterations", "1", "--elements", "4"]
            good = subprocess.run(command, text=True, capture_output=True)
            self.assertEqual(good.returncode, 0, good.stderr)
            path = Path(f"{prefix}.rank1.csv")
            with path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            for row in rows:
                if row["record_type"] == "partition":
                    if row["part"] == "0":
                        row["count"] = "3"
                    else:
                        row["offset"], row["count"] = "3", "1"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            disagreement = subprocess.run(command, text=True, capture_output=True)
            self.assertNotEqual(disagreement.returncode, 0)
            self.assertIn("rank progress or condition mismatch", disagreement.stderr)
            rows[1]["offset"] = "5"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            bad = subprocess.run(command, text=True, capture_output=True)
            self.assertNotEqual(bad.returncode, 0)
            self.assertIn("gap or overlap", bad.stderr)

    def test_existing_trace_is_not_overwritten_and_all_ranks_stop(self):
        for binary in sys.argv[1:]:
            with self.subTest(binary=binary), tempfile.TemporaryDirectory() as directory:
                existing = Path(directory) / "trace.rank0.csv"
                existing.write_text("keep me")
                result = self.run_case(binary, directory)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(existing.read_text(), "keep me")
                self.assertFalse((Path(directory) / "steps.csv").exists())
                self.assertFalse((Path(directory) / "trace.rank1.csv").exists())


if __name__ == "__main__":
    binaries = sys.argv[1:]
    if len(binaries) != 2:
        raise SystemExit("usage: test_diagnostic_trace.py KNEIGHBOR LEANMD")
    sys.argv = [sys.argv[0], *binaries]
    unittest.main(argv=[sys.argv[0]])
