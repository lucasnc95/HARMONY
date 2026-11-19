#!/usr/bin/env python3
"""
Test suite for benchmarks/analyze_super_matrix.py.
Verifies dual-format parsing (upstream #run/#timeline/#device/#process/#balance CSVs
and Historical Evidence [DCL][balance-metrics] logs), CSV reconciliation, paired
analysis, summary statistics, artifact generation, manifest failure/timeout propagation,
configurable iteration counts, strict matrix-stage filtering, and non-zero exit propagation.
"""

import csv
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
BENCHMARKS_DIR = ROOT_DIR / "benchmarks"
sys.path.insert(0, str(BENCHMARKS_DIR))

try:
    import analyze_super_matrix as asm
except ImportError:
    asm = None


SAMPLE_UPSTREAM_REPORT_CSV = """\
#run,wall_s,iterations,mpi_processes,devices,balance_attempts,balance_applied,bytes_moved
run,3.25,100,2,2,9,2,800000
#timeline,rank,device,wall_s,opencl_compute_s,opencl_other_s,halo_s,mpi_halo_s,mpi_migration_s,mpi_balance_s,mpi_sync_s,migration_s,balance_s,gather_s,unmeasured_s
timeline,1,1,3.25,2.4,0.1,0.12,0.28,0.05,0.02,0.01,0.08,0.04,0,0.15
#device,index,rank,type,name,elements,compute_s,transfer_s,idle_s
device,0,0,GPU,NVIDIA A100 80GB PCIe,600000,2.2,0.1,0.95
device,1,1,GPU,NVIDIA A100 80GB PCIe,400000,2.4,0.12,0.73
#process,rank,host,launch_s,device_wait_s,halo_s,mpi_halo_s,mpi_migration_s,mpi_balance_s,mpi_sync_s,migration_s,balance_s,gather_s,unmeasured_s
process,0,compute-1-0,0.05,2.25,0.12,0.28,0.05,0.02,0.01,0.08,0.04,0,0.35
process,1,compute-1-1,0.06,2.44,0.12,0.28,0.05,0.02,0.01,0.08,0.04,0,0.15
#balance,number,iteration,policy,applied,iteration_s_before,iteration_s_after,bytes_moved,migration_s,decision_s,warmup_probe,reason,algorithm,perturbation_factor,seed,contention_adjusted,power_capped,power_budget_watts,throttle_factor
balance,1,10,dynamic threshold,1,0.04,0.032,400000,0.04,0.02,0,division changed,erad,0,42,0,0,0,1
balance,2,20,dynamic threshold,1,0.032,0.03,400000,0.04,0.02,0,division changed,erad,0,42,0,0,0,1
"""


class TestAnalyzeSuperMatrix(unittest.TestCase):
    """Unit and integration tests for analyze_super_matrix.py."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_asm_")
        self.output_dir = Path(self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_module_importable(self):
        """Module analyze_super_matrix must exist and be importable."""
        self.assertIsNotNone(asm, "analyze_super_matrix could not be imported")

    def test_parse_upstream_timing_report_csv_all_sections(self):
        """Parses #run, #timeline, #device, #process, #balance sections from write_timing_report() CSV."""
        report_path = self.output_dir / "kneighbor_erad_42_rep1_report.csv"
        report_path.write_text(SAMPLE_UPSTREAM_REPORT_CSV, encoding="utf-8")

        parsed = asm.parse_timing_report_csv(report_path)
        self.assertIsNotNone(parsed)

        # #run
        self.assertAlmostEqual(parsed["run"]["wall_s"], 3.25)
        self.assertEqual(parsed["run"]["iterations"], 100)
        self.assertEqual(parsed["run"]["mpi_processes"], 2)
        self.assertEqual(parsed["run"]["devices"], 2)
        self.assertEqual(parsed["run"]["balance_attempts"], 9)
        self.assertEqual(parsed["run"]["balance_applied"], 2)
        self.assertEqual(parsed["run"]["bytes_moved"], 800000)

        # #timeline
        tl = parsed["timeline"]
        self.assertEqual(tl["rank"], 1)
        self.assertEqual(tl["device"], 1)
        self.assertAlmostEqual(tl["opencl_compute_s"], 2.4)
        self.assertAlmostEqual(tl["halo_s"], 0.12)
        self.assertAlmostEqual(tl["mpi_halo_s"], 0.28)
        self.assertAlmostEqual(tl["mpi_migration_s"], 0.05)
        self.assertAlmostEqual(tl["mpi_balance_s"], 0.02)
        self.assertAlmostEqual(tl["mpi_sync_s"], 0.01)
        self.assertAlmostEqual(tl["migration_s"], 0.08)
        self.assertAlmostEqual(tl["balance_s"], 0.04)

        # #device
        self.assertEqual(len(parsed["devices"]), 2)
        self.assertEqual(parsed["devices"][0]["name"], "NVIDIA A100 80GB PCIe")
        self.assertEqual(parsed["devices"][0]["type"], "GPU")
        self.assertEqual(parsed["devices"][1]["elements"], 400000)

        # #process
        self.assertEqual(len(parsed["processes"]), 2)
        self.assertEqual(parsed["processes"][0]["host"], "compute-1-0")
        self.assertEqual(parsed["processes"][1]["host"], "compute-1-1")

        # #balance
        self.assertEqual(len(parsed["balance"]), 2)
        self.assertEqual(parsed["balance"][0]["applied"], 1)
        self.assertEqual(parsed["balance"][0]["algorithm"], "erad")
        self.assertEqual(parsed["balance"][0]["bytes_moved"], 400000)

    def test_upstream_run_integrates_report_csv_without_double_counting(self):
        """Upstream run with log + per-iter CSV + _report.csv produces 1 run with timeline/device/process/balance metrics."""
        log_path = self.output_dir / "kneighbor_erad_42_rep1.log"
        csv_path = self.output_dir / "kneighbor_erad_42_rep1.csv"
        report_path = self.output_dir / "kneighbor_erad_42_rep1_report.csv"

        log_path.write_text(
            "[device] rank=0 global_index=0 name=NVIDIA A100 80GB PCIe\n"
            "[device] rank=1 global_index=1 name=NVIDIA A100 80GB PCIe\n"
            "[kneighbor] algo=erad mode=gpu_opencl total_time_s=4.000000 loop_time_s=3.250000 num_rebalances=2 avg_rebalance_gain=0.004\n",
            encoding="utf-8",
        )
        csv_rows = ["iteration,step_time_s,action,rebalanced,rebalance_gain,execution_mode\n"]
        for i in range(100):
            rebal = 1 if i in (9, 19) else 0
            csv_rows.append(f"{i},0.032,{'rebalance' if rebal else 'compute'},{rebal},0.004,gpu_opencl\n")
        csv_path.write_text("".join(csv_rows), encoding="utf-8")
        report_path.write_text(SAMPLE_UPSTREAM_REPORT_CSV, encoding="utf-8")

        runs = asm.discover_runs(self.output_dir, expected_iterations=100)
        self.assertEqual(len(runs), 1, "Must not double-count _report.csv as a second run")
        r = runs[0]
        self.assertEqual(r["status"], "SUCCESS")
        self.assertAlmostEqual(r["wall_s"], 4.0)
        self.assertAlmostEqual(r["loop_s"], 3.25)
        self.assertAlmostEqual(r["T_compute_s"], 2.4)
        self.assertAlmostEqual(r["T_comm_s"], 0.12 + 0.28 + 0.05 + 0.02 + 0.01)
        self.assertAlmostEqual(r["T_balance_s"], 0.08 + 0.04)
        self.assertEqual(r["rebalance_count"], 2)
        self.assertEqual(r["bytes_moved"], 800000)

        asm.generate_artifacts(self.output_dir, runs)
        report_md = (self.output_dir / "SUPER_JOB_REPORT.md").read_text(encoding="utf-8")
        self.assertIn("compute-1-0", report_md)
        self.assertIn("compute-1-1", report_md)
        self.assertIn("NVIDIA A100 80GB PCIe", report_md)

    def test_standalone_upstream_report_csv_discovery(self):
        """A standalone *_report.csv without per-iter CSV or log is parsed cleanly via #run/#timeline/#device."""
        report_path = self.output_dir / "leanmd_hwtopolb_123_rep2_report.csv"
        report_path.write_text(
            SAMPLE_UPSTREAM_REPORT_CSV.replace(",erad,0,42,", ",hwtopolb,0.1,123,"),
            encoding="utf-8",
        )
        runs = asm.discover_runs(self.output_dir, expected_iterations=100)
        self.assertEqual(len(runs), 1)
        r = runs[0]
        self.assertEqual(r["benchmark"], "leanmd")
        self.assertEqual(r["algo"], "hwtopolb")
        self.assertEqual(r["seed"], 123)
        self.assertEqual(r["repetition"], 2)
        self.assertEqual(r["status"], "SUCCESS")
        self.assertAlmostEqual(r["loop_s"], 3.25)
        self.assertAlmostEqual(r["T_compute_s"], 2.4)

    def test_parse_balance_metrics_historical_evidence(self):
        """Parses Historical Evidence [DCL][balance-metrics] block into structured records."""
        sample_log = """
[DCL][balance-metrics] iteration=1
  policy=off
  action=skip
  reason=balance disabled
  T_compute_interval_s=2.7648e-05
  T_comm_interval_s=0
  T_balance_total_s=0
  T_rebalance_apply_s=0
  T_compute_accum_s=2.7648e-05
  T_comm_accum_s=0
  T_balance_accum_s=0
  T_rebalance_apply_accum_s=0
  device_times_s:
    part[0]=2.3552e-05
    part[1]=2.7648e-05
  OLD loads:
    part[0] cum=50% share=50% device_global=0 rank=0 local_index=0 offset=0 count=500000
    part[1] cum=100% share=50% device_global=1 rank=1 local_index=-1 offset=500000 count=500000
    total=100%
  CURRENT loads:
    part[0] cum=40% share=40% device_global=0 rank=0 local_index=0 offset=0 count=400000
    part[1] cum=100% share=60% device_global=1 rank=1 local_index=-1 offset=400000 count=600000
    total=100%
"""
        metrics = asm.parse_balance_metrics(sample_log)
        self.assertEqual(len(metrics), 1)
        m = metrics[0]
        self.assertEqual(m["iteration"], 1)
        self.assertAlmostEqual(m["T_compute_interval_s"], 2.7648e-05)
        self.assertEqual(len(m["old_loads"]), 2)
        self.assertEqual(m["old_loads"][0]["count"], 500000)
        self.assertEqual(len(m["current_loads"]), 2)
        self.assertEqual(m["current_loads"][0]["count"], 400000)
        self.assertEqual(m["current_loads"][1]["count"], 600000)

    def test_historical_dataset_reconciliation_synthetic_fixture(self):
        """Validates Historical Evidence reconciliation against HISTORICAL_PRELIMINARY reference ratios."""
        for (bench, seed), vals in asm.HISTORICAL_PRELIMINARY.items():
            for algo, wall_key in (("erad", "erad_s"), ("hwtopolb", "hwtopolb_s")):
                stem = f"{bench}_{algo}_{seed}"
                (self.output_dir / f"{stem}.log").write_text(
                    "[device] rank=0 global_index=0 name=NVIDIA A100 80GB PCIe\n"
                    f"[{bench}] algo={algo} mode=gpu_opencl total_time_s={vals[wall_key]} "
                    "num_rebalances=9 avg_rebalance_gain=6.34934e-05\n",
                    encoding="utf-8",
                )
                csv_lines = ["iteration,step_time_s,action,rebalanced,rebalance_gain,execution_mode\n"]
                for i in range(100):
                    csv_lines.append(f"{i},0.01,compute,0,0,gpu_opencl\n")
                (self.output_dir / f"{stem}.csv").write_text("".join(csv_lines), encoding="utf-8")

        script = BENCHMARKS_DIR / "analyze_super_matrix.py"
        proc = subprocess.run(
            [sys.executable, str(script), "--output-dir", str(self.output_dir), "--summarize-only"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, f"Analyzer failed: {proc.stderr}")

        with open(self.output_dir / "super_matrix_paired.csv", newline="", encoding="utf-8") as f:
            pairs = list(csv.DictReader(f))
        self.assertEqual(len(pairs), 6)
        for p in pairs:
            expected = asm.HISTORICAL_PRELIMINARY[(p["benchmark"], int(p["seed"]))]
            self.assertAlmostEqual(float(p["ratio"]), expected["ratio"], places=2)
            self.assertAlmostEqual(float(p["speedup_pct"]), expected["speedup_pct"], places=2)

    def test_parse_hardware_telemetry_single_and_multinode(self):
        """Parses both single-node and host/rank-tagged multi-node hardware_telemetry.log."""
        telemetry_file = self.output_dir / "hardware_telemetry.log"
        telemetry_file.write_text(
            "host, rank, timestamp, name, temperature.gpu, utilization.gpu, clocks.current.graphics, clocks.current.memory, power.draw, clocks_throttle_reasons.active\n"
            "compute-1-0, 0, 2026/10/20 22:00:00.000, NVIDIA A100 80GB PCIe, 38, 12, 1410, 1215, 64.20, [N/A]\n"
            "compute-1-1, 1, 2026/10/20 22:00:00.000, NVIDIA A100 80GB PCIe, 42, 18, 1410, 1215, 78.50, [N/A]\n",
            encoding="utf-8",
        )
        records = asm.parse_hardware_telemetry(telemetry_file)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["host"], "compute-1-0")
        self.assertEqual(records[0]["rank"], 0)
        self.assertEqual(records[0]["temperature_gpu"], 38)
        self.assertAlmostEqual(records[0]["power_draw_w"], 64.20)
        self.assertEqual(records[1]["host"], "compute-1-1")
        self.assertEqual(records[1]["rank"], 1)
        self.assertEqual(records[1]["temperature_gpu"], 42)
        self.assertAlmostEqual(records[1]["power_draw_w"], 78.50)

    def test_unavailable_telemetry_marker_is_reported_not_dropped(self):
        """An UNAVAILABLE marker yields an explicit unavailable record and report row without polluting stats."""
        telemetry_file = self.output_dir / "hardware_telemetry.log"
        telemetry_file.write_text(
            "# [telemetry] block=before_warmup timestamp=2026-10-20 22:00:00\n"
            "compute-1-0, 0, 2026/10/20 22:00:00.000, NVIDIA A100 80GB PCIe, 38, 12, 1410, 1215, 64.20, [N/A]\n"
            "# [telemetry-unavailable] host=compute-1-1 rank=1 block=before_warmup reason=no ssh-free path to remote node\n",
            encoding="utf-8",
        )
        records = asm.parse_hardware_telemetry(telemetry_file)
        unavailable = [r for r in records if r.get("host") == "compute-1-1"]
        self.assertEqual(len(unavailable), 1)
        self.assertFalse(unavailable[0]["available"])

        runs = [{
            "benchmark": "kneighbor", "algo": "erad", "seed": 42, "repetition": 1, "stage": "matrix",
            "wall_s": 3.74, "step_sum_s": 2.74, "unaccounted_s": 1.0, "rebalance_count": 9, "status": "SUCCESS",
        }]
        asm.generate_artifacts(self.output_dir, runs)
        content = (self.output_dir / "SUPER_JOB_REPORT.md").read_text(encoding="utf-8")
        row = next(l for l in content.splitlines() if l.startswith("| compute-1-1 |"))
        self.assertIn("UNAVAILABLE", row)
        self.assertNotIn("| GPU Temperature (°C) | 0", content)

    def test_paired_analysis_and_statistics(self):
        """Computes ERAD / HWTOPOLB speedup ratios, percentages, and repetition statistics."""
        runs = [
            {"benchmark": "kneighbor", "algo": "erad", "seed": 42, "repetition": 1, "stage": "matrix",
             "wall_s": 3.74, "step_sum_s": 2.74, "unaccounted_s": 1.00, "rebalance_count": 9, "status": "SUCCESS"},
            {"benchmark": "kneighbor", "algo": "hwtopolb", "seed": 42, "repetition": 1, "stage": "matrix",
             "wall_s": 0.22, "step_sum_s": 0.16, "unaccounted_s": 0.06, "rebalance_count": 9, "status": "SUCCESS"},
            {"benchmark": "kneighbor", "algo": "erad", "seed": 42, "repetition": 2, "stage": "matrix",
             "wall_s": 3.76, "step_sum_s": 2.75, "unaccounted_s": 1.01, "rebalance_count": 9, "status": "SUCCESS"},
            {"benchmark": "kneighbor", "algo": "hwtopolb", "seed": 42, "repetition": 2, "stage": "matrix",
             "wall_s": 0.24, "step_sum_s": 0.17, "unaccounted_s": 0.07, "rebalance_count": 9, "status": "SUCCESS"},
        ]
        paired = asm.compute_paired_analysis(runs)
        self.assertEqual(len(paired), 2)
        self.assertAlmostEqual(paired[0]["ratio"], 3.74 / 0.22, places=2)
        self.assertAlmostEqual(paired[0]["speedup_pct"], ((3.74 - 0.22) / 3.74) * 100, places=2)

        stats = asm.compute_summary_statistics(runs)
        self.assertEqual(len(stats), 2)
        erad_stat = [s for s in stats if s["algo"] == "erad"][0]
        self.assertEqual(erad_stat["repetition_count"], 2)
        self.assertAlmostEqual(erad_stat["median_wall_s"], 3.75)

    def test_aba_thermal_drift_and_pairing_no_overwrite(self):
        """ERAD step 1 is not overwritten by ERAD step 3, and dedicated drift metrics are computed."""
        runs = [
            {"benchmark": "kneighbor", "algo": "erad", "seed": 123, "repetition": 1, "step": 1, "stage": "aba",
             "wall_s": 3.50, "status": "SUCCESS"},
            {"benchmark": "kneighbor", "algo": "hwtopolb", "seed": 123, "repetition": 2, "step": 2, "stage": "aba",
             "wall_s": 0.25, "status": "SUCCESS"},
            {"benchmark": "kneighbor", "algo": "erad", "seed": 123, "repetition": 3, "step": 3, "stage": "aba",
             "wall_s": 3.60, "status": "SUCCESS"},
        ]
        pairs = asm.compute_paired_analysis(runs)
        self.assertEqual(len(pairs), 2)
        p1 = next(p for p in pairs if p.get("step") == 1)
        p3 = next(p for p in pairs if p.get("step") == 3)
        self.assertAlmostEqual(p1["erad_wall_s"], 3.50)
        self.assertAlmostEqual(p3["erad_wall_s"], 3.60)

        drift = asm.compute_aba_drift_analysis(runs)
        self.assertEqual(len(drift), 1)
        self.assertAlmostEqual(drift[0]["drift_delta_s"], 0.10)
        self.assertAlmostEqual(drift[0]["bracketed_erad_mean_s"], 3.55)
        self.assertAlmostEqual(drift[0]["bracketed_ratio"], 3.55 / 0.25)

    def test_manifest_failure_overrides_csv_success_and_excludes_from_paired(self):
        """Manifest failure or timeout status overrides CSV/log success and excludes run from pairing."""
        erad_log = self.output_dir / "kneighbor_erad_42.log"
        erad_csv = self.output_dir / "kneighbor_erad_42.csv"
        hw_log = self.output_dir / "kneighbor_hwtopolb_42.log"
        hw_csv = self.output_dir / "kneighbor_hwtopolb_42.csv"

        erad_log.write_text("[kneighbor] algo=erad mode=gpu_opencl total_time_s=1.5 num_rebalances=0 avg_rebalance_gain=0\n", encoding="utf-8")
        erad_csv.write_text("iteration,step_time_s,action,rebalanced,rebalance_gain,execution_mode\n0,0.01,compute,0,0,gpu_opencl\n", encoding="utf-8")
        hw_log.write_text("[kneighbor] algo=hwtopolb mode=gpu_opencl total_time_s=0.5 num_rebalances=0 avg_rebalance_gain=0\n", encoding="utf-8")
        hw_csv.write_text("iteration,step_time_s,action,rebalanced,rebalance_gain,execution_mode\n0,0.01,compute,0,0,gpu_opencl\n", encoding="utf-8")

        run_hw = asm.parse_run(hw_log, hw_csv, expected_iterations=1)
        self.assertEqual(run_hw["status"], "SUCCESS")

        run_failed = asm.parse_run(erad_log, erad_csv, expected_iterations=1, manifest_run={"status": "failed", "exit_code": 1})
        self.assertEqual(run_failed["status"], "FAILED")
        self.assertEqual(asm.compute_paired_analysis([run_failed, run_hw]), [])

        run_timeout = asm.parse_run(erad_log, erad_csv, expected_iterations=1, manifest_run={"status": "timeout", "exit_code": 124})
        self.assertEqual(run_timeout["status"], "TIMEOUT")
        self.assertEqual(asm.compute_paired_analysis([run_timeout, run_hw]), [])

        run_exit2 = asm.parse_run(erad_log, erad_csv, expected_iterations=1, manifest_run={"exit_code": 2, "status": "success"})
        self.assertEqual(run_exit2["status"], "FAILED")

    def test_seed_123_verdict_filters_matrix_stage_exclusively(self):
        """Diagnostic verdict for Seed 123 must strictly evaluate rotated matrix pairs, excluding warmup and ABA."""
        runs = [
            {"benchmark": "kneighbor", "algo": "erad", "seed": 123, "repetition": 1, "stage": "warmup", "wall_s": 0.5, "status": "SUCCESS"},
            {"benchmark": "kneighbor", "algo": "hwtopolb", "seed": 123, "repetition": 1, "stage": "warmup", "wall_s": 1.0, "status": "SUCCESS"},
            {"benchmark": "kneighbor", "algo": "erad", "seed": 123, "repetition": 1, "stage": "aba", "step": 1, "wall_s": 0.7, "status": "SUCCESS"},
            {"benchmark": "kneighbor", "algo": "hwtopolb", "seed": 123, "repetition": 2, "stage": "aba", "step": 2, "wall_s": 1.0, "status": "SUCCESS"},
        ]
        for rep in range(1, 6):
            runs.append({"benchmark": "kneighbor", "algo": "erad", "seed": 123, "repetition": rep, "stage": "matrix", "wall_s": 10.0, "status": "SUCCESS"})
            runs.append({"benchmark": "kneighbor", "algo": "hwtopolb", "seed": 123, "repetition": rep, "stage": "matrix", "wall_s": 1.0, "status": "SUCCESS"})

        asm.generate_artifacts(self.output_dir, runs)
        report_text = (self.output_dir / "SUPER_JOB_REPORT.md").read_text(encoding="utf-8")
        self.assertIn("- **Evaluated Seed 123 Rotated Matrix Pairs:** 5", report_text)
        self.assertIn("Verdict: Observed HWTOPOLB Advantage Across Rotated Matrix Repetitions", report_text)
        self.assertIn("Descriptive Scope:", report_text)

    def test_no_fallback_to_other_stages_when_matrix_runs_fail(self):
        """When all matrix runs fail, statistics and report must NOT fall back to warmup or aba runs."""
        runs = [
            {"benchmark": "kneighbor", "algo": "erad", "seed": 42, "repetition": 1, "stage": "matrix", "wall_s": None, "status": "FAILED"},
            {"benchmark": "kneighbor", "algo": "erad", "seed": 42, "repetition": 2, "stage": "matrix", "wall_s": 3.5, "status": "TIMEOUT"},
            {"benchmark": "kneighbor", "algo": "erad", "seed": 42, "repetition": None, "stage": "warmup", "wall_s": 100.0, "status": "SUCCESS"},
            {"benchmark": "kneighbor", "algo": "erad", "seed": 42, "repetition": 1, "step": 1, "stage": "aba", "wall_s": 999.0, "status": "SUCCESS"},
        ]
        self.assertEqual(asm.compute_matrix_statistics(runs, stage="matrix"), [])
        self.assertEqual(asm.compute_paired_analysis(runs, stage="matrix"), [])

        asm.generate_artifacts(self.output_dir, runs)
        report_text = (self.output_dir / "SUPER_JOB_REPORT.md").read_text(encoding="utf-8")
        matrix_summary_table = report_text.split("## 2.")[1].split("###")[0]
        self.assertNotIn("100.000000", matrix_summary_table)
        self.assertNotIn("999.000000", matrix_summary_table)
        self.assertIn("*(no successful matrix samples)*", matrix_summary_table)

    def test_paired_analysis_prioritizes_loop_time_s(self):
        """Paired analysis uses loop_s instead of wall_s when present, avoiding verification inflation."""
        runs = [
            {"benchmark": "kneighbor", "algo": "erad", "seed": 42, "repetition": 1, "stage": "matrix",
             "wall_s": 12.0, "loop_s": 2.0, "status": "SUCCESS"},
            {"benchmark": "kneighbor", "algo": "hwtopolb", "seed": 42, "repetition": 1, "stage": "matrix",
             "wall_s": 10.2, "loop_s": 0.2, "status": "SUCCESS"},
        ]
        paired = asm.compute_paired_analysis(runs)
        self.assertEqual(len(paired), 1)
        self.assertAlmostEqual(paired[0]["ratio"], 10.0)
        self.assertAlmostEqual(paired[0]["speedup_pct"], 90.0)

    def test_configurable_iterations_from_manifest_and_cli(self):
        """Manifest parameters.iterations=5 marks 5-step runs SUCCESS even with default expected_iterations=100."""
        manifest_data = {
            "parameters": {"n": 1000, "iterations": 5, "balance_interval": 2, "seeds": [42]},
            "runs": [
                {"run_id": 1, "stage": "matrix", "repetition": 1, "benchmark": "kneighbor", "algorithm": "erad",
                 "seed": 42, "step": None, "csv_file": "run_erad.csv", "log_file": "run_erad.log",
                 "exit_code": 0, "status": "success"},
                {"run_id": 2, "stage": "matrix", "repetition": 1, "benchmark": "kneighbor", "algorithm": "hwtopolb",
                 "seed": 42, "step": None, "csv_file": "run_hwtopolb.csv", "log_file": "run_hwtopolb.log",
                 "exit_code": 0, "status": "success"},
            ],
        }
        (self.output_dir / "run_manifest.json").write_text(json.dumps(manifest_data), encoding="utf-8")
        csv_header = "iteration,step_time_s,action,rebalanced,rebalance_gain,execution_mode\n"
        csv_rows = "".join(f"{i},0.01,compute,0,0,gpu_opencl\n" for i in range(5))
        (self.output_dir / "run_erad.csv").write_text(csv_header + csv_rows, encoding="utf-8")
        (self.output_dir / "run_hwtopolb.csv").write_text(csv_header + csv_rows, encoding="utf-8")
        (self.output_dir / "run_erad.log").write_text(
            "[kneighbor] algo=erad mode=gpu_opencl total_time_s=0.5 loop_time_s=0.45 num_rebalances=0 avg_rebalance_gain=0\n",
            encoding="utf-8",
        )
        (self.output_dir / "run_hwtopolb.log").write_text(
            "[kneighbor] algo=hwtopolb mode=gpu_opencl total_time_s=0.4 loop_time_s=0.35 num_rebalances=0 avg_rebalance_gain=0\n",
            encoding="utf-8",
        )

        runs = asm.discover_runs(self.output_dir, expected_iterations=100)
        self.assertEqual(len(runs), 2)
        for r in runs:
            self.assertEqual(r["status"], "SUCCESS")

    def test_nonzero_exit_on_missing_dir_and_corrupt_manifest(self):
        """Analyzer exits non-zero when --output-dir is missing or run_manifest.json is malformed."""
        script = BENCHMARKS_DIR / "analyze_super_matrix.py"
        missing_proc = subprocess.run(
            [sys.executable, str(script), "--output-dir", str(self.output_dir / "does_not_exist")],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertNotEqual(missing_proc.returncode, 0)

        (self.output_dir / "run_manifest.json").write_text("{corrupt json", encoding="utf-8")
        corrupt_proc = subprocess.run(
            [sys.executable, str(script), "--output-dir", str(self.output_dir)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertNotEqual(corrupt_proc.returncode, 0)
        self.assertIn("manifest", corrupt_proc.stderr.lower())

    def test_cli_summarize_only_idempotent(self):
        """CLI --summarize-only is idempotent across repeated executions."""
        (self.output_dir / "kneighbor_erad_42.log").write_text(
            "[device] rank=0 global_index=0 name=NVIDIA A100 80GB PCIe\n"
            "[kneighbor] algo=erad mode=gpu_opencl total_time_s=3.74 num_rebalances=9 avg_rebalance_gain=6.34e-05\n",
            encoding="utf-8",
        )
        (self.output_dir / "kneighbor_erad_42.csv").write_text(
            "iteration,step_time_s,action,rebalanced,rebalance_gain,execution_mode\n"
            "0,0.02,compute,0,0,gpu_opencl\n"
            "1,0.02,compute,0,0,gpu_opencl\n",
            encoding="utf-8",
        )
        cmd = [sys.executable, str(BENCHMARKS_DIR / "analyze_super_matrix.py"),
               "--output-dir", str(self.output_dir), "--summarize-only", "--iterations", "2"]
        p1 = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.assertEqual(p1.returncode, 0, p1.stderr)
        s1 = (self.output_dir / "super_matrix_summary.csv").read_text(encoding="utf-8")

        p2 = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.assertEqual(p2.returncode, 0, p2.stderr)
        s2 = (self.output_dir / "super_matrix_summary.csv").read_text(encoding="utf-8")
        self.assertEqual(s1, s2)

    def test_cpu_fallback_and_missing_telemetry_handling(self):
        """Detects CPU_FALLBACK from CSV or standalone _report.csv and omits missing telemetry JSON."""
        csv_f = self.output_dir / "kneighbor_erad_42.csv"
        log_f = self.output_dir / "kneighbor_erad_42.log"
        csv_f.write_text("iteration,step_time_s,action,rebalanced,rebalance_gain,execution_mode\n0,0.02,compute,0,0,cpu_simulation\n", encoding="utf-8")
        log_f.write_text("[kneighbor] algo=erad mode=cpu_simulation total_time_s=1.0 num_rebalances=0 avg_rebalance_gain=0\n", encoding="utf-8")
        run_csv = asm.parse_run(log_f, csv_f, expected_iterations=1)
        self.assertEqual(run_csv["status"], "CPU_FALLBACK")

        cpu_report = self.output_dir / "leanmd_hwtopolb_42_report.csv"
        cpu_report.write_text(
            "#run,wall_s,iterations,mpi_processes,devices,balance_attempts,balance_applied,bytes_moved\n"
            "run,1.0,1,1,1,0,0,0\n"
            "#device,index,rank,type,name,elements,compute_s,transfer_s,idle_s\n"
            "device,0,0,CPU,Simulated CPU 0,1000,0.8,0,0.2\n",
            encoding="utf-8",
        )
        run_rep = asm.parse_run(None, None, expected_iterations=1, report_csv_path=cpu_report)
        self.assertEqual(run_rep["status"], "CPU_FALLBACK")

        asm.generate_artifacts(self.output_dir, [run_csv])
        self.assertTrue((self.output_dir / "super_matrix_summary.csv").exists())
        self.assertFalse((self.output_dir / "hardware_telemetry.json").exists())

    def test_unknown_hardware_and_mode_fallbacks_and_heuristic_constant(self):
        """Missing hardware/mode logs must fall back to 'unknown' labels and expose named heuristic constant."""
        self.assertEqual(asm.HEURISTIC_CLOCK_DIVERGENCE_DELTA_MHZ, 15)

        runs = [
            {"benchmark": "kneighbor", "algo": "erad", "seed": 42, "repetition": 1,
             "stage": "matrix", "wall_s": 1.0, "status": "SUCCESS"},
            {"benchmark": "kneighbor", "algo": "hwtopolb", "seed": 42, "repetition": 1,
             "stage": "matrix", "wall_s": 0.5, "status": "SUCCESS"},
        ]
        asm.generate_artifacts(self.output_dir, runs)
        report_text = (self.output_dir / "SUPER_JOB_REPORT.md").read_text(encoding="utf-8")
        self.assertIn("unknown (not observed in telemetry/logs)", report_text)
        self.assertIn("| Execution Mode | `unknown` |", report_text)
        self.assertNotIn("NVIDIA A100 80GB PCIe (unidentified)", report_text)
        self.assertNotIn("| Execution Mode | `gpu_opencl` |", report_text)


if __name__ == "__main__":
    unittest.main()
