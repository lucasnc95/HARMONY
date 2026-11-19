#!/usr/bin/env python3
"""
Offline integration test harness for benchmarks/run_super_matrix.sh and benchmarks/telemetry_probe.sh.
Verifies CLI validation, mandatory prebuilt binary enforcement (bin/ and tests/bin/),
no compute-node compilation, dry-run mode, shared 60s Preflight Verification budget,
Rotated Matrix / A/B/A schedule, multi-node telemetry, fault tolerance, and E2E analyzer integration.
"""

import hashlib
import json
import os
import shutil
import stat
import subprocess
import tempfile
import time
import unittest


class TestSuperMatrixDryRun(unittest.TestCase):
    """Offline test suite for benchmarks/run_super_matrix.sh."""

    @classmethod
    def setUpClass(cls):
        cls.root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        cls.script_path = os.path.join(cls.root_dir, "benchmarks", "run_super_matrix.sh")
        cls.mock_launcher_path = os.path.join(cls.root_dir, "tests", "mock_cluster_launcher.py")

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_super_matrix_")
        self.bin_dir = os.path.join(self.temp_dir, "bin")
        self.out_dir = os.path.join(self.temp_dir, "out")
        os.makedirs(self.bin_dir, exist_ok=True)
        os.makedirs(self.out_dir, exist_ok=True)
        self.required_binaries = [
            "kneighbor",
            "leanmd",
            "topo_probe",
            "test_async_halo_opencl",
        ]

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_dummy_binaries(self, binaries=None, target_dir=None):
        """Creates minimal executable placeholder binaries."""
        if binaries is None:
            binaries = self.required_binaries
        dest = target_dir or self.bin_dir
        os.makedirs(dest, exist_ok=True)
        for b in binaries:
            path = os.path.join(dest, b)
            with open(path, "w", encoding="utf-8") as f:
                f.write("#!/bin/sh\nexit 0\n")
            os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)

    def _run_script(self, args, env=None):
        """Helper to invoke benchmarks/run_super_matrix.sh."""
        run_env = os.environ.copy()
        if env:
            run_env.update(env)
        return subprocess.run(
            ["bash", self.script_path] + args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=run_env,
        )

    def test_missing_binary_directory_fails(self):
        """Runner must fail if --bin-dir points to a non-existent directory."""
        non_existent_dir = os.path.join(self.temp_dir, "no_such_dir")
        proc = self._run_script(["--bin-dir", non_existent_dir, "--output-dir", self.out_dir, "--dry-run"])
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("Missing", proc.stderr + proc.stdout)

    def test_missing_single_binary_fails(self):
        """Runner must fail immediately if any required binary is missing from --bin-dir."""
        self._create_dummy_binaries(["kneighbor", "leanmd", "topo_probe"])
        proc = self._run_script(["--bin-dir", self.bin_dir, "--output-dir", self.out_dir, "--dry-run"])
        self.assertNotEqual(proc.returncode, 0)
        err = proc.stderr + proc.stdout
        self.assertIn("test_async_halo_opencl", err)
        self.assertIn("Missing prebuilt binary", err)
        self.assertIn("build.sh", err)

    def test_separate_test_bin_dir_layout_supported(self):
        """Runner supports upstream split layout (--bin-dir bin/ + --test-bin-dir tests/bin/)."""
        tests_bin_dir = os.path.join(self.temp_dir, "tests_bin")
        self._create_dummy_binaries(["kneighbor", "leanmd", "topo_probe"], target_dir=self.bin_dir)
        self._create_dummy_binaries(["test_async_halo_opencl"], target_dir=tests_bin_dir)
        proc = self._run_script([
            "--bin-dir", self.bin_dir,
            "--test-bin-dir", tests_bin_dir,
            "--output-dir", self.out_dir,
            "--dry-run",
        ])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("test_async_halo_opencl", proc.stdout)

    def test_forbid_compute_node_compilation(self):
        """Runner must never invoke mpic++ or mpicxx on compute nodes."""
        self._create_dummy_binaries()
        mock_bin_dir = os.path.join(self.temp_dir, "mock_tools")
        os.makedirs(mock_bin_dir, exist_ok=True)
        for compiler in ("mpic++", "mpicxx"):
            comp_path = os.path.join(mock_bin_dir, compiler)
            with open(comp_path, "w", encoding="utf-8") as f:
                f.write(f"#!/bin/sh\necho 'ERROR: {compiler} invoked during run!' >&2\nexit 99\n")
            os.chmod(comp_path, os.stat(comp_path).st_mode | stat.S_IEXEC)

        proc = self._run_script(
            ["--bin-dir", self.bin_dir, "--output-dir", self.out_dir, "--dry-run"],
            env={"PATH": f"{mock_bin_dir}:{os.environ.get('PATH', '')}"},
        )
        self.assertEqual(proc.returncode, 0, f"Failed: {proc.stderr}")
        self.assertNotIn("invoked during run", proc.stderr)

    def test_invalid_arguments_and_help(self):
        """Runner validates --n, --iterations, --seeds and supports --help."""
        self._create_dummy_binaries()
        for invalid_n in ["0", "-100", "abc"]:
            proc = self._run_script(["--bin-dir", self.bin_dir, "--output-dir", self.out_dir, "--n", invalid_n, "--dry-run"])
            self.assertNotEqual(proc.returncode, 0)
        for invalid_iter in ["0", "-5", "xyz"]:
            proc = self._run_script(["--bin-dir", self.bin_dir, "--output-dir", self.out_dir, "--iterations", invalid_iter, "--dry-run"])
            self.assertNotEqual(proc.returncode, 0)
        for invalid_seed in ["", "abc", "42,foo"]:
            proc = self._run_script(["--bin-dir", self.bin_dir, "--output-dir", self.out_dir, "--seeds", invalid_seed, "--dry-run"])
            self.assertNotEqual(proc.returncode, 0)
        for flag in ["--help", "-h"]:
            proc = self._run_script([flag])
            self.assertEqual(proc.returncode, 0)
            self.assertIn("Usage:", proc.stdout)

    def test_output_dir_auto_creation_and_dry_run_plan(self):
        """--dry-run auto-creates --output-dir and prints the 5-stage plan without running workloads."""
        self._create_dummy_binaries()
        nested_out = os.path.join(self.temp_dir, "nested", "run_out")
        proc = self._run_script(["--bin-dir", self.bin_dir, "--output-dir", nested_out, "--dry-run"])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertTrue(os.path.isdir(nested_out))
        out = proc.stdout
        self.assertIn("[DRY-RUN]", out)
        self.assertIn("Preflight Verification", out)
        self.assertIn("topo_probe", out)
        self.assertIn("test_async_halo_opencl", out)
        self.assertIn("kneighbor", out)
        self.assertIn("leanmd", out)
        self.assertIn("123", out)

    def test_mock_launcher_execution_generates_upstream_report_csvs(self):
        """Runner executes with mock launcher and generates per-iter CSVs, _report.csvs, logs, and manifest."""
        self._create_dummy_binaries()
        launcher_cmd = f"python3 {self.mock_launcher_path}"
        proc = self._run_script([
            "--bin-dir", self.bin_dir,
            "--output-dir", self.out_dir,
            "--n", "64",
            "--iterations", "2",
            "--seeds", "42",
            "--launcher", launcher_cmd,
        ])
        self.assertEqual(proc.returncode, 0, f"Script failed:\n{proc.stderr}\n{proc.stdout}")

        topo_json = os.path.join(self.out_dir, "topo_metrics_cluster.json")
        self.assertTrue(os.path.isfile(topo_json))
        with open(topo_json, encoding="utf-8") as f:
            data = json.load(f)
        self.assertFalse(data.get("synthetic", True))
        self.assertEqual(len(data.get("pcie_latency_ns", [])), 2)

        for bench in ["kneighbor", "leanmd"]:
            for algo in ["erad", "hwtopolb"]:
                rep1_csv = os.path.join(self.out_dir, f"{bench}_{algo}_42_rep1.csv")
                rep1_report_csv = os.path.join(self.out_dir, f"{bench}_{algo}_42_rep1_report.csv")
                rep1_log = os.path.join(self.out_dir, f"{bench}_{algo}_42_rep1.log")
                warmup_csv = os.path.join(self.out_dir, f"warmup_{bench}_{algo}_42.csv")

                self.assertTrue(os.path.isfile(rep1_csv))
                self.assertTrue(os.path.isfile(rep1_report_csv))
                self.assertTrue(os.path.isfile(rep1_log))
                self.assertTrue(os.path.isfile(warmup_csv))

                with open(rep1_report_csv, encoding="utf-8") as f:
                    report_text = f.read()
                self.assertIn("#run,", report_text)
                self.assertIn("#timeline,", report_text)
                self.assertIn("#device,", report_text)
                self.assertIn("#process,", report_text)
                self.assertIn("#balance,", report_text)

                with open(rep1_log, encoding="utf-8") as f:
                    log_text = f.read()
                self.assertIn("total_time_s=", log_text)
                self.assertIn("loop_time_s=", log_text)

    def test_output_isolation_does_not_modify_benchmarks_dir(self):
        """Runner must never modify existing files in benchmarks/."""
        self._create_dummy_binaries()
        benchmarks_dir = os.path.join(self.root_dir, "benchmarks")
        watched_files = [
            f for f in os.listdir(benchmarks_dir)
            if f.endswith((".cpp", ".cl", ".hpp", ".sh", ".py"))
        ]
        self.assertGreater(len(watched_files), 0)
        hashes_before = {}
        for fname in watched_files:
            with open(os.path.join(benchmarks_dir, fname), "rb") as f:
                hashes_before[fname] = hashlib.sha256(f.read()).hexdigest()

        proc = self._run_script([
            "--bin-dir", self.bin_dir,
            "--output-dir", self.out_dir,
            "--n", "32",
            "--iterations", "2",
            "--seeds", "42",
            "--launcher", f"python3 {self.mock_launcher_path}",
        ])
        self.assertEqual(proc.returncode, 0, proc.stderr)

        for fname, before_hash in hashes_before.items():
            with open(os.path.join(benchmarks_dir, fname), "rb") as f:
                after_hash = hashlib.sha256(f.read()).hexdigest()
            self.assertEqual(after_hash, before_hash, f"Modified benchmarks/{fname}!")

    def test_preflight_failure_aborts_execution(self):
        """If preflight halo check fails, runner must abort before running benchmarks."""
        self._create_dummy_binaries()
        proc = self._run_script(
            ["--bin-dir", self.bin_dir, "--output-dir", self.out_dir, "--n", "32",
             "--iterations", "2", "--seeds", "42", "--launcher", f"python3 {self.mock_launcher_path}"],
            env={"MOCK_FAIL_PREFLIGHT_HALO": "1"},
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("Preflight", proc.stderr + proc.stdout)
        self.assertEqual([f for f in os.listdir(self.out_dir) if f.endswith(".csv")], [])

    def test_asymmetric_gpu_allocation_fails_preflight(self):
        """Asymmetric GPU allocation fails test_async_halo_opencl --require-gpu before benchmarks run."""
        self._create_dummy_binaries()
        proc = self._run_script(
            ["--bin-dir", self.bin_dir, "--output-dir", self.out_dir, "--n", "32",
             "--iterations", "2", "--seeds", "42", "--launcher", f"python3 {self.mock_launcher_path}"],
            env={"MOCK_ASYMMETRIC_GPUS": "1"},
        )
        self.assertNotEqual(proc.returncode, 0)
        output = proc.stderr + proc.stdout
        self.assertIn("ERROR: Asynchronous halo test preflight failed (requires at least 1 OpenCL GPU per rank)", output)
        self.assertEqual([f for f in os.listdir(self.out_dir) if f.endswith(".csv")], [])

    def test_asymmetric_topology_distribution_fails_topology_check(self):
        """2+0 per-rank device distribution fails the TOPOLOGY check before halo test runs."""
        self._create_dummy_binaries()
        proc = self._run_script(
            ["--bin-dir", self.bin_dir, "--output-dir", self.out_dir, "--n", "32",
             "--iterations", "2", "--seeds", "42", "--launcher", f"python3 {self.mock_launcher_path}"],
            env={"MOCK_ASYMMETRIC_TOPO": "1"},
        )
        output = proc.stderr + proc.stdout
        self.assertNotEqual(proc.returncode, 0, output)
        self.assertIn("Preflight failed: per-rank OpenCL GPU counts", output)
        self.assertNotIn("Asynchronous halo test preflight failed", output)

    def test_preflight_budget_is_shared_across_steps(self):
        """Topo probe and halo test share one preflight deadline instead of 60s each."""
        self._create_dummy_binaries()
        started = time.monotonic()
        proc = self._run_script(
            ["--bin-dir", self.bin_dir, "--output-dir", self.out_dir, "--n", "32",
             "--iterations", "2", "--seeds", "42", "--launcher", f"python3 {self.mock_launcher_path}"],
            env={"PREFLIGHT_BUDGET_SECONDS": "3", "MOCK_SLEEP_TOPO": "2", "MOCK_SLEEP_HALO": "2"},
        )
        elapsed = time.monotonic() - started
        self.assertNotEqual(proc.returncode, 0, proc.stderr + proc.stdout)
        self.assertLess(elapsed, 3.9, f"preflight overran shared 3s budget ({elapsed:.1f}s)")

    def test_preflight_budget_defaults_to_60_seconds_total(self):
        with open(self.script_path, "r", encoding="utf-8") as f:
            script_text = f.read()
        self.assertIn("PREFLIGHT_BUDGET_SECONDS:-60", script_text)
        self.assertNotIn("timeout 60s", script_text)

    def test_warmup_matrix_and_aba_execution_schedule(self):
        """Verify warmup (4), 5 repetitions (60), and ABA (6) execute in exact rotated order."""
        self._create_dummy_binaries()
        proc = self._run_script([
            "--bin-dir", self.bin_dir,
            "--output-dir", self.out_dir,
            "--n", "64",
            "--iterations", "2",
            "--seeds", "42,123,456",
            "--launcher", f"python3 {self.mock_launcher_path}",
        ])
        self.assertEqual(proc.returncode, 0, proc.stderr)

        with open(os.path.join(self.out_dir, "run_manifest.json"), encoding="utf-8") as f:
            manifest = json.load(f)
        runs = manifest["runs"]
        self.assertEqual(len(runs), 70)

        warmup_runs = [r for r in runs if r["stage"] == "warmup"]
        self.assertEqual(len(warmup_runs), 4)
        self.assertIn("report_csv_file", warmup_runs[0])

        matrix_runs = [r for r in runs if r["stage"] == "matrix"]
        self.assertEqual(len(matrix_runs), 60)

        expected_rep_seeds = {
            1: [42, 123, 456],
            2: [123, 456, 42],
            3: [456, 42, 123],
            4: [42, 123, 456],
            5: [123, 456, 42],
        }
        expected_rep_algos = {
            1: ["erad", "hwtopolb"],
            2: ["hwtopolb", "erad"],
            3: ["erad", "hwtopolb"],
            4: ["hwtopolb", "erad"],
            5: ["erad", "hwtopolb"],
        }
        for rep in range(1, 6):
            rep_runs = [r for r in matrix_runs if r["repetition"] == rep]
            self.assertEqual(len(rep_runs), 12)
            idx = 0
            for bench in ["kneighbor", "leanmd"]:
                for algo in expected_rep_algos[rep]:
                    for seed in expected_rep_seeds[rep]:
                        r = rep_runs[idx]
                        self.assertEqual((r["benchmark"], r["algorithm"], r["seed"]), (bench, algo, seed))
                        idx += 1

        aba_runs = [r for r in runs if r["stage"] == "aba"]
        self.assertEqual(len(aba_runs), 6)
        self.assertEqual(
            [(r["benchmark"], r["algorithm"], r["seed"], r["step"]) for r in aba_runs],
            [
                ("kneighbor", "erad", 123, 1),
                ("kneighbor", "hwtopolb", 123, 2),
                ("kneighbor", "erad", 123, 3),
                ("leanmd", "erad", 123, 1),
                ("leanmd", "hwtopolb", 123, 2),
                ("leanmd", "erad", 123, 3),
            ],
        )

    def test_hardware_telemetry_multinode_and_nodefile_fallback(self):
        """Multi-node telemetry via launcher and PBS_NODEFILE unavailable-marker fallback."""
        self._create_dummy_binaries()
        proc = self._run_script([
            "--bin-dir", self.bin_dir,
            "--output-dir", self.out_dir,
            "--n", "64",
            "--iterations", "2",
            "--seeds", "42",
            "--launcher", f"python3 {self.mock_launcher_path}",
        ])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        with open(os.path.join(self.out_dir, "hardware_telemetry.log"), encoding="utf-8") as f:
            telem = f.read()
        self.assertIn("compute-1-0, 0,", telem)
        self.assertIn("compute-1-1, 1,", telem)
        self.assertIn("warmup", telem)
        self.assertIn("rep", telem)
        self.assertIn("aba", telem)

        # Now test PBS_NODEFILE fallback when telemetry probe fails
        mock_bin_dir = os.path.join(self.temp_dir, "mock_smi_dir")
        os.makedirs(mock_bin_dir, exist_ok=True)
        mock_smi = os.path.join(mock_bin_dir, "nvidia-smi")
        with open(mock_smi, "w", encoding="utf-8") as f:
            f.write('#!/bin/sh\necho "2026/10/20 22:00:00.000, NVIDIA A100-PCIE-40GB, 38, 12, 1410, 1215, 78.42, [N/A]"\n')
        os.chmod(mock_smi, 0o755)

        wrapper = os.path.join(self.temp_dir, "failing_telemetry_launcher.sh")
        with open(wrapper, "w", encoding="utf-8") as f:
            f.write(
                '#!/bin/sh\ncase "$*" in *telemetry_probe*) exit 1;; esac\n'
                f'exec python3 "{self.mock_launcher_path}" "$@"\n'
            )
        os.chmod(wrapper, 0o755)

        nodefile = os.path.join(self.temp_dir, "nodefile")
        with open(nodefile, "w", encoding="utf-8") as f:
            f.write("remote-node-a[0]\nremote-node-b[0]\n")

        out2 = os.path.join(self.temp_dir, "out2")
        proc2 = self._run_script(
            ["--bin-dir", self.bin_dir, "--output-dir", out2, "--n", "32",
             "--iterations", "2", "--seeds", "42", "--launcher", wrapper, "--skip-preflight"],
            env={"PATH": f"{mock_bin_dir}:{os.environ.get('PATH', '')}", "PBS_NODEFILE": nodefile},
        )
        self.assertEqual(proc2.returncode, 0, proc2.stderr)
        with open(os.path.join(out2, "hardware_telemetry.log"), encoding="utf-8") as f:
            telem2 = f.read()
        self.assertIn("remote-node-a, 0,", telem2)
        self.assertIn("# [telemetry-unavailable] host=remote-node-b rank=1", telem2)
        self.assertNotIn("remote-node-b, 1, ", telem2)

    def test_fault_tolerance_failures_and_timeouts(self):
        """Matrix failure/timeout exits 1 after completing remaining runs; warmup failure exits 0."""
        self._create_dummy_binaries()
        launcher_cmd = f"python3 {self.mock_launcher_path}"

        # 1. Warmup failure does not fail runner
        warmup_marker = os.path.join(self.temp_dir, "fail_warmup.marker")
        out_w = os.path.join(self.temp_dir, "out_w")
        proc_w = self._run_script(
            ["--bin-dir", self.bin_dir, "--output-dir", out_w, "--n", "64", "--iterations", "2", "--seeds", "42", "--launcher", launcher_cmd],
            env={"MOCK_FAIL_WARMUP_ONCE_FILE": warmup_marker},
        )
        self.assertEqual(proc_w.returncode, 0, proc_w.stderr)

        # 2. Matrix failure exits 1 and completes all 30 runs
        fail_marker = os.path.join(self.temp_dir, "fail_once.marker")
        out_f = os.path.join(self.temp_dir, "out_f")
        proc_f = self._run_script(
            ["--bin-dir", self.bin_dir, "--output-dir", out_f, "--n", "64", "--iterations", "2", "--seeds", "42", "--launcher", launcher_cmd],
            env={"MOCK_FAIL_ONCE_FILE": fail_marker},
        )
        self.assertEqual(proc_f.returncode, 1)
        with open(os.path.join(out_f, "run_manifest.json"), encoding="utf-8") as f:
            runs_f = json.load(f)["runs"]
        self.assertEqual(len(runs_f), 30)
        self.assertEqual(len([r for r in runs_f if r["status"] == "failed"]), 1)

        # 3. Matrix timeout exits 1 with exit_code 124
        sleep_marker = os.path.join(self.temp_dir, "sleep_once.marker")
        out_t = os.path.join(self.temp_dir, "out_t")
        proc_t = self._run_script(
            ["--bin-dir", self.bin_dir, "--output-dir", out_t, "--n", "64", "--iterations", "2", "--seeds", "42", "--launcher", launcher_cmd],
            env={"SAMPLE_TIMEOUT_SECONDS": "2", "MOCK_SLEEP_ONCE_FILE": sleep_marker, "MOCK_SLEEP_SECONDS": "4"},
        )
        self.assertEqual(proc_t.returncode, 1)
        with open(os.path.join(out_t, "run_manifest.json"), encoding="utf-8") as f:
            runs_t = json.load(f)["runs"]
        timed_out = [r for r in runs_t if r["status"] == "timeout"]
        self.assertEqual(len(timed_out), 1)
        self.assertEqual(timed_out[0]["exit_code"], 124)

    def test_multinode_telemetry_dryrun_e2e_analysis(self):
        """End-to-end: runner captures telemetry + upstream report CSVs and analyzer produces JSON & report."""
        self._create_dummy_binaries()
        proc = self._run_script([
            "--bin-dir", self.bin_dir,
            "--output-dir", self.out_dir,
            "--n", "32",
            "--iterations", "2",
            "--seeds", "42",
            "--launcher", f"python3 {self.mock_launcher_path}",
            "--skip-preflight",
        ])
        self.assertEqual(proc.returncode, 0, proc.stderr)

        analyzer_script = os.path.join(self.root_dir, "benchmarks", "analyze_super_matrix.py")
        res = subprocess.run(
            ["python3", analyzer_script, "--output-dir", self.out_dir],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertEqual(res.returncode, 0, f"Analyzer failed:\n{res.stderr}\n{res.stdout}")

        with open(os.path.join(self.out_dir, "hardware_telemetry.json"), encoding="utf-8") as f:
            data = json.load(f)
        self.assertIn("compute-1-0", {e.get("host") for e in data})
        self.assertIn("compute-1-1", {e.get("host") for e in data})

        with open(os.path.join(self.out_dir, "SUPER_JOB_REPORT.md"), encoding="utf-8") as f:
            report_text = f.read()
        self.assertIn("Hardware Telemetry by Node & Rank", report_text)
        self.assertIn("Aggregate Cluster Metrics", report_text)
        self.assertIn("NVIDIA A100 80GB PCIe", report_text)


if __name__ == "__main__":
    unittest.main()
