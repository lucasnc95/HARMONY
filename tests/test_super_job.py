#!/usr/bin/env python3
"""
Unit and integration tests for the PBS Super Job script (job_harmony_super.pbs).
Verifies:
  - Bash syntax validity (bash -n)
  - Required PBS directives (name, output, walltime, vnodes chunks, place)
  - Integration with scripts/pbs/job_env.sh and scripts/pbs/rank_wrapper.sh (MIG rejection)
  - LIMC module loading rules (no module purge, correct module defaults)
  - Prebuilt binary verification (bin/ and tests/bin/, fail fast citing ./build.sh all)
  - Nodefile slot count verification (exactly 2 slots)
  - Shared 60-second Preflight Verification timeout budget
  - Pipeline execution, configurable iterations, sample/analyzer exit propagation
  - Finite-output verification in benchmarks/benchmark_common.hpp, kneighbor.cpp, leanmd.cpp
"""

import csv
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
import unittest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
JOB_SCRIPT_PATH = os.path.join(ROOT_DIR, "job_harmony_super.pbs")
MOCK_LAUNCHER_PATH = os.path.join(ROOT_DIR, "tests", "mock_cluster_launcher.py")
BENCHMARK_COMMON_HPP = os.path.join(ROOT_DIR, "benchmarks", "benchmark_common.hpp")
KNEIGHBOR_CPP_PATH = os.path.join(ROOT_DIR, "benchmarks", "kneighbor.cpp")
LEANMD_CPP_PATH = os.path.join(ROOT_DIR, "benchmarks", "leanmd.cpp")


class TestSuperJobScriptDirectivesAndSyntax(unittest.TestCase):
    """Verifies static properties of job_harmony_super.pbs."""

    def test_file_exists(self):
        """job_harmony_super.pbs must exist at repository root."""
        self.assertTrue(os.path.isfile(JOB_SCRIPT_PATH), f"Expected {JOB_SCRIPT_PATH} at repo root")

    def test_bash_syntax(self):
        """job_harmony_super.pbs must have valid bash syntax."""
        proc = subprocess.run(["bash", "-n", JOB_SCRIPT_PATH], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.assertEqual(proc.returncode, 0, f"bash -n failed:\n{proc.stderr}")

    def test_pbs_directives(self):
        """Verifies required PBS directives in job_harmony_super.pbs."""
        with open(JOB_SCRIPT_PATH, "r", encoding="utf-8") as f:
            directives = " ".join(line.strip() for line in f if line.startswith("#PBS"))

        self.assertIn("#PBS -N harmony_super", directives)
        self.assertIn("#PBS -j oe", directives)
        self.assertIn("#PBS -o harmony_super.out", directives)
        self.assertIn("#PBS -l walltime=02:00:00", directives)
        self.assertIn("#PBS -l place=free", directives)
        expected_select = (
            "#PBS -l select=1:ncpus=4:mpiprocs=1:mem=8gb:ngpus=1:vnode=compute-1-0[0]"
            "+1:ncpus=4:mpiprocs=1:mem=8gb:ngpus=1:vnode=compute-1-1[0]"
        )
        self.assertIn(expected_select, directives)

    def test_module_loading_and_pbs_env_integration(self):
        """Verifies no module purge, LIMC module defaults, and job_env.sh / rank_wrapper.sh integration."""
        with open(JOB_SCRIPT_PATH, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertFalse(
            re.search(r"^\s*module\s+purge", content, re.MULTILINE),
            "job_harmony_super.pbs must not execute 'module purge'",
        )
        self.assertIn("scripts/pbs/job_env.sh", content)
        self.assertIn("scripts/pbs/rank_wrapper.sh", content)
        self.assertIn("--check-gpus", content)
        self.assertIn("gnu13/13.2.0", content)
        self.assertIn("openmpi5/5.0.5", content)
        self.assertIn("prun/2.2", content)

    def test_analyzer_invocation_not_masked(self):
        """job_harmony_super.pbs must not mask analyze_super_matrix.py with || true."""
        with open(JOB_SCRIPT_PATH, "r", encoding="utf-8") as f:
            content = f.read()
        analyzer_lines = [l for l in content.splitlines() if "analyze_super_matrix.py" in l]
        self.assertTrue(analyzer_lines, "Expected analyze_super_matrix.py invocation")
        self.assertNotIn("|| true", analyzer_lines[0])
        self.assertIn("analyzer_exit_code", content)

    def test_configurable_iterations_and_skip_preflight_propagation(self):
        """Passes --iterations "$bench_iterations" to runner/analyzer and --skip-preflight to runner."""
        with open(JOB_SCRIPT_PATH, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertRegex(content, r'bench_iterations="\$\{BENCH_ITERATIONS:-100\}"')
        runner_match = re.search(r'run_super_matrix\.sh.*?(?:(?:\n\s*--[^\n]+)+)', content)
        self.assertIsNotNone(runner_match)
        self.assertIn('--iterations "$bench_iterations"', runner_match.group(0))
        self.assertIn("--skip-preflight", runner_match.group(0))

        analyzer_match = re.search(r'analyze_super_matrix\.py.*?(?:(?:\n\s*--[^\n]+)+)', content)
        self.assertIsNotNone(analyzer_match)
        self.assertIn('--iterations "$bench_iterations"', analyzer_match.group(0))

    def test_preflight_budget_defaults_to_60_seconds_total(self):
        with open(JOB_SCRIPT_PATH, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("PREFLIGHT_BUDGET_SECONDS:-60", content)
        self.assertNotIn("timeout 60s", content)


class TestSuperJobExecution(unittest.TestCase):
    """Dynamic execution tests for job_harmony_super.pbs with mock cluster environment."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_super_job_")
        self.mock_bin_dir = os.path.join(self.temp_dir, "mock_tools")
        self.bin_dir = os.path.join(self.temp_dir, "bin")
        self.tests_bin_dir = os.path.join(self.temp_dir, "tests", "bin")
        os.makedirs(self.mock_bin_dir, exist_ok=True)
        os.makedirs(self.bin_dir, exist_ok=True)
        os.makedirs(self.tests_bin_dir, exist_ok=True)

        shutil.copytree(os.path.join(ROOT_DIR, "scripts"), os.path.join(self.temp_dir, "scripts"))

        self.nodefile = os.path.join(self.temp_dir, "pbs_nodefile")
        with open(self.nodefile, "w", encoding="utf-8") as f:
            f.write("compute-1-0\ncompute-1-1\n")

        self.mock_module = os.path.join(self.mock_bin_dir, "module")
        with open(self.mock_module, "w", encoding="utf-8") as f:
            f.write("#!/bin/sh\nexit 0\n")
        os.chmod(self.mock_module, 0o755)

        self.mock_prun = os.path.join(self.mock_bin_dir, "prun")
        with open(self.mock_prun, "w", encoding="utf-8") as f:
            f.write(f'#!/bin/sh\nexec python3 "{MOCK_LAUNCHER_PATH}" "$@"\n')
        os.chmod(self.mock_prun, 0o755)

        self.env = os.environ.copy()
        self.env["PATH"] = f"{self.mock_bin_dir}:{self.env.get('PATH', '')}"
        self.env["PBS_O_WORKDIR"] = self.temp_dir
        self.env["PBS_NODEFILE"] = self.nodefile
        self.env["PBS_JOBID"] = "12345.headnode"
        self.env["HARMONY_ROOT"] = self.temp_dir

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_prebuilt_binaries(self, root_bins=None, include_halo=True):
        """Creates dummy prebuilt binaries in bin/ and tests/bin/."""
        if root_bins is None:
            root_bins = ["topo_probe", "kneighbor", "leanmd"]
        for b in root_bins:
            p = os.path.join(self.bin_dir, b)
            with open(p, "w", encoding="utf-8") as f:
                f.write("#!/bin/sh\nexit 0\n")
            os.chmod(p, 0o755)
        if include_halo:
            halo = os.path.join(self.tests_bin_dir, "test_async_halo_opencl")
            with open(halo, "w", encoding="utf-8") as f:
                f.write("#!/bin/sh\nexit 0\n")
            os.chmod(halo, 0o755)

    def _run_job_script(self, custom_env=None, env_override=False):
        if env_override and custom_env is not None:
            run_env = custom_env.copy()
        else:
            run_env = self.env.copy()
            if custom_env:
                run_env.update(custom_env)
        return subprocess.run(
            ["bash", JOB_SCRIPT_PATH],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=self.temp_dir,
            env=run_env,
        )

    def test_missing_pbs_env_vars_fails(self):
        """Job script must fail if PBS_O_WORKDIR or PBS_NODEFILE is missing."""
        for key in ("PBS_O_WORKDIR", "PBS_NODEFILE"):
            bad_env = self.env.copy()
            del bad_env[key]
            proc = self._run_job_script(bad_env, env_override=True)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn(key, proc.stderr + proc.stdout)

    def test_invalid_nodefile_slot_count_fails(self):
        """Job script must fail if PBS_NODEFILE does not contain exactly 2 slots."""
        with open(self.nodefile, "w", encoding="utf-8") as f:
            f.write("compute-1-0\n")
        self._create_prebuilt_binaries()
        proc = self._run_job_script()
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("two", (proc.stderr + proc.stdout).lower())

    def test_missing_prebuilt_binaries_fails_fast(self):
        """Job script must fail fast if prebuilt binaries are missing, citing build.sh."""
        proc = self._run_job_script()
        self.assertNotEqual(proc.returncode, 0)
        combined = proc.stderr + proc.stdout
        self.assertIn("Missing", combined)
        self.assertIn("build.sh", combined)

    def test_missing_single_binary_fails(self):
        """Job script must fail if any required binary (including test_async_halo_opencl) is missing."""
        self._create_prebuilt_binaries(["topo_probe", "kneighbor", "leanmd"], include_halo=False)
        proc = self._run_job_script()
        self.assertNotEqual(proc.returncode, 0)
        combined = proc.stderr + proc.stdout
        self.assertIn("test_async_halo_opencl", combined)
        self.assertIn("build.sh", combined)

    def test_mig_instance_rejected_via_rank_wrapper(self):
        """When rank_wrapper.sh detects a MIG instance, Super Job aborts before preflight/benchmarks."""
        self._create_prebuilt_binaries()
        proc = self._run_job_script({"MOCK_MIG_DETECTED": "1"})
        self.assertNotEqual(proc.returncode, 0)
        combined = proc.stderr + proc.stdout
        self.assertIn("MIG", combined)
        self.assertNotIn("Executing Super Matrix Benchmark", combined)

    def test_preflight_failures_and_asymmetric_topologies_abort(self):
        """Topo probe failure, halo failure, asymmetric GPUs, and 2+0 topology all abort preflight."""
        self._create_prebuilt_binaries()
        for env_flag, expected_msg in (
            ({"MOCK_FAIL_TOPO_PROBE": "1"}, "Preflight"),
            ({"MOCK_FAIL_PREFLIGHT_HALO": "1"}, "Asynchronous halo test preflight failed"),
            ({"MOCK_ASYMMETRIC_GPUS": "1"}, "Asynchronous halo test preflight failed"),
            ({"MOCK_ASYMMETRIC_TOPO": "1"}, "Preflight error: per-rank OpenCL GPU counts"),
        ):
            proc = self._run_job_script(env_flag)
            self.assertNotEqual(proc.returncode, 0, f"Expected failure for {env_flag}")
            self.assertIn(expected_msg, proc.stderr + proc.stdout)
            self.assertNotIn("Executing Super Matrix Benchmark", proc.stderr + proc.stdout)

    def test_preflight_budget_is_shared_across_steps(self):
        """Topo probe and halo test share one preflight deadline."""
        self._create_prebuilt_binaries()
        started = time.monotonic()
        proc = self._run_job_script(
            {"PREFLIGHT_BUDGET_SECONDS": "3", "MOCK_SLEEP_TOPO": "2", "MOCK_SLEEP_HALO": "2"}
        )
        elapsed = time.monotonic() - started
        output = proc.stderr + proc.stdout
        self.assertNotEqual(proc.returncode, 0, output)
        self.assertNotIn("Executing Super Matrix Benchmark", output)
        self.assertLess(elapsed, 3.9, f"preflight overran shared 3s budget ({elapsed:.1f}s)")

    def test_successful_mock_pipeline_execution(self):
        """Super Job runs preflight, matrix runner, and dual-format analyzer producing all artifacts."""
        self._create_prebuilt_binaries()
        shutil.copytree(os.path.join(ROOT_DIR, "benchmarks"), os.path.join(self.temp_dir, "benchmarks"))

        proc = self._run_job_script({"BENCH_SEEDS": "42", "BENCH_ITERATIONS": "5"})
        self.assertEqual(proc.returncode, 0, f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")

        expected_out = os.path.join(self.temp_dir, "results", "variance_run_12345.headnode")
        self.assertTrue(os.path.isfile(os.path.join(expected_out, "SUPER_JOB_REPORT.md")))
        self.assertTrue(os.path.isfile(os.path.join(expected_out, "topo_metrics_cluster.json")))

        with open(os.path.join(expected_out, "super_matrix_summary.csv"), newline="", encoding="utf-8") as f:
            s_rows = list(csv.DictReader(f))
        self.assertGreater(len(s_rows), 0)
        for r in s_rows:
            self.assertEqual(r["status"], "SUCCESS")

        with open(os.path.join(expected_out, "super_matrix_paired.csv"), newline="", encoding="utf-8") as f:
            p_rows = list(csv.DictReader(f))
        self.assertGreater(len(p_rows), 0)

    def test_sample_failure_propagates_exit_code_and_runs_postprocessing(self):
        """When a matrix sample fails, Super Job exits 1 but still runs post-processing."""
        self._create_prebuilt_binaries()
        shutil.copytree(os.path.join(ROOT_DIR, "benchmarks"), os.path.join(self.temp_dir, "benchmarks"))

        fail_marker = os.path.join(self.temp_dir, "fail_once.marker")
        proc = self._run_job_script({"BENCH_SEEDS": "42", "BENCH_ITERATIONS": "5", "MOCK_FAIL_ONCE_FILE": fail_marker})
        self.assertEqual(proc.returncode, 1)

        expected_out = os.path.join(self.temp_dir, "results", "variance_run_12345.headnode")
        self.assertTrue(os.path.isfile(os.path.join(expected_out, "SUPER_JOB_REPORT.md")))

    def test_analyzer_failure_propagates_exit_code(self):
        """When post-processing analyzer fails, Super Job exits 1."""
        self._create_prebuilt_binaries()
        shutil.copytree(os.path.join(ROOT_DIR, "benchmarks"), os.path.join(self.temp_dir, "benchmarks"))

        analyzer_script = os.path.join(self.temp_dir, "benchmarks", "analyze_super_matrix.py")
        with open(analyzer_script, "w", encoding="utf-8") as f:
            f.write("#!/usr/bin/env python3\nimport sys\nsys.stderr.write('Mock analyzer failed\\n')\nsys.exit(1)\n")
        os.chmod(analyzer_script, 0o755)

        proc = self._run_job_script({"BENCH_SEEDS": "42", "BENCH_ITERATIONS": "5"})
        self.assertEqual(proc.returncode, 1)
        self.assertIn("analyzer failed", (proc.stderr + proc.stdout).lower())

    def test_multiword_launcher_executes_properly(self):
        """Multi-word LAUNCHER (e.g. 'env MOCK_MULTIWORD=1 prun') must word-split cleanly without failing."""
        self._create_prebuilt_binaries()
        shutil.copytree(os.path.join(ROOT_DIR, "benchmarks"), os.path.join(self.temp_dir, "benchmarks"))

        proc = self._run_job_script({
            "LAUNCHER": "env MOCK_MULTIWORD_FLAG=1 prun",
            "BENCH_SEEDS": "42",
            "BENCH_ITERATIONS": "5",
        })
        self.assertEqual(proc.returncode, 0, f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
        expected_out = os.path.join(self.temp_dir, "results", "variance_run_12345.headnode")
        self.assertTrue(os.path.isfile(os.path.join(expected_out, "SUPER_JOB_REPORT.md")))


class TestBenchmarkNumericalVerification(unittest.TestCase):
    """Verifies that refactored benchmarks check std::isfinite via benchmark_common.hpp."""

    def test_sources_use_verify_finite_and_close_and_isolate_loop_time(self):
        with open(BENCHMARK_COMMON_HPP, "r", encoding="utf-8") as f:
            common_src = f.read()
        self.assertIn("verify_finite_and_close", common_src)
        self.assertIn("std::isfinite(actual) && std::isfinite(expected)", common_src)

        for path in (KNEIGHBOR_CPP_PATH, LEANMD_CPP_PATH):
            with open(path, "r", encoding="utf-8") as f:
                src = f.read()
            self.assertIn("harmony_bench::verify_finite_and_close", src)
            self.assertIn("loop_time_s", src)

    def test_python_finite_mismatch_contract(self):
        def accept(actual: float, expected: float) -> bool:
            return math.isfinite(actual) and math.isfinite(expected) and abs(actual - expected) <= 1e-5

        self.assertTrue(accept(1.0, 1.0))
        self.assertFalse(accept(float("nan"), 1.0))
        self.assertFalse(accept(1.0, float("nan")))
        self.assertFalse(accept(float("inf"), float("inf")))
        self.assertFalse(accept(float("-inf"), 1.0))


if __name__ == "__main__":
    unittest.main()
