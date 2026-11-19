"""Unit tests for colab/run_validation.sh Pre-Cluster Validation runner."""

import json
import os
import stat
import subprocess
import tempfile
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VALIDATION_SCRIPT = os.path.join(REPO_ROOT, "colab", "run_validation.sh")

REQUIRED_BENCH_BINARIES = [
    "kneighbor",
    "leanmd",
]
REQUIRED_TEST_BINARIES = [
    "test_topo_metrics",
    "test_algorithms",
    "test_numa_cost",
    "test_contention_power",
    "test_hierarchical",
    "test_async_halo_opencl",
]


class TestColabValidation(unittest.TestCase):
    """Test suite for colab/run_validation.sh validation runner."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.mock_bin = os.path.join(self.temp_dir.name, "mock_bin")
        self.fake_bin_dir = os.path.join(self.temp_dir.name, "bin")
        self.fake_test_bin_dir = os.path.join(self.temp_dir.name, "tests_bin")
        self.runs_dir = os.path.join(self.temp_dir.name, "runs")
        self.invocations_log = os.path.join(self.temp_dir.name, "invocations.jsonl")
        os.makedirs(self.mock_bin, exist_ok=True)
        os.makedirs(self.fake_bin_dir, exist_ok=True)
        os.makedirs(self.fake_test_bin_dir, exist_ok=True)
        os.makedirs(self.runs_dir, exist_ok=True)

        for b in REQUIRED_BENCH_BINARIES:
            path = os.path.join(self.fake_bin_dir, b)
            with open(path, "w", encoding="utf-8") as f:
                f.write("#!/bin/sh\nexit 0\n")
            os.chmod(path, 0o755)
        for tb in REQUIRED_TEST_BINARIES:
            path = os.path.join(self.fake_test_bin_dir, tb)
            with open(path, "w", encoding="utf-8") as f:
                f.write("#!/bin/sh\nexit 0\n")
            os.chmod(path, 0o755)

        self._create_mock_mpirun()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_mock_mpirun(self, fail_target=None, fail_code=1, non_finite_bench=None, skip_halo_np2=False):
        """Creates a mock mpirun script that logs all invocations and simulates behaviors."""
        path = os.path.join(self.mock_bin, "mpirun")
        script = f"""#!/usr/bin/env bash
set -e

if [[ -n "${{INVOCATIONS_LOG:-}}" ]]; then
    python3 -c '
import json, sys, os
log_file = os.environ["INVOCATIONS_LOG"]
record = {{
    "args": sys.argv[1:],
    "ompi_root": os.environ.get("OMPI_ALLOW_RUN_AS_ROOT", ""),
    "ompi_confirm": os.environ.get("OMPI_ALLOW_RUN_AS_ROOT_CONFIRM", ""),
}}
with open(log_file, "a", encoding="utf-8") as f:
    f.write(json.dumps(record) + "\\n")
' "$@"
fi

cmd_args="$*"

if [[ -n "{fail_target or ''}" && "$cmd_args" == *"{fail_target or ''}"* ]]; then
    echo "Simulated target failure for {fail_target}" >&2
    exit {fail_code}
fi

if [[ -n "{non_finite_bench or ''}" && "$cmd_args" == *"{non_finite_bench or ''}"* ]]; then
    echo "Stencil verification failed at index 42: actual=nan reference=1.0000" >&2
    exit 1
fi

if [[ "$cmd_args" == *"test_async_halo_opencl"* ]]; then
    if [[ "$cmd_args" == *"-np 1"* ]]; then
        echo "SKIP: asynchronous OpenCL halo test needs >= 1 OpenCL GPU per MPI rank"
        exit 77
    fi
    if [[ "$cmd_args" == *"-np 2"* ]]; then
        if [[ "{'1' if skip_halo_np2 else '0'}" == "1" ]]; then
            echo "SKIP: asynchronous OpenCL halo test needs >= 1 OpenCL GPU per MPI rank"
            exit 77
        fi
        echo "PASS: asynchronous OpenCL halo matches two-step CPU reference"
        exit 0
    fi
fi

if [[ "$cmd_args" == *"kneighbor"* ]]; then
    echo "[kneighbor] algo=erad mode=gpu_opencl total_time_s=1.234 loop_time_s=0.987 num_rebalances=4"
    exit 0
fi

if [[ "$cmd_args" == *"leanmd"* ]]; then
    echo "[leanmd] algo=erad mode=gpu_opencl total_time_s=1.567 loop_time_s=1.234 num_rebalances=4"
    exit 0
fi

exit 0
"""
        with open(path, "w", encoding="utf-8") as f:
            f.write(script)
        st = os.stat(path)
        os.chmod(path, st.st_mode | stat.S_IEXEC)
        return path

    def _run_validation(self, args=None, extra_env=None):
        cmd = ["bash", VALIDATION_SCRIPT]
        if args:
            cmd.extend(args)

        env = os.environ.copy()
        env["PATH"] = f"{self.mock_bin}:{env['PATH']}"
        env["INVOCATIONS_LOG"] = self.invocations_log
        env["BIN_DIR"] = self.fake_bin_dir
        env["TEST_BIN_DIR"] = self.fake_test_bin_dir
        env["RUNS_DIR"] = self.runs_dir
        if extra_env:
            env.update(extra_env)

        return subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            env=env,
        )

    def _get_invocations(self):
        if not os.path.exists(self.invocations_log):
            return []
        with open(self.invocations_log, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def test_file_exists_and_shebang(self):
        """colab/run_validation.sh must exist and declare a bash shebang."""
        self.assertTrue(os.path.isfile(VALIDATION_SCRIPT), f"Missing {VALIDATION_SCRIPT}")
        with open(VALIDATION_SCRIPT, "r", encoding="utf-8") as f:
            self.assertEqual(f.readline().strip(), "#!/usr/bin/env bash")

    def test_bash_syntax_valid(self):
        """colab/run_validation.sh must pass bash -n syntax validation."""
        result = subprocess.run(
            ["bash", "-n", VALIDATION_SCRIPT],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, f"Syntax error: {result.stderr}")

    def test_dry_run_flag(self):
        """--dry-run should print planned phases without invoking mpirun."""
        res = self._run_validation(args=["--dry-run"])
        self.assertEqual(res.returncode, 0, f"Dry-run failed: {res.stderr}\n{res.stdout}")
        combined = res.stdout + res.stderr
        self.assertIn("[DRY-RUN]", combined)
        self.assertIn("Phase 1", combined)
        self.assertIn("Phase 2", combined)
        self.assertIn("Phase 3", combined)
        self.assertEqual(len(self._get_invocations()), 0, "No mpirun calls expected in dry run")

    def test_dry_run_env_var(self):
        """COLAB_VALIDATION_DRY_RUN=1 should trigger dry run mode."""
        res = self._run_validation(extra_env={"COLAB_VALIDATION_DRY_RUN": "1"})
        self.assertEqual(res.returncode, 0, f"Dry-run env failed: {res.stderr}\n{res.stdout}")
        self.assertIn("[DRY-RUN]", res.stdout + res.stderr)
        self.assertEqual(len(self._get_invocations()), 0)

    def test_openmpi_root_execution_flags(self):
        """Must set OMPI_ALLOW_RUN_AS_ROOT=1 and OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1."""
        res = self._run_validation()
        self.assertEqual(res.returncode, 0, f"Validation run failed: {res.stderr}\n{res.stdout}")
        invocations = self._get_invocations()
        self.assertGreater(len(invocations), 0, "Expected mpirun invocations")
        for inv in invocations:
            self.assertEqual(inv["ompi_root"], "1")
            self.assertEqual(inv["ompi_confirm"], "1")
            args_str = " ".join(inv["args"])
            self.assertIn("--allow-run-as-root", args_str, f"--allow-run-as-root missing in {args_str}")

    def test_phase1_unit_tests_invocations(self):
        """Phase 1 must execute core R1-R5 unit tests from TEST_BIN_DIR."""
        res = self._run_validation()
        self.assertEqual(res.returncode, 0, f"Validation run failed: {res.stderr}\n{res.stdout}")
        invocations = self._get_invocations()

        for test_name in ("test_topo_metrics", "test_algorithms", "test_numa_cost", "test_contention_power", "test_hierarchical"):
            calls = [inv for inv in invocations if any(test_name in a for a in inv["args"])]
            self.assertEqual(len(calls), 1, f"{test_name} must be called once")
            args = calls[0]["args"]
            self.assertIn("-np", args)
            expected_np = "2" if test_name == "test_hierarchical" else "1"
            self.assertEqual(args[args.index("-np") + 1], expected_np)

    def test_phase2_halo_integration_np1_and_np2(self):
        """Phase 2 must execute test_async_halo_opencl with -np 1 and -np 2 --oversubscribe."""
        res = self._run_validation()
        self.assertEqual(res.returncode, 0, f"Validation run failed: {res.stderr}\n{res.stdout}")
        invocations = self._get_invocations()

        halo_calls = [inv for inv in invocations if any("test_async_halo_opencl" in a for a in inv["args"])]
        self.assertEqual(len(halo_calls), 2, "test_async_halo_opencl must be called twice (np 1 and np 2)")

        np1_call = [c for c in halo_calls if "-np" in c["args"] and c["args"][c["args"].index("-np") + 1] == "1"]
        self.assertEqual(len(np1_call), 1)

        np2_call = [c for c in halo_calls if "-np" in c["args"] and c["args"][c["args"].index("-np") + 1] == "2"]
        self.assertEqual(len(np2_call), 1)
        self.assertIn("--oversubscribe", np2_call[0]["args"])

    def test_phase2_skip_code_77_handling(self):
        """Exit code 77 from halo test must be handled cleanly as SKIP without failing."""
        self._create_mock_mpirun(skip_halo_np2=True)
        res = self._run_validation()
        self.assertEqual(res.returncode, 0, f"Should pass when halo test returns skip 77: {res.stderr}\n{res.stdout}")

        status_path = os.path.join(self.runs_dir, "colab_status.json")
        self.assertTrue(os.path.exists(status_path))
        with open(status_path, "r", encoding="utf-8") as f:
            status_data = json.load(f)

        self.assertEqual(status_data["status"], "PASS")
        self.assertEqual(status_data["phases"]["phase2_halo_integration"]["test_async_halo_opencl_np1"], "SKIP")
        self.assertEqual(status_data["phases"]["phase2_halo_integration"]["test_async_halo_opencl_np2"], "SKIP")

    def test_phase3_fast_benchmarks_matrix_and_flags(self):
        """Phase 3 must execute kneighbor and leanmd using refactored harmony CLI flags."""
        res = self._run_validation()
        self.assertEqual(res.returncode, 0, f"Validation run failed: {res.stderr}\n{res.stdout}")
        invocations = self._get_invocations()

        bench_calls = [
            inv for inv in invocations
            if any(os.path.basename(a) in ("kneighbor", "leanmd") for a in inv["args"])
        ]
        self.assertEqual(len(bench_calls), 4, f"Expected 4 benchmark invocations, got {len(bench_calls)}")

        for call in bench_calls:
            args = call["args"]
            self.assertIn("-np", args)
            self.assertEqual(args[args.index("-np") + 1], "2")
            self.assertIn("--oversubscribe", args)
            self.assertEqual(args[args.index("--n") + 1], "50000")
            self.assertEqual(args[args.index("--iterations") + 1], "20")
            self.assertEqual(args[args.index("--balance-interval") + 1], "5")
            self.assertIn("--balance-algo", args)
            self.assertIn("--output-csv", args)
            self.assertIn("--report-csv", args)
            self.assertNotIn("--steps", args)
            self.assertNotIn("--algo", args)

        for bench in ("kneighbor", "leanmd"):
            for algo in ("erad", "hwtopolb"):
                matching = [
                    c for c in bench_calls
                    if any(os.path.basename(a) == bench for a in c["args"])
                    and c["args"][c["args"].index("--balance-algo") + 1] == algo
                ]
                self.assertEqual(len(matching), 1, f"Missing run for {bench} with {algo}")

    def test_phase3_configurable_benchmark_dimensions(self):
        """Should respect BENCH_STEPS, BENCH_INTERVAL, and BENCH_N overrides."""
        res = self._run_validation(
            extra_env={
                "BENCH_STEPS": "35",
                "BENCH_INTERVAL": "7",
                "BENCH_N": "120000",
            }
        )
        self.assertEqual(res.returncode, 0, f"Validation run failed: {res.stderr}\n{res.stdout}")
        invocations = self._get_invocations()
        bench_calls = [
            inv for inv in invocations
            if any(os.path.basename(a) in ("kneighbor", "leanmd") for a in inv["args"])
        ]
        for call in bench_calls:
            args = call["args"]
            self.assertEqual(args[args.index("--n") + 1], "120000")
            self.assertEqual(args[args.index("--iterations") + 1], "35")
            self.assertEqual(args[args.index("--balance-interval") + 1], "7")

    def test_status_json_and_log_artifacts_emitted(self):
        """Logs and colab_status.json must be emitted in RUNS_DIR."""
        res = self._run_validation()
        self.assertEqual(res.returncode, 0, f"Validation run failed: {res.stderr}\n{res.stdout}")

        status_path = os.path.join(self.runs_dir, "colab_status.json")
        self.assertTrue(os.path.exists(status_path))

        with open(status_path, "r", encoding="utf-8") as f:
            status_data = json.load(f)

        self.assertEqual(status_data["status"], "PASS")
        self.assertIn("phases", status_data)
        self.assertIn("phase1_unit_tests", status_data["phases"])
        self.assertIn("phase2_halo_integration", status_data["phases"])
        self.assertIn("phase3_fast_benchmarks", status_data["phases"])

        for log_name in (
            "test_topo_metrics.log",
            "test_algorithms.log",
            "test_numa_cost.log",
            "test_contention_power.log",
            "test_hierarchical.log",
            "test_async_halo_opencl_np1.log",
            "test_async_halo_opencl_np2.log",
            "kneighbor_erad.log",
            "kneighbor_hwtopolb.log",
            "leanmd_erad.log",
            "leanmd_hwtopolb.log",
        ):
            self.assertTrue(
                os.path.exists(os.path.join(self.runs_dir, log_name)),
                f"Missing log artifact {log_name} in {self.runs_dir}",
            )

    def test_fails_on_unit_test_failure(self):
        """Should fail with non-zero exit code if Phase 1 unit test fails."""
        self._create_mock_mpirun(fail_target="test_algorithms", fail_code=1)
        res = self._run_validation()
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("Simulated target failure for test_algorithms", res.stderr + res.stdout)

        status_path = os.path.join(self.runs_dir, "colab_status.json")
        self.assertTrue(os.path.exists(status_path))
        with open(status_path, "r", encoding="utf-8") as f:
            status_data = json.load(f)
        self.assertEqual(status_data["status"], "FAIL")

    def test_fails_on_non_finite_rejection(self):
        """Should fail with non-zero exit code if benchmark detects non-finite output."""
        self._create_mock_mpirun(non_finite_bench="kneighbor")
        res = self._run_validation()
        self.assertNotEqual(res.returncode, 0)
        combined = res.stderr + res.stdout
        self.assertTrue("Stencil verification failed" in combined or "verification failed" in combined)

        status_path = os.path.join(self.runs_dir, "colab_status.json")
        self.assertTrue(os.path.exists(status_path))
        with open(status_path, "r", encoding="utf-8") as f:
            status_data = json.load(f)
        self.assertEqual(status_data["status"], "FAIL")

    def test_fails_on_missing_binaries(self):
        """Should fail with clear error message if required binaries are missing."""
        empty_bin_dir = os.path.join(self.temp_dir.name, "empty_bin")
        empty_test_bin_dir = os.path.join(self.temp_dir.name, "empty_test_bin")
        os.makedirs(empty_bin_dir, exist_ok=True)
        os.makedirs(empty_test_bin_dir, exist_ok=True)
        res = self._run_validation(
            extra_env={"BIN_DIR": empty_bin_dir, "TEST_BIN_DIR": empty_test_bin_dir}
        )
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("Missing required binary", res.stderr + res.stdout)


if __name__ == "__main__":
    unittest.main()
