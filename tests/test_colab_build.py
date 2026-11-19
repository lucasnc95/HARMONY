"""Unit tests for colab/build.sh Pre-Cluster Validation build wrapper."""

import json
import os
import subprocess
import tempfile
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BUILD_SCRIPT = os.path.join(REPO_ROOT, "colab", "build.sh")

EXPECTED_BINARIES = [
    "topo_probe",
    "kneighbor",
    "leanmd",
]
EXPECTED_TEST_BINARIES = [
    "test_topo_metrics",
    "test_algorithms",
    "test_async_halo_opencl",
]


class TestColabBuild(unittest.TestCase):
    """Test suite for colab/build.sh delegating to ./build.sh targets."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.bin_dir = os.path.join(self.temp_dir.name, "bin")
        self.test_bin_dir = os.path.join(self.temp_dir.name, "tests_bin")
        self.mock_build = os.path.join(self.temp_dir.name, "mock_build.sh")
        self.log_file = os.path.join(self.temp_dir.name, "build_call.json")
        os.makedirs(self.bin_dir, exist_ok=True)
        os.makedirs(self.test_bin_dir, exist_ok=True)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_mock_build(self, exit_code: int = 0, create_binaries: bool = True):
        create_block = ""
        if create_binaries:
            lines = []
            for b in EXPECTED_BINARIES:
                p = os.path.join(self.bin_dir, b)
                lines.append(f'printf "#!/bin/sh\\nexit 0\\n" > "{p}" && chmod +x "{p}"')
            for tb in EXPECTED_TEST_BINARIES:
                p = os.path.join(self.test_bin_dir, tb)
                lines.append(f'printf "#!/bin/sh\\nexit 0\\n" > "{p}" && chmod +x "{p}"')
            create_block = "\n".join(lines)

        script = f"""#!/usr/bin/env bash
set -euo pipefail
python3 -c '
import json, os, sys
record = {{
    "args": sys.argv[1:],
    "MPICXX": os.environ.get("MPICXX", ""),
    "EXTRA_CXXFLAGS": os.environ.get("EXTRA_CXXFLAGS", ""),
    "OPENCL_INCLUDE_DIR": os.environ.get("OPENCL_INCLUDE_DIR", ""),
    "OPENCL_LIBRARY_DIR": os.environ.get("OPENCL_LIBRARY_DIR", ""),
    "CL_TARGET_OPENCL_VERSION": os.environ.get("CL_TARGET_OPENCL_VERSION", ""),
}}
with open(sys.argv[1], "w", encoding="utf-8") as f:
    json.dump(record, f)
' "{self.log_file}" "$@"
{create_block}
exit {exit_code}
"""
        with open(self.mock_build, "w", encoding="utf-8") as f:
            f.write(script)
        os.chmod(self.mock_build, 0o755)

    def _run_build(self, args=None, extra_env=None):
        cmd = ["bash", BUILD_SCRIPT]
        if args:
            cmd.extend(args)

        env = os.environ.copy()
        env["BIN_DIR"] = self.bin_dir
        env["TEST_BIN_DIR"] = self.test_bin_dir
        if extra_env:
            env.update(extra_env)

        return subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            env=env,
        )

    def test_file_exists_and_shebang(self):
        """colab/build.sh must exist and declare a bash shebang."""
        self.assertTrue(os.path.isfile(BUILD_SCRIPT), f"Missing {BUILD_SCRIPT}")
        with open(BUILD_SCRIPT, "r", encoding="utf-8") as f:
            self.assertEqual(f.readline().strip(), "#!/usr/bin/env bash")

    def test_bash_syntax_valid(self):
        """colab/build.sh must pass bash -n syntax validation."""
        result = subprocess.run(
            ["bash", "-n", BUILD_SCRIPT],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, f"Syntax error: {result.stderr}")

    def test_dry_run_flag(self):
        """--dry-run should display ./build.sh targets and expected binaries without compiling."""
        self._create_mock_build()
        res = self._run_build(
            args=["--dry-run"],
            extra_env={"BUILD_SCRIPT": self.mock_build},
        )
        self.assertEqual(res.returncode, 0, f"Dry-run failed: {res.stderr}\n{res.stdout}")
        combined = res.stdout + res.stderr
        self.assertIn("[DRY-RUN]", combined)
        for target in ("lib", "tools", "benchmarks", "tests"):
            self.assertIn(target, combined)
        for binary in EXPECTED_BINARIES + EXPECTED_TEST_BINARIES:
            self.assertIn(binary, combined)
        self.assertFalse(os.path.exists(self.log_file), "Build script must not run during --dry-run")

    def test_dry_run_env_var(self):
        """COLAB_BUILD_DRY_RUN=1 should trigger dry-run mode."""
        self._create_mock_build()
        res = self._run_build(
            extra_env={
                "BUILD_SCRIPT": self.mock_build,
                "COLAB_BUILD_DRY_RUN": "1",
            }
        )
        self.assertEqual(res.returncode, 0, f"Dry-run env failed: {res.stderr}")
        self.assertIn("[DRY-RUN]", res.stdout + res.stderr)
        self.assertFalse(os.path.exists(self.log_file))

    def test_invokes_build_sh_with_refactored_targets(self):
        """Must invoke ./build.sh with lib, tools, benchmarks, and tests targets."""
        self._create_mock_build()
        res = self._run_build(extra_env={"BUILD_SCRIPT": self.mock_build})
        self.assertEqual(res.returncode, 0, f"Build failed: {res.stderr}\n{res.stdout}")
        self.assertTrue(os.path.isfile(self.log_file))

        with open(self.log_file, "r", encoding="utf-8") as f:
            record = json.load(f)
        for target in ("lib", "tools", "benchmarks", "tests"):
            self.assertIn(target, record["args"], f"Missing target {target}: {record['args']}")

    def test_forwards_compiler_and_opencl_env_vars(self):
        """Should forward CXX/MPICXX, CXXFLAGS/EXTRA_CXXFLAGS, and OpenCL env vars."""
        self._create_mock_build()
        res = self._run_build(
            args=["-d", "-j", "4"],
            extra_env={
                "BUILD_SCRIPT": self.mock_build,
                "CXX": "custom-mpicxx",
                "CXXFLAGS": "-DFOO=1",
                "OPENCL_INCLUDE_DIR": "/opt/ocl/include",
                "OPENCL_LIBRARY_DIR": "/opt/ocl/lib",
                "CL_TARGET_OPENCL_VERSION": "300",
            },
        )
        self.assertEqual(res.returncode, 0, f"Build failed: {res.stderr}\n{res.stdout}")
        with open(self.log_file, "r", encoding="utf-8") as f:
            record = json.load(f)
        self.assertIn("-d", record["args"])
        self.assertIn("-j", record["args"])
        self.assertIn("4", record["args"])
        self.assertEqual(record["MPICXX"], "custom-mpicxx")
        self.assertEqual(record["EXTRA_CXXFLAGS"], "-DFOO=1")
        self.assertEqual(record["OPENCL_INCLUDE_DIR"], "/opt/ocl/include")
        self.assertEqual(record["OPENCL_LIBRARY_DIR"], "/opt/ocl/lib")
        self.assertEqual(record["CL_TARGET_OPENCL_VERSION"], "300")

    def test_fails_when_build_script_fails(self):
        """Should exit non-zero immediately if ./build.sh returns non-zero."""
        self._create_mock_build(exit_code=2, create_binaries=False)
        res = self._run_build(extra_env={"BUILD_SCRIPT": self.mock_build})
        self.assertNotEqual(res.returncode, 0)

    def test_fails_when_expected_binary_missing(self):
        """Should fail if expected binaries are not present after build."""
        self._create_mock_build(exit_code=0, create_binaries=False)
        res = self._run_build(extra_env={"BUILD_SCRIPT": self.mock_build})
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("Missing", res.stderr + res.stdout)


if __name__ == "__main__":
    unittest.main()
