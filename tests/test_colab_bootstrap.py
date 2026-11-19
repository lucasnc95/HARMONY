"""Unit tests for colab/bootstrap.sh environment bootstrap and GPU discovery."""

import os
import stat
import subprocess
import tempfile
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BOOTSTRAP_PATH = os.path.join(REPO_ROOT, "colab", "bootstrap.sh")


class TestColabBootstrap(unittest.TestCase):
    """Test suite for Colab environment bootstrap script."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.mock_bin = os.path.join(self.temp_dir.name, "mock_bin")
        os.makedirs(self.mock_bin, exist_ok=True)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_mock(self, name: str, script_content: str):
        path = os.path.join(self.mock_bin, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"#!/usr/bin/env bash\n{script_content}\n")
        st = os.stat(path)
        os.chmod(path, st.st_mode | stat.S_IEXEC)
        return path

    def _run_bootstrap(self, extra_env=None):
        env = os.environ.copy()
        env["BOOTSTRAP_SKIP_APT"] = "1"
        env["PATH"] = f"{self.mock_bin}:{env['PATH']}"
        if extra_env:
            env.update(extra_env)

        return subprocess.run(
            ["bash", BOOTSTRAP_PATH],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            env=env,
        )

    def test_file_exists_and_shebang(self):
        """colab/bootstrap.sh must exist and declare a bash shebang."""
        self.assertTrue(os.path.isfile(BOOTSTRAP_PATH), f"Missing {BOOTSTRAP_PATH}")
        with open(BOOTSTRAP_PATH, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
        self.assertEqual(first_line, "#!/usr/bin/env bash")

    def test_bash_syntax_valid(self):
        """colab/bootstrap.sh must pass bash -n syntax validation."""
        result = subprocess.run(
            ["bash", "-n", BOOTSTRAP_PATH],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, f"Syntax error: {result.stderr}")

    def test_fails_when_nvidia_smi_missing(self):
        """Should fail with diagnostic error if nvidia-smi is not found."""
        clean_path = ":".join(
            p for p in os.environ.get("PATH", "").split(":")
            if not os.path.exists(os.path.join(p, "nvidia-smi"))
        )

        res = self._run_bootstrap(extra_env={"PATH": f"{self.mock_bin}:{clean_path}"})
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("nvidia-smi", (res.stderr + res.stdout).lower())

    def test_fails_when_nvidia_smi_errors(self):
        """Should fail if nvidia-smi exits non-zero (driver error)."""
        self._create_mock("nvidia-smi", "echo 'NVIDIA-SMI has failed' >&2; exit 1")
        self._create_mock("clinfo", "exit 0")

        res = self._run_bootstrap()
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("nvidia-smi", (res.stderr + res.stdout).lower())

    def test_fails_when_clinfo_missing(self):
        """Should fail if nvidia-smi succeeds but clinfo is not available."""
        self._create_mock(
            "nvidia-smi",
            "echo 'Tesla T4, 535.104.05, 15360 MiB'"
        )
        self._create_mock("clinfo", "echo 'clinfo: command not found' >&2; exit 127")

        res = self._run_bootstrap()
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("clinfo", (res.stderr + res.stdout).lower())

    def test_fails_when_zero_opencl_platforms(self):
        """Should fail if clinfo finds zero platforms or empty device list."""
        self._create_mock("nvidia-smi", "echo 'Tesla T4, 535.104.05, 15360 MiB'")
        self._create_mock("clinfo", "echo 'Number of platforms: 0'; echo '0 platforms'")

        res = self._run_bootstrap()
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("platform", (res.stderr + res.stdout).lower())

    def test_guards_against_cpu_only_fallback(self):
        """Should reject environment when OpenCL only provides CPU devices."""
        self._create_mock("nvidia-smi", "echo 'Tesla T4, 535.104.05, 15360 MiB'")
        mock_clinfo = (
            "if [[ \"${1:-}\" == \"-l\" ]]; then\n"
            "  echo 'Platform #0: Portable Computing Language'\n"
            "  echo ' `-- Device #0: pthread-AMD EPYC 7B12'\n"
            "else\n"
            "  echo 'Platform Name: Portable Computing Language'\n"
            "  echo 'Device Name: pthread-AMD EPYC 7B12'\n"
            "  echo 'Device Type: CL_DEVICE_TYPE_CPU'\n"
            "fi\n"
        )
        self._create_mock("clinfo", mock_clinfo)

        res = self._run_bootstrap()
        self.assertNotEqual(res.returncode, 0)
        output = (res.stderr + res.stdout).lower()
        self.assertTrue(
            "cpu-only" in output or "gpu" in output or "fallback" in output,
            f"Expected CPU fallback diagnostic, got: {res.stderr}\n{res.stdout}",
        )

    def test_valid_gpu_environment_succeeds(self):
        """Should succeed (exit 0) when valid NVIDIA GPU and OpenCL GPU device exist."""
        self._create_mock("nvidia-smi", "echo 'Tesla T4, 535.104.05, 15360 MiB'")
        mock_clinfo = (
            "if [[ \"${1:-}\" == \"-l\" ]]; then\n"
            "  echo 'Platform #0: NVIDIA CUDA'\n"
            "  echo ' `-- Device #0: Tesla T4'\n"
            "else\n"
            "  echo 'Platform Name: NVIDIA CUDA'\n"
            "  echo 'Device Name: Tesla T4'\n"
            "  echo 'Device Type: CL_DEVICE_TYPE_GPU'\n"
            "fi\n"
        )
        self._create_mock("clinfo", mock_clinfo)

        res = self._run_bootstrap()
        self.assertEqual(res.returncode, 0, f"Failed with: {res.stderr}\n{res.stdout}")
        output = res.stdout.lower()
        self.assertIn("tesla t4", output)
        self.assertIn("success", output)

    def test_ompi_root_flags_configured(self):
        """colab/bootstrap.sh must configure OMPI root execution flags."""
        with open(BOOTSTRAP_PATH, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("OMPI_ALLOW_RUN_AS_ROOT=1", content)
        self.assertIn("OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1", content)

    def test_idempotent_execution(self):
        """Running bootstrap twice in a valid environment succeeds both times."""
        self._create_mock("nvidia-smi", "echo 'Tesla T4, 535.104.05, 15360 MiB'")
        mock_clinfo = (
            "if [[ \"${1:-}\" == \"-l\" ]]; then\n"
            "  echo 'Platform #0: NVIDIA CUDA'\n"
            "  echo ' `-- Device #0: Tesla T4'\n"
            "else\n"
            "  echo 'Platform Name: NVIDIA CUDA'\n"
            "  echo 'Device Name: Tesla T4'\n"
            "  echo 'Device Type: CL_DEVICE_TYPE_GPU'\n"
            "fi\n"
        )
        self._create_mock("clinfo", mock_clinfo)

        res1 = self._run_bootstrap()
        self.assertEqual(res1.returncode, 0)

        res2 = self._run_bootstrap()
        self.assertEqual(res2.returncode, 0)

    def test_nvidia_icd_file_creation(self):
        """Verifies ensure_nvidia_icd creates nvidia.icd when missing."""
        vendor_dir = os.path.join(self.temp_dir.name, "vendors")
        self._create_mock("nvidia-smi", "echo 'Tesla T4, 535.104.05, 15360 MiB'")
        mock_clinfo = (
            "if [[ \"${1:-}\" == \"-l\" ]]; then\n"
            "  echo 'Platform #0: NVIDIA CUDA'\n"
            "  echo ' `-- Device #0: Tesla T4'\n"
            "else\n"
            "  echo 'Device Type: CL_DEVICE_TYPE_GPU'\n"
            "fi\n"
        )
        self._create_mock("clinfo", mock_clinfo)

        res = self._run_bootstrap(extra_env={"OPENCL_VENDOR_DIR": vendor_dir})
        self.assertEqual(res.returncode, 0)
        icd_file = os.path.join(vendor_dir, "nvidia.icd")
        self.assertTrue(os.path.isfile(icd_file), f"Expected {icd_file} to exist")
        with open(icd_file, "r", encoding="utf-8") as f:
            self.assertIn("libnvidia-opencl.so.1", f.read())

    def test_env_and_profile_persistence(self):
        """Verifies OMPI flags are written to mock env file and profile dir."""
        test_env_file = os.path.join(self.temp_dir.name, "test_env")
        test_profile_dir = os.path.join(self.temp_dir.name, "test_profile_d")
        os.makedirs(test_profile_dir, exist_ok=True)
        with open(test_env_file, "w", encoding="utf-8") as f:
            f.write("# initial env\n")

        self._create_mock("nvidia-smi", "echo 'Tesla T4, 535.104.05, 15360 MiB'")
        mock_clinfo = (
            "if [[ \"${1:-}\" == \"-l\" ]]; then\n"
            "  echo 'Platform #0: NVIDIA CUDA'\n"
            "  echo ' `-- Device #0: Tesla T4'\n"
            "else\n"
            "  echo 'Device Type: CL_DEVICE_TYPE_GPU'\n"
            "fi\n"
        )
        self._create_mock("clinfo", mock_clinfo)

        res = self._run_bootstrap(extra_env={
            "ENV_FILE": test_env_file,
            "OPENMPI_PROFILE_DIR": test_profile_dir,
        })
        self.assertEqual(res.returncode, 0)

        with open(test_env_file, "r", encoding="utf-8") as f:
            env_content = f.read()
        self.assertIn("OMPI_ALLOW_RUN_AS_ROOT=1", env_content)
        self.assertIn("OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1", env_content)

        profile_script = os.path.join(test_profile_dir, "openmpi_root.sh")
        self.assertTrue(os.path.isfile(profile_script))
        with open(profile_script, "r", encoding="utf-8") as f:
            profile_content = f.read()
        self.assertIn("export OMPI_ALLOW_RUN_AS_ROOT=1", profile_content)

    def test_apt_install_packages_invoked(self):
        """Verifies apt-get is called with required packages when not skipped."""
        log_file = os.path.join(self.temp_dir.name, "apt.log")
        mock_apt = (
            f"echo \"$*\" >> '{log_file}'\n"
            "exit 0\n"
        )
        self._create_mock("apt-get", mock_apt)
        self._create_mock("nvidia-smi", "echo 'Tesla T4, 535.104.05, 15360 MiB'")
        mock_clinfo = (
            "if [[ \"${1:-}\" == \"-l\" ]]; then\n"
            "  echo 'Platform #0: NVIDIA CUDA'\n"
            "  echo ' `-- Device #0: Tesla T4'\n"
            "else\n"
            "  echo 'Device Type: CL_DEVICE_TYPE_GPU'\n"
            "fi\n"
        )
        self._create_mock("clinfo", mock_clinfo)

        res = self._run_bootstrap(extra_env={"BOOTSTRAP_SKIP_APT": "0"})
        self.assertEqual(res.returncode, 0)
        self.assertTrue(os.path.isfile(log_file), "apt-get was not called")
        with open(log_file, "r", encoding="utf-8") as f:
            apt_invocations = f.read()

        for pkg in [
            "build-essential",
            "openmpi-bin",
            "libopenmpi-dev",
            "ocl-icd-opencl-dev",
            "opencl-headers",
            "clinfo",
        ]:
            self.assertIn(pkg, apt_invocations, f"Package {pkg} was not requested via apt-get")


if __name__ == "__main__":
    unittest.main()
