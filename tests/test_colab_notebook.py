"""Unit tests for colab/harmony_colab.ipynb Pre-Cluster Validation notebook.

Enforces nbformat v4 structure, metadata accelerator configuration, refactored
harmony build/CLI references, cell execution sequence, operational boundaries,
and GLOSSARY.md domain terminology compliance.
"""

import json
import os
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NOTEBOOK_PATH = os.path.join(REPO_ROOT, "colab", "harmony_colab.ipynb")


class TestColabNotebook(unittest.TestCase):
    """Test suite for the Google Colab Pre-Cluster Validation interactive notebook."""

    @classmethod
    def setUpClass(cls):
        cls.notebook_exists = os.path.isfile(NOTEBOOK_PATH)
        cls.notebook_data = None
        if cls.notebook_exists:
            with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
                cls.notebook_data = json.load(f)

    def _get_cell_source(self, cell):
        source = cell.get("source", "")
        if isinstance(source, list):
            return "".join(source)
        return str(source)

    def test_notebook_file_exists(self):
        """colab/harmony_colab.ipynb must exist in the repository."""
        self.assertTrue(
            os.path.isfile(NOTEBOOK_PATH),
            f"Expected notebook file at {NOTEBOOK_PATH} does not exist.",
        )

    def test_notebook_valid_json_and_nbformat_version(self):
        """Notebook must be valid JSON conforming to nbformat v4 with nbformat_minor >= 2."""
        self.assertIsNotNone(self.notebook_data, "Failed to parse notebook as JSON")
        self.assertEqual(self.notebook_data.get("nbformat"), 4)
        self.assertGreaterEqual(self.notebook_data.get("nbformat_minor", 0), 2)

    def test_notebook_metadata(self):
        """Notebook metadata must specify GPU accelerator hints and python language info."""
        self.assertIsNotNone(self.notebook_data)
        metadata = self.notebook_data.get("metadata", {})
        self.assertIsInstance(metadata, dict)
        accelerator = metadata.get("accelerator")
        colab_meta = metadata.get("colab", {})
        self.assertTrue(accelerator == "GPU" or colab_meta.get("gpuType") is not None)
        language_info = metadata.get("language_info", {})
        self.assertEqual(language_info.get("name", "").lower(), "python")

    def test_cells_structure(self):
        """Notebook must have non-empty cells list with valid cell types."""
        self.assertIsNotNone(self.notebook_data)
        cells = self.notebook_data.get("cells", [])
        self.assertIsInstance(cells, list)
        self.assertGreaterEqual(len(cells), 5)
        for idx, cell in enumerate(cells):
            self.assertIn(cell.get("cell_type"), ["markdown", "code"], f"Invalid cell {idx}")
            self.assertIn("source", cell)

    def test_gpu_environment_check_cell(self):
        """Must contain code cell executing nvidia-smi and clinfo."""
        self.assertIsNotNone(self.notebook_data)
        code_cells = [
            self._get_cell_source(c)
            for c in self.notebook_data.get("cells", [])
            if c.get("cell_type") == "code"
        ]
        self.assertTrue(any("nvidia-smi" in src and "clinfo" in src for src in code_cells))

    def test_git_clone_checkout_cell(self):
        """Must contain code cell cloning HARMONY repo with configurable BRANCH_OR_TAG and COMMIT_SHA."""
        self.assertIsNotNone(self.notebook_data)
        code_cells = [
            self._get_cell_source(c)
            for c in self.notebook_data.get("cells", [])
            if c.get("cell_type") == "code"
        ]
        found = any(
            "github.com/lispz1610/HARMONY" in src and "BRANCH_OR_TAG" in src and "COMMIT_SHA" in src
            for src in code_cells
        )
        self.assertTrue(found)

    def test_bootstrap_invocation_cell(self):
        """Must contain code cell executing colab/bootstrap.sh."""
        self.assertIsNotNone(self.notebook_data)
        code_cells = [
            self._get_cell_source(c)
            for c in self.notebook_data.get("cells", [])
            if c.get("cell_type") == "code"
        ]
        self.assertTrue(any("colab/bootstrap.sh" in src for src in code_cells))

    def test_build_invocation_cell(self):
        """Must contain code cell executing colab/build.sh."""
        self.assertIsNotNone(self.notebook_data)
        code_cells = [
            self._get_cell_source(c)
            for c in self.notebook_data.get("cells", [])
            if c.get("cell_type") == "code"
        ]
        self.assertTrue(any("colab/build.sh" in src for src in code_cells))

    def test_run_validation_invocation_cell(self):
        """Must contain code cell executing colab/run_validation.sh."""
        self.assertIsNotNone(self.notebook_data)
        code_cells = [
            self._get_cell_source(c)
            for c in self.notebook_data.get("cells", [])
            if c.get("cell_type") == "code"
        ]
        self.assertTrue(any("colab/run_validation.sh" in src for src in code_cells))

    def test_extended_benchmark_cell_uses_refactored_binaries_and_cli(self):
        """Must contain optional cell referencing ./bin/kneighbor, ./bin/leanmd, and harmony CLI flags."""
        self.assertIsNotNone(self.notebook_data)
        code_cells = [
            self._get_cell_source(c)
            for c in self.notebook_data.get("cells", [])
            if c.get("cell_type") == "code"
        ]
        matching = [
            src for src in code_cells
            if "./bin/kneighbor" in src and "./bin/leanmd" in src
        ]
        self.assertTrue(matching, "Extended benchmark cell must reference ./bin/kneighbor and ./bin/leanmd")
        src = matching[0]
        self.assertIn("--iterations", src)
        self.assertIn("--balance-interval", src)
        self.assertIn("--balance-algo", src)
        self.assertNotIn(".out", src)
        self.assertNotIn("--steps", src)

    def test_cell_sequence_order(self):
        """Cells must appear in logical workflow order: check -> clone -> bootstrap -> build -> validate -> extended."""
        self.assertIsNotNone(self.notebook_data)
        cells = self.notebook_data.get("cells", [])

        gpu_idx = clone_idx = bootstrap_idx = build_idx = validate_idx = extended_idx = -1
        for idx, cell in enumerate(cells):
            if cell.get("cell_type") != "code":
                continue
            src = self._get_cell_source(cell)
            if ("nvidia-smi" in src or "clinfo" in src) and gpu_idx == -1:
                gpu_idx = idx
            if ("HARMONY.git" in src or "BRANCH_OR_TAG" in src) and clone_idx == -1:
                clone_idx = idx
            if "colab/bootstrap.sh" in src and bootstrap_idx == -1:
                bootstrap_idx = idx
            if "colab/build.sh" in src and build_idx == -1:
                build_idx = idx
            if "colab/run_validation.sh" in src and validate_idx == -1:
                validate_idx = idx
            if ("./bin/kneighbor" in src and "./bin/leanmd" in src) and idx > validate_idx:
                extended_idx = idx

        self.assertGreater(gpu_idx, -1)
        self.assertGreater(clone_idx, gpu_idx)
        self.assertGreater(bootstrap_idx, clone_idx)
        self.assertGreater(build_idx, bootstrap_idx)
        self.assertGreater(validate_idx, build_idx)
        self.assertGreater(extended_idx, validate_idx)

    def test_markdown_explains_pre_cluster_validation_boundaries_and_layout(self):
        """Markdown cells must explain Pre-Cluster Validation, operational boundaries, and refactored layout."""
        self.assertIsNotNone(self.notebook_data)
        all_markdown = "\n".join(
            self._get_cell_source(c)
            for c in self.notebook_data.get("cells", [])
            if c.get("cell_type") == "markdown"
        )
        self.assertIn("Pre-Cluster Validation", all_markdown)
        self.assertIn("Runtime", all_markdown)
        self.assertIn("Change runtime type", all_markdown)
        self.assertIn("./build.sh", all_markdown)
        self.assertIn("tests/bin/", all_markdown)
        self.assertIn("bin/", all_markdown)
        self.assertTrue("does not replace" in all_markdown.lower() or "cannot substitute" in all_markdown.lower())
        self.assertNotIn("build/colab", all_markdown)
        self.assertNotIn("dcl/", all_markdown)

    def test_markdown_domain_glossary_compliance(self):
        """Markdown cells must avoid prohibited GLOSSARY.md synonyms."""
        self.assertIsNotNone(self.notebook_data)
        all_markdown = "\n".join(
            self._get_cell_source(c)
            for c in self.notebook_data.get("cells", [])
            if c.get("cell_type") == "markdown"
        ).lower()

        prohibited = [
            "cloud cluster",
            "colab benchmark",
            "production run",
            "orchestrator",
            "orchestration",
            "legacy log",
            "old run",
        ]
        for term in prohibited:
            self.assertNotIn(term, all_markdown, f"Prohibited term '{term}' found in notebook")


if __name__ == "__main__":
    unittest.main()
