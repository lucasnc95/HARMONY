#!/usr/bin/env python3
"""Documentation consistency checks for HARMONY Super Job and Pre-Cluster Validation.

Verifies that README/ADR/Standards vocabulary follows GLOSSARY.md, that documented
artifact names match benchmarks/analyze_super_matrix.py and benchmarks/run_super_matrix.sh,
that README/ADRs/CODING_STANDARDS.md/AGENTS.md accurately reflect the refactored harmony
namespace, modular library layout (include/harmony/, src/, ./build.sh, bin/, tests/bin/),
Super Job workflow, and Pre-Cluster Validation operational boundaries.
"""

import os
import re
import unittest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def read(*parts):
    with open(os.path.join(ROOT_DIR, *parts), encoding="utf-8") as handle:
        return handle.read()


def github_anchor(heading):
    anchor = heading.strip().lower()
    anchor = re.sub(r"[^\w\- ]", "", anchor)
    return anchor.replace(" ", "-")


class TestGlossaryVocabulary(unittest.TestCase):
    def test_super_job_and_colab_files_avoid_glossary_avoid_terms(self):
        """Super Job and Pre-Cluster Validation scripts/docs must not use _Avoid_ terms from GLOSSARY.md."""
        avoid_terms = []
        for line in read("GLOSSARY.md").splitlines():
            if line.startswith("_Avoid_:"):
                avoid_terms += [t.strip().lower() for t in line.split(":", 1)[1].split(",")]
        self.assertIn("orchestrator", avoid_terms)
        self.assertIn("cloud cluster", avoid_terms)
        self.assertIn("colab benchmark", avoid_terms)

        files = [
            ("docs", "adr", "0001-non-invasive-pbs-super-job.md"),
            ("docs", "adr", "0002-colab-pre-cluster-validation.md"),
            ("job_harmony_super.pbs",),
            ("benchmarks", "run_super_matrix.sh"),
            ("benchmarks", "telemetry_probe.sh"),
            ("benchmarks", "analyze_super_matrix.py"),
            ("colab", "bootstrap.sh"),
            ("colab", "build.sh"),
            ("colab", "run_validation.sh"),
        ]
        offenders = []
        for parts in files:
            if not os.path.isfile(os.path.join(ROOT_DIR, *parts)):
                offenders.append(f"{'/'.join(parts)}: missing file")
                continue
            for number, line in enumerate(read(*parts).splitlines(), 1):
                low = line.lower()
                for term in avoid_terms:
                    if re.search(r"\b" + re.escape(term) + r"\b", low):
                        offenders.append(f"{'/'.join(parts)}:{number}: {term}")
                if "orchestration" in low:
                    offenders.append(f"{'/'.join(parts)}:{number}: orchestration")

        readme = read("README.md")
        for term in ("cloud cluster", "colab benchmark", "orchestrator", "orchestration", "legacy log", "old run"):
            for number, line in enumerate(readme.splitlines(), 1):
                if term in line.lower():
                    offenders.append(f"README.md:{number}: {term}")

        self.assertEqual(offenders, [])


class TestArtifactNames(unittest.TestCase):
    def test_readme_artifact_names_match_analyzer_outputs(self):
        """README must document the exact artifact filenames emitted by analyzer and matrix runner."""
        analyzer = read("benchmarks", "analyze_super_matrix.py")
        runner = read("benchmarks", "run_super_matrix.sh")
        readme = read("README.md")

        self.assertIn('output_dir / "super_matrix_paired.csv"', analyzer)
        self.assertIn('output_dir / "hardware_telemetry.json"', analyzer)
        self.assertIn("hardware_telemetry.log", runner)
        self.assertIn("run_manifest.json", runner)
        self.assertNotIn("super_matrix_paired_ratios.csv", readme)

        self.assertIn("Artifacts generated in", readme)
        section = readme.split("Artifacts generated in", 1)[1].split("Offline recovery", 1)[0]
        for name in (
            "SUPER_JOB_REPORT.md",
            "super_matrix_summary.csv",
            "super_matrix_paired.csv",
            "hardware_telemetry.log",
            "hardware_telemetry.json",
            "run_manifest.json",
        ):
            self.assertIn(name, section, f"{name} missing from README artifact list")


class TestRefactoredStandardsAndGlossary(unittest.TestCase):
    def test_glossary_reflects_harmony_and_canonical_terms(self):
        glossary = read("GLOSSARY.md")
        for term in (
            "**Super Job**",
            "**Unprofiled Repetition**",
            "**Rotated Matrix**",
            "**A/B/A Test**",
            "**Preflight Verification**",
            "**Historical Evidence**",
            "**Pre-Cluster Validation**",
        ):
            self.assertIn(term, glossary)
        self.assertNotIn("dcl/", glossary)

    def test_coding_standards_reflects_refactored_layout(self):
        standards = read("CODING_STANDARDS.md")
        self.assertIn("include/harmony/", standards)
        self.assertIn("src/", standards)
        self.assertIn("bin/", standards)
        self.assertIn("tests/bin/", standards)
        self.assertIn("./build.sh all", standards)
        self.assertIn("8d7c3392c8d616997b342f778ca041a9a717aff2", standards)
        self.assertNotIn("dcl/", standards)
        self.assertNotIn("cluster_build.sh", standards)
        self.assertNotIn("build/cluster", standards)

    def test_agents_md_reflects_harmony_layout_and_invariants(self):
        agents = read("AGENTS.md")
        self.assertIn("include/harmony/", agents)
        self.assertIn("src/", agents)
        self.assertIn("harmony", agents)
        self.assertIn("bash tests/check_invariants.sh", agents)


class TestAdrConsistency(unittest.TestCase):
    def test_adr_0001_exists_and_has_migration_amendment(self):
        adr_path = os.path.join(ROOT_DIR, "docs", "adr", "0001-non-invasive-pbs-super-job.md")
        self.assertTrue(os.path.isfile(adr_path))
        content = read("docs", "adr", "0001-non-invasive-pbs-super-job.md")
        self.assertIn("include/harmony/", content)
        self.assertIn("src/", content)
        self.assertIn("./build.sh all", content)
        self.assertIn("scripts/pbs/job_env.sh", content)
        self.assertIn("scripts/pbs/rank_wrapper.sh", content)
        self.assertIn("8d7c3392c8d616997b342f778ca041a9a717aff2", content)
        self.assertRegex(content, r"## Amendment \(2026-11-\d{2}\)")

    def test_adr_0002_exists_and_reflects_refactored_colab_layout(self):
        adr_path = os.path.join(ROOT_DIR, "docs", "adr", "0002-colab-pre-cluster-validation.md")
        self.assertTrue(os.path.isfile(adr_path))
        content = read("docs", "adr", "0002-colab-pre-cluster-validation.md")

        self.assertIn("# Optional Google Colab Pre-Cluster Validation Testbed", content)
        for section in ("## Status", "## Context", "## Decision", "## Consequences"):
            self.assertIn(section, content)

        status_section = content.split("## Status", 1)[1].split("##", 1)[0]
        self.assertIn("accepted", status_section.lower())

        context_section = content.split("## Context", 1)[1].split("##", 1)[0]
        self.assertRegex(context_section, r"(?i)(local machine|workstation)")
        self.assertRegex(context_section, r"(?i)(queue|latency)")
        self.assertRegex(context_section, r"(?i)fail-fast")

        decision_section = content.split("## Decision", 1)[1].split("##", 1)[0]
        self.assertIn("Pre-Cluster Validation", decision_section)
        self.assertIn("colab/", decision_section)
        self.assertIn("bootstrap.sh", decision_section)
        self.assertIn("build.sh", decision_section)
        self.assertIn("run_validation.sh", decision_section)
        self.assertIn("harmony_colab.ipynb", decision_section)
        self.assertIn("./build.sh", decision_section)
        self.assertIn("tests/bin/", decision_section)
        self.assertIn("bin/", decision_section)

        consequences_section = content.split("## Consequences", 1)[1]
        self.assertIn("include/harmony/", consequences_section)
        self.assertIn("src/", consequences_section)
        self.assertRegex(consequences_section, r"(?i)(frozen|untouched)")
        self.assertRegex(consequences_section, r"(?i)(single-node|virtualized)")
        self.assertRegex(consequences_section, r"(?i)(cannot substitute|not substitute|cannot.*speedup)")
        self.assertNotIn("dcl/", content)
        self.assertNotIn("C++20", content)


class TestReadmeDocumentation(unittest.TestCase):
    def test_table_of_contents_anchors_resolve(self):
        readme = read("README.md")
        anchors = {github_anchor(m.group(1)) for m in re.finditer(r"^#{1,6} (.+)$", readme, re.M)}
        broken = [a for a in re.findall(r"\]\(#([^)]+)\)", readme) if a not in anchors]
        self.assertEqual(broken, [])

    def test_readme_open_in_colab_badge(self):
        readme = read("README.md")
        badge_img = "https://colab.research.google.com/assets/colab-badge.svg"
        target_link = "https://colab.research.google.com/github/lispz1610/HARMONY/blob/feature/migrate-to-upstream-harmony/colab/harmony_colab.ipynb"
        top_lines = "\n".join(readme.splitlines()[:40])
        self.assertIn(badge_img, top_lines)
        self.assertIn(target_link, top_lines)

    def test_readme_pre_cluster_validation_section(self):
        readme = read("README.md")
        self.assertIn("### Pre-Cluster Validation on Google Colab (Optional)", readme)
        section = readme.split("### Pre-Cluster Validation on Google Colab (Optional)", 1)[1].split("\n## ", 1)[0]
        for expected in ("Pre-Cluster Validation", "colab/harmony_colab.ipynb", "bootstrap.sh", "build.sh", "run_validation.sh"):
            self.assertIn(expected, section)
        self.assertRegex(section, r"(?i)(cannot substitute|does not replace)")
        self.assertNotIn("build/colab", section)

    def test_readme_super_job_section(self):
        readme = read("README.md")
        self.assertIn("### Super Job Variance Investigation Workflow", readme)
        section = readme.split("### Super Job Variance Investigation Workflow", 1)[1].split("\n## ", 1)[0]
        self.assertIn("./build.sh all", section)
        self.assertIn("qsub job_harmony_super.pbs", section)
        self.assertIn("benchmarks/analyze_super_matrix.py", section)
        self.assertIn("--summarize-only", section)


if __name__ == "__main__":
    unittest.main()
