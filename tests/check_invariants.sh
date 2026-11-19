#!/usr/bin/env bash
# Invariants and automated guardrail check for HARMONY Super Job, Pre-Cluster Validation, and runtime core.
# Ensures that:
# 1. include/harmony/ and src/ runtime core remain 100% frozen against post-Layer-1 baseline.
# 2. C++ library/tests build cleanly and all C++ tests pass.
# 3. All offline Python unit and integration tests pass cleanly.
# 4. Prebuilt benchmark runner verifies without recompilation.
# 5. Prohibited domain terminology from GLOSSARY.md is not introduced in diffed files.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CORE_FREEZE_BASE="${HARMONY_CORE_FREEZE_BASE:-8d7c3392c8d616997b342f778ca041a9a717aff2}"

echo "=== 1. Checking Runtime Core Invariants (include/harmony/ and src/ freeze) ==="
if git rev-parse --verify "${CORE_FREEZE_BASE}^{commit}" >/dev/null 2>&1; then
    core_diff="$(git diff "${CORE_FREEZE_BASE}" -- include/harmony/ src/ || true)"
    core_untracked="$(git ls-files --others --exclude-standard -- include/harmony/ src/ || true)"
    if [[ -n "$core_diff" || -n "$core_untracked" ]]; then
        echo "ERROR: Modifications detected in frozen runtime core (include/harmony/ or src/) against ${CORE_FREEZE_BASE}!" >&2
        [[ -n "$core_diff" ]] && echo "$core_diff" >&2
        [[ -n "$core_untracked" ]] && echo "Untracked core files:"$'\n'"$core_untracked" >&2
        exit 1
    fi
    echo "PASS: include/harmony/ and src/ runtime core are 100% frozen against ${CORE_FREEZE_BASE}."
else
    echo "ERROR: Core freeze baseline commit ${CORE_FREEZE_BASE} not found in git history." >&2
    exit 1
fi

echo "=== 2. Building and Running C++ Test Suite ==="
./build.sh -q lib tools benchmarks tests
bash tests/run_tests.sh

echo "=== 3. Running Python Test Suite Discovery ==="
python3 -m unittest discover -s tests -p 'test_*.py'

echo "=== 4. Verifying Prebuilt Benchmark Script Guard ==="
bash tests/test_compare_prebuilt.sh

echo "=== 5. Checking Domain Glossary Terminology Guard ==="
python3 - "$CORE_FREEZE_BASE" << 'PY'
import re
import subprocess
import sys

base_commit = sys.argv[1]

# Parse _Avoid_: terms dynamically from GLOSSARY.md plus orchestration variant
avoid_terms = set()
with open("GLOSSARY.md", encoding="utf-8") as glossary:
    for line in glossary:
        if line.startswith("_Avoid_:"):
            for term in line.split(":", 1)[1].split(","):
                cleaned = term.strip().lower()
                if cleaned:
                    avoid_terms.add(cleaned)
avoid_terms.update({"orchestration", "legacy log", "old run"})

diff_cmd = [
    "git", "diff", base_commit, "--",
    "*.md", "*.py", "*.sh", "*.pbs", "*.ipynb",
    ":!GLOSSARY.md",
    ":!CODING_STANDARDS.md",
    ":!AGENTS.md",
    ":!tests/check_invariants.sh",
    ":!tests/test_docs_consistency.py",
    ":!tests/test_colab_notebook.py",
]
diff_output = subprocess.check_output(diff_cmd, text=True, stderr=subprocess.DEVNULL)

violations = []
for line in diff_output.splitlines():
    if line.startswith("+") and not line.startswith("+++"):
        low = line.lower()
        for term in sorted(avoid_terms):
            if re.search(r"\b" + re.escape(term) + r"\b", low):
                violations.append(f"Found prohibited term '{term}': {line.strip()}")

untracked_cmd = [
    "git", "ls-files", "--others", "--exclude-standard", "--",
    "colab/", "docs/adr/", "benchmarks/*.sh", "benchmarks/*.py",
    "tests/test_*.py", "tests/*.sh", "*.pbs",
]
untracked_files = subprocess.check_output(untracked_cmd, text=True, stderr=subprocess.DEVNULL).splitlines()
for rel_path in untracked_files:
    if rel_path in {"tests/check_invariants.sh", "tests/test_docs_consistency.py", "tests/test_colab_notebook.py"}:
        continue
    with open(rel_path, encoding="utf-8", errors="ignore") as f:
        for idx, line in enumerate(f, 1):
            low = line.lower()
            for term in sorted(avoid_terms):
                if re.search(r"\b" + re.escape(term) + r"\b", low):
                    violations.append(f"Found prohibited term '{term}' in {rel_path}:{idx}: {line.strip()}")

if violations:
    print("ERROR: Prohibited terminology found in diff (see GLOSSARY.md):", file=sys.stderr)
    for v in violations:
        print(f"  {v}", file=sys.stderr)
    sys.exit(1)

print("PASS: No prohibited domain terms found in diff.")
PY

echo "=== All invariants and automated guardrails passed successfully ==="
exit 0
