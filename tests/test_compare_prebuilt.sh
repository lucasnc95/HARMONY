#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
work_dir="$(mktemp -d)"
trap 'rm -rf "$work_dir"' EXIT
mkdir -p "$work_dir/bin" "$work_dir/output" "$work_dir/mock"

compiler="$(command -v mpic++)"
for name in kneighbor leanmd; do
    "$compiler" -std=c++20 -O0 -DCL_TARGET_OPENCL_VERSION=300 \
        "$root_dir/benchmarks/${name}.cpp" -lOpenCL \
        -o "$work_dir/bin/${name}.out"
done

cat > "$work_dir/mock/mpic++" <<'SH'
#!/usr/bin/env bash
if [[ "${1:-}" == "--version" ]]; then
    printf 'prebuilt-mode test compiler\n'
    exit 0
fi
printf 'Unexpected compilation during benchmark execution\n' >&2
exit 99
SH
chmod +x "$work_dir/mock/mpic++"

PATH="$work_dir/mock:$PATH" \
BENCH_BIN_DIR="$work_dir/bin" \
BENCH_SKIP_BUILD=1 \
BENCH_OUTPUT_DIR="$work_dir/output" \
BENCH_N=64 \
BENCH_ITERATIONS=2 \
BENCH_BALANCE_INTERVAL=1 \
MPI_LAUNCHER='' \
    bash "$root_dir/benchmarks/compare_algos.sh"

python3 - "$work_dir/output/comparison_summary.csv" <<'PY'
import csv
import sys

with open(sys.argv[1], newline="", encoding="utf-8") as source:
    rows = list(csv.DictReader(source))
if len(rows) != 4 or {row["execution_mode"] for row in rows} != {"cpu_simulation"}:
    raise SystemExit("Prebuilt benchmark summary is missing CPU simulation results")
print("PASS: prebuilt benchmarks ran without recompilation")
PY
