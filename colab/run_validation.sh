#!/usr/bin/env bash
# ==============================================================================
# HARMONY Pre-Cluster Validation: Functional Tests and Fast Benchmarks
# Three-phase validation runner for single-node Google Colab GPU environments.
# ==============================================================================
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd "${script_dir}/.." && pwd)"

BIN_DIR="${BIN_DIR:-${COLAB_BIN_DIR:-${root_dir}/bin}}"
TEST_BIN_DIR="${TEST_BIN_DIR:-}"
RUNS_DIR="${RUNS_DIR:-${root_dir}/runs/colab_$(date +%Y%m%d_%H%M%S)}"
MPI_LAUNCHER="${MPI_LAUNCHER:-mpirun}"
BENCH_STEPS="${BENCH_STEPS:-20}"
BENCH_INTERVAL="${BENCH_INTERVAL:-5}"
BENCH_N="${BENCH_N:-50000}"
BENCH_DEVICE="${BENCH_DEVICE:-}"

dry_run=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            dry_run=1
            shift
            ;;
        --bin-dir)
            BIN_DIR="$2"
            shift 2
            ;;
        --test-bin-dir)
            TEST_BIN_DIR="$2"
            shift 2
            ;;
        --runs-dir)
            RUNS_DIR="$2"
            shift 2
            ;;
        --steps|--iterations)
            BENCH_STEPS="$2"
            shift 2
            ;;
        --interval|--balance-interval)
            BENCH_INTERVAL="$2"
            shift 2
            ;;
        --n)
            BENCH_N="$2"
            shift 2
            ;;
        --device)
            BENCH_DEVICE="$2"
            shift 2
            ;;
        --launcher)
            MPI_LAUNCHER="$2"
            shift 2
            ;;
        -h|--help)
            cat << 'EOF'
Usage: colab/run_validation.sh [OPTIONS]

Pre-Cluster Validation runner for functional checks and fast benchmarks in Google Colab.

Options:
  --bin-dir <DIR>       Directory containing benchmark binaries (default: bin/)
  --test-bin-dir <DIR>  Directory containing test binaries (default: tests/bin/)
  --runs-dir <DIR>      Directory to store logs and status (default: runs/colab_<timestamp>)
  --iterations <N>      Number of iterations for fast benchmarks (default: 20)
  --steps <N>           Alias for --iterations
  --balance-interval <N> Rebalance interval for fast benchmarks (default: 5)
  --interval <N>        Alias for --balance-interval
  --n <N>               Problem size N for fast benchmarks (default: 50000)
  --device <MODE>       Optional device mode override (auto|gpu|cpu|all|sim)
  --launcher <CMD>      MPI launcher command (default: mpirun)
  --dry-run             Display planned execution stages without executing binaries
  -h, --help            Show this help message and exit
EOF
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

if [[ "${COLAB_VALIDATION_DRY_RUN:-0}" == "1" || "${DRY_RUN:-0}" == "1" ]]; then
    dry_run=1
fi

export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

resolve_test_bin() {
    local name="$1"
    if [[ -n "${TEST_BIN_DIR}" ]]; then
        echo "${TEST_BIN_DIR}/${name}"
    elif [[ -x "${BIN_DIR}/${name}" ]]; then
        echo "${BIN_DIR}/${name}"
    else
        echo "${root_dir}/tests/bin/${name}"
    fi
}

if [[ "${dry_run}" -eq 1 ]]; then
    echo "=== [DRY-RUN] HARMONY Pre-Cluster Validation Plan ==="
    echo "Binary Directory:      ${BIN_DIR}"
    echo "Test Binary Directory: ${TEST_BIN_DIR:-${root_dir}/tests/bin}"
    echo "Run Directory:         ${RUNS_DIR}"
    echo "Launcher:              ${MPI_LAUNCHER} (with --allow-run-as-root --oversubscribe)"
    echo "Benchmark Specs:       N=${BENCH_N}, iterations=${BENCH_STEPS}, balance-interval=${BENCH_INTERVAL}"
    echo ""
    echo "[DRY-RUN] Phase 1: Unit Tests"
    echo "  - ${MPI_LAUNCHER} -np 1 $(resolve_test_bin test_topo_metrics)"
    echo "  - ${MPI_LAUNCHER} -np 1 $(resolve_test_bin test_algorithms)"
    echo "  - ${MPI_LAUNCHER} -np 1 $(resolve_test_bin test_numa_cost)"
    echo "  - ${MPI_LAUNCHER} -np 1 $(resolve_test_bin test_contention_power)"
    echo "  - ${MPI_LAUNCHER} -np 2 $(resolve_test_bin test_hierarchical)"
    echo ""
    echo "[DRY-RUN] Phase 2: OpenCL Halo Integration"
    echo "  - ${MPI_LAUNCHER} -np 1 $(resolve_test_bin test_async_halo_opencl)"
    echo "  - ${MPI_LAUNCHER} -np 2 $(resolve_test_bin test_async_halo_opencl)"
    echo ""
    echo "[DRY-RUN] Phase 3: Fast Benchmarks"
    echo "  - ${MPI_LAUNCHER} -np 2 ${BIN_DIR}/kneighbor --n ${BENCH_N} --iterations ${BENCH_STEPS} --balance-interval ${BENCH_INTERVAL} --balance-algo erad"
    echo "  - ${MPI_LAUNCHER} -np 2 ${BIN_DIR}/kneighbor --n ${BENCH_N} --iterations ${BENCH_STEPS} --balance-interval ${BENCH_INTERVAL} --balance-algo hwtopolb"
    echo "  - ${MPI_LAUNCHER} -np 2 ${BIN_DIR}/leanmd --n ${BENCH_N} --iterations ${BENCH_STEPS} --balance-interval ${BENCH_INTERVAL} --balance-algo erad"
    echo "  - ${MPI_LAUNCHER} -np 2 ${BIN_DIR}/leanmd --n ${BENCH_N} --iterations ${BENCH_STEPS} --balance-interval ${BENCH_INTERVAL} --balance-algo hwtopolb"
    echo "=== [DRY-RUN] Validation plan displayed successfully ==="
    exit 0
fi

# Verify required binaries exist and are executable
for bin_path in \
    "$(resolve_test_bin test_topo_metrics)" \
    "$(resolve_test_bin test_algorithms)" \
    "$(resolve_test_bin test_numa_cost)" \
    "$(resolve_test_bin test_contention_power)" \
    "$(resolve_test_bin test_hierarchical)" \
    "$(resolve_test_bin test_async_halo_opencl)" \
    "${BIN_DIR}/kneighbor" \
    "${BIN_DIR}/leanmd"; do
    if [[ ! -x "${bin_path}" ]]; then
        echo "ERROR: Missing required binary: ${bin_path}" >&2
        echo "Please compile all targets first using 'bash colab/build.sh'." >&2
        exit 1
    fi
done

mkdir -p "${RUNS_DIR}"

phase1_status="PENDING"
topo_status="PENDING"
algo_status="PENDING"
numa_status="PENDING"
contention_status="PENDING"
hierarchical_status="PENDING"

phase2_status="PENDING"
halo_np1_status="PENDING"
halo_np2_status="PENDING"

phase3_status="PENDING"
kneighbor_erad_status="PENDING"
kneighbor_hwtopolb_status="PENDING"
leanmd_erad_status="PENDING"
leanmd_hwtopolb_status="PENDING"

write_status_json() {
    local current_status="$1"
    local status_file="${RUNS_DIR}/colab_status.json"
    python3 - "$current_status" \
        "$phase1_status" "$topo_status" "$algo_status" "$numa_status" "$contention_status" "$hierarchical_status" \
        "$phase2_status" "$halo_np1_status" "$halo_np2_status" \
        "$phase3_status" "$kneighbor_erad_status" "$kneighbor_hwtopolb_status" "$leanmd_erad_status" "$leanmd_hwtopolb_status" \
        "${RUNS_DIR}" "$(date -Iseconds 2>/dev/null || date)" "${status_file}" <<'PY'
import json
import sys

data = {
    "status": sys.argv[1],
    "phases": {
        "phase1_unit_tests": {
            "status": sys.argv[2],
            "test_topo_metrics": sys.argv[3],
            "test_algorithms": sys.argv[4],
            "test_numa_cost": sys.argv[5],
            "test_contention_power": sys.argv[6],
            "test_hierarchical": sys.argv[7],
        },
        "phase2_halo_integration": {
            "status": sys.argv[8],
            "test_async_halo_opencl_np1": sys.argv[9],
            "test_async_halo_opencl_np2": sys.argv[10],
        },
        "phase3_fast_benchmarks": {
            "status": sys.argv[11],
            "kneighbor_erad": sys.argv[12],
            "kneighbor_hwtopolb": sys.argv[13],
            "leanmd_erad": sys.argv[14],
            "leanmd_hwtopolb": sys.argv[15],
        },
    },
    "run_dir": sys.argv[16],
    "timestamp": sys.argv[17],
}
with open(sys.argv[18], "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)
PY
}

cleanup_on_error() {
    write_status_json "FAIL"
    echo "" >&2
    echo "=== [Pre-Cluster Validation] FAILED: Execution aborted due to an error ===" >&2
}
trap cleanup_on_error ERR

run_unit_test() {
    local test_name="$1"
    local ranks="$2"
    local status_var="$3"
    local test_bin
    test_bin="$(resolve_test_bin "${test_name}")"
    local log_file="${RUNS_DIR}/${test_name}.log"
    local rc=0

    echo "--> Executing ${test_name} (-np ${ranks})..."
    ${MPI_LAUNCHER} --allow-run-as-root --oversubscribe -np "${ranks}" "${test_bin}" > "${log_file}" 2>&1 || rc=$?
    if [[ $rc -eq 0 ]]; then
        printf -v "${status_var}" '%s' "PASS"
        echo "    PASS: ${test_name}"
    else
        printf -v "${status_var}" '%s' "FAIL"
        phase1_status="FAIL"
        echo "    FAIL: ${test_name} failed (see ${log_file})" >&2
        cat "${log_file}" >&2
        cleanup_on_error
        exit 1
    fi
}

run_halo_check() {
    local ranks="$1"
    local status_var="$2"
    local halo_bin
    halo_bin="$(resolve_test_bin "test_async_halo_opencl")"
    local log_file="${RUNS_DIR}/test_async_halo_opencl_np${ranks}.log"
    local rc=0

    echo "--> Executing test_async_halo_opencl (-np ${ranks})..."
    ${MPI_LAUNCHER} --allow-run-as-root --oversubscribe -np "${ranks}" "${halo_bin}" > "${log_file}" 2>&1 || rc=$?
    if [[ $rc -eq 0 ]]; then
        printf -v "${status_var}" '%s' "PASS"
        echo "    PASS: test_async_halo_opencl (-np ${ranks})"
    elif [[ $rc -eq 77 ]]; then
        printf -v "${status_var}" '%s' "SKIP"
        echo "    SKIP: test_async_halo_opencl (-np ${ranks}, insufficient OpenCL GPUs)"
    else
        printf -v "${status_var}" '%s' "FAIL"
        phase2_status="FAIL"
        echo "    FAIL: test_async_halo_opencl (-np ${ranks}) exited with code ${rc}" >&2
        cat "${log_file}" >&2
        cleanup_on_error
        exit "${rc}"
    fi
}

run_benchmark_case() {
    local bench="$1"
    local algo_lower="$2"
    local status_var="$3"
    local bench_log="${RUNS_DIR}/${bench}_${algo_lower}.log"
    local bench_csv="${RUNS_DIR}/${bench}_${algo_lower}.csv"
    local bench_report_csv="${RUNS_DIR}/${bench}_${algo_lower}_report.csv"
    local bench_rc=0

    local cmd=(
        ${MPI_LAUNCHER} --allow-run-as-root --oversubscribe -np 2
        "${BIN_DIR}/${bench}"
        --n "${BENCH_N}"
        --iterations "${BENCH_STEPS}"
        --balance-interval "${BENCH_INTERVAL}"
        --balance-algo "${algo_lower}"
        --output-csv "${bench_csv}"
        --report-csv "${bench_report_csv}"
    )
    if [[ -n "${BENCH_DEVICE}" ]]; then
        cmd+=(--device "${BENCH_DEVICE}")
    fi

    echo "--> Running ${bench} with 2 ranks (algo=${algo_lower}, N=${BENCH_N}, iterations=${BENCH_STEPS})..."
    "${cmd[@]}" > "${bench_log}" 2>&1 || bench_rc=$?

    if [[ $bench_rc -ne 0 ]] || grep -qi "verification failed" "${bench_log}"; then
        printf -v "${status_var}" '%s' "FAIL"
        phase3_status="FAIL"
        if [[ $bench_rc -ne 0 ]]; then
            echo "    FAIL: ${bench} (${algo_lower}) exited with non-zero code ${bench_rc}" >&2
        else
            echo "    FAIL: ${bench} (${algo_lower}) verification failed or non-finite output detected!" >&2
        fi
        cat "${bench_log}" >&2
        cleanup_on_error
        exit 1
    fi

    printf -v "${status_var}" '%s' "PASS"
    echo "    PASS: ${bench} (${algo_lower}) completed and verified finite"
}

echo "=== [Pre-Cluster Validation] Phase 1: Unit Tests (R1-R5) ==="
run_unit_test "test_topo_metrics" 1 "topo_status"
run_unit_test "test_algorithms" 1 "algo_status"
run_unit_test "test_numa_cost" 1 "numa_status"
run_unit_test "test_contention_power" 1 "contention_status"
run_unit_test "test_hierarchical" 2 "hierarchical_status"
phase1_status="PASS"

echo "=== [Pre-Cluster Validation] Phase 2: OpenCL Halo Integration ==="
run_halo_check 1 "halo_np1_status"
run_halo_check 2 "halo_np2_status"
phase2_status="PASS"

echo "=== [Pre-Cluster Validation] Phase 3: Fast Benchmarks (erad & hwtopolb) ==="
run_benchmark_case "kneighbor" "erad" "kneighbor_erad_status"
run_benchmark_case "kneighbor" "hwtopolb" "kneighbor_hwtopolb_status"
run_benchmark_case "leanmd" "erad" "leanmd_erad_status"
run_benchmark_case "leanmd" "hwtopolb" "leanmd_hwtopolb_status"
phase3_status="PASS"

write_status_json "PASS"

echo ""
echo "========================================================================"
echo "=== [Pre-Cluster Validation Summary] Status: PASS ==="
echo "========================================================================"
echo "Phase 1: Unit Tests:               ${phase1_status}"
echo "  - test_topo_metrics:             ${topo_status}"
echo "  - test_algorithms:               ${algo_status}"
echo "  - test_numa_cost:                ${numa_status}"
echo "  - test_contention_power:         ${contention_status}"
echo "  - test_hierarchical:             ${hierarchical_status}"
echo "Phase 2: Halo Integration:         ${phase2_status}"
echo "  - test_async_halo_opencl (np1):  ${halo_np1_status}"
echo "  - test_async_halo_opencl (np2):  ${halo_np2_status}"
echo "Phase 3: Fast Benchmarks:          ${phase3_status}"
echo "  - kneighbor (erad):              ${kneighbor_erad_status}"
echo "  - kneighbor (hwtopolb):          ${kneighbor_hwtopolb_status}"
echo "  - leanmd (erad):                 ${leanmd_erad_status}"
echo "  - leanmd (hwtopolb):             ${leanmd_hwtopolb_status}"
echo "========================================================================"
echo "Artifacts and logs saved in: ${RUNS_DIR}"
echo "Status file emitted at:     ${RUNS_DIR}/colab_status.json"
exit 0
