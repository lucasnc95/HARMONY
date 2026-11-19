#!/usr/bin/env bash
# ==============================================================================
# HARMONY Pre-Cluster Validation: Refactored C++17 Build Automation
# Builds lib, tools, benchmarks, and tests via the unified ./build.sh system.
# ==============================================================================
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd "${script_dir}/.." && pwd)"

BUILD_SCRIPT="${BUILD_SCRIPT:-${root_dir}/build.sh}"
BIN_DIR="${BIN_DIR:-${root_dir}/bin}"
TEST_BIN_DIR="${TEST_BIN_DIR:-${root_dir}/tests/bin}"

dry_run=0
build_opts=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            dry_run=1
            shift
            ;;
        -d|--debug)
            build_opts+=("-d")
            shift
            ;;
        -r|--release)
            build_opts+=("-r")
            shift
            ;;
        -j)
            if [[ $# -lt 2 ]]; then
                echo "Error: -j requires a job count argument" >&2
                exit 1
            fi
            build_opts+=("-j" "$2")
            shift 2
            ;;
        -q|-v)
            build_opts+=("$1")
            shift
            ;;
        -h|--help)
            cat << 'EOF'
Usage: colab/build.sh [OPTIONS]

Builds HARMONY library, tools, benchmarks, and tests for Pre-Cluster Validation
using the repository's ./build.sh script (targets: lib tools benchmarks tests).

Options:
  --dry-run       Display build command and target binaries without compiling
  -d, --debug     Pass -d (debug build) to ./build.sh
  -r, --release   Pass -r (release build, default) to ./build.sh
  -j N            Pass -j N parallel jobs to ./build.sh
  -q              Quiet output
  -v              Verbose compiler output
  -h, --help      Show this help message and exit
EOF
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

if [[ "${COLAB_BUILD_DRY_RUN:-0}" == "1" || "${DRY_RUN:-0}" == "1" ]]; then
    dry_run=1
fi

if [[ -z "${MPICXX:-}" && -n "${CXX:-}" ]]; then
    export MPICXX="${CXX}"
fi
if [[ -z "${EXTRA_CXXFLAGS:-}" && -n "${CXXFLAGS:-}" ]]; then
    export EXTRA_CXXFLAGS="${CXXFLAGS}"
fi
export OPENCL_INCLUDE_DIR="${OPENCL_INCLUDE_DIR:-}"
export OPENCL_LIBRARY_DIR="${OPENCL_LIBRARY_DIR:-}"
export CL_TARGET_OPENCL_VERSION="${CL_TARGET_OPENCL_VERSION:-300}"

targets=(lib tools benchmarks tests)

echo "=== [Colab Build] Root Build Script: ${BUILD_SCRIPT} ==="
echo "=== [Colab Build] Targets: ${targets[*]} ==="
echo "=== [Colab Build] Binary Dir: ${BIN_DIR} | Test Binary Dir: ${TEST_BIN_DIR} ==="

if [[ "${dry_run}" -eq 1 ]]; then
    echo "[DRY-RUN] Command: ${BUILD_SCRIPT} ${build_opts[*]:-} ${targets[*]}"
    echo "[DRY-RUN] Expected library: ${root_dir}/lib/libharmony.a"
    echo "[DRY-RUN] Expected binaries: ${BIN_DIR}/topo_probe ${BIN_DIR}/kneighbor ${BIN_DIR}/leanmd"
    echo "[DRY-RUN] Expected tests: ${TEST_BIN_DIR}/test_topo_metrics ${TEST_BIN_DIR}/test_algorithms ${TEST_BIN_DIR}/test_async_halo_opencl"
    echo "=== [Colab Build] Dry-run completed successfully ==="
    exit 0
fi

if [[ ! -f "${BUILD_SCRIPT}" ]]; then
    echo "ERROR: Build script not found: ${BUILD_SCRIPT}" >&2
    exit 1
fi

(cd "${root_dir}" && bash "${BUILD_SCRIPT}" "${build_opts[@]}" "${targets[@]}")

required_bins=(
    "${BIN_DIR}/topo_probe"
    "${BIN_DIR}/kneighbor"
    "${BIN_DIR}/leanmd"
    "${TEST_BIN_DIR}/test_topo_metrics"
    "${TEST_BIN_DIR}/test_algorithms"
    "${TEST_BIN_DIR}/test_async_halo_opencl"
)

for bin_path in "${required_bins[@]}"; do
    if [[ ! -x "${bin_path}" ]]; then
        echo "ERROR: Missing expected binary after build: ${bin_path}" >&2
        exit 1
    fi
done

echo "=== [Colab Build] All Pre-Cluster Validation targets built successfully ==="
exit 0
