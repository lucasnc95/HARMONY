#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$root_dir"

module load gnu13/13.2.0
module load openmpi5/5.0.5
module load prun/2.2

echo "=== Frontend build toolchain ==="
module list 2>&1
command -v mpic++
mpic++ --version
printf '#include <span>\nint main() { std::span<int> value; return value.size(); }\n' \
    | mpic++ -std=c++20 -x c++ -fsyntax-only -

bin_dir="$root_dir/build/cluster"
mkdir -p "$bin_dir"
common_flags=(-std=c++20 -Wall -Wextra -O2 -DCL_TARGET_OPENCL_VERSION=300
              -I"$root_dir/third_party/opencl_headers")

opencl_library="${OPENCL_LIBRARY:-}"
if [[ -z "$opencl_library" ]] && command -v ldconfig >/dev/null 2>&1; then
    opencl_library="$(ldconfig -p 2>/dev/null | awk '$1 == "libOpenCL.so" && !found { print $NF; found = 1 }')"
fi
opencl_library="${opencl_library:--lOpenCL}"
echo "OpenCL library: $opencl_library"

probe_bin="$(mktemp "$bin_dir/.opencl-link.XXXXXX")"
trap 'rm -f "$probe_bin"' EXIT
printf '#include <CL/cl.h>\n#ifndef CL_VERSION_3_0\n#error OpenCL 3.0 headers required\n#endif\nint main() { cl_uint n = 0; return clGetPlatformIDs(0, nullptr, &n); }\n' \
    | mpic++ "${common_flags[@]}" -x c++ - -x none "$opencl_library" -o "$probe_bin"
rm -f "$probe_bin"

echo "=== Build test suites ==="
for name in topo_metrics numa_cost hierarchical algorithms contention_power \
            probe_runtime_integration async_halo_opencl; do
    mpic++ "${common_flags[@]}" "tests/test_${name}.cpp" "$opencl_library" \
        -o "$bin_dir/test_${name}.out"
done

echo "=== Build topology probe and main-program smoke executable ==="
mpic++ "${common_flags[@]}" topo_probe.cpp "$opencl_library" \
    -o "$bin_dir/topo_probe.out"
mpic++ "${common_flags[@]}" main_high_order_residual.cpp dcl/runtime.cpp \
    "$opencl_library" -o "$bin_dir/harmony_smoke.out"

echo "=== Build benchmarks ==="
BENCH_BIN_DIR="$bin_dir" BENCH_BUILD_ONLY=1 OPENCL_LIBRARY="$opencl_library" \
    bash "$root_dir/benchmarks/compare_algos.sh"

# Provenance is captured on the frontend, where git and the compiler exist.
python3 "$root_dir/benchmarks/write_build_manifest.py" "$root_dir" "$bin_dir" \
    mpic++ "$opencl_library" -std=c++20 -Wall -Wextra -Wpedantic \
    -Wno-unused-parameter -O3 -DCL_TARGET_OPENCL_VERSION=300 \
    "-I$root_dir/third_party/opencl_headers"

echo "Frontend build complete: $bin_dir"
