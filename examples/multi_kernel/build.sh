#!/usr/bin/env bash
# Builds this example against HARMONY. It does not use HARMONY's build
# system: it needs only the headers, the static library and the flags that
# HARMONY's ./build.sh recorded in lib/harmony_config.sh (compiler, OpenCL).
#
#   ./build.sh            -> bin/multi_kernel
#   ./build.sh clean
#
# Environment (each one overrides lib/harmony_config.sh):
#   HARMONY_ROOT        directory with include/harmony/ and lib/libharmony.a
#                       (default: the repository root, ../..; an install prefix also works)
#   MPICXX              MPI C++ wrapper
#   OPENCL_INCLUDE_DIR  directory containing CL/cl.h
#   OPENCL_LIBRARY_DIR  directory containing libOpenCL.so
#   CXXFLAGS            compiler flags (default: -O3)

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

NAME=multi_kernel
SOURCES="main.cpp"
KERNEL="$PWD/multi_kernel.cl"

if [ "${1:-}" = "clean" ]; then
    rm -rf bin
    exit 0
fi

HARMONY_ROOT=${HARMONY_ROOT:-$(cd ../.. && pwd)}
if [ ! -f "$HARMONY_ROOT/lib/libharmony.a" ]; then
    echo "error: $HARMONY_ROOT/lib/libharmony.a not found (run HARMONY's ./build.sh or set HARMONY_ROOT)" >&2
    exit 1
fi

# Same compiler and OpenCL as the library, unless overridden.
HARMONY_CXX="" HARMONY_OPENCL_CFLAGS="" HARMONY_OPENCL_LIBS="-lOpenCL" HARMONY_CL_TARGET_OPENCL_VERSION=300
if [ -f "$HARMONY_ROOT/lib/harmony_config.sh" ]; then
    # shellcheck disable=SC1091
    source "$HARMONY_ROOT/lib/harmony_config.sh"
fi
CXX=${MPICXX:-${HARMONY_CXX:-mpicxx}}
OPENCL_CFLAGS=$HARMONY_OPENCL_CFLAGS
OPENCL_LIBS=$HARMONY_OPENCL_LIBS
[ -n "${OPENCL_INCLUDE_DIR:-}" ] && OPENCL_CFLAGS="-I$OPENCL_INCLUDE_DIR"
[ -n "${OPENCL_LIBRARY_DIR:-}" ] && OPENCL_LIBS="-L$OPENCL_LIBRARY_DIR -Wl,-rpath,$OPENCL_LIBRARY_DIR -lOpenCL"

mkdir -p bin
# shellcheck disable=SC2086
"$CXX" -std=c++17 ${CXXFLAGS:--O3} -Wall -Wextra -DOMPI_SKIP_MPICXX -DMPICH_SKIP_MPICXX \
    -DCL_TARGET_OPENCL_VERSION=${CL_TARGET_OPENCL_VERSION:-$HARMONY_CL_TARGET_OPENCL_VERSION} \
    -DKERNEL_FILE="\"$KERNEL\"" \
    -I"$HARMONY_ROOT/include" $OPENCL_CFLAGS \
    $SOURCES -o "bin/$NAME" \
    -L"$HARMONY_ROOT/lib" -lharmony $OPENCL_LIBS

echo "OK $PWD/bin/$NAME"
