#!/usr/bin/env bash
# Detection of the MPI compiler wrapper and of OpenCL.
#
#   FIND_MPI     sets CXX to the MPI C++ wrapper ($MPICXX, mpicxx, mpic++, mpiCC)
#   FIND_OPENCL  sets OPENCL_CFLAGS / OPENCL_LIBS, using $OPENCL_INCLUDE_DIR and
#                $OPENCL_LIBRARY_DIR when given, else the usual install paths.
#                A library outside the default search path is also recorded as
#                an rpath, so executables find it at run time.

FIND_MPI() {
    local candidate
    for candidate in "${MPICXX:-}" mpicxx mpic++ mpiCC; do
        [ -n "$candidate" ] || continue
        if command -v "$candidate" > /dev/null 2>&1; then
            CXX=$(command -v "$candidate")
            PRINT_INFO "MPI C++ compiler: $CXX"
            return 0
        fi
    done
    PRINT_ERROR "no MPI C++ compiler found (set MPICXX=/path/to/mpic++)"
    return 1
}

# Compiles and links a tiny OpenCL program with the given flags.
_TRY_OPENCL() {
    local cflags=$1 libs=$2
    local tmp
    tmp=$(mktemp -d)
    cat > "$tmp/t.cpp" <<'EOF'
#include <CL/cl.h>
int main() { cl_uint n = 0; return clGetPlatformIDs(0, nullptr, &n) == CL_SUCCESS ? 0 : 0; }
EOF
    # shellcheck disable=SC2086
    "$CXX" $cflags -DCL_TARGET_OPENCL_VERSION=120 "$tmp/t.cpp" -o "$tmp/t" $libs > /dev/null 2>&1
    local status=$?
    rm -rf "$tmp"
    return $status
}

FIND_OPENCL() {
    local inc lib
    # Explicit paths first, then the default search path, then common SDKs.
    local -a include_dirs=("${OPENCL_INCLUDE_DIR:-}" "" /usr/local/cuda/include /opt/rocm/opencl/include
                           /opt/intel/oneapi/compiler/latest/include/sycl /opt/intel/opencl/include)
    local -a library_dirs=("${OPENCL_LIBRARY_DIR:-}" "" /usr/lib64 /usr/local/cuda/lib64 /opt/rocm/opencl/lib
                           /opt/intel/opencl/lib64)

    for inc in "${include_dirs[@]}"; do
        if [ -n "$inc" ] && [ ! -f "$inc/CL/cl.h" ]; then continue; fi
        for lib in "${library_dirs[@]}"; do
            if [ -n "$lib" ] && [ ! -d "$lib" ]; then continue; fi
            local cflags="" libs="-lOpenCL"
            [ -n "$inc" ] && cflags="-I$inc"
            [ -n "$lib" ] && libs="-L$lib -Wl,-rpath,$lib -lOpenCL"
            if _TRY_OPENCL "$cflags" "$libs"; then
                OPENCL_CFLAGS=$cflags
                OPENCL_LIBS=$libs
                PRINT_INFO "OpenCL: ${cflags:-default include path} ${libs}"
                return 0
            fi
        done
    done
    PRINT_ERROR "OpenCL not found (set OPENCL_INCLUDE_DIR and OPENCL_LIBRARY_DIR)"
    return 1
}
