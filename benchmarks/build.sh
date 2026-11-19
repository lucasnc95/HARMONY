# Scientific benchmarks (called by the root build.sh).

COMPILE_EXECUTABLE kneighbor "benchmarks/kneighbor.cpp" \
    "-Iinclude -Ibenchmarks -DKERNEL_FILE='\"$ROOT_DIR/benchmarks/kneighbor.cl\"'" \
    "$LIB_DIR/libharmony.a $OPENCL_LIBS"

COMPILE_EXECUTABLE leanmd "benchmarks/leanmd.cpp" \
    "-Iinclude -Ibenchmarks -DKERNEL_FILE='\"$ROOT_DIR/benchmarks/leanmd.cl\"'" \
    "$LIB_DIR/libharmony.a $OPENCL_LIBS"
