# Migration-cost profiler used by the "profiled" balancing modes
# (called by the root build.sh).

COMPILE_EXECUTABLE migration_profiler "tools/migration_profiler/migration_profiler.cpp" "" "$OPENCL_LIBS"
