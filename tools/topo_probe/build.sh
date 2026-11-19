# Standalone MPI+OpenCL topology probe (called by the root build.sh).

COMPILE_EXECUTABLE topo_probe "tools/topo_probe/topo_probe.cpp" "-Iinclude" "$OPENCL_LIBS"
