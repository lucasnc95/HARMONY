Compila no cluster:

 /opt/ohpc/pub/mpi/mpich-ofi-gnu13-ohpc/3.4.3/bin/mpic++    -std=c++20  profiling.cpp -o profiling.out   -Iinclude   -L/usr/lib64/   -L/usr/local/cuda-12.6/targets/x86_64-linux/lib/   -L/opt/intel/oneapi/2024.0/lib/   -I/usr/local/cuda-12.6/targets/x86_64-linux/include/   -I/opt/intel/oneapi/2024.0/opt/oclfpga/host/include/   -lOpenCL  -DCL_TARGET_OPENCL_VERSION=300 -O3
