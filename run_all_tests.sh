#!/bin/bash
set -e
mpic++ -std=c++20 -Wall -Wextra -O2 tests/test_topo_metrics.cpp -lOpenCL -DCL_TARGET_OPENCL_VERSION=300 -o tests/test_topo_metrics.out
mpic++ -std=c++20 -Wall -Wextra -O2 tests/test_numa_cost.cpp -lOpenCL -DCL_TARGET_OPENCL_VERSION=300 -o tests/test_numa_cost.out
mpic++ -std=c++20 -Wall -Wextra -O2 tests/test_hierarchical.cpp -lOpenCL -DCL_TARGET_OPENCL_VERSION=300 -o tests/test_hierarchical.out
mpic++ -std=c++20 -Wall -Wextra -O2 tests/test_algorithms.cpp -lOpenCL -DCL_TARGET_OPENCL_VERSION=300 -o tests/test_algorithms.out
mpic++ -std=c++20 -Wall -Wextra -O2 tests/test_contention_power.cpp -lOpenCL -DCL_TARGET_OPENCL_VERSION=300 -o tests/test_contention_power.out

mpirun --oversubscribe -n 1 ./tests/test_topo_metrics.out
mpirun --oversubscribe -n 1 ./tests/test_numa_cost.out
mpirun --oversubscribe -n 2 ./tests/test_hierarchical.out
mpirun --oversubscribe -n 1 ./tests/test_algorithms.out
mpirun --oversubscribe -n 1 ./tests/test_contention_power.out
