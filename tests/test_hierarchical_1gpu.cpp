#include "harmony/runtime.hpp"

#include <mpi.h>

#include <cassert>
#include <iostream>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    auto rt = harmony::Runtime::create(argc, argv);
    rt.set_verbose(false);
    if (size == 1) rt.set_simulated_ranks_for_testing(2);
    rt.set_simulated_devices_count(2);
    rt.set_partition({1000000, 1, sizeof(float), 1});

    harmony::AutoBalancePolicy policy;
    policy.mode = harmony::BalanceMode::hierarchical;
    policy.interval = 1;
    auto step = rt.step("hier_1gpu_hyst").with_balance(policy).build();

    // Slight imbalance (1%): inter-rank gain < 5% hysteresis -> keeps equal split without float precision issues.
    rt.set_simulated_times({0.100, 0.101});
    rt.execute(step);

    const auto& parts = rt.partitions();
    assert(parts.size() == 2);
    assert(parts[0].global_offset == 0 && parts[0].element_count == 500000);
    assert(parts[1].global_offset == 500000 && parts[1].element_count == 500000);

    if (rank == 0) {
        std::cout << "[PASS] test_hierarchical_1gpu\n";
    }
    MPI_Finalize();
    return 0;
}
