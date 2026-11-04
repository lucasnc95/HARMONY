#include <mpi.h>
#include "../dcl/runtime.hpp"
#include "../dcl/runtime.cpp"
#include <cassert>
#include <iostream>

int main(int argc, char** argv) {
    dcl::Runtime rt = dcl::Runtime::create(argc, argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    rt.set_simulated_devices_count(2);

    dcl::PartitionSpec ps;
    ps.global_elements = 1000000;
    ps.units_per_element = 1;
    ps.bytes_per_unit = 4;
    ps.granularity = 1;
    rt.set_partition(ps);

    // Initial partitions: [0, 500000) on dev 0, [500000, 1000000) on dev 1
    // Let's force an extreme imbalance first
    rt.set_simulated_times({1000000.0, 0.0001});
    rt.maybe_rebalance_hierarchical();
    
    auto p1 = rt.partitions();
    if (rank == 0) std::cout << "After first rebalance: dev 0=" << p1[0].element_count << " dev 1=" << p1[1].element_count << std::endl;

    // Now dev 1 is the fast one, dev 0 is the slow one. Dev 0 has very few (or 0).
    // What if dev 0 becomes fast?
    rt.set_simulated_times({0.0001, 10.0});
    rt.maybe_rebalance_hierarchical();

    auto p2 = rt.partitions();
    if (rank == 0) std::cout << "After second rebalance: dev 0=" << p2[0].element_count << " dev 1=" << p2[1].element_count << std::endl;

    MPI_Finalize();
    return 0;
}
