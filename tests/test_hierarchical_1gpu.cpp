#ifndef OMPI_SKIP_MPICXX
#define OMPI_SKIP_MPICXX 1
#endif
#ifndef MPICH_SKIP_MPICXX
#define MPICH_SKIP_MPICXX 1
#endif

#include <mpi.h>
#include "../dcl/runtime.hpp"
#include "../dcl/runtime.cpp"
#include "../dcl/topo_metrics_io.hpp"
#include <iostream>

int main(int argc, char** argv) {
    dcl::Runtime rt = dcl::Runtime::create(argc, argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    rt.set_simulated_devices_count(2); // 1 per rank
    
    dcl::PartitionSpec ps;
    ps.global_elements = 1000000;
    ps.units_per_element = 1;
    ps.bytes_per_unit = 4;
    ps.granularity = 1;
    rt.set_partition(ps);

    // Give a slight time imbalance to trigger the float precision issue if any, but keep hysteresis
    rt.set_simulated_times({0.100, 0.101});
    rt.maybe_rebalance_hierarchical();

    const auto& parts = rt.partitions();
    for(size_t i=0; i<parts.size(); i++){
        std::cout << "Rank " << rank << " part " << i << " offset=" << parts[i].global_offset << " count=" << parts[i].element_count << "\n";
    }

    MPI_Finalize();
    return 0;
}
