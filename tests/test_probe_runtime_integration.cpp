#ifndef OMPI_SKIP_MPICXX
#define OMPI_SKIP_MPICXX 1
#endif
#ifndef MPICH_SKIP_MPICXX
#define MPICH_SKIP_MPICXX 1
#endif
#include "../dcl/runtime.hpp"
#include "../dcl/runtime.cpp"
#include "../dcl/topo_metrics_io.hpp"
#include <cassert>
#include <iostream>

int main(int argc, char** argv) {
    dcl::Runtime runtime = dcl::Runtime::create(argc, argv);
    if (argc != 2) {
        if (runtime.rank() == 0) std::cerr << "Usage: test_probe_runtime_integration probe.json\n";
        MPI_Finalize();
        return 1;
    }
    const auto metrics = dcl::load_topo_metrics(argv[1]);
    runtime.set_simulated_devices_count(static_cast<int>(metrics.pcie_latency_ns.size()));
    runtime.set_topo_metrics(metrics);
    assert(runtime.topo_metrics().has_value());
    assert(metrics.mpi_latency_ns.size() ==
           static_cast<std::size_t>(runtime.size() * runtime.size()));
    if (runtime.rank() == 0) {
        std::cout << "Probe JSON loaded with " << runtime.size()
                  << " ranks and " << metrics.pcie_latency_ns.size()
                  << " devices; synthetic=" << metrics.synthetic << "\n";
    }
    MPI_Finalize();
    return 0;
}
