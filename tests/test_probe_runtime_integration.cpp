#include "harmony/runtime.hpp"
#include "harmony/topo_metrics_io.hpp"

#include <mpi.h>

#include <cassert>
#include <iostream>

int main(int argc, char** argv) {
    harmony::Runtime runtime = harmony::Runtime::create(argc, argv);
    if (argc != 2) {
        if (runtime.rank() == 0) {
            std::cerr << "Usage: test_probe_runtime_integration probe.json\n";
        }
        return 1;
    }

    const harmony::TopoMetrics metrics = harmony::load_topo_metrics(argv[1]);
    runtime.set_simulated_devices_count(static_cast<int>(metrics.pcie_latency_ns.size()));
    runtime.set_topo_metrics(metrics);

    assert(runtime.topo_metrics().has_value());
    assert(metrics.mpi_latency_ns.size() ==
           static_cast<std::size_t>(runtime.size()) * static_cast<std::size_t>(runtime.size()));
    assert(metrics.mpi_bandwidth_gbps.size() == metrics.mpi_latency_ns.size());
    assert(runtime.topo_metrics()->synthetic == metrics.synthetic);

    if (runtime.rank() == 0) {
        std::cout << "Probe JSON loaded with " << runtime.size()
                  << " ranks and " << metrics.pcie_latency_ns.size()
                  << " devices; synthetic=" << metrics.synthetic << "\n";
    }
    return 0;
}
