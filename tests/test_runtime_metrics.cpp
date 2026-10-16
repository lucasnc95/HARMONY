#ifndef OMPI_SKIP_MPICXX
#define OMPI_SKIP_MPICXX 1
#endif
#include <mpi.h>
#include "../dcl/runtime.hpp"
#include "../dcl/runtime.cpp"

#include <cassert>
#include <chrono>
#include <iostream>
#include <sstream>
#include <vector>

struct Observation {
    std::vector<std::size_t> counts;
    std::string output;
    double elapsed_seconds{0.0};
};

Observation run_case(int& argc, char**& argv, bool quiet, int iterations) {
    auto runtime = dcl::Runtime::create(argc, argv);
    runtime.set_simulated_devices_count(2);
    if (quiet) runtime.set_verbose_metrics(false);
    dcl::PartitionSpec partition;
    partition.global_elements = 100;
    partition.units_per_element = 1;
    partition.bytes_per_unit = sizeof(float);
    partition.granularity = 1;
    runtime.set_partition(partition);
    std::vector<float> data(100, 1.0f);
    dcl::FieldSpec field{"input", 100, 1, sizeof(float), dcl::BufferUsage::read_only,
                         data.data(), dcl::RedistributionDependency::proportional};
    const auto handle = runtime.create_field(field);
    std::ostringstream captured;
    auto* old = std::cout.rdbuf(captured.rdbuf());
    Observation result;
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
        runtime.set_simulated_times((i % 2 == 0)
            ? std::vector<double>{0.020, 0.010}
            : std::vector<double>{0.010, 0.020});
        runtime.maybe_rebalance_from_timings({handle}, 0.0f);
        for (const auto& part : runtime.partitions()) result.counts.push_back(part.element_count);
    }
    result.elapsed_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    std::cout.rdbuf(old);
    result.output = captured.str();
    return result;
}

int main(int argc, char** argv) {
    const bool measure = argc > 1 && std::string(argv[1]) == "--measure";
    const int iterations = measure ? 1000 : 2;
    const auto verbose = run_case(argc, argv, false, iterations);
    const auto quiet = run_case(argc, argv, true, iterations);
    assert(verbose.counts == quiet.counts);
    assert(verbose.counts.size() == static_cast<std::size_t>(iterations) * 2);
    for (std::size_t i = 0; i < verbose.counts.size(); i += 2) {
        assert(verbose.counts[i] + verbose.counts[i + 1] == 100);
    }
    if (quiet.output.find("[DCL][balance-metrics]") != std::string::npos ||
        quiet.output.find("OLD loads") != std::string::npos ||
        quiet.output.find("DCL loads by partition") != std::string::npos) return 1;
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0 && verbose.output.find("[DCL][balance-metrics]") == std::string::npos) return 1;
    if (measure && rank == 0) {
        std::cout << "simulated_rebalance_iterations=" << iterations
                  << " verbose_s=" << verbose.elapsed_seconds
                  << " quiet_s=" << quiet.elapsed_seconds << "\n";
    }
    std::cout << "Runtime metric modes: PASS\n";
    MPI_Finalize();
}
