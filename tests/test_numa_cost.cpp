#ifndef OMPI_SKIP_MPICXX
#define OMPI_SKIP_MPICXX 1
#endif
#ifndef MPICH_SKIP_MPICXX
#define MPICH_SKIP_MPICXX 1
#endif

#include <mpi.h>
#if __has_include("dcl/runtime.hpp")
#include "dcl/runtime.hpp"
#include "dcl/runtime.cpp"
#include "dcl/topo_metrics_io.hpp"
#else
#include "../dcl/runtime.hpp"
#include "../dcl/runtime.cpp"
#include "../dcl/topo_metrics_io.hpp"
#endif

#include <cassert>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

void test_numa_distance_matrix_and_helper() {
    std::cout << "[TEST] Running test_numa_distance_matrix_and_helper..." << std::endl;

    dcl::TopoMetrics m;
    // 2-node NUMA distance matrix: 2x2
    m.numa_distance = {10, 20, 20, 10};

    // Verify distance between all pairs
    assert(dcl::numa_distance_between(m, 0, 0) == 10);
    assert(dcl::numa_distance_between(m, 0, 1) == 20);
    assert(dcl::numa_distance_between(m, 1, 0) == 20);
    assert(dcl::numa_distance_between(m, 1, 1) == 10);

    // Verify symmetry property
    assert(dcl::numa_distance_between(m, 0, 1) == dcl::numa_distance_between(m, 1, 0));

    // Verify diagonal is minimum distance
    assert(dcl::numa_distance_between(m, 0, 0) <= dcl::numa_distance_between(m, 0, 1));
    assert(dcl::numa_distance_between(m, 1, 1) <= dcl::numa_distance_between(m, 1, 0));

    // Verify out-of-bounds node index throws dcl::Error
    bool caught = false;
    try {
        (void)dcl::numa_distance_between(m, -1, 0);
    } catch (const dcl::Error&) {
        caught = true;
    }
    assert(caught && "Expected dcl::Error for negative node_a");

    caught = false;
    try {
        (void)dcl::numa_distance_between(m, 0, 2);
    } catch (const dcl::Error&) {
        caught = true;
    }
    assert(caught && "Expected dcl::Error for node_b >= num_nodes");

    // Verify empty numa_distance throws dcl::Error
    dcl::TopoMetrics empty_m;
    caught = false;
    try {
        (void)dcl::numa_distance_between(empty_m, 0, 0);
    } catch (const dcl::Error&) {
        caught = true;
    }
    assert(caught && "Expected dcl::Error for empty numa_distance");

    std::cout << "  -> PASS" << std::endl;
}

void test_cross_vs_same_numa_migration_cost() {
    std::cout << "[TEST] Running test_cross_vs_same_numa_migration_cost..." << std::endl;

    dcl::TopoMetrics m;
    // 2-node NUMA system, 2 global devices on different nodes
    m.numa_distance = {10, 25, 25, 10}; // cross-NUMA distance is 25
    m.device_numa_node = {0, 1};        // dev 0 on node 0, dev 1 on node 1
    m.pcie_latency_ns = {4000.0, 4000.0};
    m.pcie_bandwidth_gbps = {12.0, 12.0};

    const double penalty_ns = 50.0;

    // Test 1: Zero bytes transfer
    {
        const double same_cost = dcl::estimate_migration_cost_bytes(m, 0, 0, -1, -1, 0, penalty_ns);
        const double cross_cost = dcl::estimate_migration_cost_bytes(m, 0, 1, -1, -1, 0, penalty_ns);
        assert(cross_cost > same_cost);
        const double expected_diff = (25.0 * penalty_ns) * 1e-9;
        assert(std::fabs((cross_cost - same_cost) - expected_diff) < 1e-12);
    }

    // Test 2: 1 MiB transfer (1048576 bytes)
    {
        const std::size_t bytes = 1048576;
        const double same_cost = dcl::estimate_migration_cost_bytes(m, 0, 0, -1, -1, bytes, penalty_ns);
        const double cross_cost = dcl::estimate_migration_cost_bytes(m, 0, 1, -1, -1, bytes, penalty_ns);

        assert(cross_cost > same_cost);

        // Difference must be exactly the NUMA distance penalty
        const double expected_diff = (25.0 * penalty_ns) * 1e-9;
        assert(std::fabs((cross_cost - same_cost) - expected_diff) < 1e-12);
    }

    // Test 3: 16 MiB transfer
    {
        const std::size_t bytes = 16 * 1024 * 1024;
        const double same_cost = dcl::estimate_migration_cost_bytes(m, 0, 0, -1, -1, bytes, penalty_ns);
        const double cross_cost = dcl::estimate_migration_cost_bytes(m, 0, 1, -1, -1, bytes, penalty_ns);
        assert(cross_cost > same_cost);
    }

    // Test 4: Reverse direction (device 1 to device 0 vs device 1 to device 1)
    {
        const std::size_t bytes = 2 * 1024 * 1024;
        const double same_cost = dcl::estimate_migration_cost_bytes(m, 1, 1, -1, -1, bytes, penalty_ns);
        const double cross_cost = dcl::estimate_migration_cost_bytes(m, 1, 0, -1, -1, bytes, penalty_ns);
        assert(cross_cost > same_cost);
    }

    // Test 5: Negative device index throws dcl::Error
    {
        bool caught = false;
        try {
            (void)dcl::estimate_migration_cost_bytes(m, -1, 0, -1, -1, 1024);
        } catch (const dcl::Error&) {
            caught = true;
        }
        assert(caught && "Expected dcl::Error for negative device index");
    }

    std::cout << "  -> PASS" << std::endl;
}

void test_mpi_network_cost_calculation() {
    std::cout << "[TEST] Running test_mpi_network_cost_calculation..." << std::endl;

    dcl::TopoMetrics m;
    m.pcie_latency_ns = {4000.0, 4000.0};
    m.pcie_bandwidth_gbps = {12.0, 12.0};
    m.device_numa_node = {0, 0};
    m.numa_distance = {10, 10, 10, 10};

    // 2x2 MPI matrix (2 ranks)
    // rank 0 to rank 1 has astronomical latency
    m.mpi_latency_ns = {0.0, 1e9, 1e9, 0.0}; // 1 second latency!
    m.mpi_bandwidth_gbps = {10.0, 10.0, 10.0, 10.0};

    const double penalty_ns = 50.0;
    const std::size_t bytes = 1048576; // 1 MiB

    // Intra-rank cost
    const double intra_cost = dcl::estimate_migration_cost_bytes(m, 0, 1, 0, 0, bytes, penalty_ns);

    // Inter-rank cost (rank 0 to rank 1)
    const double inter_cost = dcl::estimate_migration_cost_bytes(m, 0, 1, 0, 1, bytes, penalty_ns);

    assert(inter_cost > intra_cost);
    // Should be at least 1 second due to MPI latency
    assert(inter_cost >= 1.0);

    std::cout << "  -> PASS" << std::endl;
}

void test_invalid_device_numa_node_throws(dcl::Runtime& rt) {
    std::cout << "[TEST] Running test_invalid_device_numa_node_throws..." << std::endl;
    rt.set_simulated_devices_count(2);

    dcl::TopoMetrics base;
    base.pcie_latency_ns = {4000.0, 4000.0};
    base.pcie_bandwidth_gbps = {15.0, 15.0};
    base.mpi_latency_ns = (rt.size() == 1 ? std::vector<double>{0.0} : std::vector<double>{0.0, 1000.0, 1000.0, 0.0});
    base.mpi_bandwidth_gbps = (rt.size() == 1 ? std::vector<double>{0.0} : std::vector<double>{0.0, 3.0, 3.0, 0.0});
    base.numa_distance = {10, 20, 20, 10}; // 2-node system: valid nodes are 0, 1

    // Case 1: device_numa_node contains node index >= num_nodes
    {
        dcl::TopoMetrics bad = base;
        bad.device_numa_node = {0, 2}; // 2 >= 2
        bool caught = false;
        try {
            rt.set_topo_metrics(bad);
        } catch (const dcl::Error&) {
            caught = true;
        }
        assert(caught && "Expected dcl::Error for node index >= num_nodes");
    }

    // Case 2: device_numa_node contains negative index
    {
        dcl::TopoMetrics bad = base;
        bad.device_numa_node = {-1, 0};
        bool caught = false;
        try {
            rt.set_topo_metrics(bad);
        } catch (const dcl::Error&) {
            caught = true;
        }
        assert(caught && "Expected dcl::Error for negative node index");
    }

    // Case 3: device_numa_node provided without numa_distance matrix
    {
        dcl::TopoMetrics bad = base;
        bad.numa_distance.clear();
        bad.device_numa_node = {0, 1};
        bool caught = false;
        try {
            rt.set_topo_metrics(bad);
        } catch (const dcl::Error&) {
            caught = true;
        }
        assert(caught && "Expected dcl::Error for device_numa_node without numa_distance");
    }

    // Case 4: Non-square numa_distance matrix (3 elements instead of 4)
    {
        dcl::TopoMetrics bad = base;
        bad.numa_distance = {10, 20, 20};
        bad.device_numa_node = {0, 1};
        bool caught = false;
        try {
            rt.set_topo_metrics(bad);
        } catch (const dcl::Error&) {
            caught = true;
        }
        assert(caught && "Expected dcl::Error for non-square numa_distance");
    }

    // Case 5: Non-positive value in numa_distance matrix
    {
        dcl::TopoMetrics bad = base;
        bad.numa_distance = {10, 0, 0, 10};
        bad.device_numa_node = {0, 1};
        bool caught = false;
        try {
            rt.set_topo_metrics(bad);
        } catch (const dcl::Error&) {
            caught = true;
        }
        assert(caught && "Expected dcl::Error for non-positive numa_distance value");
    }

    // Case 6: Diagonal not minimum in numa_distance matrix
    {
        dcl::TopoMetrics bad = base;
        bad.numa_distance = {30, 10, 10, 30}; // diagonal (30) > off-diagonal (10)
        bad.device_numa_node = {0, 1};
        bool caught = false;
        try {
            rt.set_topo_metrics(bad);
        } catch (const dcl::Error&) {
            caught = true;
        }
        assert(caught && "Expected dcl::Error for non-minimal diagonal in numa_distance");
    }

    std::cout << "  -> PASS" << std::endl;
}

void test_numa_skip_rebalance_when_cost_high(dcl::Runtime& rt) {
    std::cout << "[TEST] Running test_numa_skip_rebalance_when_cost_high..." << std::endl;

    // 1. Configure simulated 2-device runtime
    rt.set_simulated_devices_count(2);

    dcl::PartitionSpec ps;
    ps.global_elements = 1000000;
    ps.units_per_element = 1;
    ps.bytes_per_unit = 4; // 4 MB total data
    ps.granularity = 1;
    rt.set_partition(ps);

    // Initial partitions: [0, 500000) on dev 0, [500000, 1000000) on dev 1
    assert(rt.partitions().size() == 2);

    // Create a registered proportional field
    dcl::FieldSpec fs;
    fs.name = "test_field";
    fs.global_elements = 1000000;
    fs.units_per_element = 1;
    fs.bytes_per_unit = 4;
    fs.usage = dcl::BufferUsage::read_write;
    fs.redistribution = dcl::RedistributionDependency::proportional;
    dcl::FieldHandle fh = rt.create_field(fs);

    // 2. Configure TopoMetrics with very low PCIe bandwidth across NUMA nodes
    //    so migration cost is high compared to gain.
    dcl::TopoMetrics m;
    m.pcie_latency_ns = {5000.0, 5000.0};
    m.pcie_bandwidth_gbps = {0.001, 0.001}; // 1 MB/s -> 400 KB takes 0.4 seconds!
    m.mpi_latency_ns = (rt.size() == 1 ? std::vector<double>{0.0} : std::vector<double>{0.0, 1000.0, 1000.0, 0.0});
    m.mpi_bandwidth_gbps = (rt.size() == 1 ? std::vector<double>{0.0} : std::vector<double>{0.0, 1.0, 1.0, 0.0});
    m.numa_distance = {10, 100, 100, 10}; // 2 nodes, large distance
    m.device_numa_node = {0, 1};          // dev 0 on node 0, dev 1 on node 1
    rt.set_topo_metrics(m);

    // 3. Inject simulated times with a small imbalance:
    //    Device 0: 0.0010s, Device 1: 0.0012s
    //    Expected gain is tiny (~0.0001s), while migration cost is ~0.4s!
    rt.set_simulated_times({0.0010, 0.0012});

    // Capture stdout to verify the exact log message appears
    std::stringstream captured;
    std::streambuf* old_buf = std::cout.rdbuf(captured.rdbuf());

    // Call maybe_rebalance_from_timings with threshold = 0.01 and numa_ratio = 0.50
    const bool rebalanced = rt.maybe_rebalance_from_timings({fh}, 0.01f, 0.50);

    // Restore stdout
    std::cout.rdbuf(old_buf);
    const std::string out_str = captured.str();

    // Echo to stdout so test logs and auditor can see it
    std::cout << out_str;

    // Verify rebalance was skipped
    assert(!rebalanced && "Expected rebalance to be skipped due to NUMA migration cost");

    int current_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &current_rank);
    if (current_rank == 0) {
        // Verify exact log line was emitted
        const std::string expected_log = "[NUMA] migration cost exceeds gain threshold, skipping rebalance";
        assert(out_str.find(expected_log) != std::string::npos &&
               "Expected log line '[NUMA] migration cost exceeds gain threshold, skipping rebalance' not found");
    }

    // 4. Now test the opposite case: fast PCIe bandwidth, migration cost is tiny compared to gain
    m.pcie_bandwidth_gbps = {100.0, 100.0}; // 100 GB/s -> transfer takes only ~4 microseconds
    m.numa_distance = {10, 12, 12, 10};
    rt.set_topo_metrics(m);

    // Large timing imbalance: Dev 0 takes 0.001s, Dev 1 takes 0.010s
    // Expected gain is ~0.004s >> migration cost ~4 microseconds
    rt.set_simulated_times({0.001, 0.010});

    const bool rebalanced_accepted = rt.maybe_rebalance_from_timings({fh}, 0.01f, 0.50);
    assert(rebalanced_accepted && "Expected rebalance to be accepted when migration cost is low");

    // Cleanup simulated times
    rt.clear_simulated_times();

    std::cout << "  -> PASS" << std::endl;
}

void test_mpi_skip_rebalance_when_cost_high(dcl::Runtime& rt) {
    if (rt.size() < 2) return;
    std::cout << "[TEST] Running test_mpi_skip_rebalance_when_cost_high..." << std::endl;

    // Trick the runtime into thinking there are 2 ranks for the purpose of partitioning
    rt.set_simulated_ranks_for_testing(2);
    rt.set_simulated_devices_count(2); // 2 devices across 2 ranks: {1, 1}

    dcl::PartitionSpec ps;
    ps.global_elements = 1000000;
    ps.units_per_element = 1;
    ps.bytes_per_unit = 4;
    ps.granularity = 1;
    rt.set_partition(ps);

    dcl::FieldSpec fs;
    fs.name = "test_field_mpi";
    fs.global_elements = 1000000;
    fs.units_per_element = 1;
    fs.bytes_per_unit = 4;
    fs.usage = dcl::BufferUsage::read_write;
    fs.redistribution = dcl::RedistributionDependency::proportional;
    dcl::FieldHandle fh = rt.create_field(fs);

    // Fast PCIe, no NUMA penalty, but astronomical MPI latency
    dcl::TopoMetrics m;
    m.pcie_latency_ns = {1.0, 1.0};
    m.pcie_bandwidth_gbps = {1000.0, 1000.0};
    m.mpi_latency_ns = (rt.size() == 1 ? std::vector<double>{0.0} : std::vector<double>{0.0, 10e9, 10e9, 0.0}); // 10 seconds latency
    m.mpi_bandwidth_gbps = (rt.size() == 1 ? std::vector<double>{0.0} : std::vector<double>{10.0, 10.0, 10.0, 10.0});
    m.numa_distance = {10, 10, 10, 10}; 
    m.device_numa_node = {0, 0};          
    rt.set_topo_metrics(m);

    // Imbalance creates small gain (~0.009s) << 10s cost
    rt.set_simulated_times({0.0010, 0.010});

    std::stringstream captured;
    std::streambuf* old_buf = std::cout.rdbuf(captured.rdbuf());

    const bool rebalanced = rt.maybe_rebalance_from_timings({fh}, 0.01f, 0.50);

    std::cout.rdbuf(old_buf);
    const std::string out_str = captured.str();
    std::cout << out_str;

    assert(!rebalanced && "Expected rebalance to be skipped due to MPI network migration cost");

    int current_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &current_rank);
    if (current_rank == 0) {
        const std::string expected_log = "[NUMA] migration cost exceeds gain threshold, skipping rebalance";
        assert(out_str.find(expected_log) != std::string::npos &&
               "Expected log line not found");
    }

    // Restore runtime back to 1 rank
    rt.set_simulated_ranks_for_testing(1);
    rt.set_simulated_devices_count(2);
    rt.clear_simulated_times();

    std::cout << "  -> PASS" << std::endl;
}

void test_divergent_numa_consensus(dcl::Runtime& rt) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0) std::cout << "[TEST] Skipping test_divergent_numa_consensus, needs at least 2 ranks." << std::endl;
        return;
    }

    if (rank == 0) std::cout << "[TEST] Running test_divergent_numa_consensus..." << std::endl;

    rt.set_simulated_devices_count(2);
    
    dcl::PartitionSpec ps;
    ps.global_elements = 1000000;
    ps.units_per_element = 1;
    ps.bytes_per_unit = 4;
    ps.granularity = 1;
    rt.set_partition(ps);

    dcl::FieldSpec fs;
    fs.name = "consensus_field";
    fs.global_elements = 1000000;
    fs.units_per_element = 1;
    fs.bytes_per_unit = 4;
    fs.usage = dcl::BufferUsage::read_write;
    fs.redistribution = dcl::RedistributionDependency::proportional;
    dcl::FieldHandle fh = rt.create_field(fs);

    dcl::TopoMetrics m;
    m.pcie_latency_ns = {5000.0, 5000.0};
    m.mpi_latency_ns = (rt.size() == 1 ? std::vector<double>{0.0} : std::vector<double>{0.0, 1000.0, 1000.0, 0.0});
    m.mpi_bandwidth_gbps = (rt.size() == 1 ? std::vector<double>{0.0} : std::vector<double>{0.0, 1.0, 1.0, 0.0});
    m.numa_distance = {10, 20, 20, 10};
    m.device_numa_node = {0, 0};

    // Divergent configuration
    if (rank == 0) {
        m.pcie_bandwidth_gbps = {1000.0, 1000.0}; // Very fast, would accept rebalance
    } else {
        m.pcie_bandwidth_gbps = {0.001, 0.001};  // Very slow, would reject rebalance
    }

    rt.set_topo_metrics(m);

    // Timings that suggest rebalance is beneficial
    rt.set_simulated_times({0.0010, 0.010});

    // The logic inside maybe_rebalance_from_timings will trigger global consensus.
    // If one rank rejects, all ranks must reject to avoid deadlocks in subsequent collectives.
    const bool rebalanced = rt.maybe_rebalance_from_timings({fh}, 0.01f, 0.50);

    // Because rank 1 rejects, consensus means both should reject
    assert(!rebalanced && "Expected divergent consensus to result in skipping rebalance globally");

    rt.clear_simulated_times();

    if (rank == 0) std::cout << "  -> PASS" << std::endl;
}

void test_divergent_contention_proposal(dcl::Runtime& rt) {
    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size < 2) return;

    rt.set_simulated_devices_count(2);
    dcl::PartitionSpec ps;
    ps.global_elements = 1000000;
    ps.units_per_element = 1;
    ps.bytes_per_unit = sizeof(float);
    ps.granularity = 1;
    rt.set_partition(ps);

    dcl::FieldSpec fs;
    fs.name = "divergent_contention_field";
    fs.global_elements = ps.global_elements;
    fs.units_per_element = 1;
    fs.bytes_per_unit = sizeof(float);
    fs.redistribution = dcl::RedistributionDependency::proportional;
    const dcl::FieldHandle field = rt.create_field(fs);

    dcl::TopoMetrics metrics;
    metrics.pcie_latency_ns = {4000.0, 4000.0};
    metrics.pcie_bandwidth_gbps = {15.0, 15.0};
    metrics.mpi_latency_ns = (rt.size() == 1 ? std::vector<double>{0.0} : std::vector<double>{0.0, 1000.0, 1000.0, 0.0});
    metrics.mpi_bandwidth_gbps = (rt.size() == 1 ? std::vector<double>{0.0} : std::vector<double>{0.0, 3.0, 3.0, 0.0});
    metrics.memory_contention_factor = rank == 0
        ? std::vector<double>{1.0, 1.0}
        : std::vector<double>{1.0, 100.0};
    rt.set_topo_metrics(metrics);
    rt.set_simulated_times({1.0, 1.0});

    dcl::AutoBalancePolicy policy;
    policy.mode = dcl::BalanceMode::dynamic_threshold;
    policy.threshold = 0.01f;
    policy.use_contention_adjustment = true;
    const bool changed = rt.maybe_rebalance_from_timings({field}, policy);
    assert(!changed);
    assert(rt.partitions()[0].element_count == 500000);
    assert(rt.partitions()[1].element_count == 500000);
    rt.clear_simulated_times();

    if (rank == 0) std::cout << "[TEST] Divergent contention proposal reached a common decision: PASS\n";
}

void test_divergent_balance_inputs_fail_collectively(dcl::Runtime& rt) {
    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size < 2) return;

    rt.set_simulated_devices_count(2);
    dcl::PartitionSpec ps;
    ps.global_elements = 1000;
    ps.units_per_element = 1;
    ps.bytes_per_unit = sizeof(float);
    ps.granularity = 1;
    rt.set_partition(ps);

    dcl::FieldSpec fs;
    fs.name = "collective_input_field";
    fs.global_elements = ps.global_elements;
    fs.units_per_element = 1;
    fs.bytes_per_unit = sizeof(float);
    fs.redistribution = dcl::RedistributionDependency::proportional;
    const dcl::FieldHandle field = rt.create_field(fs);

    dcl::AutoBalancePolicy policy;
    policy.mode = dcl::BalanceMode::dynamic_threshold;
    policy.threshold = 0.01f;

    if (rank == 0) rt.set_simulated_times({1.0, 1.0});
    bool caught = false;
    try {
        rt.maybe_rebalance_from_timings({field}, policy);
    } catch (const dcl::Error&) {
        caught = true;
    }
    assert(caught && "Mismatched timing sources must fail on every rank");
    rt.clear_simulated_times();

    caught = false;
    try {
        rt.maybe_rebalance_from_timings(rank == 0 ? std::vector<dcl::FieldHandle>{field}
                                                  : std::vector<dcl::FieldHandle>{}, policy);
    } catch (const dcl::Error&) {
        caught = true;
    }
    assert(caught && "Mismatched field lists must fail on every rank");

    dcl::FieldSpec other_spec = fs;
    other_spec.name = "other_collective_input_field";
    const dcl::FieldHandle other_field = rt.create_field(other_spec);
    caught = false;
    try {
        rt.maybe_rebalance_from_timings(
            rank == 0 ? std::vector<dcl::FieldHandle>{field}
                      : std::vector<dcl::FieldHandle>{other_field},
            policy
        );
    } catch (const dcl::Error&) {
        caught = true;
    }
    assert(caught && "Different fields with the same count must fail on every rank");
    if (rank == 0) std::cout << "[TEST] Divergent balance inputs failed collectively: PASS\n";
}

} // namespace

int main(int argc, char** argv) {
    dcl::Runtime rt = dcl::Runtime::create(argc, argv);

    test_numa_distance_matrix_and_helper();
    test_cross_vs_same_numa_migration_cost();
    test_mpi_network_cost_calculation();
    test_invalid_device_numa_node_throws(rt);
    test_numa_skip_rebalance_when_cost_high(rt);
    test_mpi_skip_rebalance_when_cost_high(rt);
    test_divergent_numa_consensus(rt);
    test_divergent_contention_proposal(rt);
    test_divergent_balance_inputs_fail_collectively(rt);

    std::cout << "\nAll test_numa_cost unit tests PASSED successfully!" << std::endl;
    MPI_Finalize();
    return 0;
}
