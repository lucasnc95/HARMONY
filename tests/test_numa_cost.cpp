#include "harmony/algorithms.hpp"
#include "harmony/runtime.hpp"

#include <mpi.h>

#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace {

void test_numa_distance_matrix_and_helper() {
    harmony::TopoMetrics m;
    m.numa_distance = {10, 20, 20, 10};

    assert(harmony::numa_distance_between(m, 0, 0) == 10);
    assert(harmony::numa_distance_between(m, 0, 1) == 20);
    assert(harmony::numa_distance_between(m, 1, 0) == 20);
    assert(harmony::numa_distance_between(m, 1, 1) == 10);

    // Asymmetric matrix returns symmetric max and clamps against diagonal minimum.
    harmony::TopoMetrics asym;
    asym.numa_distance = {10, 15, 25, 12};
    assert(harmony::numa_distance_between(asym, 0, 1) == 25);
    assert(harmony::numa_distance_between(asym, 1, 0) == 25);

    auto expect_error = [](auto fn) {
        bool caught = false;
        try {
            (void)fn();
        } catch (const harmony::Error&) {
            caught = true;
        }
        assert(caught);
    };

    expect_error([&]() { return harmony::numa_distance_between(m, -1, 0); });
    expect_error([&]() { return harmony::numa_distance_between(m, 0, 2); });

    harmony::TopoMetrics empty_m;
    expect_error([&]() { return harmony::numa_distance_between(empty_m, 0, 0); });

    harmony::TopoMetrics non_square;
    non_square.numa_distance = {10, 20, 20};
    expect_error([&]() { return harmony::numa_distance_between(non_square, 0, 0); });
}

void test_cross_vs_same_numa_migration_cost() {
    harmony::TopoMetrics m;
    m.numa_distance = {10, 25, 25, 10};
    m.device_numa_node = {0, 1};
    m.pcie_latency_ns = {4000.0, 4000.0};
    m.pcie_bandwidth_gbps = {12.0, 12.0};

    const double penalty_ns = 50.0;
    const double expected_diff = (25.0 * penalty_ns) * 1e-9;

    // Zero bytes transfer: only latency + NUMA hop penalty differs.
    {
        const double same_cost = harmony::estimate_migration_cost_bytes(m, 0, 0, -1, -1, 0, penalty_ns);
        const double cross_cost = harmony::estimate_migration_cost_bytes(m, 0, 1, -1, -1, 0, penalty_ns);
        assert(cross_cost > same_cost);
        assert(std::fabs((cross_cost - same_cost) - expected_diff) < 1e-12);
    }

    // 1 MiB transfer (both 7-arg and 5-arg overload).
    {
        const std::size_t bytes = 1048576;
        const double same_cost = harmony::estimate_migration_cost_bytes(m, 0, 0, bytes, penalty_ns);
        const double cross_cost = harmony::estimate_migration_cost_bytes(m, 0, 1, bytes, penalty_ns);
        assert(cross_cost > same_cost);
        assert(std::fabs((cross_cost - same_cost) - expected_diff) < 1e-12);
    }

    // Reverse direction.
    {
        const std::size_t bytes = 2 * 1024 * 1024;
        const double same_cost = harmony::estimate_migration_cost_bytes(m, 1, 1, -1, -1, bytes, penalty_ns);
        const double cross_cost = harmony::estimate_migration_cost_bytes(m, 1, 0, -1, -1, bytes, penalty_ns);
        assert(cross_cost > same_cost);
    }

    // Negative device index throws harmony::Error.
    bool caught = false;
    try {
        (void)harmony::estimate_migration_cost_bytes(m, -1, 0, -1, -1, 1024);
    } catch (const harmony::Error&) {
        caught = true;
    }
    assert(caught);
}

void test_mpi_network_cost_calculation() {
    harmony::TopoMetrics m;
    m.pcie_latency_ns = {4000.0, 4000.0};
    m.pcie_bandwidth_gbps = {12.0, 12.0};
    m.device_numa_node = {0, 0};
    m.numa_distance = {10, 10, 10, 10};
    m.mpi_latency_ns = {0.0, 1e9, 1e9, 0.0};
    m.mpi_bandwidth_gbps = {10.0, 10.0, 10.0, 10.0};

    const double penalty_ns = 50.0;
    const std::size_t bytes = 1048576;

    const double intra_cost = harmony::estimate_migration_cost_bytes(m, 0, 1, 0, 0, bytes, penalty_ns);
    const double inter_cost = harmony::estimate_migration_cost_bytes(m, 0, 1, 0, 1, bytes, penalty_ns);

    assert(inter_cost > intra_cost);
    assert(inter_cost >= 1.0);

    // Verify full route cost: PCIe(src) + MPI(src,dst) + PCIe(dst).
    const double expected_inter =
        (4000.0 + 4000.0 + 1e9) * 1e-9 +
        static_cast<double>(bytes) / (12.0 * 1e9) +
        static_cast<double>(bytes) / (12.0 * 1e9) +
        static_cast<double>(bytes) / (10.0 * 1e9);
    assert(std::fabs(inter_cost - expected_inter) < 1e-12);
}

harmony::TopoMetrics make_base_metrics(int num_ranks) {
    const std::size_t nr = static_cast<std::size_t>(std::max(1, num_ranks));
    harmony::TopoMetrics m;
    m.pcie_latency_ns = {4000.0, 4000.0};
    m.pcie_bandwidth_gbps = {15.0, 15.0};
    m.mpi_latency_ns.assign(nr * nr, 0.0);
    m.mpi_bandwidth_gbps.assign(nr * nr, 0.0);
    for (std::size_t s = 0; s < nr; ++s) {
        for (std::size_t d = 0; d < nr; ++d) {
            if (s != d) {
                m.mpi_latency_ns[s * nr + d] = 1000.0;
                m.mpi_bandwidth_gbps[s * nr + d] = 3.0;
            }
        }
    }
    m.numa_distance = {10, 20, 20, 10};
    return m;
}

void test_invalid_device_numa_node_throws(int argc, char** argv) {
    auto rt = harmony::Runtime::create(argc, argv);
    rt.set_simulated_devices_count(2);
    const harmony::TopoMetrics base = make_base_metrics(rt.size());

    auto expect_set_error = [&](const harmony::TopoMetrics& bad) {
        bool caught = false;
        try {
            rt.set_topo_metrics(bad);
        } catch (const harmony::Error&) {
            caught = true;
        }
        assert(caught);
    };

    {
        auto bad = base;
        bad.device_numa_node = {0, 2};
        expect_set_error(bad);
    }
    {
        auto bad = base;
        bad.device_numa_node = {-1, 0};
        expect_set_error(bad);
    }
    {
        auto bad = base;
        bad.numa_distance.clear();
        bad.device_numa_node = {0, 1};
        expect_set_error(bad);
    }
    {
        auto bad = base;
        bad.numa_distance = {10, 20, 20};
        bad.device_numa_node = {0, 1};
        expect_set_error(bad);
    }
    {
        auto bad = base;
        bad.numa_distance = {10, 0, 0, 10};
        bad.device_numa_node = {0, 1};
        expect_set_error(bad);
    }
    {
        auto bad = base;
        bad.numa_distance = {30, 10, 10, 30};
        bad.device_numa_node = {0, 1};
        expect_set_error(bad);
    }
}

void test_numa_skip_rebalance_when_cost_high(int argc, char** argv) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    auto rt = harmony::Runtime::create(argc, argv);
    rt.set_simulated_devices_count(2);
    rt.set_partition({1000000, 1, sizeof(float), 1});
    auto fh = rt.create_field({"test_field", 1000000, 1, sizeof(float),
                               harmony::BufferUsage::read_write, nullptr,
                               harmony::RedistributionDependency::proportional});

    // Default threshold is 0.50; runtime setter/getter works too.
    assert(std::fabs(rt.numa_cost_gain_ratio_threshold() - 0.50) < 1e-9);
    rt.set_numa_cost_gain_ratio_threshold(0.40);
    assert(std::fabs(rt.numa_cost_gain_ratio_threshold() - 0.40) < 1e-9);

    harmony::TopoMetrics m = make_base_metrics(rt.size());
    m.pcie_latency_ns = {5000.0, 5000.0};
    m.pcie_bandwidth_gbps = {0.001, 0.001};  // 1 MB/s -> migration cost >> gain
    m.numa_distance = {10, 100, 100, 10};
    m.device_numa_node = {0, 1};
    rt.set_topo_metrics(m);

    harmony::AutoBalancePolicy policy;
    policy.mode = harmony::BalanceMode::dynamic_threshold;
    policy.interval = 1;
    policy.threshold = 0.01f;
    policy.numa_cost_gain_ratio_threshold = 0.50;

    auto step = rt.step("numa_gated")
        .with_balance(policy)
        .tag_field(fh, harmony::StepFieldRole::rebalance_source)
        .build();

    rt.start_timing();
    rt.set_simulated_times({0.0010, 0.0012});

    std::stringstream captured;
    std::streambuf* old_buf = std::cout.rdbuf(captured.rdbuf());
    rt.execute(step);
    std::cout.rdbuf(old_buf);
    const std::string out_str = captured.str();

    assert(rt.partitions()[0].element_count == 500000);
    assert(rt.partitions()[1].element_count == 500000);
    if (rank == 0) {
        assert(out_str.find("[NUMA] migration cost exceeds gain threshold, skipping rebalance") != std::string::npos);
    }

    // Fast PCIe and low NUMA penalty -> rebalance accepted.
    m.pcie_bandwidth_gbps = {100.0, 100.0};
    m.numa_distance = {10, 12, 12, 10};
    rt.set_topo_metrics(m);
    rt.set_verbose(false);
    rt.set_simulated_times({0.001, 0.010});
    rt.execute(step);
    assert(rt.partitions()[0].element_count > 500000);
    assert(rt.partitions()[0].element_count + rt.partitions()[1].element_count == 1000000);

    (void)rt.stop_timing();
    const auto report = rt.timing_report();
    assert(report.balance.size() == 2);
    assert(!report.balance[0].applied);
    assert(report.balance[0].reason.find("[NUMA] migration cost exceeds gain threshold") != std::string::npos);
    assert(report.balance[1].applied);
    assert(report.ranks[static_cast<std::size_t>(rank)].balance_seconds >= 0.0);
    assert(report.ranks[static_cast<std::size_t>(rank)].mpi_balance_seconds >= 0.0);
}

void test_mpi_skip_rebalance_when_cost_high(int argc, char** argv) {
    int size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size < 2) return;

    auto rt = harmony::Runtime::create(argc, argv);
    rt.set_verbose(false);
    rt.set_simulated_devices_count(2);
    rt.set_partition({1000000, 1, sizeof(float), 1});
    auto fh = rt.create_field({"test_field_mpi", 1000000, 1, sizeof(float),
                               harmony::BufferUsage::read_write, nullptr,
                               harmony::RedistributionDependency::proportional});

    harmony::TopoMetrics m = make_base_metrics(rt.size());
    m.pcie_latency_ns = {1.0, 1.0};
    m.pcie_bandwidth_gbps = {1000.0, 1000.0};
    const std::size_t nr = static_cast<std::size_t>(rt.size());
    m.mpi_latency_ns.assign(nr * nr, 0.0);
    m.mpi_bandwidth_gbps.assign(nr * nr, 0.0);
    for (std::size_t s = 0; s < nr; ++s) {
        for (std::size_t d = 0; d < nr; ++d) {
            if (s != d) {
                m.mpi_latency_ns[s * nr + d] = 10e9;  // 10 s latency
                m.mpi_bandwidth_gbps[s * nr + d] = 10.0;
            }
        }
    }
    m.numa_distance = {10, 10, 10, 10};
    m.device_numa_node = {0, 0};
    rt.set_topo_metrics(m);

    harmony::AutoBalancePolicy policy;
    policy.mode = harmony::BalanceMode::dynamic_threshold;
    policy.interval = 1;
    policy.threshold = 0.01f;
    policy.numa_cost_gain_ratio_threshold = 0.50;

    auto step = rt.step("mpi_gated")
        .with_balance(policy)
        .tag_field(fh, harmony::StepFieldRole::rebalance_source)
        .build();

    rt.set_simulated_times({0.0010, 0.010});
    rt.execute(step);

    assert(rt.partitions()[0].element_count == 500000);
    assert(rt.partitions()[1].element_count == 500000);
    const auto report = rt.timing_report();
    assert(report.balance.size() == 1);
    assert(!report.balance[0].applied);
    assert(report.balance[0].reason.find("[NUMA] migration cost exceeds gain threshold") != std::string::npos);
}

void test_divergent_numa_consensus(int argc, char** argv) {
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size < 2) return;

    auto rt = harmony::Runtime::create(argc, argv);
    rt.set_verbose(false);
    rt.set_simulated_devices_count(2);
    rt.set_partition({1000000, 1, sizeof(float), 1});
    auto fh = rt.create_field({"consensus_field", 1000000, 1, sizeof(float),
                               harmony::BufferUsage::read_write, nullptr,
                               harmony::RedistributionDependency::proportional});

    harmony::TopoMetrics m = make_base_metrics(rt.size());
    m.pcie_latency_ns = {5000.0, 5000.0};
    m.device_numa_node = {0, 0};
    if (rank == 0) {
        m.pcie_bandwidth_gbps = {1000.0, 1000.0};  // fast locally -> would accept
    } else {
        m.pcie_bandwidth_gbps = {0.001, 0.001};   // slow locally -> rejects
    }
    rt.set_topo_metrics(m);

    harmony::AutoBalancePolicy policy;
    policy.mode = harmony::BalanceMode::dynamic_threshold;
    policy.interval = 1;
    policy.threshold = 0.01f;
    policy.numa_cost_gain_ratio_threshold = 0.50;

    auto step = rt.step("divergent_numa")
        .with_balance(policy)
        .tag_field(fh, harmony::StepFieldRole::rebalance_source)
        .build();

    rt.set_simulated_times({0.0010, 0.010});
    rt.execute(step);

    assert(rt.partitions()[0].element_count == 500000);
    assert(rt.partitions()[1].element_count == 500000);
    const auto report = rt.timing_report();
    assert(report.balance.size() == 1);
    assert(!report.balance[0].applied);
}

void test_divergent_contention_proposal(int argc, char** argv) {
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size < 2) return;

    auto rt = harmony::Runtime::create(argc, argv);
    rt.set_verbose(false);
    rt.set_simulated_devices_count(2);
    rt.set_partition({1000000, 1, sizeof(float), 1});
    auto fh = rt.create_field({"divergent_contention_field", 1000000, 1, sizeof(float),
                               harmony::BufferUsage::read_write, nullptr,
                               harmony::RedistributionDependency::proportional});

    harmony::TopoMetrics metrics = make_base_metrics(rt.size());
    metrics.memory_contention_factor = (rank == 0)
        ? std::vector<double>{1.0, 1.0}
        : std::vector<double>{1.0, 100.0};
    rt.set_topo_metrics(metrics);
    rt.set_simulated_times({1.0, 1.0});

    harmony::AutoBalancePolicy policy;
    policy.mode = harmony::BalanceMode::dynamic_threshold;
    policy.interval = 1;
    policy.threshold = 0.01f;
    policy.use_contention_adjustment = true;
    auto step = rt.step("divergent_contention")
        .with_balance(policy)
        .tag_field(fh, harmony::StepFieldRole::rebalance_source)
        .build();

    rt.execute(step);
    assert(rt.partitions()[0].element_count == 500000);
    assert(rt.partitions()[1].element_count == 500000);
}

void test_divergent_balance_inputs_fail_collectively(int argc, char** argv) {
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size < 2) return;

    auto rt = harmony::Runtime::create(argc, argv);
    rt.set_verbose(false);
    rt.set_simulated_devices_count(2);
    rt.set_partition({1000, 1, sizeof(float), 1});
    auto field = rt.create_field({"collective_input_field", 1000, 1, sizeof(float),
                                  harmony::BufferUsage::read_write, nullptr,
                                  harmony::RedistributionDependency::proportional});
    auto other_field = rt.create_field({"other_collective_input_field", 1000, 1, sizeof(float),
                                        harmony::BufferUsage::read_write, nullptr,
                                        harmony::RedistributionDependency::proportional});

    harmony::AutoBalancePolicy policy;
    policy.mode = harmony::BalanceMode::dynamic_threshold;
    policy.interval = 1;
    policy.threshold = 0.01f;

    // 1. Mismatched simulated_times_ across ranks.
    if (rank == 0) rt.set_simulated_times({1.0, 1.0});
    auto step = rt.step("mismatched_times")
        .with_balance(policy)
        .tag_field(field, harmony::StepFieldRole::rebalance_source)
        .build();
    bool caught = false;
    try {
        rt.execute(step);
    } catch (const harmony::Error&) {
        caught = true;
    }
    assert(caught && "Mismatched timing sources must fail on every rank");
    rt.clear_simulated_times();

    // 2. Mismatched field counts across ranks.
    auto sb = rt.step("mismatched_fields").with_balance(policy);
    if (rank == 0) sb.tag_field(field, harmony::StepFieldRole::rebalance_source);
    caught = false;
    try {
        rt.execute(sb.build());
    } catch (const harmony::Error&) {
        caught = true;
    }
    assert(caught && "Mismatched field lists must fail on every rank");

    // 3. Different field handles with same count across ranks.
    auto step_diff = rt.step("diff_fields")
        .with_balance(policy)
        .tag_field((rank == 0) ? field : other_field, harmony::StepFieldRole::rebalance_source)
        .build();
    caught = false;
    try {
        rt.execute(step_diff);
    } catch (const harmony::Error&) {
        caught = true;
    }
    assert(caught && "Different fields with the same count must fail on every rank");
}

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    test_numa_distance_matrix_and_helper();
    test_cross_vs_same_numa_migration_cost();
    test_mpi_network_cost_calculation();
    test_invalid_device_numa_node_throws(argc, argv);
    test_numa_skip_rebalance_when_cost_high(argc, argv);
    test_mpi_skip_rebalance_when_cost_high(argc, argv);
    test_divergent_numa_consensus(argc, argv);
    test_divergent_contention_proposal(argc, argv);
    test_divergent_balance_inputs_fail_collectively(argc, argv);

    if (rank == 0) {
        std::cout << "[PASS] test_numa_cost\n";
    }
    MPI_Finalize();
    return 0;
}
