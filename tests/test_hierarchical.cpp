#include "harmony/runtime.hpp"

#include <mpi.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

harmony::ExecutionStep make_hier_step(harmony::Runtime& rt,
                                      const std::string& name,
                                      const std::vector<harmony::FieldHandle>& fields = {},
                                      int interval = 1) {
    harmony::AutoBalancePolicy policy;
    policy.mode = harmony::BalanceMode::hierarchical;
    policy.interval = interval;
    auto sb = rt.step(name).with_balance(policy);
    for (harmony::FieldHandle fh : fields) {
        sb.tag_field(fh, harmony::StepFieldRole::rebalance_source);
    }
    return sb.build();
}

void test_balance_mode_enum_and_policy() {
    harmony::AutoBalancePolicy pol;
    pol.mode = harmony::BalanceMode::hierarchical;
    pol.interval = 10;
    pol.threshold = 0.05f;

    assert(pol.mode == harmony::BalanceMode::hierarchical);
    assert(pol.interval == 10);

    int mode_id = -1;
    switch (pol.mode) {
        case harmony::BalanceMode::off: mode_id = 0; break;
        case harmony::BalanceMode::static_threshold: mode_id = 1; break;
        case harmony::BalanceMode::dynamic_threshold: mode_id = 2; break;
        case harmony::BalanceMode::static_profiled: mode_id = 3; break;
        case harmony::BalanceMode::dynamic_profiled: mode_id = 4; break;
        case harmony::BalanceMode::hierarchical: mode_id = 5; break;
    }
    assert(mode_id == 5);
}

void test_hierarchical_timing_imbalance(int argc, char** argv) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    auto rt = harmony::Runtime::create(argc, argv);
    rt.set_simulated_devices_count(2);
    rt.set_partition({1000000, 1, sizeof(float), 1});

    rt.start_timing();
    rt.set_simulated_times({0.010, 0.020});
    auto step = make_hier_step(rt, "hier_imbalance");

    std::stringstream captured;
    std::streambuf* old_buf = nullptr;
    if (rank == 0) old_buf = std::cout.rdbuf(captured.rdbuf());
    rt.execute(step);
    std::string out_str;
    if (rank == 0) {
        std::cout.rdbuf(old_buf);
        out_str = captured.str();
        assert(out_str.find("[HIER] intra-node loads:") != std::string::npos);
        assert(out_str.find("[HIER] inter-node loads:") != std::string::npos);
        assert(out_str.find("[HIER] final partition:") != std::string::npos);
    }

    const auto& parts = rt.partitions();
    assert(parts.size() == 2);
    assert(parts[0].global_offset == 0);
    assert(parts[0].element_count == 666667);
    assert(parts[1].global_offset == 666667);
    assert(parts[1].element_count == 333333);
    assert(parts[0].element_count + parts[1].element_count == 1000000);

    (void)rt.stop_timing();
    const auto report = rt.timing_report();
    assert(report.balance.size() == 1);
    assert(report.balance[0].policy == "hierarchical");
    assert(report.balance[0].applied);
    assert((report.balance[0].elements_before == std::vector<std::size_t>{500000, 500000}));
    assert((report.balance[0].elements_after == std::vector<std::size_t>{666667, 333333}));

    std::ostringstream text_report;
    rt.print_timing_report(text_report);
    if (rank == 0) {
        const std::string txt = text_report.str();
        assert(txt.find("Load balancing: hierarchical") != std::string::npos);
        assert(txt.find("static profiled") == std::string::npos);
        assert(txt.find("(single attempt)") == std::string::npos);
    }

    const std::string csv_path =
        "/tmp/harmony_test_hier_" + std::to_string(static_cast<long>(getpid())) + ".csv";
    rt.write_timing_report(csv_path);
    if (rank == 0) {
        std::ifstream ifs(csv_path);
        assert(ifs.is_open());
        std::string all((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
        assert(all.find(",hierarchical,") != std::string::npos);
        std::remove(csv_path.c_str());
    }
}

void test_hierarchical_reverse_imbalance(int argc, char** argv) {
    auto rt = harmony::Runtime::create(argc, argv);
    rt.set_verbose(false);
    rt.set_simulated_devices_count(2);
    rt.set_partition({1000000, 1, sizeof(float), 1});
    rt.set_simulated_times({0.020, 0.010});
    rt.execute(make_hier_step(rt, "hier_reverse"));

    const auto& parts = rt.partitions();
    assert(parts.size() == 2);
    assert(parts[0].global_offset == 0 && parts[0].element_count == 333333);
    assert(parts[1].global_offset == 333333 && parts[1].element_count == 666667);
}

void test_hierarchical_equal_times(int argc, char** argv) {
    auto rt = harmony::Runtime::create(argc, argv);
    rt.set_verbose(false);
    rt.set_simulated_devices_count(2);
    rt.set_partition({1000000, 1, sizeof(float), 1});
    rt.set_simulated_times({0.015, 0.015});
    rt.execute(make_hier_step(rt, "hier_equal"));

    const auto& parts = rt.partitions();
    assert(parts.size() == 2);
    assert(parts[0].element_count == 500000);
    assert(parts[1].element_count == 500000);
}

void test_hierarchical_with_granularity(int argc, char** argv) {
    auto rt = harmony::Runtime::create(argc, argv);
    rt.set_verbose(false);
    rt.set_simulated_devices_count(2);
    rt.set_partition({1000000, 1, sizeof(float), 1000});
    rt.set_simulated_times({0.010, 0.020});
    rt.execute(make_hier_step(rt, "hier_gran"));

    const auto& parts = rt.partitions();
    assert(parts.size() == 2);
    assert(parts[0].global_offset % 1000 == 0);
    assert(parts[0].element_count % 1000 == 0);
    assert(parts[1].global_offset % 1000 == 0);
    assert(parts[1].element_count % 1000 == 0);
    assert(parts[0].element_count + parts[1].element_count == 1000000);
    assert(parts[0].element_count == 667000 || parts[0].element_count == 666000);
}

void test_hierarchical_with_registered_field(int argc, char** argv) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    auto rt = harmony::Runtime::create(argc, argv);
    rt.set_verbose(false);
    rt.set_simulated_devices_count(2);
    rt.set_partition({1000000, 1, sizeof(float), 1});
    auto fh = rt.create_field({"hier_test_field", 1000000, 1, sizeof(float),
                               harmony::BufferUsage::read_write, nullptr,
                               harmony::RedistributionDependency::proportional});

    rt.start_timing();
    rt.set_simulated_times({0.010, 0.020});
    rt.execute(make_hier_step(rt, "hier_field", {fh}));
    (void)rt.stop_timing();

    const auto& parts = rt.partitions();
    assert(parts.size() == 2);
    assert(parts[0].element_count == 666667);
    assert(parts[1].element_count == 333333);

    const auto report = rt.timing_report();
    assert(report.balance.size() == 1);
    assert(report.balance[0].applied);
    assert(report.balance[0].bytes_moved == 166667 * sizeof(float));
    assert(report.ranks[static_cast<std::size_t>(rank)].balance_seconds >= 0.0);
    assert(report.ranks[static_cast<std::size_t>(rank)].mpi_balance_seconds >= 0.0);
}

void test_hysteresis_prevents_oscillation(int argc, char** argv) {
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    auto rt = harmony::Runtime::create(argc, argv);
    if (size == 1) rt.set_simulated_ranks_for_testing(2);
    rt.set_simulated_devices_count(2);
    rt.set_partition({1000000, 1, sizeof(float), 1});
    auto fh = rt.create_field({"hyst_field", 1000000, 1, sizeof(float),
                               harmony::BufferUsage::read_write, nullptr,
                               harmony::RedistributionDependency::proportional});

    auto step = make_hier_step(rt, "hier_hyst", {fh});
    rt.set_verbose(false);
    rt.set_simulated_times({0.100, 0.100});
    rt.execute(step);

    rt.start_timing();
    rt.set_verbose(true);
    rt.set_simulated_times({0.100, 0.102});

    std::stringstream captured;
    std::streambuf* old_buf = nullptr;
    if (rank == 0) old_buf = std::cout.rdbuf(captured.rdbuf());
    rt.execute(step);
    std::string out_str;
    if (rank == 0) {
        std::cout.rdbuf(old_buf);
        out_str = captured.str();
        assert(out_str.find("aborted due to hysteresis") != std::string::npos);
    }

    (void)rt.stop_timing();
    const auto& parts = rt.partitions();
    assert(parts[0].element_count == 500000);
    assert(parts[1].element_count == 500000);

    const auto report = rt.timing_report();
    assert(report.balance.size() == 1);
    assert(!report.balance[0].applied);
    assert(report.balance[0].bytes_moved == 0);
    assert(report.balance[0].reason.find("hysteresis") != std::string::npos);
    assert(report.ranks[static_cast<std::size_t>(rank)].mpi_migration_seconds == 0.0);
}

void test_c1_no_element_loss_with_more_ranks_than_devices(int argc, char** argv) {
    int size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    for (int dev_count : {1, 3}) {
        if (dev_count >= size && size > 1) continue;
        auto rt = harmony::Runtime::create(argc, argv);
        rt.set_verbose(false);
        rt.set_simulated_devices_count(dev_count);
        rt.set_partition({1200000, 1, sizeof(float), 1});

        std::vector<double> times(static_cast<std::size_t>(dev_count), 0.010);
        if (dev_count == 3) times = {0.010, 0.020, 0.030};
        rt.set_simulated_times(times);
        rt.execute(make_hier_step(rt, "hier_c1"));

        const auto& parts = rt.partitions();
        assert(parts.size() == static_cast<std::size_t>(dev_count));
        std::size_t covered = 0;
        for (const auto& dp : parts) {
            assert(dp.global_offset == covered);
            covered += dp.element_count;
        }
        assert(covered == 1200000);
    }
}

void test_local_rebalance_keeps_global_partition_view(int argc, char** argv) {
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 2) return;

    auto rt = harmony::Runtime::create(argc, argv);
    rt.set_verbose(false);
    rt.set_simulated_devices_count(4);
    rt.set_partition({1000000, 1, sizeof(float), 1});
    auto fh = rt.create_field({"local_rebal_field", 1000000, 1, sizeof(float),
                               harmony::BufferUsage::read_write, nullptr,
                               harmony::RedistributionDependency::proportional});

    rt.start_timing();
    // Both ranks have equal total capacity, while devices inside each rank differ ->
    // inter-node migration aborted, intra-node local equalization runs without MPI barriers.
    rt.set_simulated_times(rank == 0
        ? std::vector<double>{0.01, 0.02, 0.01, 0.01}
        : std::vector<double>{0.01, 0.01, 0.02, 0.01});
    rt.execute(make_hier_step(rt, "hier_local", {fh}));
    (void)rt.stop_timing();

    const auto& partitions = rt.partitions();
    assert(partitions.size() == 4);

    unsigned long long local[8] = {};
    unsigned long long minimum[8] = {};
    unsigned long long maximum[8] = {};
    std::size_t next_offset = 0;
    for (std::size_t i = 0; i < partitions.size(); ++i) {
        local[2 * i] = static_cast<unsigned long long>(partitions[i].global_offset);
        local[2 * i + 1] = static_cast<unsigned long long>(partitions[i].element_count);
        assert(partitions[i].global_offset == next_offset);
        next_offset += partitions[i].element_count;
    }
    assert(next_offset == 1000000);

    MPI_Allreduce(local, minimum, 8, MPI_UNSIGNED_LONG_LONG, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(local, maximum, 8, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
    for (int i = 0; i < 8; ++i) {
        assert(minimum[i] == maximum[i] && "Ranks must share a global partition map");
    }

    // Rank 0 had {0.01, 0.02} -> {2/3, 1/3} of its 500k elements.
    assert(partitions[0].element_count == 333333);
    assert(partitions[1].element_count == 166667);
    // Rank 1 had {0.02, 0.01} -> {1/3, 2/3} of its 500k elements.
    assert(partitions[2].element_count == 166667);
    assert(partitions[3].element_count == 333333);

    const auto report = rt.timing_report();
    assert(report.balance.size() == 1);
    assert(report.balance[0].applied);
    assert(report.ranks[static_cast<std::size_t>(rank)].mpi_migration_seconds == 0.0);
}

void test_constant_throughput_converges(int argc, char** argv) {
    int size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 2) return;

    auto rt = harmony::Runtime::create(argc, argv);
    rt.set_verbose(false);
    rt.set_simulated_devices_count(4);
    rt.set_partition({1200000, 1, sizeof(float), 1});

    const std::vector<double> speeds = {60e6, 30e6, 20e6, 10e6};
    auto observed_times = [&]() {
        std::vector<double> times;
        for (std::size_t i = 0; i < speeds.size(); ++i) {
            times.push_back(static_cast<double>(rt.partitions()[i].element_count) / speeds[i]);
        }
        return times;
    };

    auto step = make_hier_step(rt, "hier_converge");
    rt.set_simulated_times(observed_times());
    rt.execute(step);

    std::vector<std::size_t> first_counts;
    for (const auto& part : rt.partitions()) first_counts.push_back(part.element_count);

    rt.set_simulated_times(observed_times());
    rt.execute(step);
    for (std::size_t i = 0; i < first_counts.size(); ++i) {
        assert(rt.partitions()[i].element_count == first_counts[i]);
    }
}

void test_consecutive_micro_partitions_are_removed_before_halo(int argc, char** argv) {
    auto rt = harmony::Runtime::create(argc, argv);
    rt.set_verbose(false);
    rt.set_simulated_devices_count(3);
    rt.set_partition({10, 1, sizeof(float), 1});
    auto fh = rt.create_field({"halo_f", 10, 1, sizeof(float),
                               harmony::BufferUsage::read_write, nullptr,
                               harmony::RedistributionDependency::proportional});

    rt.rebalance_to({0.1f, 0.2f, 1.0f});
    assert(rt.partitions()[0].element_count == 1);
    assert(rt.partitions()[1].element_count == 1);
    assert(rt.partitions()[2].element_count == 8);

    // Step with halo width 3 and an unknown kernel: micro-partitions must be pruned
    // before kernel validation/execution, without losing any of the 10 elements.
    harmony::ExecutionStep step;
    step.invocations.push_back(harmony::KernelInvocation{});
    step.halos.push_back(harmony::HaloSpec{3, {fh}});
    bool reached_kernel_validation = false;
    try {
        rt.execute(step);
    } catch (const harmony::Error& error) {
        reached_kernel_validation = std::string(error.what()).find("unknown kernel") != std::string::npos;
    }
    assert(reached_kernel_validation);

    std::size_t covered = 0;
    for (const auto& part : rt.partitions()) {
        assert(part.element_count == 0 || part.element_count >= 3);
        assert(part.global_offset == covered);
        covered += part.element_count;
    }
    assert(covered == 10);
}

void test_hierarchical_hwtopolb_seed_sync_and_record(int argc, char** argv) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    auto check_rank_agreement = [&](const harmony::Runtime& rt) {
        const auto& parts = rt.partitions();
        assert(parts.size() == 2);
        assert(parts[0].element_count + parts[1].element_count == 1000000);
        unsigned long long local_counts[2] = {
            static_cast<unsigned long long>(parts[0].element_count),
            static_cast<unsigned long long>(parts[1].element_count)
        };
        unsigned long long root_counts[2] = {local_counts[0], local_counts[1]};
        MPI_Bcast(root_counts, 2, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
        assert(local_counts[0] == root_counts[0]);
        assert(local_counts[1] == root_counts[1]);
    };

    // Case A: seed == 0 auto-generates on rank 0, broadcasts via MPI_Bcast, and populates BalanceRecord.
    {
        auto rt = harmony::Runtime::create(argc, argv);
        rt.set_verbose(false);
        rt.set_simulated_devices_count(2);
        rt.set_partition({1000000, 1, sizeof(float), 1});
        auto fh = rt.create_field({"hw_auto", 1000000, 1, sizeof(float),
                                   harmony::BufferUsage::read_write, nullptr,
                                   harmony::RedistributionDependency::proportional});

        harmony::AutoBalancePolicy policy;
        policy.mode = harmony::BalanceMode::hierarchical;
        policy.interval = 1;
        policy.algorithm = harmony::BalanceAlgorithm::hwtopolb;
        policy.perturbation_factor = 0.25f;
        policy.seed = 0;

        auto step = rt.step("hier_hwtopolb_auto")
            .with_balance(policy)
            .tag_field(fh, harmony::StepFieldRole::rebalance_source)
            .build();

        rt.set_simulated_times({0.010, 0.020});
        rt.execute(step);
        check_rank_agreement(rt);

        const auto report = rt.timing_report();
        assert(report.balance.size() == 1);
        assert(report.balance[0].policy == "hierarchical");
        assert(report.balance[0].algorithm == "hwtopolb");
        assert(report.balance[0].applied);
        assert(std::fabs(report.balance[0].perturbation_factor - 0.25f) < 1e-6f);
        assert(report.balance[0].seed != 0);

        unsigned int root_seed = report.balance[0].seed;
        MPI_Bcast(&root_seed, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
        assert(report.balance[0].seed == root_seed);
    }

    // Case B: explicit seed is deterministic across runs and perturbs away from ERAD's 666667/333333.
    auto run_seeded = [&](unsigned int seed) {
        auto rt = harmony::Runtime::create(argc, argv);
        rt.set_verbose(false);
        rt.set_simulated_devices_count(2);
        rt.set_partition({1000000, 1, sizeof(float), 1});
        auto fh = rt.create_field({"hw_seed", 1000000, 1, sizeof(float),
                                   harmony::BufferUsage::read_write, nullptr,
                                   harmony::RedistributionDependency::proportional});

        harmony::AutoBalancePolicy policy;
        policy.mode = harmony::BalanceMode::hierarchical;
        policy.interval = 1;
        policy.algorithm = harmony::BalanceAlgorithm::hwtopolb;
        policy.perturbation_factor = 0.25f;
        policy.seed = seed;

        auto step = rt.step("hier_hwtopolb_seeded")
            .with_balance(policy)
            .tag_field(fh, harmony::StepFieldRole::rebalance_source)
            .build();

        rt.set_simulated_times({0.010, 0.020});
        rt.execute(step);
        check_rank_agreement(rt);
        const auto report = rt.timing_report();
        assert(report.balance.size() == 1);
        assert(report.balance[0].seed == seed);
        assert(report.balance[0].algorithm == "hwtopolb");
        assert(report.balance[0].applied);
        return rt.partitions()[0].element_count;
    };

    const std::size_t run1 = run_seeded(123u);
    const std::size_t run2 = run_seeded(123u);
    assert(run1 == run2);
    assert(run1 != 666667);
    (void)rank;
}

void test_hierarchical_power_cap_and_record(int argc, char** argv) {
    auto rt = harmony::Runtime::create(argc, argv);
    rt.set_verbose(false);
    rt.set_simulated_devices_count(2);
    rt.set_partition({1000000, 1, sizeof(float), 1});
    auto fh = rt.create_field({"hier_pcap", 1000000, 1, sizeof(float),
                               harmony::BufferUsage::read_write, nullptr,
                               harmony::RedistributionDependency::proportional});

    const std::size_t n_ranks = static_cast<std::size_t>(rt.size());
    harmony::TopoMetrics m;
    m.pcie_latency_ns = {4000.0, 4000.0};
    m.pcie_bandwidth_gbps = {15.0, 15.0};
    m.mpi_latency_ns.assign(n_ranks * n_ranks, 0.0);
    m.mpi_bandwidth_gbps.assign(n_ranks * n_ranks, 0.0);
    for (std::size_t s = 0; s < n_ranks; ++s) {
        for (std::size_t d = 0; d < n_ranks; ++d) {
            if (s != d) {
                m.mpi_latency_ns[s * n_ranks + d] = 1000.0;
                m.mpi_bandwidth_gbps[s * n_ranks + d] = 3.0;
            }
        }
    }
    m.memory_contention_factor = {1.0, 2.0};
    m.thermal_tdp_watts = {200.0, 200.0};
    m.current_power_watts = {220.0, 50.0};
    rt.set_topo_metrics(m);

    harmony::AutoBalancePolicy policy;
    policy.mode = harmony::BalanceMode::hierarchical;
    policy.interval = 1;
    policy.algorithm = harmony::BalanceAlgorithm::erad;
    policy.use_contention_adjustment = true;
    policy.use_power_cap = true;
    policy.power_budget_watts = 350.0;

    auto step = rt.step("hier_power")
        .with_balance(policy)
        .tag_field(fh, harmony::StepFieldRole::rebalance_source)
        .build();

    rt.start_timing();
    rt.set_simulated_times({0.005, 0.010});
    rt.execute(step);
    (void)rt.stop_timing();

    // Dev 0 drew 220W > 200W TDP, so apply_power_cap must shed load from Dev 0 to Dev 1.
    assert(rt.partitions()[0].element_count < 500000);
    assert(rt.partitions()[1].element_count > 500000);
    assert(rt.partitions()[0].element_count + rt.partitions()[1].element_count == 1000000);

    const auto report = rt.timing_report();
    assert(report.balance.size() == 1);
    assert(report.balance[0].policy == "hierarchical");
    assert(report.balance[0].algorithm == "erad");
    assert(report.balance[0].applied);
    assert(report.balance[0].contention_adjusted);
    assert(report.balance[0].power_capped);
    assert(std::fabs(report.balance[0].power_budget_watts - 350.0) < 1e-9);
    assert(std::fabs(report.balance[0].throttle_factor - 1.0) < 1e-9);

    // Infeasible power cap must throw harmony::Error collectively across all ranks.
    m.current_power_watts = {400.0, 400.0};
    rt.set_topo_metrics(m);
    rt.set_simulated_times({0.010, 0.010});
    bool rejected = false;
    try {
        rt.execute(step);
    } catch (const harmony::Error&) {
        rejected = true;
    }
    assert(rejected);
}

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    test_balance_mode_enum_and_policy();
    test_hierarchical_timing_imbalance(argc, argv);
    test_hierarchical_reverse_imbalance(argc, argv);
    test_hierarchical_equal_times(argc, argv);
    test_hierarchical_with_granularity(argc, argv);
    test_hierarchical_with_registered_field(argc, argv);
    test_hysteresis_prevents_oscillation(argc, argv);
    test_c1_no_element_loss_with_more_ranks_than_devices(argc, argv);
    test_local_rebalance_keeps_global_partition_view(argc, argv);
    test_constant_throughput_converges(argc, argv);
    test_consecutive_micro_partitions_are_removed_before_halo(argc, argv);
    test_hierarchical_hwtopolb_seed_sync_and_record(argc, argv);
    test_hierarchical_power_cap_and_record(argc, argv);

    if (rank == 0) {
        std::cout << "[PASS] test_hierarchical\n";
    }
    MPI_Finalize();
    return 0;
}
