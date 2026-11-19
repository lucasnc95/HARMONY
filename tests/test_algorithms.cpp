#include "harmony/algorithms.hpp"
#include "harmony/runtime.hpp"

#include <mpi.h>

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

namespace {

harmony::DevicePartition make_part(int g_idx, int rank, int loc_idx,
                                   std::size_t off, std::size_t cnt) {
    harmony::DevicePartition p;
    p.device_global_index = g_idx;
    p.owning_rank = rank;
    p.local_index = loc_idx;
    p.global_offset = off;
    p.element_count = cnt;
    return p;
}

void assert_valid_cumulative_loads(const std::vector<float>& loads, std::size_t expected_size) {
    assert(loads.size() == expected_size);
    for (std::size_t i = 0; i < loads.size(); ++i) {
        assert(std::isfinite(loads[i]));
        assert(loads[i] >= 0.0f);
        assert(loads[i] <= 1.0f);
        if (i > 0) {
            assert(loads[i] >= loads[i - 1]);
        }
    }
    assert(loads.back() == 1.0f);
}

void test_erad_loads_throughput_and_edge_cases() {
    // Case A: 2 partitions with unequal throughput (ratio 2:1 -> {2/3, 1})
    {
        const std::vector<harmony::DevicePartition> parts = {
            make_part(0, 0, 0, 0, 500000),
            make_part(1, 0, 1, 500000, 500000)
        };
        const std::vector<double> times = {0.005, 0.010};
        const std::vector<float> erad = harmony::erad_loads(times, parts);
        assert_valid_cumulative_loads(erad, 2);
        assert(std::fabs(erad[0] - (2.0f / 3.0f)) < 1e-5f);
        assert(erad[1] == 1.0f);
    }

    // Case B: 3 partitions with varying elements and times
    {
        const std::vector<harmony::DevicePartition> parts = {
            make_part(0, 0, 0, 0, 300000),
            make_part(1, 0, 1, 300000, 400000),
            make_part(2, 0, 2, 700000, 300000)
        };
        const std::vector<double> times = {0.003, 0.002, 0.005};
        // Throughputs: 100M, 200M, 60M -> total 360M -> shares 10/36, 20/36, 6/36
        const std::vector<float> erad = harmony::erad_loads(times, parts);
        assert_valid_cumulative_loads(erad, 3);
        assert(std::fabs(erad[0] - (10.0f / 36.0f)) < 1e-5f);
        assert(std::fabs(erad[1] - (30.0f / 36.0f)) < 1e-5f);
        assert(erad[2] == 1.0f);
    }

    // Case C: Zero-time / zero-element fallback yields equal loads
    {
        const std::vector<harmony::DevicePartition> parts = {
            make_part(0, 0, 0, 0, 250000),
            make_part(1, 0, 1, 250000, 250000),
            make_part(2, 0, 2, 500000, 250000),
            make_part(3, 0, 3, 750000, 250000)
        };
        const std::vector<double> zero_times = {0.0, 0.0, 0.0, 0.0};
        const std::vector<float> erad = harmony::erad_loads(zero_times, parts);
        assert_valid_cumulative_loads(erad, 4);
        assert(std::fabs(erad[0] - 0.25f) < 1e-5f);
        assert(std::fabs(erad[1] - 0.50f) < 1e-5f);
        assert(std::fabs(erad[2] - 0.75f) < 1e-5f);
        assert(erad[3] == 1.0f);
    }

    // Case D: Empty and mismatched sizes return empty
    {
        assert(harmony::erad_loads({}, {}).empty());
        const std::vector<harmony::DevicePartition> parts = {
            make_part(0, 0, 0, 0, 100),
            make_part(1, 0, 1, 100, 100)
        };
        assert(harmony::erad_loads({0.01}, parts).empty());
    }
}

void test_hwtopolb_zero_perturbation_matches_erad_bit_for_bit() {
    const std::vector<harmony::DevicePartition> parts = {
        make_part(0, 0, 0, 0, 400000),
        make_part(1, 0, 1, 400000, 600000)
    };
    const std::vector<double> times = {0.008, 0.004};

    const std::vector<float> erad = harmony::erad_loads(times, parts);
    const std::vector<float> hw0 = harmony::hwtopolb_loads(times, parts, 0.0f, 123);
    const std::vector<float> hw_neg = harmony::hwtopolb_loads(times, parts, -0.05f, 123);

    assert(erad.size() == hw0.size());
    assert(erad.size() == hw_neg.size());
    for (std::size_t i = 0; i < erad.size(); ++i) {
        assert(erad[i] == hw0[i]);
        assert(erad[i] == hw_neg[i]);
    }
}

void test_hwtopolb_stochastic_perturbation_and_seed_determinism() {
    const std::vector<harmony::DevicePartition> parts = {
        make_part(0, 0, 0, 0, 1000000),
        make_part(1, 0, 1, 1000000, 1000000),
        make_part(2, 0, 2, 2000000, 1000000)
    };
    const std::vector<double> times = {0.01, 0.01, 0.01};
    const std::vector<float> erad = harmony::erad_loads(times, parts);

    // Explicit non-zero seed must be deterministic across repeated calls.
    const std::vector<float> seeded_a1 = harmony::hwtopolb_loads(times, parts, 0.20f, 42);
    const std::vector<float> seeded_a2 = harmony::hwtopolb_loads(times, parts, 0.20f, 42);
    const std::vector<float> seeded_b = harmony::hwtopolb_loads(times, parts, 0.20f, 123);
    assert_valid_cumulative_loads(seeded_a1, 3);
    assert_valid_cumulative_loads(seeded_b, 3);
    assert(seeded_a1 == seeded_a2);
    assert(seeded_a1 != seeded_b);

    // Random runs (seed = 0) produce valid monotone loads and differ from ERAD.
    int differ_count = 0;
    for (int run = 0; run < 50; ++run) {
        const std::vector<float> hw = harmony::hwtopolb_loads(times, parts, 0.20f, 0);
        assert_valid_cumulative_loads(hw, 3);
        for (std::size_t i = 0; i < hw.size(); ++i) {
            if (std::fabs(hw[i] - erad[i]) > 1e-5f) {
                ++differ_count;
                break;
            }
        }
    }
    assert(differ_count > 0);
}

void test_hwtopolb_runtime_mpi_bcast_seed_sync(int argc, char** argv) {
    int rank = 0;
    int comm_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    auto check_rank_partition_agreement = [&](const harmony::Runtime& rt) {
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

    // Case A: seed == 0 generates on rank 0 and broadcasts via MPI_Bcast so all ranks match.
    {
        auto rt = harmony::Runtime::create(argc, argv);
        rt.set_verbose(false);
        rt.set_simulated_devices_count(2);
        rt.set_partition({1000000, 1, sizeof(float), 1});
        auto f = rt.create_field({"u", 1000000, 1, sizeof(float), harmony::BufferUsage::read_write, nullptr,
                                  harmony::RedistributionDependency::proportional});

        harmony::AutoBalancePolicy policy;
        policy.mode = harmony::BalanceMode::dynamic_threshold;
        policy.interval = 1;
        policy.threshold = 0.0f;
        policy.algorithm = harmony::BalanceAlgorithm::hwtopolb;
        policy.perturbation_factor = 0.25f;
        policy.seed = 0;

        auto step = rt.step("hwtopolb_auto_seed")
            .with_balance(policy)
            .tag_field(f, harmony::StepFieldRole::rebalance_source)
            .build();

        rt.set_simulated_times({0.01, 0.01});
        rt.execute(step);
        check_rank_partition_agreement(rt);
        rt.execute(step);
        check_rank_partition_agreement(rt);

        const auto report = rt.timing_report();
        assert(report.balance.size() == 2);
        assert(report.balance[0].algorithm == "hwtopolb");
        assert(report.balance[0].seed != 0);
        assert(std::fabs(report.balance[0].perturbation_factor - 0.25f) < 1e-6f);

        unsigned int root_seed = report.balance[0].seed;
        MPI_Bcast(&root_seed, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
        assert(report.balance[0].seed == root_seed);
    }

    // Case B: explicit seed is deterministic across runs and identical across ranks.
    auto run_explicit_seed = [&](unsigned int seed) {
        auto rt = harmony::Runtime::create(argc, argv);
        rt.set_verbose(false);
        rt.set_simulated_devices_count(2);
        rt.set_partition({1000000, 1, sizeof(float), 1});
        auto f = rt.create_field({"u", 1000000, 1, sizeof(float), harmony::BufferUsage::read_write, nullptr,
                                  harmony::RedistributionDependency::proportional});

        harmony::AutoBalancePolicy policy;
        policy.mode = harmony::BalanceMode::dynamic_threshold;
        policy.interval = 1;
        policy.threshold = 0.0f;
        policy.algorithm = harmony::BalanceAlgorithm::hwtopolb;
        policy.perturbation_factor = 0.25f;
        policy.seed = seed;

        auto step = rt.step("hwtopolb_seeded")
            .with_balance(policy)
            .tag_field(f, harmony::StepFieldRole::rebalance_source)
            .build();

        rt.set_simulated_times({0.01, 0.01});
        rt.execute(step);
        check_rank_partition_agreement(rt);
        const std::size_t first0 = rt.partitions()[0].element_count;
        rt.execute(step);
        check_rank_partition_agreement(rt);
        const auto report = rt.timing_report();
        assert(report.balance[0].seed == seed);
        return std::pair<std::size_t, std::size_t>{first0, rt.partitions()[0].element_count};
    };

    const auto run1 = run_explicit_seed(123u);
    const auto run2 = run_explicit_seed(123u);
    assert(run1 == run2);

    // Case C: hwtopolb with perturbation_factor == 0.0f matches erad in Runtime.
    auto run_unperturbed = [&](harmony::BalanceAlgorithm algo) {
        auto rt = harmony::Runtime::create(argc, argv);
        rt.set_verbose(false);
        rt.set_simulated_devices_count(2);
        rt.set_partition({1000000, 1, sizeof(float), 1});
        auto f = rt.create_field({"u", 1000000, 1, sizeof(float), harmony::BufferUsage::read_write, nullptr,
                                  harmony::RedistributionDependency::proportional});

        harmony::AutoBalancePolicy policy;
        policy.mode = harmony::BalanceMode::static_threshold;
        policy.interval = 1;
        policy.threshold = 0.01f;
        policy.algorithm = algo;
        policy.perturbation_factor = 0.0f;
        policy.seed = 123u;

        auto step = rt.step("compare_unperturbed")
            .with_balance(policy)
            .tag_field(f, harmony::StepFieldRole::rebalance_source)
            .build();
        rt.set_simulated_times({0.005, 0.010});
        rt.execute(step);
        check_rank_partition_agreement(rt);
        return rt.partitions()[0].element_count;
    };

    const std::size_t erad_p0 = run_unperturbed(harmony::BalanceAlgorithm::erad);
    const std::size_t hw0_p0 = run_unperturbed(harmony::BalanceAlgorithm::hwtopolb);
    assert(erad_p0 == hw0_p0);
}

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    test_erad_loads_throughput_and_edge_cases();
    test_hwtopolb_zero_perturbation_matches_erad_bit_for_bit();
    test_hwtopolb_stochastic_perturbation_and_seed_determinism();
    test_hwtopolb_runtime_mpi_bcast_seed_sync(argc, argv);

    if (rank == 0) {
        std::cout << "[PASS] test_algorithms\n";
    }
    MPI_Finalize();
    return 0;
}
