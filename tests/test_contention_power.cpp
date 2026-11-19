#include "harmony/algorithms.hpp"
#include "harmony/runtime.hpp"

#include <mpi.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

void test_adjusted_capacity() {
    harmony::TopoMetrics m;
    m.memory_contention_factor = {1.0, 2.0, 4.0};

    assert(std::fabs(harmony::adjusted_capacity(m, 0, 1000.0) - 1000.0) < 1e-9);
    assert(std::fabs(harmony::adjusted_capacity(m, 1, 1000.0) - 500.0) < 1e-9);
    assert(std::fabs(harmony::adjusted_capacity(m, 2, 1000.0) - 250.0) < 1e-9);

    auto expect_error = [](auto fn) {
        bool caught = false;
        try {
            (void)fn();
        } catch (const harmony::Error&) {
            caught = true;
        }
        assert(caught);
    };

    expect_error([&]() { return harmony::adjusted_capacity(m, -1, 1000.0); });
    expect_error([&]() { return harmony::adjusted_capacity(m, 3, 1000.0); });

    harmony::TopoMetrics bad;
    bad.memory_contention_factor = {0.8};
    expect_error([&]() { return harmony::adjusted_capacity(bad, 0, 1000.0); });

    bad.memory_contention_factor = {std::numeric_limits<double>::quiet_NaN()};
    expect_error([&]() { return harmony::adjusted_capacity(bad, 0, 1000.0); });

    expect_error([&]() { return harmony::adjusted_capacity(m, 0, -5.0); });
}

void test_apply_power_cap_redistribution_and_budget() {
    // 2-device case: Dev 0 overloaded (250W > 200W TDP), Dev 1 cool (150W < 200W TDP).
    {
        const std::vector<float> initial = {0.5f, 1.0f};
        harmony::TopoMetrics m;
        m.thermal_tdp_watts = {200.0, 200.0};
        m.current_power_watts = {250.0, 150.0};

        double throttle = 0.0;
        const std::vector<float> capped = harmony::apply_power_cap(initial, m, 0.0, &throttle);
        assert(capped.size() == 2);
        assert(std::fabs(throttle - 1.0) < 1e-9);
        assert(std::fabs(capped[0] - 0.4f) < 1e-5f);
        assert(capped[1] == 1.0f);
    }

    // Reverse 2-device case: Dev 0 cool, Dev 1 overloaded.
    {
        const std::vector<float> initial = {0.5f, 1.0f};
        harmony::TopoMetrics m;
        m.thermal_tdp_watts = {200.0, 200.0};
        m.current_power_watts = {150.0, 250.0};

        double throttle = 0.0;
        const std::vector<float> capped = harmony::apply_power_cap(initial, m, 0.0, &throttle);
        assert(capped.size() == 2);
        assert(std::fabs(throttle - 1.0) < 1e-9);
        assert(std::fabs(capped[0] - 0.6f) < 1e-5f);
        assert(capped[1] == 1.0f);
    }

    // 3-device case: Dev 0 overloaded, Dev 1 & 2 share freed load by thermal headroom.
    {
        const std::vector<float> initial = {1.0f / 3.0f, 2.0f / 3.0f, 1.0f};
        harmony::TopoMetrics m;
        m.thermal_tdp_watts = {200.0, 200.0, 200.0};
        m.current_power_watts = {250.0, 150.0, 100.0};

        double throttle = 0.0;
        const std::vector<float> capped = harmony::apply_power_cap(initial, m, 0.0, &throttle);
        assert(capped.size() == 3);
        assert(std::fabs(throttle - 1.0) < 1e-9);
        const float s0 = capped[0];
        const float s1 = capped[1] - capped[0];
        const float s2 = capped[2] - capped[1];
        assert(s0 < 1.0f / 3.0f);
        assert(s1 > 1.0f / 3.0f);
        assert(s2 > s1);
        assert(capped[0] > 0.0f && capped[0] < capped[1] && capped[1] < capped[2]);
        assert(capped[2] == 1.0f);
    }

    // Feasible system-wide power budget:
    // Current loads {0.5, 1.0}, current power {300, 100} -> slopes {600, 200} W/unit.
    // Proposed {0.5, 1.0} predicts 400W, exceeding budget 320W.
    // Minimum-power feasible allocation shifts to Dev 1 up to its TDP capacity (200/200 = 1.0),
    // which predicts 200W <= 320W, or shifts enough to Dev 1 to satisfy budget.
    {
        const std::vector<float> current = {0.5f, 1.0f};
        const std::vector<float> proposed = {0.5f, 1.0f};
        harmony::TopoMetrics m;
        m.thermal_tdp_watts = {400.0, 200.0};
        m.current_power_watts = {300.0, 100.0};

        double throttle = 0.0;
        const std::vector<float> capped = harmony::apply_power_cap(proposed, m, 320.0, &throttle, &current);
        assert(capped.size() == 2);
        assert(std::fabs(throttle - 1.0) < 1e-9);
        assert(capped[0] < 0.5f);
        assert(capped[1] == 1.0f);
        const double predicted = 600.0 * static_cast<double>(capped[0]) +
                                 200.0 * static_cast<double>(1.0f - capped[0]);
        assert(predicted <= 320.0 + 1e-6);
    }

    // Explicit current_loads differs from proposed loads:
    // Measured under {0.1, 1.0} with power {300, 100}: slopes {3000, 111.11}.
    // Proposed {0.9, 1.0} must NOT blow up Dev 0; it caps Dev 0 to 200/3000 = 0.0667.
    {
        const std::vector<float> current = {0.1f, 1.0f};
        const std::vector<float> proposed = {0.9f, 1.0f};
        harmony::TopoMetrics m;
        m.thermal_tdp_watts = {200.0, 200.0};
        m.current_power_watts = {300.0, 100.0};

        const std::vector<float> capped = harmony::apply_power_cap(proposed, m, 0.0, nullptr, &current);
        assert(capped.size() == 2);
        assert(std::fabs(capped[0] - (200.0f / 3000.0f)) < 1e-4f);
        assert(capped[1] == 1.0f);
    }

    // Infeasible TDP saturation reports global throttle factor > 1.0 and throws harmony::Error.
    {
        const std::vector<float> initial = {0.5f, 1.0f};
        harmony::TopoMetrics m;
        m.thermal_tdp_watts = {200.0, 200.0};
        m.current_power_watts = {400.0, 400.0};

        double throttle = 1.0;
        bool rejected = false;
        try {
            (void)harmony::apply_power_cap(initial, m, 0.0, &throttle);
        } catch (const harmony::Error&) {
            rejected = true;
        }
        assert(rejected);
        assert(std::fabs(throttle - 2.0) < 1e-5);
    }

    // Infeasible system-wide power budget reports global throttle factor > 1.0 and throws harmony::Error.
    {
        const std::vector<float> initial = {0.5f, 1.0f};
        harmony::TopoMetrics m;
        m.thermal_tdp_watts = {200.0, 200.0};
        m.current_power_watts = {200.0, 200.0};

        double throttle = 1.0;
        bool rejected = false;
        try {
            (void)harmony::apply_power_cap(initial, m, 300.0, &throttle);
        } catch (const harmony::Error&) {
            rejected = true;
        }
        assert(rejected);
        assert(std::fabs(throttle - (400.0 / 300.0)) < 1e-5);
    }

    // Edge cases: empty loads, single device, missing power metrics.
    {
        harmony::TopoMetrics m;
        assert(harmony::apply_power_cap({}, m, 0.0).empty());
        const std::vector<float> single = {1.0f};
        assert((harmony::apply_power_cap(single, m, 0.0) == single));
        const std::vector<float> two = {0.5f, 1.0f};
        assert((harmony::apply_power_cap(two, m, 0.0) == two));
    }
}

void test_runtime_contention_power_and_reports(int argc, char** argv) {
    int rank = 0;
    int comm_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    // 1. Verify default AutoBalancePolicy & BalanceRecord fields.
    {
        harmony::AutoBalancePolicy def_policy;
        assert(def_policy.algorithm == harmony::BalanceAlgorithm::erad);
        assert(def_policy.perturbation_factor == 0.0f);
        assert(def_policy.seed == 0u);
        assert(!def_policy.use_contention_adjustment);
        assert(!def_policy.use_power_cap);
        assert(std::fabs(def_policy.power_budget_watts) < 1e-9);

        harmony::BalanceRecord def_rec;
        assert(def_rec.algorithm == "erad");
        assert(!def_rec.contention_adjusted);
        assert(!def_rec.power_capped);
        assert(std::fabs(def_rec.throttle_factor - 1.0) < 1e-9);
    }

    auto rt = harmony::Runtime::create(argc, argv);
    rt.set_verbose(false);
    rt.set_simulated_devices_count(2);
    rt.set_partition({1000000, 1, sizeof(float), 1});
    auto fh = rt.create_field({"state_data", 1000000, 1, sizeof(float),
                               harmony::BufferUsage::read_write, nullptr,
                               harmony::RedistributionDependency::proportional});

    const std::size_t n_ranks = static_cast<std::size_t>(rt.size());
    harmony::TopoMetrics m;
    m.pcie_latency_ns = {4000.0, 4000.0};
    m.pcie_bandwidth_gbps = {15.0, 15.0};
    m.mpi_latency_ns.assign(n_ranks * n_ranks, 0.0);
    m.mpi_bandwidth_gbps.assign(n_ranks * n_ranks, 0.0);
    for (std::size_t src = 0; src < n_ranks; ++src) {
        for (std::size_t dst = 0; dst < n_ranks; ++dst) {
            if (src != dst) {
                m.mpi_latency_ns[src * n_ranks + dst] = 1000.0;
                m.mpi_bandwidth_gbps[src * n_ranks + dst] = 3.0;
            }
        }
    }
    m.memory_contention_factor = {1.0, 2.0};
    m.thermal_tdp_watts = {200.0, 200.0};
    m.current_power_watts = {150.0, 150.0};
    rt.set_topo_metrics(m);

    // 2. Contention adjustment: raw 2:1 -> adjusted 4:1 (800k / 200k).
    harmony::AutoBalancePolicy policy;
    policy.mode = harmony::BalanceMode::dynamic_threshold;
    policy.interval = 1;
    policy.threshold = 0.01f;
    policy.algorithm = harmony::BalanceAlgorithm::erad;
    policy.use_contention_adjustment = true;
    policy.use_power_cap = false;

    auto step_contention = rt.step("contention_step")
        .with_balance(policy)
        .tag_field(fh, harmony::StepFieldRole::rebalance_source)
        .build();

    rt.start_timing();
    rt.set_simulated_times({0.005, 0.010});
    rt.execute(step_contention);
    assert(rt.partitions()[0].element_count == 800000);
    assert(rt.partitions()[1].element_count == 200000);

    // 3. Power cap: Dev 0 overloaded (220W > 200W), Dev 1 cool (50W < 200W).
    m.current_power_watts = {220.0, 50.0};
    rt.set_topo_metrics(m);
    rt.set_simulated_times({0.010, 0.005});

    policy.use_power_cap = true;
    policy.power_budget_watts = 350.0;
    auto step_power = rt.step("power_step")
        .with_balance(policy)
        .tag_field(fh, harmony::StepFieldRole::rebalance_source)
        .build();

    rt.execute(step_power);
    assert(rt.partitions()[0].element_count < 800000);
    assert(rt.partitions()[1].element_count > 200000);
    assert(rt.partitions()[0].element_count + rt.partitions()[1].element_count == 1000000);

    const double wall = rt.stop_timing();
    assert(wall >= 0.0);

    const auto report = rt.timing_report();
    assert(report.balance.size() == 2);
    assert(report.balance[0].applied);
    assert(report.balance[0].algorithm == "erad");
    assert(report.balance[0].contention_adjusted);
    assert(!report.balance[0].power_capped);

    assert(report.balance[1].applied);
    assert(report.balance[1].contention_adjusted);
    assert(report.balance[1].power_capped);
    assert(std::fabs(report.balance[1].power_budget_watts - 350.0) < 1e-9);
    assert(std::fabs(report.balance[1].throttle_factor - 1.0) < 1e-9);

    // Verify timing report text and CSV expose algorithm, contention, and power-cap fields.
    std::ostringstream text_out;
    rt.print_timing_report(text_out);
    if (rank == 0) {
        const std::string txt = text_out.str();
        assert(txt.find("erad") != std::string::npos);
        assert(txt.find("contention") != std::string::npos);
        assert(txt.find("power cap") != std::string::npos);
    }

    const std::string csv_path =
        "/tmp/harmony_test_cp_report_" + std::to_string(static_cast<long>(getpid())) + ".csv";
    rt.write_timing_report(csv_path);
    if (rank == 0) {
        std::ifstream ifs(csv_path);
        assert(ifs.is_open());
        std::string all((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
        assert(all.find("#balance,number,iteration,policy,applied,iteration_s_before,iteration_s_after,"
                        "bytes_moved,migration_s,decision_s,warmup_probe,reason,"
                        "algorithm,perturbation_factor,seed,contention_adjusted,power_capped,"
                        "power_budget_watts,throttle_factor") != std::string::npos);
        assert(all.find(",erad,0,0,1,1,350,1") != std::string::npos);
        std::remove(csv_path.c_str());
    }

    // 4. Collective infeasible power cap throws harmony::Error on all ranks without deadlock.
    m.current_power_watts = {400.0, 400.0};
    rt.set_topo_metrics(m);
    rt.set_simulated_times({0.010, 0.010});
    policy.use_contention_adjustment = false;
    policy.use_power_cap = true;
    policy.power_budget_watts = 0.0;
    auto step_fail = rt.step("infeasible_power_step")
        .with_balance(policy)
        .tag_field(fh, harmony::StepFieldRole::rebalance_source)
        .build();

    bool rejected = false;
    try {
        rt.execute(step_fail);
    } catch (const harmony::Error&) {
        rejected = true;
    }
    assert(rejected);
    rt.clear_simulated_times();
    (void)comm_size;
}

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    test_adjusted_capacity();
    test_apply_power_cap_redistribution_and_budget();
    test_runtime_contention_power_and_reports(argc, argv);

    if (rank == 0) {
        std::cout << "[PASS] test_contention_power\n";
    }
    MPI_Finalize();
    return 0;
}
