#include "../benchmarks/benchmark_common.hpp"
#include "harmony/runtime.hpp"

#include <mpi.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

void test_verify_finite_and_close_rejects_non_finite() {
    const float nan_val = std::numeric_limits<float>::quiet_NaN();
    const float inf_val = std::numeric_limits<float>::infinity();
    const float ninf_val = -std::numeric_limits<float>::infinity();

    assert(harmony_bench::verify_finite_and_close(1.0f, 1.0f, 1e-5f, 0.0f));
    assert(harmony_bench::verify_finite_and_close(1.000001f, 1.0f, 1e-5f, 0.0f));
    assert(!harmony_bench::verify_finite_and_close(1.1f, 1.0f, 1e-5f, 0.0f));

    assert(!harmony_bench::verify_finite_and_close(nan_val, 0.0f, 1e-5f, 1e-4f));
    assert(!harmony_bench::verify_finite_and_close(0.0f, nan_val, 1e-5f, 1e-4f));
    assert(!harmony_bench::verify_finite_and_close(nan_val, nan_val, 1e-5f, 1e-4f));

    assert(!harmony_bench::verify_finite_and_close(inf_val, inf_val, 1e-5f, 1e-4f));
    assert(!harmony_bench::verify_finite_and_close(ninf_val, 0.0f, 1e-5f, 1e-4f));
    assert(!harmony_bench::verify_finite_and_close(0.0f, inf_val, 1e-5f, 1e-4f));
}

void test_parse_benchmark_options_and_policy() {
    {
        const char* argv[] = {
            "kneighbor",
            "--n", "2048",
            "--k", "3",
            "--iterations", "25",
            "--balance-algo", "hwtopolb",
            "--perturbation", "0.15",
            "--balance-interval", "5",
            "--seed", "123",
            "--output-csv", "/tmp/kn.csv",
            "--report-csv", "/tmp/kn_report.csv",
            "--device", "sim",
            "--verbose", "0"
        };
        int argc = static_cast<int>(sizeof(argv) / sizeof(argv[0]));
        const auto opt = harmony_bench::parse_options(argc, const_cast<char**>(argv), "kneighbor", 5);
        assert(opt.n == 2048);
        assert(opt.k == 3);
        assert(opt.iterations == 25);
        assert(opt.balance_algo == "hwtopolb");
        assert(std::fabs(opt.perturbation - 0.15f) < 1e-6f);
        assert(opt.balance_interval == 5);
        assert(opt.seed == 123u);
        assert(opt.output_csv == "/tmp/kn.csv");
        assert(opt.report_csv == "/tmp/kn_report.csv");
        assert(opt.device_mode == harmony_bench::DeviceMode::sim);
        assert(!opt.verbose);

        const harmony::AutoBalancePolicy policy = harmony_bench::build_balance_policy(opt);
        assert(policy.mode == harmony::BalanceMode::dynamic_threshold);
        assert(policy.algorithm == harmony::BalanceAlgorithm::hwtopolb);
        assert(std::fabs(policy.perturbation_factor - 0.15f) < 1e-6f);
        assert(policy.interval == 5);
        assert(policy.seed == 123u);
        assert(policy.total_iterations == 25);
    }

    {
        const char* argv[] = {"leanmd", "--balance-interval", "0", "--balance-algo", "erad"};
        int argc = static_cast<int>(sizeof(argv) / sizeof(argv[0]));
        const auto opt = harmony_bench::parse_options(argc, const_cast<char**>(argv), "leanmd", 4);
        const harmony::AutoBalancePolicy policy = harmony_bench::build_balance_policy(opt);
        assert(policy.mode == harmony::BalanceMode::off);
        assert(policy.algorithm == harmony::BalanceAlgorithm::erad);
    }

    auto expect_invalid = [](std::initializer_list<const char*> args) {
        std::vector<char*> argv;
        for (const char* a : args) argv.push_back(const_cast<char*>(a));
        bool threw = false;
        try {
            (void)harmony_bench::parse_options(static_cast<int>(argv.size()), argv.data(), "kneighbor", 5);
        } catch (const std::exception&) {
            threw = true;
        }
        assert(threw);
    };

    expect_invalid({"kneighbor", "--n", "0"});
    expect_invalid({"kneighbor", "--iterations", "0"});
    expect_invalid({"kneighbor", "--k", "-1"});
    expect_invalid({"kneighbor", "--balance-algo", "unknown"});
    expect_invalid({"kneighbor", "--unknown-flag", "1"});
}

void test_reference_worked_examples() {
    const std::vector<float> in{0.0f, 1.0f, 2.0f, 3.0f};
    assert(std::fabs(harmony_bench::kneighbor_ref_at(in, 0, 1) - 0.5f) < 1e-6f);
    assert(std::fabs(harmony_bench::kneighbor_ref_at(in, 1, 1) - 1.0f) < 1e-6f);
    assert(std::fabs(harmony_bench::kneighbor_ref_at(in, 2, 1) - 2.0f) < 1e-6f);
    assert(std::fabs(harmony_bench::kneighbor_ref_at(in, 3, 1) - 2.5f) < 1e-6f);

    const std::vector<float> px{0.0f, 1.0f};
    const std::vector<float> py{0.0f, 0.0f};
    const std::vector<float> pz{0.0f, 0.0f};
    const float f0 = harmony_bench::leanmd_ref_at(px, py, pz, 0, 1);
    const float f1 = harmony_bench::leanmd_ref_at(px, py, pz, 1, 1);
    assert(std::fabs(f0 - 1.0f) < 1e-4f);
    assert(std::fabs(f1 - 1.0f) < 1e-4f);
}

void test_runtime_benchmark_policy_and_report_determinism(int argc, char** argv) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    auto run_hwtopolb_once = [&](unsigned int seed, const std::string& report_path) {
        harmony::Runtime rt = harmony::Runtime::create(argc, argv);
        rt.set_verbose(false);
        rt.set_simulated_devices_count(2);
        rt.set_partition({1000, 1, sizeof(float), 1});
        auto f_out = rt.create_field({"out_data", 1000, 1, sizeof(float),
                                      harmony::BufferUsage::write_only, nullptr,
                                      harmony::RedistributionDependency::proportional});

        const char* args[] = {
            "kneighbor", "--n", "1000", "--iterations", "6",
            "--balance-interval", "2", "--balance-algo", "hwtopolb",
            "--perturbation", "0.20", "--seed", "123",
            "--report-csv", report_path.c_str()
        };
        auto opt = harmony_bench::parse_options(
            static_cast<int>(sizeof(args) / sizeof(args[0])),
            const_cast<char**>(args), "kneighbor", 5);
        opt.seed = seed;

        const auto policy = harmony_bench::build_balance_policy(opt);
        const auto step = rt.step("kneighbor_step")
            .with_balance(policy)
            .tag_field(f_out, harmony::StepFieldRole::rebalance_source)
            .build();

        rt.start_timing();
        for (int it = 0; it < opt.iterations; ++it) {
            rt.set_simulated_times({0.02, 0.01});
            rt.execute(step);
        }
        const double loop_s = rt.stop_timing();
        assert(std::isfinite(loop_s) && loop_s >= 0.0);

        const auto report = rt.timing_report();
        assert(report.balance.size() == 2);
        assert(report.balance[0].algorithm == "hwtopolb");
        assert(report.balance[0].seed == seed);

        rt.write_timing_report(report_path);
        return rt.partitions()[0].element_count;
    };

    const std::string csv_a = "/tmp/harmony_bench_test_a_" + std::to_string(getpid()) + ".csv";
    const std::string csv_b = "/tmp/harmony_bench_test_b_" + std::to_string(getpid()) + ".csv";
    const std::size_t elems_a = run_hwtopolb_once(123u, csv_a);
    const std::size_t elems_b = run_hwtopolb_once(123u, csv_b);
    assert(elems_a == elems_b);

    if (rank == 0) {
        std::ifstream ifs(csv_a);
        assert(ifs.is_open());
        const std::string content((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
        assert(content.find("#run,wall_s,") != std::string::npos);
        assert(content.find("#timeline,") != std::string::npos);
        assert(content.find("#device,") != std::string::npos);
        assert(content.find("#process,") != std::string::npos);
        assert(content.find("#balance,") != std::string::npos);
        assert(content.find(",hwtopolb,") != std::string::npos);
        std::remove(csv_a.c_str());
        std::remove(csv_b.c_str());
    }
}

void test_run_benchmark_loop_balance_off_records_iterations(int argc, char** argv) {
    harmony::Runtime rt = harmony::Runtime::create(argc, argv);
    rt.set_verbose(false);
    rt.set_simulated_devices_count(2);
    rt.set_partition({200, 1, sizeof(float), 1});
    auto f_out = rt.create_field({"out_data", 200, 1, sizeof(float),
                                  harmony::BufferUsage::write_only, nullptr,
                                  harmony::RedistributionDependency::proportional});

    const char* args[] = {
        "kneighbor", "--n", "200", "--iterations", "4",
        "--balance-interval", "0", "--device", "sim"
    };
    const auto opt = harmony_bench::parse_options(
        static_cast<int>(sizeof(args) / sizeof(args[0])),
        const_cast<char**>(args), "kneighbor", 5);
    const auto policy = harmony_bench::build_balance_policy(opt);
    assert(policy.mode == harmony::BalanceMode::off);

    const auto step = rt.step("kneighbor_off")
        .with_balance(policy)
        .tag_field(f_out, harmony::StepFieldRole::rebalance_source)
        .build();

    std::vector<float> in_data(200, 1.0f);
    std::vector<float> out_data(200, 0.0f);
    const std::vector<double> step_times = harmony_bench::run_benchmark_loop(
        rt, step, policy, opt, /*opencl_available=*/false,
        [&](std::size_t i) { out_data[i] = harmony_bench::kneighbor_ref_at(in_data, i, opt.k); });
    const double loop_s = rt.stop_timing();
    assert(std::isfinite(loop_s) && loop_s >= 0.0);
    assert(step_times.size() == 4);

    const auto report = rt.timing_report();
    assert(report.iterations == 4);
    assert(report.balance.empty());
}

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    test_verify_finite_and_close_rejects_non_finite();
    test_parse_benchmark_options_and_policy();
    test_reference_worked_examples();
    test_runtime_benchmark_policy_and_report_determinism(argc, argv);
    test_run_benchmark_loop_balance_off_records_iterations(argc, argv);

    if (rank == 0) {
        std::cout << "[PASS] test_benchmarks\n";
    }
    MPI_Finalize();
    return 0;
}
