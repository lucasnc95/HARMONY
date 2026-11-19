#include "detail/opencl_resources.hpp"
#include "harmony/runtime.hpp"

#include <mpi.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

struct MockResource {
    int id{0};
};

struct MockReleaseLog {
    std::vector<int>* released{nullptr};
    void operator()(MockResource* res) const noexcept {
        if (res != nullptr && released != nullptr) {
            released->push_back(res->id);
        }
    }
};

using ScopedMock = harmony::detail::ScopedHandle<MockResource*, MockReleaseLog>;

void test_raii_scoped_handle_lifecycle() {
    std::vector<int> released;
    MockResource r1{101};
    MockResource r2{202};
    MockResource r3{303};

    // Out-of-scope destruction releases once.
    {
        ScopedMock h(&r1, MockReleaseLog{&released});
        assert(static_cast<bool>(h));
        assert(h.get() == &r1);
    }
    assert((released == std::vector<int>{101}));
    released.clear();

    // Move construction and move assignment avoid double release and release overwritten handle.
    {
        ScopedMock a(&r1, MockReleaseLog{&released});
        ScopedMock b(std::move(a));
        assert(!static_cast<bool>(a));
        assert(b.get() == &r1);
        assert(released.empty());

        ScopedMock c(&r2, MockReleaseLog{&released});
        c = std::move(b);
        assert(!static_cast<bool>(b));
        assert(c.get() == &r1);
        assert((released == std::vector<int>{202}));
    }
    assert((released == std::vector<int>{202, 101}));
    released.clear();

    // release() relinquishes ownership; out_ptr() resets old handle before C-style output write.
    {
        ScopedMock h(&r1, MockReleaseLog{&released});
        MockResource* raw = h.release();
        assert(raw == &r1);
        assert(!static_cast<bool>(h));
        assert(released.empty());

        h.reset(&r2);
        *h.out_ptr() = &r3;
        assert((released == std::vector<int>{202}));
    }
    assert((released == std::vector<int>{202, 303}));
}

void test_raii_mid_loop_exception_and_rollback() {
    std::vector<int> released;
    std::vector<int> registry{1, 2, 3};
    MockResource r1{11};
    MockResource r2{22};

    bool caught = false;
    try {
        std::vector<ScopedMock> list;
        list.emplace_back(&r1, MockReleaseLog{&released});
        list.emplace_back(&r2, MockReleaseLog{&released});

        registry.push_back(99);
        auto rollback = harmony::detail::make_scope_exit([&]() {
            if (!registry.empty() && registry.back() == 99) {
                registry.pop_back();
            }
        });

        throw std::runtime_error("simulated mid-loop failure");
        rollback.dismiss();
    } catch (const std::runtime_error&) {
        caught = true;
    }

    assert(caught);
    assert((registry == std::vector<int>{1, 2, 3}));
    assert(released.size() == 2);
    assert((released == std::vector<int>{22, 11}) || (released == std::vector<int>{11, 22}));

    // Dismissed rollback must not fire.
    {
        registry.push_back(42);
        auto rollback = harmony::detail::make_scope_exit([&]() { registry.pop_back(); });
        rollback.dismiss();
    }
    assert((registry == std::vector<int>{1, 2, 3, 42}));
}

void test_simulated_discovery_and_partitioning(int argc, char** argv) {
    int comm_size = 1;
    int comm_rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);

    // 1. Multi-rank or single-rank simulated discovery across real MPI ranks.
    {
        auto rt = harmony::Runtime::create(argc, argv);
        assert(rt.rank() == comm_rank);
        assert(rt.size() == comm_size);

        rt.set_simulated_devices_count(4);
        assert(rt.global_devices().size() == 4);
        assert(rt.devices().size() == static_cast<std::size_t>(4 / comm_size));
        assert(rt.device_timings().size() == rt.devices().size());

        for (std::size_t g = 0; g < rt.global_devices().size(); ++g) {
            assert(rt.global_devices()[g].global_index == static_cast<int>(g));
        }

        rt.set_partition({100, 1, sizeof(float), 1});
        const auto& parts = rt.partitions();
        assert(parts.size() == 4);
        std::size_t expected_offset = 0;
        for (std::size_t i = 0; i < parts.size(); ++i) {
            assert(parts[i].device_global_index == static_cast<int>(i));
            assert(parts[i].global_offset == expected_offset);
            assert(parts[i].element_count == 25);
            expected_offset += parts[i].element_count;
            if (parts[i].owning_rank == comm_rank) {
                assert(parts[i].local_index >= 0);
            } else {
                assert(parts[i].local_index == -1);
            }
        }
        assert(expected_offset == 100);

        auto field = rt.create_field({"u", 100, 1, sizeof(float), harmony::BufferUsage::read_write, nullptr,
                                      harmony::RedistributionDependency::proportional});
        assert(field.value >= 0);

        // Clearing via count = 0 leaves clean empty state.
        rt.set_simulated_devices_count(0);
        assert(rt.global_devices().empty());
        assert(rt.devices().empty());
        assert(rt.partitions().empty());
    }

    // 2. Simulated multiple ranks on a single MPI rank (-n 1).
    if (comm_size == 1) {
        auto rt = harmony::Runtime::create(argc, argv);
        rt.set_simulated_ranks_for_testing(2);
        rt.set_simulated_devices_count(4);
        assert(rt.size() == 2);
        assert(rt.global_devices().size() == 4);
        assert(rt.devices().size() == 2);

        assert(rt.global_devices()[0].rank == 0 && rt.global_devices()[0].local_index == 0);
        assert(rt.global_devices()[1].rank == 0 && rt.global_devices()[1].local_index == 1);
        assert(rt.global_devices()[2].rank == 1 && rt.global_devices()[2].local_index == 0);
        assert(rt.global_devices()[3].rank == 1 && rt.global_devices()[3].local_index == 1);

        rt.set_partition({100, 1, sizeof(float), 1});
        const auto& parts = rt.partitions();
        assert(parts.size() == 4);
        assert(parts[0].owning_rank == 0 && parts[0].local_index == 0);
        assert(parts[1].owning_rank == 0 && parts[1].local_index == 1);
        assert(parts[2].owning_rank == 1 && parts[2].local_index == -1);
        assert(parts[3].owning_rank == 1 && parts[3].local_index == -1);
    }
}

void test_simulated_rebalance_to(int argc, char** argv) {
    int comm_size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    {
        auto rt = harmony::Runtime::create(argc, argv);
        rt.set_verbose(false);
        rt.set_simulated_devices_count(2);
        rt.set_partition({1000, 1, sizeof(float), 1});

        auto prop = rt.create_field({"prop", 1000, 1, sizeof(float), harmony::BufferUsage::read_write, nullptr,
                                     harmony::RedistributionDependency::proportional});
        auto param = rt.create_field({"param", 1000, 1, sizeof(float), harmony::BufferUsage::read_only, nullptr,
                                      harmony::RedistributionDependency::none});
        assert(prop.value >= 0 && param.value >= 0);

        rt.rebalance_to({0.25f, 1.0f});
        assert(rt.partitions().size() == 2);
        assert(rt.partitions()[0].global_offset == 0 && rt.partitions()[0].element_count == 250);
        assert(rt.partitions()[1].global_offset == 250 && rt.partitions()[1].element_count == 750);

        auto report = rt.timing_report();
        assert(report.balance.size() == 1);
        assert(report.balance[0].applied);
        assert(report.balance[0].bytes_moved == 250 * sizeof(float));
        assert((report.balance[0].elements_before == std::vector<std::size_t>{500, 500}));
        assert((report.balance[0].elements_after == std::vector<std::size_t>{250, 750}));

        // Same division -> not applied.
        rt.rebalance_to({0.25f, 1.0f});
        report = rt.timing_report();
        assert(report.balance.size() == 2);
        assert(!report.balance[1].applied);

        // Invalid loads throw harmony::Error.
        for (const std::vector<float>& bad :
             std::vector<std::vector<float>>{{1.0f},
                                             {-0.1f, 1.0f},
                                             {std::numeric_limits<float>::quiet_NaN(), 1.0f}}) {
            bool caught = false;
            try {
                rt.rebalance_to(bad);
            } catch (const harmony::Error&) {
                caught = true;
            }
            assert(caught);
        }
    }

    if (comm_size == 1) {
        auto rt = harmony::Runtime::create(argc, argv);
        rt.set_verbose(false);
        rt.set_simulated_ranks_for_testing(2);
        rt.set_simulated_devices_count(2);
        rt.set_partition({1000, 1, sizeof(float), 1});
        rt.create_field({"prop", 1000, 1, sizeof(float), harmony::BufferUsage::read_write, nullptr,
                         harmony::RedistributionDependency::proportional});
        rt.rebalance_to({0.4f, 1.0f});
        assert(rt.partitions()[0].owning_rank == 0 && rt.partitions()[0].element_count == 400);
        assert(rt.partitions()[1].owning_rank == 1 && rt.partitions()[1].element_count == 600);
        auto report = rt.timing_report();
        assert(report.devices.size() == 2);
        assert(report.balance.size() == 1 && report.balance[0].bytes_moved == 100 * sizeof(float));
    }
}

void test_simulated_times_and_auto_rebalance(int argc, char** argv) {
    int rank = 0;
    int comm_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    // Prepare a tiny temporary kernel source path shared across ranks on the host.
    std::string kernel_path;
    {
        int pid = 0;
        if (rank == 0) pid = static_cast<int>(getpid());
        MPI_Bcast(&pid, 1, MPI_INT, 0, MPI_COMM_WORLD);
        kernel_path = "/tmp/harmony_test_sim_" + std::to_string(pid) + ".cl";
        if (rank == 0) {
            std::ofstream kf(kernel_path);
            kf << "__kernel void noop(__global const float* in, __global float* out) {}\n";
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    auto cleanup_kernel = harmony::detail::make_scope_exit([&]() {
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) std::remove(kernel_path.c_str());
    });

    {
        auto rt = harmony::Runtime::create(argc, argv);
        rt.set_verbose(false);
        rt.set_simulated_devices_count(2);
        rt.set_partition({1000, 1, sizeof(float), 1});

        auto in_f = rt.create_field({"in", 1000, 1, sizeof(float), harmony::BufferUsage::read_write, nullptr,
                                     harmony::RedistributionDependency::proportional});
        auto out_f = rt.create_field({"out", 1000, 1, sizeof(float), harmony::BufferUsage::read_write, nullptr,
                                      harmony::RedistributionDependency::proportional});
        auto kern = rt.create_kernel({kernel_path, "noop", ""});

        auto bind = rt.bind(kern)
            .arg(0, in_f, harmony::Access::read)
            .arg(1, out_f, harmony::Access::write)
            .build();

        auto step = rt.step("sim_step")
            .invoke(bind)
            .with_halo_exchange({1, {in_f}})
            .with_balance({harmony::BalanceMode::dynamic_threshold, 2, 0.05f, 10, "", 0})
            .tag_field(out_f, harmony::StepFieldRole::rebalance_source)
            .build();

        rt.set_simulated_times({0.01, 0.03});
        rt.start_timing();
        rt.execute(step);  // iter 1: before interval
        assert(rt.partitions()[0].element_count == 500);
        assert(rt.partitions()[1].element_count == 500);

        rt.execute(step);  // iter 2: triggers rebalance (3x throughput on dev 0 -> 750 / 250)
        assert(rt.partitions()[0].element_count == 750);
        assert(rt.partitions()[1].element_count == 250);
        assert(!rt.device_timings().empty());
        if (comm_size == 1) {
            assert(std::fabs(rt.device_timings()[0].kernel_seconds_last - 0.01) < 1e-12);
            assert(std::fabs(rt.device_timings()[1].kernel_seconds_last - 0.03) < 1e-12);
        } else if (comm_size == 2) {
            const double expected = (rank == 0) ? 0.01 : 0.03;
            assert(std::fabs(rt.device_timings()[0].kernel_seconds_last - expected) < 1e-12);
        }

        // Iter 4 with {0.01, 0.01}: throughputs are 750/0.01 (75k) vs 250/0.01 (25k), so layout stays {750, 250}.
        rt.set_simulated_times({0.01, 0.01});
        rt.execute(step);  // iter 3
        rt.execute(step);  // iter 4
        assert(rt.partitions()[0].element_count == 750);
        assert(rt.partitions()[1].element_count == 250);

        // Reset to equal partition and clear_simulated_times() -> falls back to 0.0 window times.
        rt.rebalance_to({0.5f, 1.0f});
        assert(rt.partitions()[0].element_count == 500);
        rt.clear_simulated_times();
        rt.execute(step);  // iter 5
        rt.execute(step);  // iter 6
        assert(rt.partitions()[0].element_count == 500);
        assert(rt.partitions()[1].element_count == 500);

        const double wall = rt.stop_timing();
        assert(wall >= 0.0);
        const auto report = rt.timing_report();
        assert(report.iterations == 6);
        assert(report.devices.size() == 2);
        assert(report.ranks.size() == static_cast<std::size_t>(comm_size));
        assert(report.balance.size() == 4);
        assert(report.balance[0].applied);
        assert(report.balance[0].bytes_moved == 250 * sizeof(float));
        assert((report.balance[0].window_seconds == std::vector<double>{0.01, 0.03}));
        assert(!report.balance[3].applied);
        assert((report.balance[3].window_seconds == std::vector<double>{0.0, 0.0}));

        std::ostringstream oss;
        rt.print_timing_report(oss);
        if (rank == 0) {
            assert(oss.str().find("HARMONY - TIMING SUMMARY") != std::string::npos);
        }
    }

    // Also verify invocation-less simulated step with tag_field works on simulated ranks.
    if (comm_size == 1) {
        auto rt = harmony::Runtime::create(argc, argv);
        rt.set_verbose(false);
        rt.set_simulated_ranks_for_testing(2);
        rt.set_simulated_devices_count(2);
        rt.set_partition({1000, 1, sizeof(float), 1});
        auto f = rt.create_field({"u", 1000, 1, sizeof(float), harmony::BufferUsage::read_write, nullptr,
                                  harmony::RedistributionDependency::proportional});
        auto step = rt.step("sim_only")
            .with_balance({harmony::BalanceMode::static_threshold, 1, 0.05f, 5, "", 0})
            .tag_field(f, harmony::StepFieldRole::rebalance_source)
            .build();
        rt.set_simulated_times({0.01, 0.03});
        rt.execute(step);
        assert(rt.partitions()[0].element_count == 750);
        assert(rt.partitions()[1].element_count == 250);
    }
}

void test_simulated_balance_off_counts_iterations(int argc, char** argv) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    auto rt = harmony::Runtime::create(argc, argv);
    rt.set_verbose(false);
    rt.set_simulated_devices_count(2);
    rt.set_partition({1000, 1, sizeof(float), 1});
    auto f = rt.create_field({"u", 1000, 1, sizeof(float), harmony::BufferUsage::read_write, nullptr,
                              harmony::RedistributionDependency::proportional});

    harmony::AutoBalancePolicy off_policy;
    off_policy.mode = harmony::BalanceMode::off;
    auto step = rt.step("sim_off")
        .with_balance(off_policy)
        .tag_field(f, harmony::StepFieldRole::rebalance_source)
        .synchronize_at_end(true)
        .build();

    rt.start_timing();
    for (int it = 0; it < 5; ++it) {
        rt.set_simulated_times({0.01, 0.02});
        rt.execute(step);
    }
    const double wall = rt.stop_timing();
    assert(wall >= 0.0);

    const auto report = rt.timing_report();
    assert(report.iterations == 5);
    assert(report.balance.empty());
    assert(rt.partitions()[0].element_count == 500);
    assert(rt.partitions()[1].element_count == 500);

    const std::string csv_path =
        "/tmp/harmony_sim_off_" + std::to_string(static_cast<long>(getpid())) + ".csv";
    rt.write_timing_report(csv_path);
    if (rank == 0) {
        std::ifstream ifs(csv_path);
        assert(ifs.is_open());
        const std::string content((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
        assert(content.find(",5," + std::to_string(rt.size()) + ",2,0,0,0\n") != std::string::npos);
        std::remove(csv_path.c_str());
    }
}

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    test_raii_scoped_handle_lifecycle();
    test_raii_mid_loop_exception_and_rollback();
    test_simulated_discovery_and_partitioning(argc, argv);
    test_simulated_rebalance_to(argc, argv);
    test_simulated_times_and_auto_rebalance(argc, argv);
    test_simulated_balance_off_counts_iterations(argc, argv);

    if (rank == 0) {
        std::cout << "[PASS] test_runtime_simulated\n";
    }

    MPI_Finalize();
    return 0;
}
