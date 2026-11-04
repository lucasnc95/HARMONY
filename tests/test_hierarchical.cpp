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

// Helper: get current MPI rank in MPI_COMM_WORLD
static int get_rank() {
    int r = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &r);
    return r;
}

void test_hierarchical_timing_imbalance(dcl::Runtime& rt) {
    const int rank = get_rank();
    if (rank == 0) std::cout << "[TEST] Running test_hierarchical_timing_imbalance..." << std::endl;

    // 1. Configure simulated 2-device runtime
    rt.set_simulated_devices_count(2);

    dcl::PartitionSpec ps;
    ps.global_elements = 1000000;
    ps.units_per_element = 1;
    ps.bytes_per_unit = 4;
    ps.granularity = 1;
    rt.set_partition(ps);

    // Initial partitions: [0, 500000) on dev 0, [500000, 1000000) on dev 1
    assert(rt.partitions().size() == 2);
    assert(rt.partitions()[0].element_count == 500000);
    assert(rt.partitions()[1].element_count == 500000);

    // 2. Inject simulated kernel times to simulate imbalance:
    //    Device 0 time = 10.0 ms (0.010 s)
    //    Device 1 time = 20.0 ms (0.020 s)
    rt.set_simulated_times({0.010, 0.020});

    // 3. Capture stdout on rank 0 to verify [HIER] log lines appear
    std::stringstream captured;
    std::streambuf* old_buf = nullptr;
    if (rank == 0) {
        old_buf = std::cout.rdbuf(captured.rdbuf());
    }

    // 4. Call hierarchical balancing path
    const bool ok = rt.maybe_rebalance_hierarchical();

    std::string out_str;
    if (rank == 0) {
        std::cout.rdbuf(old_buf);
        out_str = captured.str();
        // Echo captured output so test logs are visible
        std::cout << out_str;
    }

    assert(ok && "Expected maybe_rebalance_hierarchical to return true");

    // 5. Verify all [HIER] log lines appear — only rank 0 emits them
    if (rank == 0) {
        assert(out_str.find("[HIER] intra-node loads:") != std::string::npos &&
               "Expected log line '[HIER] intra-node loads:' not found");
        assert(out_str.find("[HIER] inter-node loads:") != std::string::npos &&
               "Expected log line '[HIER] inter-node loads:' not found");
        assert(out_str.find("[HIER] final partition:") != std::string::npos &&
               "Expected log line '[HIER] final partition:' not found");
    }

    // 6. Verify resulting partitions reflect per-device timing ratios
    //    Capacities: C0 = 1/10 = 0.1, C1 = 1/20 = 0.05
    //    Proportions: p0 = 0.1 / 0.15 = 2/3 (~66.67%), p1 = 0.05 / 0.15 = 1/3 (~33.33%)
    //    Expected elements: Dev 0 = 666667, Dev 1 = 333333
    const auto& parts = rt.partitions();
    assert(parts.size() == 2);
    assert(parts[0].global_offset == 0);
    assert(parts[0].element_count == 666667);
    assert(parts[1].global_offset == 666667);
    assert(parts[1].element_count == 333333);
    assert(parts[0].element_count + parts[1].element_count == 1000000);

    const double ratio0 = static_cast<double>(parts[0].element_count) / 1000000.0;
    const double ratio1 = static_cast<double>(parts[1].element_count) / 1000000.0;
    assert(std::fabs(ratio0 - (2.0 / 3.0)) < 1e-4);
    assert(std::fabs(ratio1 - (1.0 / 3.0)) < 1e-4);

    const double timing_ratio = static_cast<double>(parts[0].element_count) /
                                static_cast<double>(parts[1].element_count);
    assert(std::fabs(timing_ratio - 2.0) < 1e-3);

    rt.clear_simulated_times();
    if (rank == 0) std::cout << "  -> PASS" << std::endl;
}

void test_hierarchical_reverse_imbalance(dcl::Runtime& rt) {
    const int rank = get_rank();
    if (rank == 0) std::cout << "[TEST] Running test_hierarchical_reverse_imbalance..." << std::endl;

    rt.set_simulated_devices_count(2);

    dcl::PartitionSpec ps;
    ps.global_elements = 1000000;
    ps.units_per_element = 1;
    ps.bytes_per_unit = 4;
    ps.granularity = 1;
    rt.set_partition(ps);

    // Reversed times: Device 0 = 20.0 ms, Device 1 = 10.0 ms
    // Device 1 is faster, so Device 0 gets 1/3 and Device 1 gets 2/3
    rt.set_simulated_times({0.020, 0.010});

    std::stringstream captured;
    std::streambuf* old_buf = nullptr;
    if (rank == 0) {
        old_buf = std::cout.rdbuf(captured.rdbuf());
    }

    const bool ok = rt.maybe_rebalance_hierarchical();

    std::string out_str;
    if (rank == 0) {
        std::cout.rdbuf(old_buf);
        out_str = captured.str();
        std::cout << out_str;
    }

    assert(ok);
    if (rank == 0) {
        assert(out_str.find("[HIER] intra-node loads:") != std::string::npos);
        assert(out_str.find("[HIER] inter-node loads:") != std::string::npos);
        assert(out_str.find("[HIER] final partition:") != std::string::npos);
    }

    const auto& parts = rt.partitions();
    assert(parts.size() == 2);
    assert(parts[0].global_offset == 0);
    assert(parts[0].element_count == 333333);
    assert(parts[1].global_offset == 333333);
    assert(parts[1].element_count == 666667);
    assert(parts[0].element_count + parts[1].element_count == 1000000);

    const double ratio0 = static_cast<double>(parts[0].element_count) / 1000000.0;
    const double ratio1 = static_cast<double>(parts[1].element_count) / 1000000.0;
    assert(std::fabs(ratio0 - (1.0 / 3.0)) < 1e-4);
    assert(std::fabs(ratio1 - (2.0 / 3.0)) < 1e-4);

    rt.clear_simulated_times();
    if (rank == 0) std::cout << "  -> PASS" << std::endl;
}

void test_hierarchical_equal_times(dcl::Runtime& rt) {
    const int rank = get_rank();
    if (rank == 0) std::cout << "[TEST] Running test_hierarchical_equal_times..." << std::endl;

    rt.set_simulated_devices_count(2);

    dcl::PartitionSpec ps;
    ps.global_elements = 1000000;
    ps.units_per_element = 1;
    ps.bytes_per_unit = 4;
    ps.granularity = 1;
    rt.set_partition(ps);

    // Equal times: both devices get 50%
    rt.set_simulated_times({0.015, 0.015});

    std::stringstream captured;
    std::streambuf* old_buf = nullptr;
    if (rank == 0) {
        old_buf = std::cout.rdbuf(captured.rdbuf());
    }

    const bool ok = rt.maybe_rebalance_hierarchical();

    std::string out_str;
    if (rank == 0) {
        std::cout.rdbuf(old_buf);
        out_str = captured.str();
        std::cout << out_str;
    }

    assert(ok);
    if (rank == 0) {
        assert(out_str.find("[HIER] intra-node loads:") != std::string::npos);
        assert(out_str.find("[HIER] inter-node loads:") != std::string::npos);
        assert(out_str.find("[HIER] final partition:") != std::string::npos);
    }

    const auto& parts = rt.partitions();
    assert(parts.size() == 2);
    assert(parts[0].element_count == 500000);
    assert(parts[1].element_count == 500000);

    rt.clear_simulated_times();
    if (rank == 0) std::cout << "  -> PASS" << std::endl;
}

void test_hierarchical_with_granularity(dcl::Runtime& rt) {
    const int rank = get_rank();
    if (rank == 0) std::cout << "[TEST] Running test_hierarchical_with_granularity..." << std::endl;

    rt.set_simulated_devices_count(2);

    dcl::PartitionSpec ps;
    ps.global_elements = 1000000;
    ps.units_per_element = 1;
    ps.bytes_per_unit = 4;
    ps.granularity = 1000; // Alignment to 1000
    rt.set_partition(ps);

    rt.set_simulated_times({0.010, 0.020});

    std::stringstream captured;
    std::streambuf* old_buf = nullptr;
    if (rank == 0) {
        old_buf = std::cout.rdbuf(captured.rdbuf());
    }

    const bool ok = rt.maybe_rebalance_hierarchical();

    std::string out_str;
    if (rank == 0) {
        std::cout.rdbuf(old_buf);
        out_str = captured.str();
        std::cout << out_str;
    }

    assert(ok);
    const auto& parts = rt.partitions();
    assert(parts.size() == 2);

    // Cuts must be aligned to granularity 1000
    assert(parts[0].global_offset % 1000 == 0);
    assert(parts[0].element_count % 1000 == 0);
    assert(parts[1].global_offset % 1000 == 0);
    assert(parts[1].element_count % 1000 == 0);
    assert(parts[0].element_count + parts[1].element_count == 1000000);

    // 2/3 of 1000000 rounded to 1000 is 667000 or 666000
    assert(parts[0].element_count == 667000 || parts[0].element_count == 666000);

    rt.clear_simulated_times();
    if (rank == 0) std::cout << "  -> PASS" << std::endl;
}

void test_hierarchical_with_registered_field(dcl::Runtime& rt) {
    const int rank = get_rank();
    if (rank == 0) std::cout << "[TEST] Running test_hierarchical_with_registered_field..." << std::endl;

    rt.set_simulated_devices_count(2);

    dcl::PartitionSpec ps;
    ps.global_elements = 1000000;
    ps.units_per_element = 1;
    ps.bytes_per_unit = 4;
    ps.granularity = 1;
    rt.set_partition(ps);

    dcl::FieldSpec fs;
    fs.name = "hier_test_field";
    fs.global_elements = 1000000;
    fs.units_per_element = 1;
    fs.bytes_per_unit = 4;
    fs.usage = dcl::BufferUsage::read_write;
    fs.redistribution = dcl::RedistributionDependency::proportional;
    dcl::FieldHandle fh = rt.create_field(fs);

    rt.set_simulated_times({0.010, 0.020});

    std::stringstream captured;
    std::streambuf* old_buf = nullptr;
    if (rank == 0) {
        old_buf = std::cout.rdbuf(captured.rdbuf());
    }

    const bool ok = rt.maybe_rebalance_hierarchical({fh});

    std::string out_str;
    if (rank == 0) {
        std::cout.rdbuf(old_buf);
        out_str = captured.str();
        std::cout << out_str;
    }

    assert(ok);
    if (rank == 0) {
        assert(out_str.find("[HIER] intra-node loads:") != std::string::npos);
        assert(out_str.find("[HIER] inter-node loads:") != std::string::npos);
        assert(out_str.find("[HIER] final partition:") != std::string::npos);
    }

    const auto& parts = rt.partitions();
    assert(parts.size() == 2);
    assert(parts[0].element_count == 666667);
    assert(parts[1].element_count == 333333);

    rt.clear_simulated_times();
    if (rank == 0) std::cout << "  -> PASS" << std::endl;
}

void test_balance_mode_enum_and_policy() {
    const int rank = get_rank();
    if (rank == 0) std::cout << "[TEST] Running test_balance_mode_enum_and_policy..." << std::endl;

    dcl::AutoBalancePolicy pol;
    pol.mode = dcl::BalanceMode::hierarchical;
    pol.interval = 10;
    pol.threshold = 0.05f;

    assert(pol.mode == dcl::BalanceMode::hierarchical);
    assert(pol.interval == 10);

    // Switch statement covering all BalanceMode enumerators without compiler warnings
    int mode_id = -1;
    switch (pol.mode) {
        case dcl::BalanceMode::off:
            mode_id = 0;
            break;
        case dcl::BalanceMode::static_threshold:
            mode_id = 1;
            break;
        case dcl::BalanceMode::dynamic_threshold:
            mode_id = 2;
            break;
        case dcl::BalanceMode::static_profiled:
            mode_id = 3;
            break;
        case dcl::BalanceMode::dynamic_profiled:
            mode_id = 4;
            break;
        case dcl::BalanceMode::hierarchical:
            mode_id = 5;
            break;
    }
    assert(mode_id == 5);

    if (rank == 0) std::cout << "  -> PASS" << std::endl;
}

// ---------------------------------------------------------------------------
// C1 fix test: run with np=2 but only 1 total simulated device.
// Before the fix, rank 1 (no device) received 1e-9 as its time, resulting in
// capacity = 1/1e-9 = 1e9, and almost all elements were assigned to it.
// Because rank 1 has no device to publish via MPI_Allgatherv, those elements
// vanished, making sum(element_count) < global_elements.
// After the fix, rank 1 has capacity = 0, so all elements go to rank 0's device.
//
// Uses a fresh Runtime so that all_device_counts_ starts empty (unaffected by
// earlier tests that set it to [1,1] for 2 total devices).
// Runtime::create checks MPI_Initialized() and skips MPI_Init_thread if MPI is
// already up; the destructor will not call MPI_Finalize (mpi_initialized_by_runtime_=false).
// ---------------------------------------------------------------------------
void test_hysteresis_prevents_oscillation(dcl::Runtime& rt) {
    const int rank = get_rank();
    if (rank == 0) std::cout << "[TEST] Running test_hysteresis_prevents_oscillation..." << std::endl;

    rt.set_simulated_devices_count(2);

    dcl::PartitionSpec ps;
    ps.global_elements = 1000000;
    ps.units_per_element = 1;
    ps.bytes_per_unit = 4;
    ps.granularity = 1;
    rt.set_partition(ps);

    // Initial perfectly balanced state (simulates exactly after a previous balance)
    rt.set_simulated_times({0.100, 0.100});
    rt.maybe_rebalance_hierarchical(); 

    // Now introduce a very tiny imbalance, e.g. 0.100 and 0.102
    rt.set_simulated_times({0.100, 0.102});

    std::stringstream captured;
    std::streambuf* old_buf = nullptr;
    if (rank == 0) {
        old_buf = std::cout.rdbuf(captured.rdbuf());
    }

    const bool ok = rt.maybe_rebalance_hierarchical();

    std::string out_str;
    if (rank == 0) {
        std::cout.rdbuf(old_buf);
        out_str = captured.str();
        std::cout << out_str;
    }

    assert(ok);
    const auto& parts = rt.partitions();
    assert(parts.size() == 2);
    assert(parts[0].element_count == 500000);
    assert(parts[1].element_count == 500000);
    
    if (rank == 0) {
        assert(out_str.find("aborted due to hysteresis") != std::string::npos);
    }

    rt.clear_simulated_times();
    if (rank == 0) std::cout << "  -> PASS" << std::endl;
}

void test_c1_no_element_loss_with_more_ranks_than_devices(int argc, char** argv) {
    const int rank = get_rank();
    int size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        std::cout << "[TEST] Running test_c1_no_element_loss_with_more_ranks_than_devices"
                  << " (size=" << size << ")..." << std::endl;
    }

    // Create a fresh Runtime so all_device_counts_ starts empty.
    dcl::Runtime rt2 = dcl::Runtime::create(argc, argv);

    // 1 total device across 'size' ranks — only rank 0 has a device.
    rt2.set_simulated_devices_count(1);

    dcl::PartitionSpec ps;
    ps.global_elements = 1200000; // reproduces the audit scenario (4 ranks, 3 devices)
    ps.units_per_element = 1;
    ps.bytes_per_unit = 4;
    ps.granularity = 1;
    rt2.set_partition(ps);

    // Only 1 partition exists (1 device).
    const auto& init_parts = rt2.partitions();
    if (rank == 0) {
        assert(init_parts.size() == 1 && "Expected 1 partition for 1 total device");
        assert(init_parts[0].element_count == 1200000 && "Initial partition must cover all elements");
    }

    // Simulate a kernel time only for device 0 (rank 0 owns it).
    // Rank 1+ has no device, so local_max_time must become 0.0 (zero capacity).
    rt2.set_simulated_times({0.010}); // only 1 device has timing

    std::stringstream captured;
    std::streambuf* old_buf = nullptr;
    if (rank == 0) {
        old_buf = std::cout.rdbuf(captured.rdbuf());
    }

    const bool ok = rt2.maybe_rebalance_hierarchical();

    std::string out_str;
    if (rank == 0) {
        std::cout.rdbuf(old_buf);
        out_str = captured.str();
        std::cout << out_str;
    }

    // The rebalance must succeed (return true).
    assert(ok && "maybe_rebalance_hierarchical must return true");

    // Critical invariant: sum(element_count) == global_elements.
    // With the old code this would be 0 (all elements lost); with the fix it's 1200000.
    const auto& parts = rt2.partitions();
    std::size_t covered = 0;
    for (const auto& dp : parts) {
        covered += dp.element_count;
    }
    if (rank == 0) {
        assert(covered == 1200000 &&
               "C1 regression: sum(element_count) != global_elements — elements were lost!");
        assert(parts.size() == 1 && "Expected exactly 1 partition (1 device)");
        // All elements must be assigned to the only device (rank 1 has zero capacity).
        assert(parts[0].element_count == 1200000 &&
               "All elements must go to rank 0's device — rank 1 has zero capacity");
        assert(parts[0].global_offset == 0);
        std::cout << "[C1 CHECK] covered=" << covered << " expected=1200000 OK\n";
    }

    if (rank == 0) std::cout << "  -> PASS" << std::endl;
}

void test_local_rebalance_keeps_global_partition_view(int argc, char** argv) {
    int size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 2) return;

    dcl::Runtime rt = dcl::Runtime::create(argc, argv);
    const int rank = rt.rank();
    rt.set_simulated_devices_count(4);

    dcl::PartitionSpec ps;
    ps.global_elements = 1000000;
    ps.units_per_element = 1;
    ps.bytes_per_unit = sizeof(float);
    ps.granularity = 1;
    rt.set_partition(ps);
    rt.set_simulated_times(rank == 0
        ? std::vector<double>{0.01, 0.02, 0.01, 0.01}
        : std::vector<double>{0.01, 0.01, 0.02, 0.01});

    assert(rt.maybe_rebalance_hierarchical());
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
    assert(next_offset == ps.global_elements);

    MPI_Allreduce(local, minimum, 8, MPI_UNSIGNED_LONG_LONG, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(local, maximum, 8, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
    for (int i = 0; i < 8; ++i) {
        assert(minimum[i] == maximum[i] && "Ranks must share a global partition map");
    }
    if (rank == 0) std::cout << "[TEST] Local rebalance preserves global partition view: PASS\n";
}

void test_consecutive_micro_partitions_are_removed_before_halo(int argc, char** argv) {
    dcl::Runtime rt = dcl::Runtime::create(argc, argv);
    rt.set_simulated_devices_count(3);
    dcl::PartitionSpec ps;
    ps.global_elements = 10;
    ps.units_per_element = 1;
    ps.bytes_per_unit = sizeof(float);
    ps.granularity = 1;
    rt.set_partition(ps);
    rt.rebalance_to({0.1f, 0.2f, 1.0f});
    assert(rt.partitions()[0].element_count == 1);
    assert(rt.partitions()[1].element_count == 1);

    dcl::ExecutionStep step;
    step.invocations.push_back(dcl::KernelInvocation{});
    step.halo.width_elements = 3;
    step.halo.fields.push_back(dcl::FieldHandle{});
    bool reached_kernel_validation = false;
    try {
        rt.execute(step);
    } catch (const dcl::Error& error) {
        reached_kernel_validation = std::string(error.what()).find("Unknown kernel") != std::string::npos;
    }
    assert(reached_kernel_validation);

    std::size_t covered = 0;
    for (const auto& part : rt.partitions()) {
        assert(part.element_count == 0 || part.element_count >= step.halo.width_elements);
        assert(part.global_offset == covered);
        covered += part.element_count;
    }
    assert(covered == ps.global_elements);
    if (rt.rank() == 0) std::cout << "[TEST] Consecutive micro partitions removed before halo: PASS\n";
}

void test_constant_throughput_converges(int& argc, char**& argv) {
    dcl::Runtime rt = dcl::Runtime::create(argc, argv);
    if (rt.size() != 2) return;
    rt.set_simulated_devices_count(4);
    dcl::PartitionSpec ps;
    ps.global_elements = 1200000;
    ps.units_per_element = 1;
    ps.bytes_per_unit = sizeof(float);
    ps.granularity = 1;
    rt.set_partition(ps);

    const std::vector<double> speeds = {60e6, 30e6, 20e6, 10e6};
    auto observed_times = [&]() {
        std::vector<double> times;
        for (std::size_t i = 0; i < speeds.size(); ++i) {
            times.push_back(static_cast<double>(rt.partitions()[i].element_count) / speeds[i]);
        }
        return times;
    };
    rt.set_simulated_times(observed_times());
    assert(rt.maybe_rebalance_hierarchical());
    std::vector<std::size_t> first_counts;
    for (const auto& part : rt.partitions()) first_counts.push_back(part.element_count);

    rt.set_simulated_times(observed_times());
    assert(rt.maybe_rebalance_hierarchical());
    for (std::size_t i = 0; i < first_counts.size(); ++i) {
        assert(rt.partitions()[i].element_count == first_counts[i]);
    }
    if (rt.rank() == 0) std::cout << "[TEST] Constant throughput converges: PASS\n";
}

} // namespace

int main(int argc, char** argv) {
    dcl::Runtime rt = dcl::Runtime::create(argc, argv);
    const int rank = get_rank();

    test_balance_mode_enum_and_policy();
    test_hierarchical_timing_imbalance(rt);
    test_hierarchical_reverse_imbalance(rt);
    test_hierarchical_equal_times(rt);
    test_hierarchical_with_granularity(rt);
    test_hierarchical_with_registered_field(rt);
    test_hysteresis_prevents_oscillation(rt);
    test_c1_no_element_loss_with_more_ranks_than_devices(argc, argv);
    test_local_rebalance_keeps_global_partition_view(argc, argv);
    test_consecutive_micro_partitions_are_removed_before_halo(argc, argv);
    test_constant_throughput_converges(argc, argv);

    if (rank == 0) std::cout << "\nAll test_hierarchical unit tests PASSED successfully!" << std::endl;
    MPI_Finalize();
    return 0;
}
