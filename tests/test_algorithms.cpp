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
#include "dcl/algorithms.hpp"
#else
#include "../dcl/runtime.hpp"
#include "../dcl/runtime.cpp"
#include "../dcl/algorithms.hpp"
#endif

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

namespace {

dcl::DevicePartition make_part(int g_idx, int rank, int loc_idx, std::size_t off, std::size_t cnt) {
    dcl::DevicePartition p;
    p.device_global_index = g_idx;
    p.owning_rank = rank;
    p.local_index = loc_idx;
    p.global_offset = off;
    p.element_count = cnt;
    return p;
}

void test_erad_matches_throughput() {
    std::cout << "[TEST] Running test_erad_matches_throughput (Test 1)..." << std::endl;

    // Case A: 2 partitions with unequal throughput (ratio 2:1)
    {
        std::vector<dcl::DevicePartition> parts = {
            make_part(0, 0, 0, 0, 500000),
            make_part(1, 0, 1, 500000, 500000)
        };
        std::vector<double> times = {0.005, 0.010};

        std::vector<float> erad = dcl::erad_loads(times, parts);
        std::vector<float> orig = dcl::detail::compute_loads_from_partition_throughput(times, parts);

        assert(erad.size() == 2);
        assert(orig.size() == 2);
        assert(std::fabs(erad[0] - orig[0]) < 1e-6f);
        assert(std::fabs(erad[1] - orig[1]) < 1e-6f);
        assert(std::fabs(erad[0] - (2.0f / 3.0f)) < 1e-5f);
        assert(erad.back() == 1.0f);
    }

    // Case B: 3 partitions with varying elements and times
    {
        std::vector<dcl::DevicePartition> parts = {
            make_part(0, 0, 0, 0, 300000),
            make_part(1, 0, 1, 300000, 400000),
            make_part(2, 0, 2, 700000, 300000)
        };
        std::vector<double> times = {0.003, 0.002, 0.005};

        std::vector<float> erad = dcl::erad_loads(times, parts);
        std::vector<float> orig = dcl::detail::compute_loads_from_partition_throughput(times, parts);

        assert(erad.size() == 3);
        assert(orig.size() == 3);
        for (std::size_t i = 0; i < erad.size(); ++i) {
            assert(std::fabs(erad[i] - orig[i]) < 1e-6f);
            if (i > 0) {
                assert(erad[i] >= erad[i - 1]);
            }
        }
        assert(erad.back() == 1.0f);
    }

    // Case C: 4 partitions with equal elements and times
    {
        std::vector<dcl::DevicePartition> parts = {
            make_part(0, 0, 0, 0, 250000),
            make_part(1, 0, 1, 250000, 250000),
            make_part(2, 0, 2, 500000, 250000),
            make_part(3, 0, 3, 750000, 250000)
        };
        std::vector<double> times = {0.01, 0.01, 0.01, 0.01};

        std::vector<float> erad = dcl::erad_loads(times, parts);
        std::vector<float> orig = dcl::detail::compute_loads_from_partition_throughput(times, parts);

        assert(erad.size() == 4);
        assert(orig.size() == 4);
        for (std::size_t i = 0; i < erad.size(); ++i) {
            assert(std::fabs(erad[i] - orig[i]) < 1e-6f);
        }
        assert(std::fabs(erad[0] - 0.25f) < 1e-5f);
        assert(std::fabs(erad[1] - 0.50f) < 1e-5f);
        assert(std::fabs(erad[2] - 0.75f) < 1e-5f);
        assert(erad[3] == 1.0f);
    }

    // Case D: Edge cases (empty, size mismatch)
    {
        std::vector<dcl::DevicePartition> empty_parts;
        std::vector<double> empty_times;
        assert(dcl::erad_loads(empty_times, empty_parts).empty());

        std::vector<double> mismatch_times = {0.01};
        std::vector<dcl::DevicePartition> parts = {
            make_part(0, 0, 0, 0, 100),
            make_part(1, 0, 1, 100, 100)
        };
        assert(dcl::erad_loads(mismatch_times, parts).empty());
    }

    std::cout << "  -> PASS" << std::endl;
}

void test_hwtopolb_zero_perturbation() {
    std::cout << "[TEST] Running test_hwtopolb_zero_perturbation (Test 2)..." << std::endl;

    std::vector<dcl::DevicePartition> parts = {
        make_part(0, 0, 0, 0, 400000),
        make_part(1, 0, 1, 400000, 600000)
    };
    std::vector<double> times = {0.008, 0.004};

    std::vector<float> erad = dcl::erad_loads(times, parts);
    std::vector<float> hw0 = dcl::hwtopolb_loads(times, parts, 0.0f);

    assert(erad.size() == hw0.size());
    for (std::size_t i = 0; i < erad.size(); ++i) {
        assert(erad[i] == hw0[i]);
    }

    // Negative perturbation factor also treated as unperturbed
    std::vector<float> hw_neg = dcl::hwtopolb_loads(times, parts, -0.05f);
    assert(erad.size() == hw_neg.size());
    for (std::size_t i = 0; i < erad.size(); ++i) {
        assert(erad[i] == hw_neg[i]);
    }

    // 3 partitions case
    std::vector<dcl::DevicePartition> parts3 = {
        make_part(0, 0, 0, 0, 200000),
        make_part(1, 0, 1, 200000, 300000),
        make_part(2, 0, 2, 500000, 500000)
    };
    std::vector<double> times3 = {0.002, 0.003, 0.005};
    std::vector<float> erad3 = dcl::erad_loads(times3, parts3);
    std::vector<float> hw3 = dcl::hwtopolb_loads(times3, parts3, 0.0f);

    assert(erad3.size() == hw3.size());
    for (std::size_t i = 0; i < erad3.size(); ++i) {
        assert(erad3[i] == hw3[i]);
    }

    std::cout << "  -> PASS" << std::endl;
}

void test_hwtopolb_stochastic_perturbation() {
    std::cout << "[TEST] Running test_hwtopolb_stochastic_perturbation (Test 3)..." << std::endl;

    std::vector<dcl::DevicePartition> parts = {
        make_part(0, 0, 0, 0, 1000000),
        make_part(1, 0, 1, 1000000, 1000000),
        make_part(2, 0, 2, 2000000, 1000000)
    };
    std::vector<double> times = {0.01, 0.01, 0.01};

    std::vector<float> erad = dcl::erad_loads(times, parts);
    assert(erad.size() == 3);

    const float perturbation = 0.20f;
    int differ_count = 0;
    const int total_runs = 100;

    for (int run = 0; run < total_runs; ++run) {
        std::vector<float> hw = dcl::hwtopolb_loads(times, parts, perturbation);
        assert(hw.size() == parts.size());
        assert(hw.back() == 1.0f);

        // Verify monotone non-decreasing and valid bounds
        for (std::size_t i = 0; i < hw.size(); ++i) {
            assert(std::isfinite(hw[i]));
            assert(hw[i] >= 0.0f);
            assert(hw[i] <= 1.0f);
            if (i > 0) {
                assert(hw[i] >= hw[i - 1]);
            }
        }

        // Check if this run differs from ERAD
        bool differs = false;
        for (std::size_t i = 0; i < hw.size(); ++i) {
            if (std::fabs(hw[i] - erad[i]) > 1e-5f) {
                differs = true;
                break;
            }
        }
        if (differs) {
            ++differ_count;
        }
    }

    std::cout << "  Stochastic runs differing from ERAD: " << differ_count << " / " << total_runs << std::endl;
    // Over 100 runs with 0.20 perturbation on 3 devices, at least one must differ (in fact almost all will differ)
    assert(differ_count > 0 && "Expected at least one stochastic run to differ from ERAD");

    std::cout << "  -> PASS" << std::endl;
}

void test_runtime_rebalance_consensus(int& argc, char**& argv) {
    dcl::Runtime runtime = dcl::Runtime::create(argc, argv);
    if (runtime.size() != 2) return;
    runtime.set_simulated_devices_count(2);
    dcl::PartitionSpec spec;
    spec.global_elements = 100;
    spec.units_per_element = 1;
    spec.bytes_per_unit = sizeof(float);
    runtime.set_partition(spec);

    const std::vector<float> local_proposal =
        runtime.rank() == 0 ? std::vector<float>{0.7f, 1.0f}
                            : std::vector<float>{0.2f, 1.0f};
    runtime.rebalance_to(local_proposal);
    assert(runtime.partitions()[0].element_count == 70);
    assert(runtime.partitions()[1].element_count == 30);

    bool rejected = false;
    try {
        runtime.rebalance_to(runtime.rank() == 0
            ? std::vector<float>{} : std::vector<float>{0.5f, 1.0f});
    } catch (const dcl::Error&) {
        rejected = true;
    }
    assert(rejected);
}

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    test_erad_matches_throughput();
    test_hwtopolb_zero_perturbation();
    test_hwtopolb_stochastic_perturbation();
    test_runtime_rebalance_consensus(argc, argv);

    std::cout << "\nAll test_algorithms unit tests PASSED successfully!" << std::endl;
    MPI_Finalize();
    return 0;
}
