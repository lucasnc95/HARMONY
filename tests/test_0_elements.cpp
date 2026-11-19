#include "harmony/runtime.hpp"

#include <mpi.h>

#include <cassert>
#include <iostream>
#include <vector>

namespace {

void test_zero_element_rank_recovery(int argc, char** argv) {
    auto rt = harmony::Runtime::create(argc, argv);
    rt.set_verbose(false);
    rt.set_simulated_devices_count(2);
    rt.set_partition({1000000, 1, sizeof(float), 1});

    auto fh = rt.create_field({"recovery_field", 1000000, 1, sizeof(float),
                               harmony::BufferUsage::read_write, nullptr,
                               harmony::RedistributionDependency::proportional});

    harmony::AutoBalancePolicy policy;
    policy.mode = harmony::BalanceMode::hierarchical;
    policy.interval = 1;
    auto step = rt.step("zero_recovery")
        .with_balance(policy)
        .tag_field(fh, harmony::StepFieldRole::rebalance_source)
        .build();

    // Extreme imbalance drives device 0 / rank 0 to 0 elements.
    rt.set_simulated_times({1000000.0, 0.0001});
    rt.execute(step);

    const auto& p1 = rt.partitions();
    assert(p1.size() == 2);
    assert(p1[0].element_count == 0);
    assert(p1[1].element_count == 1000000);
    assert(p1[0].element_count + p1[1].element_count == 1000000);

    // Device 0 becomes fast while holding 0 elements; unit-element floor enables recovery.
    rt.set_simulated_times({0.0001, 10.0});
    rt.execute(step);

    const auto& p2 = rt.partitions();
    assert(p2.size() == 2);
    assert(p2[0].element_count > 0 && "0-element rank must recover without starvation");
    assert(p2[1].element_count > 0);
    assert(p2[0].global_offset == 0);
    assert(p2[1].global_offset == p2[0].element_count);
    assert(p2[0].element_count + p2[1].element_count == 1000000);
}

void test_explicit_zero_partition_recovery(int argc, char** argv) {
    auto rt = harmony::Runtime::create(argc, argv);
    rt.set_verbose(false);
    rt.set_simulated_devices_count(2);
    rt.set_partition({1000000, 1, sizeof(float), 1});

    rt.rebalance_to({0.0f, 1.0f});
    assert(rt.partitions()[0].element_count == 0);
    assert(rt.partitions()[1].element_count == 1000000);

    harmony::AutoBalancePolicy policy;
    policy.mode = harmony::BalanceMode::hierarchical;
    policy.interval = 1;
    auto step = rt.step("explicit_zero_recovery").with_balance(policy).build();

    rt.set_simulated_times({0.0001, 10.0});
    rt.execute(step);
    assert(rt.partitions()[0].element_count > 0);
    assert(rt.partitions()[0].element_count + rt.partitions()[1].element_count == 1000000);
}

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    test_zero_element_rank_recovery(argc, argv);
    test_explicit_zero_partition_recovery(argc, argv);

    if (rank == 0) {
        std::cout << "[PASS] test_0_elements\n";
    }
    MPI_Finalize();
    return 0;
}
