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
#include "dcl/types.hpp"
#include "dcl/topo_metrics_io.hpp"
#include "dcl/algorithms.hpp"
#else
#include "../dcl/runtime.hpp"
#include "../dcl/runtime.cpp"
#include "../dcl/types.hpp"
#include "../dcl/topo_metrics_io.hpp"
#include "../dcl/algorithms.hpp"
#endif

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

namespace {

void test_adjusted_capacity() {
    std::cout << "[TEST] Running test_adjusted_capacity (R5.2)..." << std::endl;

    dcl::TopoMetrics m;
    m.memory_contention_factor = {1.0, 2.0, 4.0};

    // Test 1: factor = 2.0 returns exactly half the raw throughput
    const double raw_throughput = 1000.0;
    const double cap1 = dcl::adjusted_capacity(m, 1, raw_throughput);
    assert(std::fabs(cap1 - 500.0) < 1e-9);

    // factor = 1.0 (no contention) returns raw throughput exactly
    const double cap0 = dcl::adjusted_capacity(m, 0, raw_throughput);
    assert(std::fabs(cap0 - 1000.0) < 1e-9);

    // factor = 4.0 returns 250.0 (one quarter)
    const double cap2 = dcl::adjusted_capacity(m, 2, raw_throughput);
    assert(std::fabs(cap2 - 250.0) < 1e-9);

    // Out of bounds / invalid indices must throw dcl::Error
    bool caught_neg = false;
    try {
        (void)dcl::adjusted_capacity(m, -1, raw_throughput);
    } catch (const dcl::Error&) {
        caught_neg = true;
    }
    assert(caught_neg && "Expected dcl::Error for negative device index");

    bool caught_oob = false;
    try {
        (void)dcl::adjusted_capacity(m, 3, raw_throughput);
    } catch (const dcl::Error&) {
        caught_oob = true;
    }
    assert(caught_oob && "Expected dcl::Error for device index >= size");

    // Invalid contention factor < 1.0 must throw dcl::Error
    dcl::TopoMetrics bad_metrics;
    bad_metrics.memory_contention_factor = {0.8};
    bool caught_bad_factor = false;
    try {
        (void)dcl::adjusted_capacity(bad_metrics, 0, raw_throughput);
    } catch (const dcl::Error&) {
        caught_bad_factor = true;
    }
    assert(caught_bad_factor && "Expected dcl::Error for memory_contention_factor < 1.0");

    std::cout << "  -> PASS" << std::endl;
}

void test_apply_power_cap() {
    std::cout << "[TEST] Running test_apply_power_cap (R5.3)..." << std::endl;

    // 2-device case: equal initial load [0.5, 1.0]
    // Device 0: TDP 200W, Current 250W (overloaded by 25%)
    // Device 1: TDP 200W, Current 150W (underloaded, 50W headroom)
    {
        const std::vector<float> initial_loads = {0.5f, 1.0f};
        dcl::TopoMetrics m;
        m.thermal_tdp_watts = {200.0, 200.0};
        m.current_power_watts = {250.0, 150.0};

        const std::vector<float> capped = dcl::apply_power_cap(initial_loads, m, 0.0);

        assert(capped.size() == 2);
        // Overloaded device 0 share must be reduced below initial share 0.5f
        assert(capped[0] < 0.5f);
        // Underloaded device 1 share must be increased above initial share 0.5f
        assert((1.0f - capped[0]) > 0.5f);
        // Specifically: scale = 200/250 = 0.8, share0 = 0.4, share1 = 0.6
        assert(std::fabs(capped[0] - 0.4f) < 1e-5f);
        assert(std::fabs(capped[1] - 1.0f) < 1e-5f);

        // Monotonicity verification
        assert(capped[0] > 0.0f);
        assert(capped[0] < capped[1]);
        assert(std::fabs(capped.back() - 1.0f) < 1e-6f);
    }

    // Reverse 2-device case: Device 0 underloaded, Device 1 overloaded
    {
        const std::vector<float> initial_loads = {0.5f, 1.0f};
        dcl::TopoMetrics m;
        m.thermal_tdp_watts = {200.0, 200.0};
        m.current_power_watts = {150.0, 250.0};

        const std::vector<float> capped = dcl::apply_power_cap(initial_loads, m, 0.0);

        assert(capped.size() == 2);
        // Device 0 share increased
        assert(capped[0] > 0.5f);
        // Device 1 share reduced
        assert((1.0f - capped[0]) < 0.5f);
        assert(std::fabs(capped[0] - 0.6f) < 1e-5f);
        assert(std::fabs(capped[1] - 1.0f) < 1e-5f);

        // Monotonicity verification
        assert(capped[0] > 0.0f);
        assert(capped[0] < capped[1]);
        assert(std::fabs(capped.back() - 1.0f) < 1e-6f);
    }

    // 3-device case: Device 0 overloaded, Devices 1 & 2 underloaded
    {
        const std::vector<float> initial_loads = {0.333333f, 0.666667f, 1.0f};
        dcl::TopoMetrics m;
        m.thermal_tdp_watts = {200.0, 200.0, 200.0};
        m.current_power_watts = {250.0, 150.0, 100.0};

        const std::vector<float> capped = dcl::apply_power_cap(initial_loads, m, 0.0);

        assert(capped.size() == 3);
        const float share0 = capped[0];
        const float share1 = capped[1] - capped[0];
        const float share2 = capped[2] - capped[1];

        // Device 0 share reduced
        assert(share0 < 0.333333f);
        // Devices 1 & 2 shares increased
        assert(share1 > 0.333333f);
        assert(share2 > 0.333333f);
        // Device 2 had more headroom (100W vs 50W), so it should receive more freed share
        assert(share2 > share1);

        // Strictly monotone non-decreasing
        assert(capped[0] > 0.0f);
        assert(capped[0] < capped[1]);
        assert(capped[1] < capped[2]);
        assert(std::fabs(capped.back() - 1.0f) < 1e-6f);
    }

    // System-wide power budget cap
    {
        const std::vector<float> initial_loads = {0.5f, 1.0f};
        dcl::TopoMetrics m;
        m.thermal_tdp_watts = {200.0, 200.0};
        m.current_power_watts = {200.0, 200.0}; // total = 400W

        // A 300W budget is infeasible when every device needs 400W at full load.
        bool rejected = false;
        try {
            (void)dcl::apply_power_cap(initial_loads, m, 300.0);
        } catch (const dcl::Error&) {
            rejected = true;
        }
        assert(rejected);
    }

    // Edge cases
    {
        dcl::TopoMetrics m;
        assert(dcl::apply_power_cap({}, m, 0.0).empty());

        const std::vector<float> single = {1.0f};
        const auto res_single = dcl::apply_power_cap(single, m, 0.0);
        assert(res_single.size() == 1 && std::fabs(res_single[0] - 1.0f) < 1e-6f);

        // No power metrics: returns input loads
        const std::vector<float> two = {0.5f, 1.0f};
        const auto res_two = dcl::apply_power_cap(two, m, 0.0);
        assert(res_two.size() == 2 && res_two[0] == 0.5f && res_two[1] == 1.0f);
    }

    // A4 - Strict rate limit decoupling test (total system saturation)
    {
        const std::vector<float> initial_loads = {0.5f, 1.0f};
        dcl::TopoMetrics m;
        m.thermal_tdp_watts = {200.0, 200.0};
        m.current_power_watts = {400.0, 400.0}; // Both saturated at 2x
        
        bool rejected = false;
        try {
            (void)dcl::apply_power_cap(initial_loads, m, 0.0);
        } catch (const dcl::Error&) {
            rejected = true;
        }
        assert(rejected);
    }

    std::cout << "  -> PASS" << std::endl;
}

void test_contention_factor_validation(dcl::Runtime& rt) {
    std::cout << "[TEST] Running test_contention_factor_validation (R5.1)..." << std::endl;
    rt.set_simulated_devices_count(2);

    dcl::TopoMetrics base;
    base.pcie_latency_ns = {4000.0, 4000.0};
    base.pcie_bandwidth_gbps = {15.0, 15.0};
    base.mpi_latency_ns = (rt.size() == 1 ? std::vector<double>{0.0} : std::vector<double>{0.0, 1000.0, 1000.0, 0.0});
    base.mpi_bandwidth_gbps = (rt.size() == 1 ? std::vector<double>{0.0} : std::vector<double>{0.0, 3.0, 3.0, 0.0});

    // Factor 0.8 < 1.0 must fail
    {
        dcl::TopoMetrics bad = base;
        bad.memory_contention_factor = {0.8, 1.0};
        bool caught = false;
        try {
            rt.set_topo_metrics(bad);
        } catch (const dcl::Error&) {
            caught = true;
        }
        assert(caught && "Expected dcl::Error for memory_contention_factor < 1.0");
    }

    // Factor 0.999 < 1.0 must fail
    {
        dcl::TopoMetrics bad = base;
        bad.memory_contention_factor = {1.0, 0.999};
        bool caught = false;
        try {
            rt.set_topo_metrics(bad);
        } catch (const dcl::Error&) {
            caught = true;
        }
        assert(caught && "Expected dcl::Error for memory_contention_factor < 1.0");
    }

    // Factor 1.0 and 2.5 >= 1.0 must succeed
    {
        dcl::TopoMetrics good = base;
        good.memory_contention_factor = {1.0, 2.5};
        rt.set_topo_metrics(good);
        assert(rt.topo_metrics().has_value());
    }

    std::cout << "  -> PASS" << std::endl;
}

void test_autobalance_integration(dcl::Runtime& rt) {
    std::cout << "[TEST] Running test_autobalance_integration (R5.4)..." << std::endl;

    // 1. Verify default values of AutoBalancePolicy
    dcl::AutoBalancePolicy default_policy;
    assert(!default_policy.use_contention_adjustment);
    assert(!default_policy.use_power_cap);
    assert(std::fabs(default_policy.power_budget_watts - 0.0) < 1e-9);

    // 2. Configure simulated 2-device runtime with equal initial partitions (500k each)
    rt.set_simulated_devices_count(2);

    dcl::PartitionSpec ps;
    ps.global_elements = 1000000;
    ps.units_per_element = 1;
    ps.bytes_per_unit = 4;
    ps.granularity = 1;
    rt.set_partition(ps);
    assert(rt.partitions().size() == 2);

    dcl::FieldSpec fs;
    fs.name = "state_data";
    fs.global_elements = 1000000;
    fs.units_per_element = 1;
    fs.bytes_per_unit = 4;
    fs.usage = dcl::BufferUsage::read_write;
    fs.redistribution = dcl::RedistributionDependency::proportional;
    dcl::FieldHandle fh = rt.create_field(fs);

    // 3. Test Contention Adjustment:
    //    Device 0 has memory_contention_factor = 1.0 (no contention)
    //    Device 1 has memory_contention_factor = 2.0 (2x contention)
    //    Execution times: Dev 0 takes 0.005s, Dev 1 takes 0.010s
    //    Without contention adjustment, raw throughput ratio is 100M:50M = 2:1 -> 66.7% / 33.3%
    //    With contention adjustment, adjusted throughput is 100M:25M = 4:1 -> 80% / 20%
    dcl::TopoMetrics m;
    m.pcie_latency_ns = {4000.0, 4000.0};
    m.pcie_bandwidth_gbps = {15.0, 15.0};
    m.mpi_latency_ns = (rt.size() == 1 ? std::vector<double>{0.0} : std::vector<double>{0.0, 1000.0, 1000.0, 0.0});
    m.mpi_bandwidth_gbps = (rt.size() == 1 ? std::vector<double>{0.0} : std::vector<double>{0.0, 3.0, 3.0, 0.0});
    m.memory_contention_factor = {1.0, 2.0};
    m.thermal_tdp_watts = {200.0, 200.0};
    m.current_power_watts = {150.0, 150.0};
    rt.set_topo_metrics(m);

    rt.set_simulated_times({0.005, 0.010});

    dcl::AutoBalancePolicy policy;
    policy.mode = dcl::BalanceMode::dynamic_threshold;
    policy.threshold = 0.01f;
    policy.use_contention_adjustment = true;
    policy.use_power_cap = false;

    const bool rebalanced = rt.maybe_rebalance_from_timings({fh}, policy);
    assert(rebalanced && "Expected rebalance due to contention difference");

    const auto parts = rt.partitions();
    assert(parts.size() == 2);
    // Device 0 partition should have 80% of the work (800,000 elements)
    // Device 1 partition should have 20% of the work (200,000 elements)
    assert(parts[0].element_count == 800000);
    assert(parts[1].element_count == 200000);
    assert(parts[0].element_count + parts[1].element_count == 1000000);

    // 4. Test Power Cap Integration:
    //    Now Device 0 is thermally overloaded (300W > 200W TDP)
    //    Device 1 is cool (100W < 200W TDP)
    //    Power capping must reduce Device 0 share and shift load back to Device 1.
    m.current_power_watts = {220.0, 50.0};
    rt.set_topo_metrics(m);
    rt.set_simulated_times({0.010, 0.005});

    policy.use_power_cap = true;
    const bool power_rebalanced = rt.maybe_rebalance_from_timings({fh}, policy);
    assert(power_rebalanced && "Expected rebalance due to power cap");

    const auto parts_power = rt.partitions();
    assert(parts_power.size() == 2);
    // Device 0 share was reduced relative to 800,000
    assert(parts_power[0].element_count < parts[0].element_count);
    // Device 1 share was increased relative to 200,000
    assert(parts_power[1].element_count > parts[1].element_count);
    assert(parts_power[0].element_count + parts_power[1].element_count == 1000000);

    // 5. Test overloaded function signature with individual arguments
    rt.set_simulated_times({0.010, 0.030});
    const bool overload_call = rt.maybe_rebalance_from_timings({fh}, 0.01f, 0.50, true, true, 0.0);
    (void)overload_call;

    // 6. An infeasible cap must fail collectively instead of delaying the process.
    m.current_power_watts = {400.0, 400.0}; 
    rt.set_topo_metrics(m);
    
    // Set simulated times 0.01 and 0.01
    rt.set_simulated_times({0.010, 0.010});
    policy.use_power_cap = true;
    policy.use_contention_adjustment = false;
    
    bool rejected = false;
    try {
        rt.maybe_rebalance_from_timings({fh}, policy);
    } catch (const dcl::Error&) {
        rejected = true;
    }
    assert(rejected);

    // Cleanup simulated times
    rt.clear_simulated_times();

    std::cout << "  -> PASS" << std::endl;
}

} // namespace

int main(int argc, char** argv) {
    dcl::Runtime rt = dcl::Runtime::create(argc, argv);

    test_adjusted_capacity();
    test_apply_power_cap();
    test_contention_factor_validation(rt);
    test_autobalance_integration(rt);

    std::cout << "\nAll test_contention_power unit tests PASSED successfully!" << std::endl;
    MPI_Finalize();
    return 0;
}
