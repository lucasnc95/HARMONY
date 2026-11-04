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
#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>

namespace {

void test_accept_correct_dimensions(dcl::Runtime& rt) {
    std::cout << "[TEST] Running test_accept_correct_dimensions..." << std::endl;
    rt.set_simulated_devices_count(2);

    dcl::TopoMetrics m;
    m.pcie_latency_ns = {4200.5, 4350.2};
    m.pcie_bandwidth_gbps = {15.8, 15.6};
    m.mpi_latency_ns = {0.0};
    m.mpi_bandwidth_gbps = {0.0};
    m.memory_contention_factor = {1.0, 1.25};

    rt.set_topo_metrics(m);

    const auto& stored = rt.topo_metrics();
    assert(stored.has_value());
    assert(stored->pcie_latency_ns.size() == 2);
    assert(std::fabs(stored->pcie_latency_ns[0] - 4200.5) < 1e-6);
    assert(stored->mpi_latency_ns.size() == 1);
    std::cout << "  -> PASS" << std::endl;
}

void test_reject_wrong_dimensions(dcl::Runtime& rt) {
    std::cout << "[TEST] Running test_reject_wrong_dimensions..." << std::endl;
    rt.set_simulated_devices_count(2);

    dcl::TopoMetrics base;
    base.pcie_latency_ns = {4200.0, 4300.0};
    base.pcie_bandwidth_gbps = {15.0, 15.0};
    base.mpi_latency_ns = {0.0};
    base.mpi_bandwidth_gbps = {0.0};

    // Case 1: mpi_latency_ns wrong size (3 instead of 4)
    {
        dcl::TopoMetrics bad = base;
        bad.mpi_latency_ns = {0.0, 1000.0, 1000.0};
        bool caught = false;
        try {
            rt.set_topo_metrics(bad);
        } catch (const dcl::Error&) {
            caught = true;
        }
        assert(caught && "Expected dcl::Error for invalid mpi_latency_ns dimension");
    }

    // Case 2: mpi_bandwidth_gbps wrong size (5 instead of 4)
    {
        dcl::TopoMetrics bad = base;
        bad.mpi_bandwidth_gbps = {0.0, 3.0, 3.0, 0.0, 0.0};
        bool caught = false;
        try {
            rt.set_topo_metrics(bad);
        } catch (const dcl::Error&) {
            caught = true;
        }
        assert(caught && "Expected dcl::Error for invalid mpi_bandwidth_gbps dimension");
    }

    // Case 3: pcie_latency_ns wrong size (3 instead of 2)
    {
        dcl::TopoMetrics bad = base;
        bad.pcie_latency_ns = {4200.0, 4300.0, 4400.0};
        bool caught = false;
        try {
            rt.set_topo_metrics(bad);
        } catch (const dcl::Error&) {
            caught = true;
        }
        assert(caught && "Expected dcl::Error for invalid pcie_latency_ns dimension");
    }

    // Case 4: memory_contention_factor wrong size (1 instead of 2)
    {
        dcl::TopoMetrics bad = base;
        bad.memory_contention_factor = {1.5};
        bool caught = false;
        try {
            rt.set_topo_metrics(bad);
        } catch (const dcl::Error&) {
            caught = true;
        }
        assert(caught && "Expected dcl::Error for invalid memory_contention_factor dimension");
    }
    std::cout << "  -> PASS" << std::endl;
}

void test_contention_factor_validation(dcl::Runtime& rt) {
    std::cout << "[TEST] Running test_contention_factor_validation..." << std::endl;
    rt.set_simulated_devices_count(2);

    dcl::TopoMetrics m;
    m.pcie_latency_ns = {4000.0, 4000.0};
    m.pcie_bandwidth_gbps = {15.0, 15.0};
    m.mpi_latency_ns = (rt.size() == 1 ? std::vector<double>{0.0} : std::vector<double>{0.0, 1000.0, 1000.0, 0.0});
    m.mpi_bandwidth_gbps = (rt.size() == 1 ? std::vector<double>{0.0} : std::vector<double>{0.0, 3.0, 3.0, 0.0});
    m.memory_contention_factor = {0.95, 1.0}; // 0.95 < 1.0 must fail

    bool caught = false;
    try {
        rt.set_topo_metrics(m);
    } catch (const dcl::Error&) {
        caught = true;
    }
    assert(caught && "Expected dcl::Error for memory_contention_factor < 1.0");
    m.memory_contention_factor = {std::numeric_limits<double>::quiet_NaN(), 1.0};
    caught = false;
    try {
        rt.set_topo_metrics(m);
    } catch (const dcl::Error&) {
        caught = true;
    }
    assert(caught && "Expected dcl::Error for non-finite contention");
    std::cout << "  -> PASS" << std::endl;
}

void test_device_numa_node_validation(dcl::Runtime& rt) {
    std::cout << "[TEST] Running test_device_numa_node_validation..." << std::endl;
    rt.set_simulated_devices_count(2);

    dcl::TopoMetrics m;
    m.pcie_latency_ns = {4000.0, 4000.0};
    m.pcie_bandwidth_gbps = {15.0, 15.0};
    m.mpi_latency_ns = (rt.size() == 1 ? std::vector<double>{0.0} : std::vector<double>{0.0, 1000.0, 1000.0, 0.0});
    m.mpi_bandwidth_gbps = (rt.size() == 1 ? std::vector<double>{0.0} : std::vector<double>{0.0, 3.0, 3.0, 0.0});
    m.numa_distance = {10, 20, 20, 10}; // 2x2 -> num_nodes = 2
    m.device_numa_node = {0, 5}; // node 5 is out of range [0, 2)

    bool caught = false;
    try {
        rt.set_topo_metrics(m);
    } catch (const dcl::Error&) {
        caught = true;
    }
    assert(caught && "Expected dcl::Error for device_numa_node out of range");
    std::cout << "  -> PASS" << std::endl;
}

void test_save_load_round_trip(dcl::Runtime& rt) {
    std::cout << "[TEST] Running test_save_load_round_trip..." << std::endl;
    const std::string test_file = "test_roundtrip_" + std::to_string(rt.rank()) + ".json";

    dcl::TopoMetrics original;
    original.pcie_latency_ns = {3120.25, 4560.75};
    original.pcie_bandwidth_gbps = {15.75, 12.5};
    original.mpi_latency_ns = {0.0, 850.5, 850.5, 0.0};
    original.mpi_bandwidth_gbps = {0.0, 10.25, 10.25, 0.0};
    original.numa_distance = {10, 21, 21, 10};
    original.device_numa_node = {0, 1};
    original.memory_contention_factor = {1.0, 1.5};
    original.thermal_tdp_watts = {250.0, 300.0};
    original.current_power_watts = {115.5, 198.2};
    original.synthetic = true;

    dcl::save_topo_metrics(original, test_file);
    dcl::TopoMetrics loaded = dcl::load_topo_metrics(test_file);
    assert(loaded.synthetic);

    assert(loaded.pcie_latency_ns.size() == original.pcie_latency_ns.size());
    for (std::size_t i = 0; i < loaded.pcie_latency_ns.size(); ++i) {
        assert(std::fabs(loaded.pcie_latency_ns[i] - original.pcie_latency_ns[i]) < 1e-6);
        assert(std::fabs(loaded.pcie_bandwidth_gbps[i] - original.pcie_bandwidth_gbps[i]) < 1e-6);
    }

    assert(loaded.mpi_latency_ns.size() == original.mpi_latency_ns.size());
    for (std::size_t i = 0; i < loaded.mpi_latency_ns.size(); ++i) {
        assert(std::fabs(loaded.mpi_latency_ns[i] - original.mpi_latency_ns[i]) < 1e-6);
        assert(std::fabs(loaded.mpi_bandwidth_gbps[i] - original.mpi_bandwidth_gbps[i]) < 1e-6);
    }

    assert(loaded.numa_distance == original.numa_distance);
    assert(loaded.device_numa_node == original.device_numa_node);

    assert(loaded.memory_contention_factor.size() == original.memory_contention_factor.size());
    for (std::size_t i = 0; i < loaded.memory_contention_factor.size(); ++i) {
        assert(std::fabs(loaded.memory_contention_factor[i] - original.memory_contention_factor[i]) < 1e-6);
        assert(std::fabs(loaded.thermal_tdp_watts[i] - original.thermal_tdp_watts[i]) < 1e-6);
        assert(std::fabs(loaded.current_power_watts[i] - original.current_power_watts[i]) < 1e-6);
    }

    std::remove(test_file.c_str());
    std::cout << "  -> PASS" << std::endl;
}

void test_error_handling(dcl::Runtime& rt) {
    std::cout << "[TEST] Running test_error_handling..." << std::endl;
    // Non-existent file
    bool caught = false;
    try {
        (void)dcl::load_topo_metrics("non_existent_file_xyz_123.json");
    } catch (const dcl::Error&) {
        caught = true;
    }
    assert(caught && "Expected dcl::Error on missing file");

    auto check_bad_json = [&](const std::string& json_content, const std::string& msg) {
        const std::string bad_file = "test_bad_" + std::to_string(rt.rank()) + ".json";
        std::ofstream(bad_file) << json_content;
        bool caught = false;
        try {
            (void)dcl::load_topo_metrics(bad_file);
        } catch (const dcl::Error&) {
            caught = true;
        }
        std::remove(bad_file.c_str());
        if (!caught) {
            std::cerr << "Failed to catch error for: " << msg << std::endl;
            assert(false && "Failed to catch expected parser error");
        }
    };

    check_bad_json("{ \"pcie_latency_ns\": [ 123.4, abc ] }", "Expected dcl::Error on malformed JSON");
    check_bad_json("{ \"pcie_latency_ns\": [ 123.4, 456.7 ", "Expected dcl::Error on broken array (truncated)");
    check_bad_json("{ \"pcie_latency_ns\": [ NaN, 456.7 ] }", "Expected dcl::Error on NaN");
    check_bad_json("{ \"pcie_latency_ns\": [ inf, 456.7 ] }", "Expected dcl::Error on infinity");
    check_bad_json("{ \"pcie_latency_ns\": [ 1.2.3 ] }", "Expected dcl::Error on malformed number");
    check_bad_json("{ \"numa_distance\": [ 1, 2.5, 3, 4 ] }", "Expected dcl::Error on float parsed as int");
    check_bad_json("{ \"pcie_latency_ns\": [ 123.4 ] \"mpi_latency_ns\": [ 0.0 ] }", "Expected dcl::Error on missing comma between keys");
    check_bad_json("{ \"pcie_latency_ns\": [ 123.4 ] } trailing_trash", "Expected dcl::Error on trailing characters");
    check_bad_json("{ \"pcie_latency_ns\"", "Expected dcl::Error on truncated key/value");
    check_bad_json("{ \"pcie_latency_ns\": [1,] }", "Trailing array comma");
    check_bad_json("{ \"pcie_latency_ns\": [+1] }", "Leading plus sign");
    check_bad_json("{ \"pcie_latency_ns\": [01] }", "Leading zero");
    check_bad_json("{ \"pcie_latency_ns\": [1], \"pcie_latency_ns\": [2] }", "Duplicate key");
    check_bad_json("{ \"unknown\": [true,] }", "Invalid unknown value");
    check_bad_json("{ \"unknown\": \"bad\\q\" }", "Invalid string escape");

    // This should NOT throw an error, it's valid JSON!
    // But check_bad_json expects an error. Let's write a positive test for skip_json_value.
    const std::string good_file = "test_good_skip_" + std::to_string(rt.rank()) + ".json";
    std::ofstream(good_file) << "{ \"unknown\": [\"[\", \"}\"], \"pcie_latency_ns\": [ 123.4 ] }";
    try {
        (void)dcl::load_topo_metrics(good_file);
    } catch (const dcl::Error& e) {
        std::cerr << "Failed to parse valid JSON with brackets in string: " << e.what() << std::endl;
        assert(false && "Parser rejected valid JSON due to poor skip_json_value");
    }
    std::remove(good_file.c_str());

    std::cout << "  -> PASS" << std::endl;
}

void test_helpers() {
    std::cout << "[TEST] Running test_helpers..." << std::endl;
    dcl::TopoMetrics m;
    m.pcie_latency_ns = {5000.0, 5000.0};
    m.pcie_bandwidth_gbps = {10.0, 10.0};
    m.numa_distance = {10, 20, 20, 10};
    m.device_numa_node = {0, 1};
    m.memory_contention_factor = {1.0, 2.0};

    assert(dcl::numa_distance_between(m, 0, 0) == 10);
    assert(dcl::numa_distance_between(m, 0, 1) == 20);

    assert(std::fabs(dcl::adjusted_capacity(m, 0, 1000.0) - 1000.0) < 1e-6);
    assert(std::fabs(dcl::adjusted_capacity(m, 1, 1000.0) - 500.0) < 1e-6);

    double same_cost = dcl::estimate_migration_cost_bytes(m, 0, 0, 1024 * 1024);
    double diff_cost = dcl::estimate_migration_cost_bytes(m, 0, 1, 1024 * 1024);
    assert(diff_cost > same_cost);
    std::cout << "  -> PASS" << std::endl;
}

void test_topology_larger_than_local(dcl::Runtime& rt) {
    std::cout << "[TEST] Running test_topology_larger_than_local..." << std::endl;
    // Set simulated devices to 4 (representing 2 per rank for 2 ranks, for example)
    // The local rank might only have 2, but the topology should represent all 4.
    rt.set_simulated_devices_count(4);

    dcl::TopoMetrics m;
    m.pcie_latency_ns = {4000.0, 4000.0, 4000.0, 4000.0};
    m.pcie_bandwidth_gbps = {15.0, 15.0, 15.0, 15.0};
    m.mpi_latency_ns = {0.0};
    m.mpi_bandwidth_gbps = {0.0};
    m.memory_contention_factor = {1.0, 1.0, 1.0, 1.0};

    // Before the fix, this would fail if the local rank had fewer than 4 devices
    // and simulated_total_devices_ wasn't used or fell back incorrectly.
    // Now it correctly uses the global device count.
    rt.set_topo_metrics(m);

    const auto& stored = rt.topo_metrics();
    assert(stored.has_value());
    assert(stored->pcie_latency_ns.size() == 4);
    std::cout << "  -> PASS" << std::endl;
}

} // namespace

int main(int argc, char** argv) {
    dcl::Runtime rt = dcl::Runtime::create(argc, argv);

    test_accept_correct_dimensions(rt);
    test_reject_wrong_dimensions(rt);
    test_contention_factor_validation(rt);
    test_device_numa_node_validation(rt);
    test_save_load_round_trip(rt);
    test_error_handling(rt);
    test_helpers();
    test_topology_larger_than_local(rt);

    std::cout << "\nAll test_topo_metrics unit tests PASSED successfully!" << std::endl;
    MPI_Finalize();
    return 0;
}
