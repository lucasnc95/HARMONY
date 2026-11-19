#include "harmony/runtime.hpp"
#include "harmony/topo_metrics_io.hpp"
#include "harmony/types.hpp"

#include <mpi.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <unistd.h>

namespace {

std::string temp_json_path(int rank, const std::string& tag) {
    return "/tmp/harmony_test_topo_" + std::to_string(static_cast<long>(getpid())) +
           "_r" + std::to_string(rank) + "_" + tag + ".json";
}

void test_save_load_round_trip(int rank) {
    const std::string test_file = temp_json_path(rank, "roundtrip");
    std::remove(test_file.c_str());

    harmony::TopoMetrics original;
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

    harmony::save_topo_metrics(original, test_file);
    harmony::TopoMetrics loaded = harmony::load_topo_metrics(test_file);
    std::remove(test_file.c_str());

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
}

void test_missing_file_throws() {
    bool caught = false;
    try {
        (void)harmony::load_topo_metrics("/tmp/harmony_non_existent_file_xyz_123.json");
    } catch (const harmony::Error&) {
        caught = true;
    }
    assert(caught && "Expected harmony::Error on missing file");
}

void test_json_parser_edge_cases(int rank) {
    auto check_bad_json = [&](const std::string& json_content, const std::string& msg) {
        const std::string bad_file = temp_json_path(rank, "bad");
        {
            std::ofstream ofs(bad_file, std::ios::binary);
            ofs << json_content;
        }
        bool caught = false;
        try {
            (void)harmony::load_topo_metrics(bad_file);
        } catch (const harmony::Error&) {
            caught = true;
        }
        std::remove(bad_file.c_str());
        if (!caught) {
            std::cerr << "Failed to catch error for: " << msg << "\n";
            assert(false && "Failed to catch expected parser error");
        }
    };

    check_bad_json("{ \"pcie_latency_ns\": [ 123.4, abc ] }", "malformed JSON token");
    check_bad_json("{ \"pcie_latency_ns\": [ 123.4, 456.7 ", "broken array (truncated)");
    check_bad_json("{ \"pcie_latency_ns\": [ NaN, 456.7 ] }", "NaN");
    check_bad_json("{ \"pcie_latency_ns\": [ inf, 456.7 ] }", "inf");
    check_bad_json("{ \"pcie_latency_ns\": [ 1.2.3 ] }", "malformed number");
    check_bad_json("{ \"numa_distance\": [ 1, 2.5, 3, 4 ] }", "float parsed as int");
    check_bad_json("{ \"numa_distance\": [ 1, 1e2 ] }", "scientific notation parsed as int");
    check_bad_json("{ \"pcie_latency_ns\": [ 123.4 ] \"mpi_latency_ns\": [ 0.0 ] }", "missing comma between keys");
    check_bad_json("{ \"pcie_latency_ns\": [ 123.4 ] } trailing_trash", "trailing characters");
    check_bad_json("{ \"pcie_latency_ns\"", "truncated key/value");
    check_bad_json("{ \"pcie_latency_ns\": [1,] }", "trailing array comma");
    check_bad_json("{ \"pcie_latency_ns\": [1], }", "trailing object comma");
    check_bad_json("{ \"pcie_latency_ns\": [+1] }", "leading plus sign");
    check_bad_json("{ \"pcie_latency_ns\": [01] }", "leading zero");
    check_bad_json("{ \"pcie_latency_ns\": [-01] }", "negative leading zero");
    check_bad_json("{ \"pcie_latency_ns\": [1], \"pcie_latency_ns\": [2] }", "duplicate key");
    check_bad_json("{ \"unknown\": [true,] }", "invalid unknown value");
    check_bad_json("{ \"unknown\": \"bad\\q\" }", "invalid string escape");
    check_bad_json(std::string("{ \"unknown\": \"line1\nline2\" }"), "unescaped control character");
    check_bad_json("{ \"synthetic\": truex }", "invalid boolean suffix");

    const std::string good_file = temp_json_path(rank, "good_skip");
    {
        std::ofstream ofs(good_file);
        ofs << "{ \"unknown_arr\": [\"[\", \"}\", true, false, null, -0, 1.5e-2], "
            << "\"unknown_obj\": {\"nested\": [1, 2], \"esc\": \"a\\\"b\\\\c\"}, "
            << "\"pcie_latency_ns\": [ 123.4 ] }";
    }
    harmony::TopoMetrics loaded = harmony::load_topo_metrics(good_file);
    std::remove(good_file.c_str());
    assert(loaded.pcie_latency_ns.size() == 1);
    assert(std::fabs(loaded.pcie_latency_ns[0] - 123.4) < 1e-6);

    // Semantic and dimensional validation at JSON deserialization time (load_topo_metrics).
    check_bad_json("{ \"pcie_latency_ns\": [ -1.0 ] }", "negative pcie_latency_ns");
    check_bad_json("{ \"pcie_bandwidth_gbps\": [ 0.0 ] }", "zero pcie_bandwidth_gbps");
    check_bad_json("{ \"pcie_bandwidth_gbps\": [ -5.0 ] }", "negative pcie_bandwidth_gbps");
    check_bad_json("{ \"mpi_latency_ns\": [ -0.5 ], \"mpi_bandwidth_gbps\": [ 0.0 ] }", "negative mpi_latency_ns");
    check_bad_json("{ \"mpi_latency_ns\": [ 0.0 ], \"mpi_bandwidth_gbps\": [ -1.0 ] }", "negative mpi_bandwidth_gbps");
    check_bad_json("{ \"mpi_latency_ns\": [ 0.0, 100.0, 100.0, 0.0 ], \"mpi_bandwidth_gbps\": [ 0.0, 0.0, 10.0, 0.0 ] }",
                   "zero off-diagonal mpi_bandwidth_gbps");
    check_bad_json("{ \"memory_contention_factor\": [ 0.9 ] }", "memory_contention_factor < 1.0");
    check_bad_json("{ \"thermal_tdp_watts\": [ 0.0 ] }", "zero thermal_tdp_watts");
    check_bad_json("{ \"current_power_watts\": [ -2.0 ] }", "negative current_power_watts");
    check_bad_json("{ \"mpi_latency_ns\": [ 0.0, 10.0, 10.0 ], \"mpi_bandwidth_gbps\": [ 0.0, 10.0, 10.0 ] }",
                   "non-square mpi_latency_ns");
    check_bad_json("{ \"mpi_latency_ns\": [ 0.0, 10.0, 10.0, 0.0 ], \"mpi_bandwidth_gbps\": [ 0.0 ] }",
                   "mismatched mpi_latency_ns and mpi_bandwidth_gbps dimensions");
    check_bad_json("{ \"numa_distance\": [ 10, 20, 20 ] }", "non-square numa_distance");
    check_bad_json("{ \"numa_distance\": [ 0, 20, 20, 10 ] }", "non-positive numa_distance");
    check_bad_json("{ \"numa_distance\": [ 30, 10, 10, 30 ] }", "numa_distance diagonal > off-diagonal");
    check_bad_json("{ \"device_numa_node\": [ 0, 1 ] }", "device_numa_node without numa_distance");
    check_bad_json("{ \"numa_distance\": [ 10, 20, 20, 10 ], \"device_numa_node\": [ 0, 2 ] }",
                   "device_numa_node out of range");
    check_bad_json("{ \"pcie_latency_ns\": [ 100.0, 200.0 ], \"pcie_bandwidth_gbps\": [ 15.0 ] }",
                   "mismatched device vector dimensions");
}

void test_save_validation_errors(int rank) {
    const std::string out_file = temp_json_path(rank, "save_err");
    auto check_bad_save = [&](const harmony::TopoMetrics& m, const std::string& path, const char* msg) {
        bool caught = false;
        try {
            harmony::save_topo_metrics(m, path);
        } catch (const harmony::Error&) {
            caught = true;
        }
        std::remove(out_file.c_str());
        if (!caught) {
            std::cerr << "Failed to catch save error for: " << msg << "\n";
            assert(false && "Failed to catch expected save error");
        }
    };

    harmony::TopoMetrics nan_lat;
    nan_lat.pcie_latency_ns = {std::numeric_limits<double>::quiet_NaN()};
    check_bad_save(nan_lat, out_file, "NaN in pcie_latency_ns");

    harmony::TopoMetrics inf_bw;
    inf_bw.mpi_bandwidth_gbps = {std::numeric_limits<double>::infinity()};
    check_bad_save(inf_bw, out_file, "Inf in mpi_bandwidth_gbps");

    harmony::TopoMetrics inf_pow;
    inf_pow.current_power_watts = {-std::numeric_limits<double>::infinity()};
    check_bad_save(inf_pow, out_file, "-Inf in current_power_watts");

    harmony::TopoMetrics valid;
    valid.pcie_latency_ns = {1.0};
    check_bad_save(valid, "/tmp/harmony_nonexistent_dir_99999/metrics.json", "unwritable directory");
}

harmony::TopoMetrics make_valid_metrics(int num_devices, int num_ranks) {
    harmony::TopoMetrics m;
    const std::size_t devs = static_cast<std::size_t>(num_devices);
    const std::size_t ranks = static_cast<std::size_t>(num_ranks);
    m.pcie_latency_ns.assign(devs, 4200.5);
    m.pcie_bandwidth_gbps.assign(devs, 15.8);
    m.mpi_latency_ns.assign(ranks * ranks, 0.0);
    m.mpi_bandwidth_gbps.assign(ranks * ranks, 0.0);
    for (std::size_t s = 0; s < ranks; ++s) {
        for (std::size_t d = 0; d < ranks; ++d) {
            if (s != d) {
                m.mpi_latency_ns[s * ranks + d] = 1000.0;
                m.mpi_bandwidth_gbps[s * ranks + d] = 10.0;
            }
        }
    }
    return m;
}

void test_accept_correct_dimensions(int argc, char** argv) {
    auto rt = harmony::Runtime::create(argc, argv);
    assert(!rt.topo_metrics().has_value());
    rt.set_simulated_devices_count(2);

    harmony::TopoMetrics m = make_valid_metrics(2, rt.size());
    m.pcie_latency_ns = {4200.5, 4350.2};
    m.pcie_bandwidth_gbps = {15.8, 15.6};
    m.memory_contention_factor = {1.0, 1.25};
    m.thermal_tdp_watts = {250.0, 300.0};
    m.current_power_watts = {120.0, 140.0};
    m.numa_distance = {10, 20, 20, 10};
    m.device_numa_node = {0, 1};

    rt.set_topo_metrics(m);

    const auto& stored = rt.topo_metrics();
    assert(stored.has_value());
    assert(stored->pcie_latency_ns.size() == 2);
    assert(std::fabs(stored->pcie_latency_ns[0] - 4200.5) < 1e-6);
    assert(stored->mpi_latency_ns.size() == static_cast<std::size_t>(rt.size() * rt.size()));
}

void test_topology_larger_than_local(int argc, char** argv) {
    auto rt = harmony::Runtime::create(argc, argv);
    rt.set_simulated_devices_count(4);

    harmony::TopoMetrics m = make_valid_metrics(4, rt.size());
    m.memory_contention_factor = {1.0, 1.0, 1.0, 1.0};
    rt.set_topo_metrics(m);

    const auto& stored = rt.topo_metrics();
    assert(stored.has_value());
    assert(stored->pcie_latency_ns.size() == 4);
}

void test_multi_rank_simulated_topology(int argc, char** argv) {
    int comm_size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    if (comm_size != 1) return;

    auto rt = harmony::Runtime::create(argc, argv);
    rt.set_simulated_ranks_for_testing(3);
    rt.set_simulated_devices_count(6);

    harmony::TopoMetrics m = make_valid_metrics(6, 3);
    rt.set_topo_metrics(m);

    const auto& stored = rt.topo_metrics();
    assert(stored.has_value());
    assert(stored->mpi_latency_ns.size() == 9);
    assert(stored->pcie_latency_ns.size() == 6);
}

void expect_set_topo_error(harmony::Runtime& rt, const harmony::TopoMetrics& bad, const char* what) {
    bool caught = false;
    try {
        rt.set_topo_metrics(bad);
    } catch (const harmony::Error&) {
        caught = true;
    }
    if (!caught) {
        std::cerr << "Expected harmony::Error for: " << what << "\n";
        assert(false && "Expected harmony::Error in set_topo_metrics");
    }
}

void test_reject_undiscovered_devices(int argc, char** argv) {
    auto rt = harmony::Runtime::create(argc, argv);
    const harmony::TopoMetrics m = make_valid_metrics(2, rt.size());
    expect_set_topo_error(rt, m, "set_topo_metrics before device discovery");

    rt.set_simulated_devices_count(2);
    rt.set_topo_metrics(m);
    assert(rt.topo_metrics().has_value());

    rt.set_simulated_devices_count(0);
    assert(!rt.topo_metrics().has_value());
    expect_set_topo_error(rt, m, "set_topo_metrics after clearing simulated devices");
}

void test_reject_wrong_dimensions(int argc, char** argv) {
    auto rt = harmony::Runtime::create(argc, argv);
    rt.set_simulated_devices_count(2);
    const harmony::TopoMetrics base = make_valid_metrics(2, rt.size());
    rt.set_topo_metrics(base);

    {
        harmony::TopoMetrics bad = base;
        bad.mpi_latency_ns = {0.0, 1000.0, 1000.0};
        expect_set_topo_error(rt, bad, "mpi_latency_ns size 3");
    }
    {
        harmony::TopoMetrics bad = base;
        bad.mpi_bandwidth_gbps = {0.0, 3.0, 3.0, 0.0, 0.0};
        expect_set_topo_error(rt, bad, "mpi_bandwidth_gbps size 5");
    }
    {
        harmony::TopoMetrics bad = base;
        bad.pcie_latency_ns = {4200.0, 4300.0, 4400.0};
        expect_set_topo_error(rt, bad, "pcie_latency_ns size 3");
    }
    {
        harmony::TopoMetrics bad = base;
        bad.pcie_bandwidth_gbps = {15.0};
        expect_set_topo_error(rt, bad, "pcie_bandwidth_gbps size 1");
    }
    {
        harmony::TopoMetrics bad = base;
        bad.numa_distance = {10};
        bad.device_numa_node = {0};
        expect_set_topo_error(rt, bad, "device_numa_node size 1");
    }
    {
        harmony::TopoMetrics bad = base;
        bad.memory_contention_factor = {1.5};
        expect_set_topo_error(rt, bad, "memory_contention_factor size 1");
    }
    {
        harmony::TopoMetrics bad = base;
        bad.thermal_tdp_watts = {250.0, 250.0, 250.0};
        expect_set_topo_error(rt, bad, "thermal_tdp_watts size 3");
    }
    {
        harmony::TopoMetrics bad = base;
        bad.current_power_watts = {120.0};
        expect_set_topo_error(rt, bad, "current_power_watts size 1");
    }

    // Prior valid metrics must be preserved when invalid updates throw.
    assert(rt.topo_metrics().has_value());
    assert(rt.topo_metrics()->pcie_latency_ns.size() == 2);
}

void test_range_and_mpi_validation(int argc, char** argv) {
    auto rt = harmony::Runtime::create(argc, argv);
    rt.set_simulated_devices_count(2);
    const harmony::TopoMetrics base = make_valid_metrics(2, rt.size());

    const double nan_val = std::numeric_limits<double>::quiet_NaN();
    const double inf_val = std::numeric_limits<double>::infinity();

    {
        harmony::TopoMetrics bad = base;
        bad.pcie_latency_ns = {-1.0, 100.0};
        expect_set_topo_error(rt, bad, "negative pcie_latency_ns");
    }
    {
        harmony::TopoMetrics bad = base;
        bad.pcie_bandwidth_gbps = {0.0, 15.0};
        expect_set_topo_error(rt, bad, "zero pcie_bandwidth_gbps");
    }
    {
        harmony::TopoMetrics bad = base;
        bad.mpi_latency_ns[0] = -0.5;
        expect_set_topo_error(rt, bad, "negative mpi_latency_ns");
    }
    {
        harmony::TopoMetrics bad = base;
        bad.mpi_bandwidth_gbps[0] = -1.0;
        expect_set_topo_error(rt, bad, "negative mpi_bandwidth_gbps");
    }
    {
        harmony::TopoMetrics bad = base;
        bad.memory_contention_factor = {0.95, 1.0};
        expect_set_topo_error(rt, bad, "memory_contention_factor < 1.0");
    }
    {
        harmony::TopoMetrics bad = base;
        bad.memory_contention_factor = {nan_val, 1.0};
        expect_set_topo_error(rt, bad, "NaN memory_contention_factor");
    }
    {
        harmony::TopoMetrics bad = base;
        bad.thermal_tdp_watts = {0.0, 250.0};
        expect_set_topo_error(rt, bad, "zero thermal_tdp_watts");
    }
    {
        harmony::TopoMetrics bad = base;
        bad.current_power_watts = {-5.0, 100.0};
        expect_set_topo_error(rt, bad, "negative current_power_watts");
    }
    {
        harmony::TopoMetrics bad = base;
        bad.pcie_bandwidth_gbps = {inf_val, 15.0};
        expect_set_topo_error(rt, bad, "Inf pcie_bandwidth_gbps");
    }

    // Off-diagonal MPI bandwidth must be strictly positive (verified on 2 ranks).
    {
        auto rt2 = harmony::Runtime::create(argc, argv);
        if (rt2.size() == 1) {
            rt2.set_simulated_ranks_for_testing(2);
        }
        rt2.set_simulated_devices_count(2);
        harmony::TopoMetrics bad = make_valid_metrics(2, rt2.size());
        bad.mpi_bandwidth_gbps[1] = 0.0;
        expect_set_topo_error(rt2, bad, "zero off-diagonal mpi_bandwidth_gbps");
    }
}

void test_numa_validation(int argc, char** argv) {
    auto rt = harmony::Runtime::create(argc, argv);
    rt.set_simulated_devices_count(2);
    const harmony::TopoMetrics base = make_valid_metrics(2, rt.size());

    {
        harmony::TopoMetrics bad = base;
        bad.device_numa_node = {0, 0};
        expect_set_topo_error(rt, bad, "device_numa_node without numa_distance");
    }
    {
        harmony::TopoMetrics bad = base;
        bad.numa_distance = {10, 20, 20};
        bad.device_numa_node = {0, 1};
        expect_set_topo_error(rt, bad, "non-square numa_distance");
    }
    {
        harmony::TopoMetrics bad = base;
        bad.numa_distance = {0, 20, 20, 10};
        bad.device_numa_node = {0, 1};
        expect_set_topo_error(rt, bad, "non-positive numa_distance diagonal");
    }
    {
        harmony::TopoMetrics bad = base;
        bad.numa_distance = {10, 0, 0, 10};
        bad.device_numa_node = {0, 1};
        expect_set_topo_error(rt, bad, "non-positive numa_distance off-diagonal");
    }
    {
        harmony::TopoMetrics bad = base;
        bad.numa_distance = {30, 10, 10, 30};
        bad.device_numa_node = {0, 1};
        expect_set_topo_error(rt, bad, "numa_distance diagonal > off-diagonal");
    }
    {
        harmony::TopoMetrics bad = base;
        bad.numa_distance = {10, 20, 20, 10};
        bad.device_numa_node = {-1, 0};
        expect_set_topo_error(rt, bad, "negative device_numa_node");
    }
    {
        harmony::TopoMetrics bad = base;
        bad.numa_distance = {10, 20, 20, 10};
        bad.device_numa_node = {0, 2};
        expect_set_topo_error(rt, bad, "device_numa_node >= num_nodes");
    }
}

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    test_save_load_round_trip(rank);
    test_missing_file_throws();
    test_json_parser_edge_cases(rank);
    test_save_validation_errors(rank);
    test_accept_correct_dimensions(argc, argv);
    test_topology_larger_than_local(argc, argv);
    test_multi_rank_simulated_topology(argc, argv);
    test_reject_undiscovered_devices(argc, argv);
    test_reject_wrong_dimensions(argc, argv);
    test_range_and_mpi_validation(argc, argv);
    test_numa_validation(argc, argv);

    if (rank == 0) {
        std::cout << "[PASS] test_topo_metrics\n";
    }

    MPI_Finalize();
    return 0;
}
