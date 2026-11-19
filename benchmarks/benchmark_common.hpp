#ifndef HARMONY_BENCHMARK_COMMON_HPP
#define HARMONY_BENCHMARK_COMMON_HPP

#include "harmony/runtime.hpp"
#include "harmony/topo_metrics_io.hpp"
#include "harmony/types.hpp"

#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace harmony_bench {

enum class DeviceMode {
    auto_select,
    gpu,
    cpu,
    all,
    sim
};

struct BenchmarkOptions {
    std::size_t n{1000000};
    int k{5};
    int iterations{100};
    std::string balance_algo{"erad"};
    std::string balance_mode_name{"auto"};
    float perturbation{0.10f};
    int balance_interval{10};
    int warmup{0};
    float threshold{0.0f};
    std::string profiling_file;
    std::string topo_json;
    bool use_contention{false};
    bool use_power_cap{false};
    double power_budget_watts{0.0};
    double numa_threshold{0.50};
    unsigned int seed{0};
    std::string output_csv;
    std::string report_csv;
    std::string kernel_file;
    DeviceMode device_mode{DeviceMode::auto_select};
    bool verbose{false};
    int sim_devices{2};
};

inline std::string derive_report_csv_path(const std::string& output_csv) {
    if (output_csv.empty()) return {};
    const std::filesystem::path p(output_csv);
    const std::string stem = p.stem().string();
    const std::string ext = p.extension().string();
    const std::filesystem::path dir = p.parent_path();
    const std::string filename = stem + "_report" + (ext.empty() ? ".csv" : ext);
    return dir.empty() ? filename : (dir / filename).string();
}

inline DeviceMode parse_device_mode(const std::string& value) {
    if (value == "auto") return DeviceMode::auto_select;
    if (value == "gpu") return DeviceMode::gpu;
    if (value == "cpu") return DeviceMode::cpu;
    if (value == "all") return DeviceMode::all;
    if (value == "sim") return DeviceMode::sim;
    throw std::invalid_argument("Unknown --device mode: " + value + " (expected auto|gpu|cpu|all|sim)");
}

inline harmony::BalanceMode parse_balance_mode(const std::string& name, int interval) {
    if (name == "auto") {
        return (interval > 0) ? harmony::BalanceMode::dynamic_threshold : harmony::BalanceMode::off;
    }
    if (name == "off") return harmony::BalanceMode::off;
    if (name == "static" || name == "static_threshold") return harmony::BalanceMode::static_threshold;
    if (name == "dynamic" || name == "dynamic_threshold") return harmony::BalanceMode::dynamic_threshold;
    if (name == "static_profiled") return harmony::BalanceMode::static_profiled;
    if (name == "dynamic_profiled") return harmony::BalanceMode::dynamic_profiled;
    if (name == "hierarchical") return harmony::BalanceMode::hierarchical;
    throw std::invalid_argument("Unknown --balance-mode: " + name);
}

inline BenchmarkOptions parse_options(int argc, char** argv, const std::string& bench_name, int default_k) {
    BenchmarkOptions opt;
    opt.k = default_k;
    bool output_csv_set = false;
    bool report_csv_set = false;

    auto require_value = [&](int& i, const std::string& flag) -> std::string {
        if (i + 1 >= argc) throw std::invalid_argument("Missing value for " + flag);
        return argv[++i];
    };

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--n") {
            opt.n = static_cast<std::size_t>(std::stoull(require_value(i, arg)));
        } else if (arg == "--k" || arg == "--cutoff") {
            opt.k = std::stoi(require_value(i, arg));
        } else if (arg == "--iterations") {
            opt.iterations = std::stoi(require_value(i, arg));
        } else if (arg == "--balance-algo") {
            opt.balance_algo = require_value(i, arg);
        } else if (arg == "--balance-mode") {
            opt.balance_mode_name = require_value(i, arg);
        } else if (arg == "--perturbation") {
            opt.perturbation = std::stof(require_value(i, arg));
        } else if (arg == "--balance-interval" || arg == "--rebalance-interval") {
            opt.balance_interval = std::stoi(require_value(i, arg));
        } else if (arg == "--warmup" || arg == "--balance-warmup") {
            opt.warmup = std::stoi(require_value(i, arg));
        } else if (arg == "--threshold" || arg == "--rebalance-threshold") {
            opt.threshold = std::stof(require_value(i, arg));
        } else if (arg == "--profiling-file") {
            opt.profiling_file = require_value(i, arg);
        } else if (arg == "--topo-json") {
            opt.topo_json = require_value(i, arg);
        } else if (arg == "--contention" || arg == "--use-contention") {
            const std::string v = require_value(i, arg);
            opt.use_contention = (v == "1" || v == "true" || v == "yes");
        } else if (arg == "--power-cap") {
            opt.power_budget_watts = std::stod(require_value(i, arg));
            opt.use_power_cap = (opt.power_budget_watts > 0.0);
        } else if (arg == "--numa-threshold") {
            opt.numa_threshold = std::stod(require_value(i, arg));
        } else if (arg == "--seed") {
            opt.seed = static_cast<unsigned int>(std::stoul(require_value(i, arg)));
        } else if (arg == "--output-csv") {
            opt.output_csv = require_value(i, arg);
            output_csv_set = true;
        } else if (arg == "--report-csv") {
            opt.report_csv = require_value(i, arg);
            report_csv_set = true;
        } else if (arg == "--kernel-file") {
            opt.kernel_file = require_value(i, arg);
        } else if (arg == "--device") {
            opt.device_mode = parse_device_mode(require_value(i, arg));
        } else if (arg == "--sim-devices") {
            opt.sim_devices = std::stoi(require_value(i, arg));
        } else if (arg == "--verbose") {
            const std::string v = require_value(i, arg);
            opt.verbose = (v == "1" || v == "true" || v == "yes");
        } else {
            throw std::invalid_argument("Unknown CLI option: " + arg);
        }
    }

    if (opt.n == 0 || opt.n > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("--n must be in [1, INT_MAX]");
    }
    if (opt.iterations <= 0) {
        throw std::invalid_argument("--iterations must be > 0");
    }
    if (opt.k < 0) {
        throw std::invalid_argument("--k must be >= 0");
    }
    if (bench_name == "leanmd" && opt.k < 1) {
        throw std::invalid_argument("--k (cutoff) for leanmd must be >= 1");
    }
    if (opt.balance_algo != "erad" && opt.balance_algo != "hwtopolb") {
        throw std::invalid_argument("--balance-algo must be erad or hwtopolb");
    }
    if (opt.sim_devices <= 0) {
        throw std::invalid_argument("--sim-devices must be > 0");
    }
    (void)parse_balance_mode(opt.balance_mode_name, opt.balance_interval);

    if (!output_csv_set && !report_csv_set) {
        opt.output_csv = bench_name + ".csv";
        opt.report_csv = bench_name + "_report.csv";
    } else if (output_csv_set && !report_csv_set && !opt.output_csv.empty()) {
        opt.report_csv = derive_report_csv_path(opt.output_csv);
    }

    return opt;
}

inline harmony::AutoBalancePolicy build_balance_policy(const BenchmarkOptions& opt) {
    harmony::AutoBalancePolicy policy;
    policy.mode = parse_balance_mode(opt.balance_mode_name, opt.balance_interval);
    policy.interval = opt.balance_interval;
    policy.threshold = opt.threshold;
    policy.total_iterations = opt.iterations;
    policy.profiling_file = opt.profiling_file;
    policy.warmup = opt.warmup;
    policy.algorithm = (opt.balance_algo == "hwtopolb")
        ? harmony::BalanceAlgorithm::hwtopolb
        : harmony::BalanceAlgorithm::erad;
    policy.perturbation_factor = opt.perturbation;
    policy.seed = opt.seed;
    policy.use_contention_adjustment = opt.use_contention;
    policy.use_power_cap = opt.use_power_cap;
    policy.power_budget_watts = opt.power_budget_watts;
    policy.numa_cost_gain_ratio_threshold = opt.numa_threshold;
    return policy;
}

// Discovers OpenCL devices according to opt.device_mode. Returns true if real
// OpenCL devices are active, or false when using CPU simulation fallback.
inline bool init_benchmark_devices(harmony::Runtime& runtime, const BenchmarkOptions& opt) {
    bool opencl_available = false;
    if (opt.device_mode == DeviceMode::sim) {
        runtime.set_simulated_devices_count(opt.sim_devices);
        return false;
    }

    const harmony::DeviceKind kind =
        (opt.device_mode == DeviceMode::cpu)   ? harmony::DeviceKind::cpu
        : (opt.device_mode == DeviceMode::all) ? harmony::DeviceKind::all
                                               : harmony::DeviceKind::gpu;

    try {
        runtime.discover_devices({kind, 0});
        opencl_available = !runtime.global_devices().empty();
    } catch (const harmony::Error& err) {
        if (opt.device_mode != DeviceMode::auto_select) {
            throw;
        }
        if (runtime.rank() == 0) {
            std::cerr << "[benchmark] OpenCL GPU unavailable (" << err.what()
                      << "); falling back to CPU simulation (" << opt.sim_devices
                      << " simulated devices)\n";
        }
        runtime.set_simulated_devices_count(opt.sim_devices);
        opencl_available = false;
    }

    if (opencl_available) {
        for (const auto& device : runtime.devices()) {
            std::cout << "[device] rank=" << runtime.rank()
                      << " global_index=" << device.global_index
                      << " name=" << device.name << "\n";
        }
    }
    return opencl_available;
}

inline std::string resolve_kernel_path(const std::string& cli_path,
                                       const std::string& default_compiled_path,
                                       const std::string& filename) {
    if (!cli_path.empty()) return cli_path;
    if (!default_compiled_path.empty() && std::filesystem::exists(default_compiled_path)) {
        return default_compiled_path;
    }
    const std::string rel_bench = "benchmarks/" + filename;
    if (std::filesystem::exists(rel_bench)) return rel_bench;
    const std::string rel_parent = "../benchmarks/" + filename;
    if (std::filesystem::exists(rel_parent)) return rel_parent;
    return filename;
}

inline bool verify_finite_and_close(float actual, float expected, float abs_tol, float rel_tol) {
    return std::isfinite(actual) && std::isfinite(expected) &&
           std::fabs(actual - expected) <= abs_tol + rel_tol * std::fabs(expected);
}

inline bool should_reset_sim_window(const harmony::AutoBalancePolicy& policy,
                                    int completed_iter,
                                    int total_iterations,
                                    bool& static_attempted) {
    if (policy.mode == harmony::BalanceMode::off) return false;
    if (total_iterations > 0 && completed_iter >= total_iterations) return false;
    const int interval = std::max(1, policy.interval);
    const bool is_dynamic = (policy.mode == harmony::BalanceMode::hierarchical ||
                             policy.mode == harmony::BalanceMode::dynamic_threshold ||
                             policy.mode == harmony::BalanceMode::dynamic_profiled);
    const bool scheduled = (completed_iter % interval == 0) && (is_dynamic || !static_attempted);
    if (scheduled && !is_dynamic) static_attempted = true;
    const bool probe = !scheduled && policy.warmup > 0 && completed_iter == policy.warmup;
    return scheduled || probe;
}

inline float kneighbor_ref_at(const std::vector<float>& in_data, std::size_t idx, int k) {
    const std::size_t n = in_data.size();
    float sum = 0.0f;
    int valid_neighbors = 0;
    for (int o = -k; o <= k; ++o) {
        const long long j = static_cast<long long>(idx) + o;
        if (j >= 0 && j < static_cast<long long>(n)) {
            sum += in_data[static_cast<std::size_t>(j)];
            ++valid_neighbors;
        }
    }
    return sum / static_cast<float>(valid_neighbors);
}

inline float leanmd_ref_at(const std::vector<float>& pos_x,
                           const std::vector<float>& pos_y,
                           const std::vector<float>& pos_z,
                           std::size_t i,
                           int cutoff) {
    const std::size_t n = pos_x.size();
    const std::size_t start = i > static_cast<std::size_t>(cutoff) ? i - static_cast<std::size_t>(cutoff) : 0;
    const std::size_t end = std::min(n - 1, i + static_cast<std::size_t>(cutoff));
    float expected = 0.0f;
    for (std::size_t j = start; j <= end; ++j) {
        if (i == j) continue;
        const float dx = pos_x[i] - pos_x[j];
        const float dy = pos_y[i] - pos_y[j];
        const float dz = pos_z[i] - pos_z[j];
        const float r2 = dx * dx + dy * dy + dz * dz + 1.0e-6f;
        const float inv = 1.0f / r2;
        const float inv6 = inv * inv * inv;
        expected += inv * inv6 * (2.0f * inv6 - 1.0f);
    }
    return expected;
}

template <typename ElementWorkFn>
inline void run_simulated_partition_work(const std::vector<harmony::DevicePartition>& partitions,
                                         int rank,
                                         ElementWorkFn&& work_at,
                                         std::vector<double>& local_device_times) {
    for (std::size_t p_idx = 0; p_idx < partitions.size(); ++p_idx) {
        const auto& p = partitions[p_idx];
        if (p.owning_rank != rank || p.element_count == 0) continue;

        const std::size_t offset = p.global_offset;
        const std::size_t count = p.element_count;
        const int work_passes = (p.device_global_index == 0 ? 2 : 1);

        const auto t0 = std::chrono::steady_clock::now();
        for (int pass = 0; pass < work_passes; ++pass) {
            for (std::size_t i = offset; i < offset + count; ++i) {
                work_at(i);
            }
        }
        const auto t1 = std::chrono::steady_clock::now();
        local_device_times[p_idx] = std::chrono::duration<double>(t1 - t0).count();
    }
}

template <typename ElementWorkFn>
inline std::vector<double> run_benchmark_loop(harmony::Runtime& runtime,
                                              const harmony::ExecutionStep& step,
                                              const harmony::AutoBalancePolicy& policy,
                                              const BenchmarkOptions& opt,
                                              bool opencl_available,
                                              ElementWorkFn&& sim_work_at) {
    const int rank = runtime.rank();
    std::vector<double> step_times(static_cast<std::size_t>(opt.iterations), 0.0);
    std::vector<double> sim_window_times;
    bool sim_static_attempted = false;

    runtime.start_timing();
    for (int it = 0; it < opt.iterations; ++it) {
        const auto step_start = std::chrono::steady_clock::now();

        if (!opencl_available) {
            const auto& partitions = runtime.partitions();
            if (sim_window_times.size() != partitions.size()) {
                sim_window_times.assign(partitions.size(), 0.0);
            }
            std::vector<double> local_device_times(partitions.size(), 0.0);
            run_simulated_partition_work(partitions, rank, sim_work_at, local_device_times);
            std::vector<double> iter_device_times(partitions.size(), 0.0);
            MPI_Allreduce(local_device_times.data(), iter_device_times.data(),
                          static_cast<int>(partitions.size()), MPI_DOUBLE, MPI_MAX, runtime.communicator());
            for (std::size_t i = 0; i < partitions.size(); ++i) {
                sim_window_times[i] += iter_device_times[i];
            }
            runtime.set_simulated_times(sim_window_times);
        }

        runtime.execute(step);

        if (!opencl_available &&
            should_reset_sim_window(policy, it + 1, opt.iterations, sim_static_attempted)) {
            std::fill(sim_window_times.begin(), sim_window_times.end(), 0.0);
        }

        const auto step_end = std::chrono::steady_clock::now();
        step_times[static_cast<std::size_t>(it)] =
            std::chrono::duration<double>(step_end - step_start).count();
    }
    return step_times;
}

inline bool write_benchmark_outputs(harmony::Runtime& runtime,
                                    const std::string& bench_name,
                                    const BenchmarkOptions& opt,
                                    const std::vector<double>& step_times,
                                    double total_duration_s,
                                    double loop_time_s,
                                    bool opencl_available) {
    const harmony::TimingReport report = runtime.timing_report();
    if (!opt.report_csv.empty()) {
        runtime.write_timing_report(opt.report_csv);
    }
    if (opt.verbose) {
        runtime.print_timing_report();
    }

    int write_ok = 1;
    if (runtime.rank() == 0) {
        bool has_gpu = false;
        for (const auto& d : runtime.global_devices()) {
            if (d.kind == harmony::DeviceKind::gpu) {
                has_gpu = true;
                break;
            }
        }
        const char* mode_str = opencl_available ? (has_gpu ? "gpu_opencl" : "cpu_opencl") : "cpu_simulation";
        const std::size_t n_iters = step_times.size();
        std::vector<int> rebalanced(n_iters, 0);
        std::vector<double> gains(n_iters, 0.0);

        int num_rebalances = 0;
        double sum_gain = 0.0;
        for (const harmony::BalanceRecord& b : report.balance) {
            if (!b.applied) continue;
            const double gain = std::max(0.0, b.iteration_seconds_before - b.iteration_seconds_after);
            ++num_rebalances;
            sum_gain += gain;
            if (b.iteration > 0 && b.iteration <= n_iters) {
                rebalanced[b.iteration - 1] = 1;
                gains[b.iteration - 1] = gain;
            }
        }

        if (!opt.output_csv.empty()) {
            std::ofstream csv(opt.output_csv);
            if (!csv.is_open()) {
                std::cerr << "Failed to open output CSV: " << opt.output_csv << "\n";
                write_ok = 0;
            } else {
                csv << "iteration,step_time_s,action,rebalanced,rebalance_gain,execution_mode\n";
                for (std::size_t it = 0; it < n_iters; ++it) {
                    const char* action = rebalanced[it] ? "rebalance" : "compute";
                    csv << it << ","
                        << std::fixed << std::setprecision(9) << step_times[it] << ","
                        << action << ","
                        << rebalanced[it] << ","
                        << std::fixed << std::setprecision(9) << gains[it] << ","
                        << mode_str << "\n";
                }
            }
        }

        if (write_ok != 0) {
            const double avg_gain = (num_rebalances > 0) ? (sum_gain / num_rebalances) : 0.0;
            std::cout << "[" << bench_name << "] algo=" << opt.balance_algo
                      << " mode=" << mode_str
                      << " total_time_s=" << total_duration_s
                      << " loop_time_s=" << loop_time_s
                      << " num_rebalances=" << num_rebalances
                      << " avg_rebalance_gain=" << avg_gain << "\n";
        }
    }

    MPI_Bcast(&write_ok, 1, MPI_INT, 0, runtime.communicator());
    return write_ok != 0;
}

} // namespace harmony_bench

#endif // HARMONY_BENCHMARK_COMMON_HPP
