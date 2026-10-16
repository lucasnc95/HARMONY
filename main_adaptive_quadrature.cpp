#include "dcl/runtime.hpp"
#include <mpi.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

template <typename T>
static dcl::ScalarArg scalar_arg(const T& value) {
    dcl::ScalarArg arg;
    arg.bytes.resize(sizeof(T));
    std::memcpy(arg.bytes.data(), &value, sizeof(T));
    return arg;
}

static int parse_int_arg(char** begin, char** end, const std::string& name, int default_value) {
    for (char** it = begin; it != end; ++it) {
        if (name == *it && (it + 1) != end) return std::atoi(*(it + 1));
    }
    return default_value;
}

static float parse_float_arg(char** begin, char** end, const std::string& name, float default_value) {
    for (char** it = begin; it != end; ++it) {
        if (name == *it && (it + 1) != end) return std::strtof(*(it + 1), nullptr);
    }
    return default_value;
}

static std::string parse_string_arg(char** begin, char** end, const std::string& name, const std::string& default_value) {
    for (char** it = begin; it != end; ++it) {
        if (name == *it && (it + 1) != end) return std::string(*(it + 1));
    }
    return default_value;
}

static dcl::BalanceMode parse_balance_mode(const std::string& mode, const std::string& strategy) {
    if (mode == "off") return dcl::BalanceMode::off;
    if (mode == "static" && strategy == "threshold") return dcl::BalanceMode::static_threshold;
    if (mode == "dynamic" && strategy == "threshold") return dcl::BalanceMode::dynamic_threshold;
    if (mode == "static" && strategy == "profiled") return dcl::BalanceMode::static_profiled;
    if (mode == "dynamic" && strategy == "profiled") return dcl::BalanceMode::dynamic_profiled;
    if (mode == "hierarchical" || strategy == "hierarchical") return dcl::BalanceMode::hierarchical;
    throw std::runtime_error("Invalid balance mode/strategy");
}

static std::size_t checked_mul(std::size_t a, std::size_t b, const char* what) {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
        throw std::overflow_error(std::string("overflow computing ") + what);
    }
    return a * b;
}

static void print_partitions(const std::vector<dcl::DevicePartition>& parts) {
    std::cout << "=== Global partitions ===\n";
    for (std::size_t i = 0; i < parts.size(); ++i) {
        const dcl::DevicePartition& p = parts[i];
        std::cout << "part[" << i << "] device_global=" << p.device_global_index
                  << " rank=" << p.owning_rank
                  << " local_index=" << p.local_index
                  << " offset=" << p.global_offset
                  << " count=" << p.element_count << "\n";
    }
    std::cout << std::flush;
}

int main(int argc, char** argv) {
    try {
        using clock_t = std::chrono::steady_clock;

        auto runtime = dcl::Runtime::create(argc, argv);
        runtime.discover_devices({dcl::DeviceKind::all, 0});

        const int x = parse_int_arg(argv + 1, argv + argc, "--x", 50);
        const int y = parse_int_arg(argv + 1, argv + argc, "--y", 50);
        const int z = parse_int_arg(argv + 1, argv + argc, "--z", 6400);
        const int iterations = parse_int_arg(argv + 1, argv + argc, "--iterations", 10000);
        const int rebalance_interval = parse_int_arg(argv + 1, argv + argc, "--rebalance-interval", 1000);
        const float rebalance_threshold = parse_float_arg(argv + 1, argv + argc, "--rebalance-threshold", 0.0003125f);
        const std::string balance_mode_str = parse_string_arg(argv + 1, argv + argc, "--balance-mode", "dynamic");
        const std::string balance_strategy_str = parse_string_arg(argv + 1, argv + argc, "--balance-strategy", "threshold");
        const std::string profiling_file = parse_string_arg(argv + 1, argv + argc, "--profiling-file", "profiling_results.txt");
        const std::string timing_file = parse_string_arg(argv + 1, argv + argc, "--timing-file", "");
        const bool gather_final = parse_int_arg(argv + 1, argv + argc, "--gather-final", 0) != 0;

        const unsigned max_depth = static_cast<unsigned>(parse_int_arg(argv + 1, argv + argc, "--max-depth", 18));
        const unsigned heavy_repeat = static_cast<unsigned>(parse_int_arg(argv + 1, argv + argc, "--heavy-repeat", 64));
        const float base_tol = parse_float_arg(argv + 1, argv + argc, "--base-tol", 1.0e-4f);
        const float zoom = parse_float_arg(argv + 1, argv + argc, "--zoom", 3.5f);
        const float center_x = parse_float_arg(argv + 1, argv + argc, "--center-x", 0.0f);
        const float center_y = parse_float_arg(argv + 1, argv + argc, "--center-y", 0.0f);

        if (x <= 0 || y <= 0 || z <= 0) throw std::runtime_error("mesh dimensions must be positive");
        if (iterations <= 0) throw std::runtime_error("iterations must be positive");

        const std::size_t width = static_cast<std::size_t>(x);
        const std::size_t height = checked_mul(static_cast<std::size_t>(y), static_cast<std::size_t>(z), "y*z");
        const std::size_t total_points = checked_mul(width, height, "width*height");
        const std::size_t granularity = checked_mul(static_cast<std::size_t>(x), static_cast<std::size_t>(y), "x*y");

        if (total_points > static_cast<std::size_t>(std::numeric_limits<unsigned>::max())) {
            throw std::runtime_error("total_points exceeds uint range used by the kernel");
        }

        if (runtime.rank() == 0) {
            std::cout << "=== Adaptive quadrature CPU-friendly irregular kernel ===\n";
            std::cout << "mesh=" << x << "x" << y << "x" << z
                      << " total_points=" << total_points
                      << " max_depth=" << max_depth
                      << " heavy_repeat=" << heavy_repeat << "\n";
        }

        std::vector<float> result_init(total_points, 0.0f);
        std::vector<unsigned> work_init(total_points, 0u);
        std::vector<unsigned> iparams(1, 0u);

        auto kernel = runtime.create_kernel({"adaptive_quadrature_cpu_friendly.cl", "adaptive_quadrature_cpu_friendly", ""});

        runtime.set_partition({total_points, 1, sizeof(float), granularity});

        auto result = runtime.create_field({"result", total_points, 1, sizeof(float), dcl::BufferUsage::read_write, result_init.data(), dcl::RedistributionDependency::proportional});
        auto work_count = runtime.create_field({"work_count", total_points, 1, sizeof(unsigned), dcl::BufferUsage::read_write, work_init.data(), dcl::RedistributionDependency::proportional});
        auto params = runtime.create_field({"iparams", iparams.size(), 1, sizeof(unsigned), dcl::BufferUsage::read_write, iparams.data(), dcl::RedistributionDependency::none});

        auto binding = runtime.bind(kernel)
            .arg(0, result)
            .arg(1, work_count)
            .arg(2, params)
            .arg(3, scalar_arg(static_cast<unsigned>(total_points)))
            .arg(4, scalar_arg(static_cast<unsigned>(width)))
            .arg(5, scalar_arg(static_cast<unsigned>(height)))
            .arg(6, scalar_arg(max_depth))
            .arg(7, scalar_arg(heavy_repeat))
            .arg(8, scalar_arg(base_tol))
            .arg(9, scalar_arg(zoom))
            .arg(10, scalar_arg(center_x))
            .arg(11, scalar_arg(center_y))
            .build();

        const dcl::BalanceMode balance_mode = parse_balance_mode(balance_mode_str, balance_strategy_str);

        dcl::ExecutionStep step = runtime.step("adaptive-quadrature")
            .invoke(binding, dcl::LaunchGeometry{0, total_points, std::optional<std::size_t>()})
            .with_balance(dcl::AutoBalancePolicy{balance_mode, rebalance_interval, rebalance_threshold, iterations, profiling_file})
            .tag_field(result, dcl::StepFieldRole::write_target)
            .tag_field(work_count, dcl::StepFieldRole::write_target)
            .tag_field(params, dcl::StepFieldRole::read_source)
            .tag_field(params, dcl::StepFieldRole::write_target)
            .tag_field(result, dcl::StepFieldRole::rebalance_source)
            .tag_field(work_count, dcl::StepFieldRole::rebalance_source)
            .synchronize_at_end(false)
            .build();

        if (runtime.rank() == 0) {
            std::cout << "=== Initial partitions ===\n";
            print_partitions(runtime.partitions());
        }

        auto start = clock_t::now();
        for (int it = 0; it < iterations; ++it) runtime.execute(step);
        runtime.synchronize(true);
        auto end = clock_t::now();

        if (gather_final) {
            runtime.gather(result, result_init.data(), result_init.size() * sizeof(float));
        }

        if (runtime.rank() == 0) {
            const double seconds = std::chrono::duration<double>(end - start).count();
            if (!timing_file.empty()) {
                std::ofstream out(timing_file, std::ios::app);
                if (!out) throw std::runtime_error("could not open timing file");
                out << std::fixed << std::setprecision(9) << seconds << "\n";
            }
            std::cout << "=== Final partitions ===\n";
            print_partitions(runtime.partitions());
            std::cout << "Execution time: " << seconds << "s\n";
        }
        return 0;
    } catch (const dcl::Error& e) {
        std::cerr << "dcl error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "std error: " << e.what() << std::endl;
        return 2;
    }
}
