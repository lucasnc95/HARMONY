#include "dcl/runtime.hpp"
#include <mpi.h>

#include <chrono>
#include <cmath>
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

static std::size_t idx3(std::size_t x, std::size_t y, std::size_t z, std::size_t nx, std::size_t ny) {
    return z * nx * ny + y * nx + x;
}

static void initialize_u(std::vector<float>& u, std::size_t nx, std::size_t ny, std::size_t nz) {
    const float fx = 6.28318530718f / static_cast<float>(nx);
    const float fy = 6.28318530718f / static_cast<float>(ny);
    const float fz = 6.28318530718f / static_cast<float>(nz);
    for (std::size_t z = 0; z < nz; ++z) {
        for (std::size_t y = 0; y < ny; ++y) {
            for (std::size_t x = 0; x < nx; ++x) {
                float vx = std::sin(fx * static_cast<float>(x));
                float vy = std::cos(fy * static_cast<float>(y));
                float vz = std::sin(fz * static_cast<float>(z) * 0.5f);
                u[idx3(x, y, z, nx, ny)] = vx * vy + 0.25f * vz;
            }
        }
    }
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

static int run_harmony(int argc, char** argv) {
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

        const unsigned repeat = static_cast<unsigned>(parse_int_arg(argv + 1, argv + argc, "--repeat", 64));
        const float alpha = parse_float_arg(argv + 1, argv + argc, "--alpha", 1.0f);
        const float beta = parse_float_arg(argv + 1, argv + argc, "--beta", 0.25f);

        if (x <= 8 || y <= 8 || z <= 8) throw std::runtime_error("all mesh dimensions must be greater than 8 for radius-4 stencil");
        if (iterations <= 0) throw std::runtime_error("iterations must be positive");

        const std::size_t nx = static_cast<std::size_t>(x);
        const std::size_t ny = static_cast<std::size_t>(y);
        const std::size_t nz = static_cast<std::size_t>(z);
        const std::size_t xy = checked_mul(nx, ny, "nx*ny");
        const std::size_t total_points = checked_mul(xy, nz, "nx*ny*nz");
        const std::size_t granularity = xy;
        const std::size_t halo_width = 4u * xy;

        if (total_points > static_cast<std::size_t>(std::numeric_limits<unsigned>::max())) {
            throw std::runtime_error("total_points exceeds uint range used by the kernel");
        }

        if (runtime.rank() == 0) {
            std::cout << "=== High-order residual halo kernel, no swap buffers ===\n";
            std::cout << "mesh=" << x << "x" << y << "x" << z
                      << " total_points=" << total_points
                      << " repeat=" << repeat
                      << " halo_width=" << halo_width << " elements\n";
        }

        std::vector<float> u_init(total_points, 0.0f);
        std::vector<float> residual_init(total_points, 0.0f);
        std::vector<unsigned> iparams(1, 0u);
        initialize_u(u_init, nx, ny, nz);

        auto kernel = runtime.create_kernel({"high_order_residual_halo.cl", "high_order_residual_halo", ""});

        runtime.set_partition({total_points, 1, sizeof(float), granularity});

        auto residual = runtime.create_field({"residual", total_points, 1, sizeof(float), dcl::BufferUsage::read_write, residual_init.data(), dcl::RedistributionDependency::none});
        auto u = runtime.create_field({"u", total_points, 1, sizeof(float), dcl::BufferUsage::read_write, u_init.data(), dcl::RedistributionDependency::proportional});
        auto params = runtime.create_field({"iparams", iparams.size(), 1, sizeof(unsigned), dcl::BufferUsage::read_write, iparams.data(), dcl::RedistributionDependency::none});

        auto binding = runtime.bind(kernel)
            .arg(0, residual)
            .arg(1, u)
            .arg(2, params)
            .arg(3, scalar_arg(static_cast<unsigned>(total_points)))
            .arg(4, scalar_arg(static_cast<unsigned>(nx)))
            .arg(5, scalar_arg(static_cast<unsigned>(ny)))
            .arg(6, scalar_arg(static_cast<unsigned>(nz)))
            .arg(7, scalar_arg(repeat))
            .arg(8, scalar_arg(alpha))
            .arg(9, scalar_arg(beta))
            .build();

        const dcl::BalanceMode balance_mode = parse_balance_mode(balance_mode_str, balance_strategy_str);

        dcl::ExecutionStep step = runtime.step("high-order-residual-halo")
            .invoke(binding, dcl::LaunchGeometry{0, total_points, std::optional<std::size_t>()})
            .with_halo_exchange(dcl::HaloSpec{halo_width, std::vector<dcl::FieldHandle>{u}})
            .with_balance(dcl::AutoBalancePolicy{balance_mode, rebalance_interval, rebalance_threshold, iterations, profiling_file})
            .tag_field(u, dcl::StepFieldRole::read_source)
            .tag_field(residual, dcl::StepFieldRole::write_target)
            .tag_field(params, dcl::StepFieldRole::read_source)
            .tag_field(params, dcl::StepFieldRole::write_target)
            .tag_field(u, dcl::StepFieldRole::halo_source)
            .tag_field(u, dcl::StepFieldRole::rebalance_source)
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
            runtime.gather(residual, residual_init.data(), residual_init.size() * sizeof(float));
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

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    const int init_status = MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
    if (init_status != MPI_SUCCESS) {
        std::cerr << "MPI_Init_thread failed: " << init_status << '\n';
        return 3;
    }

    const int run_status = run_harmony(argc, argv);
    const int finalize_status = MPI_Finalize();
    if (finalize_status != MPI_SUCCESS) {
        std::cerr << "MPI_Finalize failed: " << finalize_status << '\n';
        return 3;
    }
    return run_status;
}
