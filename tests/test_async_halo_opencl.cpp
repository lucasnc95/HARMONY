#include "harmony/runtime.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef HARMONY_ROOT_DIR
#define HARMONY_ROOT_DIR "."
#endif

namespace {

struct PreflightOptions {
    bool require_gpu{false};
    bool allow_cpu_fallback{false};
    int min_devices{2};
    double timeout_seconds{60.0};
    std::string kernel_file;
};

bool env_flag_enabled(const char* name) {
    const char* val = std::getenv(name);
    if (val == nullptr) return false;
    const std::string s(val);
    return s == "1" || s == "true" || s == "yes";
}

PreflightOptions parse_options(int argc, char** argv) {
    PreflightOptions opt;
    opt.require_gpu = env_flag_enabled("HARMONY_PREFLIGHT_REQUIRE_GPU");
    opt.allow_cpu_fallback = env_flag_enabled("HARMONY_ALLOW_CPU_OPENCL");
    if (const char* t = std::getenv("HARMONY_PREFLIGHT_TIMEOUT_SECONDS")) {
        opt.timeout_seconds = std::stod(t);
    }

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--require-gpu") {
            opt.require_gpu = true;
            opt.allow_cpu_fallback = false;
        } else if (arg == "--allow-cpu-fallback") {
            opt.allow_cpu_fallback = true;
        } else if (arg == "--min-devices" && i + 1 < argc) {
            opt.min_devices = std::stoi(argv[++i]);
        } else if (arg == "--timeout-seconds" && i + 1 < argc) {
            opt.timeout_seconds = std::stod(argv[++i]);
        } else if (arg == "--kernel-file" && i + 1 < argc) {
            opt.kernel_file = argv[++i];
        } else {
            throw std::invalid_argument("Unknown option: " + arg);
        }
    }
    if (opt.min_devices < 1) {
        throw std::invalid_argument("--min-devices must be >= 1");
    }
    if (opt.timeout_seconds <= 0.0) {
        throw std::invalid_argument("--timeout-seconds must be > 0");
    }
    return opt;
}

std::string resolve_kneighbor_kernel(const std::string& cli_path) {
    if (!cli_path.empty()) return cli_path;
    const std::string from_root = std::string(HARMONY_ROOT_DIR) + "/benchmarks/kneighbor.cl";
    if (std::filesystem::exists(from_root)) return from_root;
    if (std::filesystem::exists("benchmarks/kneighbor.cl")) return "benchmarks/kneighbor.cl";
    if (std::filesystem::exists("../benchmarks/kneighbor.cl")) return "../benchmarks/kneighbor.cl";
    return "benchmarks/kneighbor.cl";
}

std::vector<float> cpu_stencil_reference(const std::vector<float>& input, int radius) {
    std::vector<float> output(input.size(), 0.0f);
    for (std::size_t i = 0; i < input.size(); ++i) {
        float sum = 0.0f;
        int count = 0;
        for (int offset = -radius; offset <= radius; ++offset) {
            const long long neighbor = static_cast<long long>(i) + offset;
            if (neighbor >= 0 && neighbor < static_cast<long long>(input.size())) {
                sum += input[static_cast<std::size_t>(neighbor)];
                ++count;
            }
        }
        output[i] = sum / static_cast<float>(count);
    }
    return output;
}

bool check_timeout_budget(double start_wtime, double timeout_seconds, MPI_Comm comm, int rank, const char* stage) {
    const double local_elapsed = MPI_Wtime() - start_wtime;
    double max_elapsed = local_elapsed;
    MPI_Allreduce(&local_elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, comm);
    if (max_elapsed > timeout_seconds) {
        if (rank == 0) {
            std::cerr << "Preflight failed: timeout budget of " << timeout_seconds
                      << "s exceeded at " << stage << " (" << max_elapsed << "s)\n";
        }
        return false;
    }
    return true;
}

struct TopologyCheckResult {
    bool ready{false};
    int exit_code{0};
    int min_per_rank{0};
    int total_devices{0};
    bool using_gpu{false};
};

TopologyCheckResult discover_and_verify_topology(harmony::Runtime& runtime, const PreflightOptions& opt) {
    TopologyCheckResult res;
    std::string last_error;

    auto try_kind = [&](harmony::DeviceKind kind) -> bool {
        try {
            runtime.discover_devices({kind, 0});
            return true;
        } catch (const harmony::Error& e) {
            last_error = e.what();
            return false;
        }
    };

    bool discovered = try_kind(harmony::DeviceKind::gpu);
    res.using_gpu = discovered;
    if (!discovered && !opt.require_gpu && opt.allow_cpu_fallback) {
        discovered = try_kind(harmony::DeviceKind::all);
        res.using_gpu = false;
    }

    if (discovered) {
        const int local_count = static_cast<int>(runtime.devices().size());
        MPI_Allreduce(&local_count, &res.min_per_rank, 1, MPI_INT, MPI_MIN, runtime.communicator());
        MPI_Allreduce(&local_count, &res.total_devices, 1, MPI_INT, MPI_SUM, runtime.communicator());
    }

    if (!discovered || res.min_per_rank < 1 || res.total_devices < opt.min_devices) {
        if (opt.require_gpu) {
            if (runtime.rank() == 0) {
                std::cerr << "Preflight failed: requires >= 1 verified OpenCL GPU per MPI rank"
                          << " and >= " << opt.min_devices << " total GPUs (found min_per_rank="
                          << res.min_per_rank << ", total=" << res.total_devices << ")";
                if (!last_error.empty()) std::cerr << " [" << last_error << "]";
                std::cerr << "\n";
            }
            res.ready = false;
            res.exit_code = 1;
            return res;
        }

        if (runtime.rank() == 0) {
            std::cout << "SKIP: asynchronous OpenCL halo test needs >= 1 OpenCL "
                      << (opt.allow_cpu_fallback ? "device" : "GPU")
                      << " per MPI rank and >= " << opt.min_devices
                      << " total devices (found min=" << res.min_per_rank
                      << ", total=" << res.total_devices << ")\n";
        }
        res.ready = false;
        res.exit_code = 77;
        return res;
    }

    res.ready = true;
    res.exit_code = 0;
    return res;
}

} // namespace

int main(int argc, char** argv) {
    try {
        harmony::Runtime runtime = harmony::Runtime::create(argc, argv);
        const int rank = runtime.rank();
        const double start_wtime = MPI_Wtime();

        PreflightOptions opt;
        try {
            opt = parse_options(argc, argv);
        } catch (const std::exception& e) {
            if (rank == 0) std::cerr << "Preflight argument error: " << e.what() << "\n";
            return 1;
        }

        const TopologyCheckResult topo = discover_and_verify_topology(runtime, opt);
        if (!topo.ready) {
            return topo.exit_code;
        }

        if (!check_timeout_budget(start_wtime, opt.timeout_seconds, runtime.communicator(), rank,
                                  "device discovery")) {
            return 1;
        }

        constexpr std::size_t elements = 1024;
        constexpr int radius = 3;
        std::vector<float> input(elements), output(elements, 0.0f);
        for (std::size_t i = 0; i < elements; ++i) {
            input[i] = static_cast<float>(i % 37) * 0.03125f;
        }

        runtime.set_partition({elements, 1, sizeof(float), 1});

        auto in_field = runtime.create_field({"input", elements, 1, sizeof(float),
                                              harmony::BufferUsage::read_write, input.data(),
                                              harmony::RedistributionDependency::proportional});
        auto out_field = runtime.create_field({"output", elements, 1, sizeof(float),
                                               harmony::BufferUsage::read_write, output.data(),
                                               harmony::RedistributionDependency::proportional});

        const std::string kernel_path = resolve_kneighbor_kernel(opt.kernel_file);
        const auto kernel = runtime.create_kernel({kernel_path, "kneighbor_stencil", ""});

        const auto first_binding = runtime.bind(kernel)
            .arg(0, in_field, harmony::Access::read)
            .arg(1, out_field, harmony::Access::write)
            .arg(2, harmony::ScalarArg(static_cast<int>(elements)))
            .arg(3, harmony::ScalarArg(radius))
            .build();
        const auto second_binding = runtime.bind(kernel)
            .arg(0, out_field, harmony::Access::read)
            .arg(1, in_field, harmony::Access::write)
            .arg(2, harmony::ScalarArg(static_cast<int>(elements)))
            .arg(3, harmony::ScalarArg(radius))
            .build();

        const auto first_step = runtime.step("async_first")
            .invoke(first_binding, {0, elements, std::nullopt})
            .with_halo_exchange({static_cast<std::size_t>(radius), {in_field}})
            .synchronize_at_end(false)
            .build();
        const auto second_step = runtime.step("halo_second")
            .invoke(second_binding, {0, elements, std::nullopt})
            .with_halo_exchange({static_cast<std::size_t>(radius), {out_field}})
            .synchronize_at_end(true)
            .build();

        runtime.execute(first_step);
        runtime.execute(second_step);

        if (!check_timeout_budget(start_wtime, opt.timeout_seconds, runtime.communicator(), rank,
                                  "halo execution")) {
            return 1;
        }

        runtime.gather(in_field, input.data(), elements * sizeof(float));

        int mismatch = 0;
        if (rank == 0) {
            std::vector<float> initial(elements);
            for (std::size_t i = 0; i < elements; ++i) {
                initial[i] = static_cast<float>(i % 37) * 0.03125f;
            }
            const auto expected = cpu_stencil_reference(cpu_stencil_reference(initial, radius), radius);
            for (std::size_t i = 0; i < elements; ++i) {
                if (!std::isfinite(input[i]) || !std::isfinite(expected[i]) ||
                    std::fabs(input[i] - expected[i]) > 1e-5f) {
                    std::cerr << "Async halo mismatch at index " << i
                              << ": actual=" << input[i]
                              << " expected=" << expected[i] << "\n";
                    mismatch = 1;
                    break;
                }
            }
        }
        MPI_Bcast(&mismatch, 1, MPI_INT, 0, runtime.communicator());

        if (!check_timeout_budget(start_wtime, opt.timeout_seconds, runtime.communicator(), rank,
                                  "verification")) {
            return 1;
        }

        if (rank == 0 && mismatch == 0) {
            std::cout << "PASS: asynchronous OpenCL halo matches two-step CPU reference"
                      << " (ranks=" << runtime.size()
                      << ", min_per_rank=" << topo.min_per_rank
                      << ", total_devices=" << topo.total_devices
                      << ", gpu=" << (topo.using_gpu ? 1 : 0) << ")\n";
        }
        return mismatch;
    } catch (const std::exception& e) {
        std::cerr << "Preflight failed: " << e.what() << "\n";
        int initialized = 0;
        int finalized = 0;
        MPI_Initialized(&initialized);
        MPI_Finalized(&finalized);
        if (initialized && !finalized) MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
