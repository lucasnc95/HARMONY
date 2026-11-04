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
#include "sample_admission.hpp"
#include "diagnostic_trace.hpp"
#include "benchmark_results.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    std::size_t n = 1000000;
    int iterations = 100;
    std::string balance_algo = "erad";
    float perturbation = 0.10f;
    int balance_interval = 10;
    std::string output_csv = "leanmd.csv";
    int cutoff = 4;
    unsigned int seed = 0;
    bool quiet_runtime_metrics = false;
    bool require_two_rank_gpu = false;
    std::string device_inventory_json;
    std::string diagnostics_prefix;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--n" && i + 1 < argc) {
            n = static_cast<std::size_t>(std::strtoull(argv[++i], nullptr, 10));
        } else if (arg == "--iterations" && i + 1 < argc) {
            iterations = std::atoi(argv[++i]);
        } else if (arg == "--balance-algo" && i + 1 < argc) {
            balance_algo = argv[++i];
        } else if (arg == "--perturbation" && i + 1 < argc) {
            perturbation = std::strtof(argv[++i], nullptr);
        } else if (arg == "--balance-interval" && i + 1 < argc) {
            balance_interval = std::atoi(argv[++i]);
        } else if (arg == "--output-csv" && i + 1 < argc) {
            output_csv = argv[++i];
        } else if (arg == "--k" && i + 1 < argc) {
            cutoff = std::max(1, std::atoi(argv[++i]));
        } else if (arg == "--seed" && i + 1 < argc) {
            seed = static_cast<unsigned int>(std::stoul(argv[++i]));
        } else if (arg == "--quiet-runtime-metrics") {
            quiet_runtime_metrics = true;
        } else if (arg == "--require-two-rank-gpu") {
            require_two_rank_gpu = true;
        } else if (arg == "--device-inventory-json" && i + 1 < argc) {
            device_inventory_json = argv[++i];
        } else if (arg == "--diagnostics-prefix" && i + 1 < argc) {
            diagnostics_prefix = argv[++i];
        }
    }

    dcl::Runtime runtime = dcl::Runtime::create(argc, argv);
    runtime.set_verbose_metrics(!quiet_runtime_metrics);
    const int rank = runtime.rank();
    if (n == 0 || n > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        iterations <= 0) {
        if (rank == 0) std::cerr << "Invalid benchmark dimensions or iteration count\n";
        MPI_Finalize();
        return 1;
    }

    runtime.discover_devices({dcl::DeviceKind::gpu, 0});
    int local_devices = static_cast<int>(runtime.devices().size());
    int total_devices = 0;
    dcl::detail::check_mpi(MPI_Allreduce(&local_devices, &total_devices, 1, MPI_INT, MPI_SUM, runtime.communicator()),
                           "MPI_Allreduce(total GPUs)");
    const bool opencl_available = total_devices > 0;
    if (!benchmark::admit_sample(runtime, require_two_rank_gpu, device_inventory_json)) {
        MPI_Finalize();
        return 1;
    }
    benchmark::DiagnosticTrace trace(diagnostics_prefix, rank, seed, balance_algo);
    for (const auto& device : runtime.devices()) {
        std::cout << "[device] rank=" << rank
                  << " global_index=" << device.global_index
                  << " name=" << device.name << "\n";
    }

    if (!opencl_available) {
        // CPU simulation fallback: 2 simulated devices with heterogeneous performance
        runtime.set_simulated_devices_count(2);
    }

    dcl::PartitionSpec ps;
    ps.global_elements = n;
    ps.units_per_element = 1;
    ps.bytes_per_unit = sizeof(float);
    ps.granularity = 1;
    runtime.set_partition(ps);

    // Setup load calculation algorithm callback
    using LoadAlgoFunc = std::function<std::vector<float>(
        const std::vector<double>& times,
        const std::vector<dcl::DevicePartition>& partitions
    )>;

    LoadAlgoFunc algo_fn;
    if (balance_algo == "erad") {
        algo_fn = [](const std::vector<double>& t, const std::vector<dcl::DevicePartition>& p) {
            return dcl::erad_loads(t, p);
        };
    } else if (balance_algo == "hwtopolb") {
        algo_fn = [perturbation, seed](const std::vector<double>& t, const std::vector<dcl::DevicePartition>& p) {
            return dcl::hwtopolb_loads(t, p, perturbation, seed);
        };
    } else {
        if (rank == 0) {
            std::cerr << "Unknown balance-algo: " << balance_algo << "\n";
        }
        MPI_Finalize();
        return 1;
    }

    // Allocate synthetic MD particle data: position vectors and forces
    std::vector<float> pos_x(n), pos_y(n), pos_z(n), force_out(n, 0.0f);
    for (std::size_t i = 0; i < n; ++i) {
        const float fi = static_cast<float>(i);
        pos_x[i] = fi * 0.1f;
        pos_y[i] = std::sin(fi * 0.05f);
        pos_z[i] = std::cos(fi * 0.05f);
    }

    dcl::FieldHandle h_pos_x, h_pos_y, h_pos_z, h_force;
    dcl::KernelHandle h_kernel;
    dcl::KernelBinding binding;
    if (opencl_available) {
        dcl::FieldSpec f_x{"pos_x", n, 1, sizeof(float), dcl::BufferUsage::read_only, pos_x.data(), dcl::RedistributionDependency::proportional};
        dcl::FieldSpec f_y{"pos_y", n, 1, sizeof(float), dcl::BufferUsage::read_only, pos_y.data(), dcl::RedistributionDependency::proportional};
        dcl::FieldSpec f_z{"pos_z", n, 1, sizeof(float), dcl::BufferUsage::read_only, pos_z.data(), dcl::RedistributionDependency::proportional};
        dcl::FieldSpec f_f{"force_out", n, 1, sizeof(float), dcl::BufferUsage::write_only, force_out.data(), dcl::RedistributionDependency::proportional};
        h_pos_x = runtime.create_field(f_x);
        h_pos_y = runtime.create_field(f_y);
        h_pos_z = runtime.create_field(f_z);
        h_force = runtime.create_field(f_f);

        dcl::KernelSpec k_spec{"benchmarks/leanmd.cl", "leanmd_pair_force", ""};
        h_kernel = runtime.create_kernel(k_spec);

        binding = runtime.bind(h_kernel)
            .arg(0, h_pos_x)
            .arg(1, h_pos_y)
            .arg(2, h_pos_z)
            .arg(3, h_force)
            .arg(4, dcl::ScalarArg(static_cast<int>(n)))
            .arg(5, dcl::ScalarArg(cutoff))
            .build();
    }

    if (!trace.prepare(runtime.communicator())) {
        MPI_Finalize();
        return 1;
    }
    trace.inventory(runtime.devices());

    std::vector<benchmark::StepRecord> records;
    records.reserve(static_cast<std::size_t>(iterations));

    std::vector<double> prev_device_times;
    const auto total_bench_start = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point loop_end;

    for (int it = 0; it < iterations; ++it) {
        const auto step_start = std::chrono::steady_clock::now();
        const auto old_snapshot = trace.enabled() ? runtime.partitions() : std::vector<dcl::DevicePartition>{};
        trace.partitions(it, "old", runtime.partitions());
        bool do_rebalance = (balance_interval > 0 && it > 0 && (it % balance_interval == 0) && !prev_device_times.empty());
        double gain = 0.0;

        if (do_rebalance) {
            const auto& old_parts = runtime.partitions();
            const auto plan_start = std::chrono::steady_clock::now();
            double total_cap = 0.0;
            std::vector<double> capacity(old_parts.size(), 0.0);
            double t_max_current = 0.0;

            for (std::size_t i = 0; i < old_parts.size(); ++i) {
                if (prev_device_times[i] > t_max_current) {
                    t_max_current = prev_device_times[i];
                }
                if (old_parts[i].element_count > 0 && prev_device_times[i] > 1e-12) {
                    capacity[i] = static_cast<double>(old_parts[i].element_count) / prev_device_times[i];
                    total_cap += capacity[i];
                }
            }

            std::vector<float> new_loads;
            if (rank == 0) {
                new_loads = algo_fn(prev_device_times, runtime.partitions());
            }
            const auto plan_end = std::chrono::steady_clock::now();
            trace.phase(it, "rebalance_plan", plan_start, plan_end);
            runtime.rebalance_to(new_loads);
            const auto apply_end = std::chrono::steady_clock::now();
            trace.phase(it, "rebalance_apply", plan_end, apply_end);

            const auto& new_parts = runtime.partitions();
            double t_max_projected = t_max_current;
            if (total_cap > 0.0) {
                t_max_projected = 0.0;
                for (std::size_t i = 0; i < new_parts.size(); ++i) {
                    if (new_parts[i].element_count == 0) continue;
                    if (i >= capacity.size() || capacity[i] <= 0.0) {
                        t_max_projected = t_max_current;
                        break;
                    }
                    const double proj = static_cast<double>(new_parts[i].element_count) / capacity[i];
                    if (proj > t_max_projected) {
                        t_max_projected = proj;
                    }
                }
            }
            gain = std::max(0.0, t_max_current - t_max_projected);
        }

        trace.partitions(it, "new", runtime.partitions(), trace.enabled() ? &old_snapshot : nullptr);
        const auto& partitions = runtime.partitions();
        std::vector<double> local_device_times(partitions.size(), 0.0);

        const auto execution_start = std::chrono::steady_clock::now();
        if (opencl_available) {
            auto builder = runtime.step("leanmd_step");
            dcl::AutoBalancePolicy timing_policy;
            timing_policy.mode = dcl::BalanceMode::off;
            timing_policy.interval = 1;
            builder.with_balance(timing_policy);
            builder.tag_field(h_pos_x, dcl::StepFieldRole::read_source);
            builder.tag_field(h_pos_y, dcl::StepFieldRole::read_source);
            builder.tag_field(h_pos_z, dcl::StepFieldRole::read_source);
            builder.tag_field(h_force, dcl::StepFieldRole::write_target);
            builder.invoke(binding, dcl::LaunchGeometry{0, n, std::nullopt});
            builder.synchronize_at_end(true);
            const auto execute_start = std::chrono::steady_clock::now();
            runtime.execute(builder.build());
            trace.phase(it, "runtime_execute", execute_start, std::chrono::steady_clock::now());
            
            for (std::size_t p_idx = 0; p_idx < partitions.size(); ++p_idx) {
                const auto& p = partitions[p_idx];
                if (p.owning_rank == rank && p.local_index >= 0 && p.local_index < static_cast<int>(runtime.device_timings().size())) {
                    local_device_times[p_idx] = runtime.device_timings()[p.local_index].kernel_seconds_last;
                }
            }
        } else {
            // CPU fallback simulation
            for (std::size_t p_idx = 0; p_idx < partitions.size(); ++p_idx) {
                const auto& p = partitions[p_idx];
                if (p.owning_rank != rank) continue;
                
                const std::size_t offset = p.global_offset;
                const std::size_t count = p.element_count;
                const int dev_id = p.device_global_index;
                const int work_passes = (dev_id == 0 ? 2 : 1);
                
                const auto t0_cpu = std::chrono::steady_clock::now();
                for (int pass = 0; pass < work_passes; ++pass) {
                    for (std::size_t i = offset; i < offset + count; ++i) {
                        const float xi = pos_x[i];
                        const float yi = pos_y[i];
                        const float zi = pos_z[i];
                        float f_total = 0.0f;

                        const std::size_t s = (i > static_cast<std::size_t>(cutoff)) ? (i - cutoff) : 0;
                        const std::size_t e = std::min(n - 1, i + cutoff);

                        for (std::size_t j = s; j <= e; ++j) {
                            if (i == j) continue;
                            const float dx = xi - pos_x[j];
                            const float dy = yi - pos_y[j];
                            const float dz = zi - pos_z[j];
                            const float r2 = dx * dx + dy * dy + dz * dz + 1.0e-6f;
                            const float r2inv = 1.0f / r2;
                            const float r6inv = r2inv * r2inv * r2inv;
                            f_total += r2inv * r6inv * (2.0f * r6inv - 1.0f);
                        }
                        force_out[i] = f_total;
                    }
                }
                const auto t1_cpu = std::chrono::steady_clock::now();
                local_device_times[p_idx] = std::chrono::duration<double>(t1_cpu - t0_cpu).count();
            }
        }

        if (!opencl_available) trace.phase(it, "cpu_simulation", execution_start, std::chrono::steady_clock::now());
        trace.device_events(it, partitions, local_device_times, opencl_available);
        const auto step_end = std::chrono::steady_clock::now();
        double local_step_time = std::chrono::duration<double>(step_end - step_start).count();

        double t_max_step = 0.0;
        dcl::detail::check_mpi(MPI_Allreduce(&local_step_time, &t_max_step, 1, MPI_DOUBLE, MPI_MAX, runtime.communicator()),
                               "MPI_Allreduce(step time)");
        const auto step_reduce_end = std::chrono::steady_clock::now();
        trace.phase(it, "mpi_step_max", step_end, step_reduce_end);

        std::vector<double> device_times(partitions.size(), 0.0);
        dcl::detail::check_mpi(MPI_Allreduce(local_device_times.data(), device_times.data(),
                               static_cast<int>(partitions.size()), MPI_DOUBLE, MPI_MAX, runtime.communicator()),
                               "MPI_Allreduce(device times)");

        const auto device_reduce_end = std::chrono::steady_clock::now();
        trace.phase(it, "mpi_device_times", step_reduce_end, device_reduce_end);
        if (it == iterations - 1) loop_end = device_reduce_end;

        if (do_rebalance) {
            records.push_back({it, t_max_step, "rebalance", 1, gain});
        } else {
            records.push_back({it, t_max_step, "compute", 0, 0.0});
        }

        prev_device_times = device_times;
        if (!opencl_available) runtime.set_simulated_times(device_times);
    }

    trace.phase(iterations, "loop_total", total_bench_start, loop_end);
    const auto validation_start = std::chrono::steady_clock::now();
    const auto& partitions = runtime.partitions();
    if (opencl_available) runtime.gather(h_force, force_out.data(), n * sizeof(float));
    const auto gather_end = std::chrono::steady_clock::now();
    trace.phase(iterations, "gather", validation_start, gather_end);
    int local_mismatch = 0;
    for (const auto& part : partitions) {
        if (!opencl_available && part.owning_rank != rank) continue;
        if (opencl_available && rank != 0) continue;
        for (std::size_t i = part.global_offset;
             i < part.global_offset + part.element_count; ++i) {
            float expected = 0.0f;
            const std::size_t start = i > static_cast<std::size_t>(cutoff)
                ? i - static_cast<std::size_t>(cutoff) : 0;
            const std::size_t end = std::min(n - 1, i + static_cast<std::size_t>(cutoff));
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
            if (!std::isfinite(force_out[i]) || !std::isfinite(expected) ||
                std::fabs(force_out[i] - expected) >
                    1e-4f + 1e-4f * std::fabs(expected)) {
                std::cerr << "LeanMD verification failed at index " << i << "\n";
                local_mismatch = 1;
                break;
            }
        }
    }
    const auto reference_end = std::chrono::steady_clock::now();
    trace.phase(iterations, "reference", gather_end, reference_end);
    int any_mismatch = 0;
    dcl::detail::check_mpi(MPI_Allreduce(&local_mismatch, &any_mismatch, 1, MPI_INT, MPI_MAX,
                           runtime.communicator()), "MPI_Allreduce(LeanMD verification)");
    trace.phase(iterations, "mpi_validation", reference_end, std::chrono::steady_clock::now());
    if (any_mismatch != 0) MPI_Abort(runtime.communicator(), 1);
    const auto validation_end = std::chrono::steady_clock::now();
    const auto total_bench_end = std::chrono::steady_clock::now();
    const benchmark::Durations durations = benchmark::reduce_durations(
        runtime.communicator(), total_bench_start, loop_end,
        validation_start, validation_end, total_bench_end);

    if (!trace.finish(runtime.communicator())) {
        MPI_Finalize();
        return 1;
    }

    if (rank == 0 && !benchmark::write_results(
            output_csv, "leanmd", balance_algo, opencl_available, records, durations)) {
        MPI_Finalize();
        return 1;
    }

    MPI_Finalize();
    return 0;
}
