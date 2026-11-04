
#ifndef DCL_RUNTIME_IMPL_HPP
#define DCL_RUNTIME_IMPL_HPP

#include "runtime.hpp"
#include <CL/cl.h>
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>
#include <iostream>
#include <chrono>
#include <climits>

namespace dcl {

namespace detail {

static inline void check_mpi(int code, const char* what) {
    if (code == MPI_SUCCESS) return;
    char err_string[MPI_MAX_ERROR_STRING];
    int len = 0;
    MPI_Error_string(code, err_string, &len);
    std::ostringstream oss;
    oss << what << " failed: " << std::string(err_string, len);
    throw Error(oss.str());
}

static inline void check_cl(cl_int code, const char* what) {
    if (code == CL_SUCCESS) return;
    std::ostringstream oss;
    oss << "OpenCL failure at " << what << " code=" << code;
    throw Error(oss.str());
}

static inline std::string slurp_file(const std::string& path) {
    std::ifstream in(path.c_str(), std::ios::binary);
    if (!in) {
        throw Error("Could not open kernel source file: " + path);
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}


inline std::vector<float> cumulative_to_individual(
    const std::vector<float>& cumulative
) {
    std::vector<float> individual(cumulative.size(), 0.0f);

    if (cumulative.empty()) return individual;

    float prev = 0.0f;
    for (std::size_t i = 0; i < cumulative.size(); ++i) {
        individual[i] = cumulative[i] - prev;
        prev = cumulative[i];
    }

    return individual;
}

inline void print_loads_debug(const std::vector<float>& cumulative) {
    if (cumulative.empty()) return;

    const std::vector<float> individual =
        detail::cumulative_to_individual(cumulative);

    float sum = 0.0f;

    std::cout << "  ---- Loads ----\n";
    for (std::size_t i = 0; i < cumulative.size(); ++i) {
        sum += individual[i];

        std::cout
            << "  part[" << i << "] "
            << "cum=" << (100.0f * cumulative[i]) << "% "
            << "share=" << (100.0f * individual[i]) << "%\n";
    }

    std::cout << "  total=" << (100.0f * sum) << "%\n";
}


static inline DeviceKind classify_device_kind(cl_device_type type) {
    if (type & CL_DEVICE_TYPE_GPU) return DeviceKind::gpu;
    if (type & CL_DEVICE_TYPE_CPU) return DeviceKind::cpu;
    if (type & CL_DEVICE_TYPE_ACCELERATOR) return DeviceKind::accelerator;
    return DeviceKind::all;
}

static inline bool matches(DeviceKind wanted, DeviceKind actual) {
    return wanted == DeviceKind::all || wanted == actual;
}

static inline cl_mem_flags to_opencl_flags(BufferUsage usage) {
    switch (usage) {
        case BufferUsage::read_only:  return CL_MEM_READ_ONLY;
        case BufferUsage::write_only: return CL_MEM_WRITE_ONLY;
        case BufferUsage::read_write: return CL_MEM_READ_WRITE;
    }
    return CL_MEM_READ_WRITE;
}

static inline bool intersect_1d(
    std::size_t off1,
    std::size_t len1,
    std::size_t off2,
    std::size_t len2,
    std::size_t& out_off,
    std::size_t& out_len
) {
    const std::size_t a0 = off1;
    const std::size_t a1 = off1 + len1;
    const std::size_t b0 = off2;
    const std::size_t b1 = off2 + len2;
    const std::size_t lo = std::max(a0, b0);
    const std::size_t hi = std::min(a1, b1);
    if (hi <= lo) {
        out_off = 0;
        out_len = 0;
        return false;
    }
    out_off = lo;
    out_len = hi - lo;
    return true;
}

static inline float max_abs_diff(const std::vector<float>& a,
                                 const std::vector<float>& b) {
    if (a.size() != b.size()) return std::numeric_limits<float>::infinity();
    float m = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i) {
        m = std::max(m, std::fabs(a[i] - b[i]));
    }
    return m;
}


inline std::vector<float> compute_loads_from_times_prefix_inverse(
    const std::vector<double>& tempos_medidos
) {
    const int participantes = static_cast<int>(tempos_medidos.size());
    std::vector<float> cargas_novas(static_cast<std::size_t>(participantes), 1.0f);

    if (participantes <= 0) {
        return {};
    }

    if (participantes == 1) {
        cargas_novas[0] = 1.0f;
        return cargas_novas;
    }

    std::vector<double> capacidade(static_cast<std::size_t>(participantes), 0.0);
    double capacidade_total = 0.0;

    for (int i = 0; i < participantes; ++i) {
        const double t = std::max(tempos_medidos[static_cast<std::size_t>(i)], 1.0e-9);
        capacidade[static_cast<std::size_t>(i)] = 1.0 / t;
        capacidade_total += capacidade[static_cast<std::size_t>(i)];
    }

    if (capacidade_total <= 0.0) {
        const float passo = 1.0f / static_cast<float>(participantes);
        float acumulada = 0.0f;
        for (int i = 0; i < participantes; ++i) {
            acumulada += passo;
            cargas_novas[static_cast<std::size_t>(i)] = acumulada;
        }
        cargas_novas.back() = 1.0f;
        return cargas_novas;
    }

    double carga_acumulada = 0.0;
    for (int i = 0; i < participantes; ++i) {
        const double fatia =
            capacidade[static_cast<std::size_t>(i)] / capacidade_total;
        carga_acumulada += fatia;
        cargas_novas[static_cast<std::size_t>(i)] =
            static_cast<float>(carga_acumulada);
    }

    cargas_novas.back() = 1.0f;
    return cargas_novas;
}

inline std::vector<float> compute_loads_from_partition_throughput(
    const std::vector<double>& times,
    const std::vector<DevicePartition>& partitions
) {
    if (times.empty() || times.size() != partitions.size()) {
        return {};
    }

    std::vector<double> capacity(times.size(), 0.0);
    double total_capacity = 0.0;

    for (std::size_t i = 0; i < times.size(); ++i) {
        if (partitions[i].element_count == 0 || times[i] <= 1.0e-12) {
            continue;
        }

        capacity[i] =
            static_cast<double>(partitions[i].element_count) / times[i];
        total_capacity += capacity[i];
    }

    std::vector<float> loads(times.size(), 0.0f);
    if (total_capacity <= 0.0) {
        const float step = 1.0f / static_cast<float>(times.size());
        float acc = 0.0f;
        for (std::size_t i = 0; i < loads.size(); ++i) {
            acc += step;
            loads[i] = acc;
        }
        loads.back() = 1.0f;
        return loads;
    }

    double cumulative = 0.0;
    for (std::size_t i = 0; i < capacity.size(); ++i) {
        cumulative += capacity[i] / total_capacity;
        loads[i] = static_cast<float>(cumulative);
    }

    loads.back() = 1.0f;
    return loads;
}

static inline std::vector<float> compute_loads_from_times(const std::vector<double>& times) {
    std::vector<float> loads(times.size(), 0.0f);
    if (times.empty()) return loads;

    double sum_inv = 0.0;
    for (double t : times) {
        if (t < 1.0e-12) t = 1.0e-12;
        sum_inv += 1.0 / t;
    }

    if (sum_inv <= 0.0) {
        const float eq = 1.0f / static_cast<float>(times.size());
        std::fill(loads.begin(), loads.end(), eq);
        return loads;
    }

    for (std::size_t i = 0; i < times.size(); ++i) {
        double t = times[i];
        if (t < 1.0e-12) t = 1.0e-12;
        loads[i] = static_cast<float>((1.0 / t) / sum_inv);
    }
    return loads;
}

static inline float l2_norm_diff(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return std::numeric_limits<float>::infinity();
    double acc = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        acc += d * d;
    }
    return static_cast<float>(std::sqrt(acc));
}

struct PlatformContext {
    cl_platform_id platform{nullptr};
    cl_context context{nullptr};
    std::vector<cl_device_id> devices;
};

struct LocalDevice {
    int platform_index{-1};
    int device_index_in_platform{-1};
    int local_index{-1};
    int global_index{-1};

    cl_device_id device{nullptr};
    cl_context context{nullptr};
    cl_command_queue kernel_queue{nullptr};
    cl_command_queue transfer_queue{nullptr};

    cl_uint compute_units{0};
    DeviceKind kind{DeviceKind::all};
    std::string name;
};

struct RegisteredField {
    FieldSpec spec;
    std::vector<cl_mem> replicas;
};

struct RegisteredKernel {
    KernelSpec spec;
    std::vector<cl_program> programs_per_platform;
    std::vector<cl_kernel> kernels_per_local_device;
};

} // namespace detail


struct ProfileSegment {
    std::size_t max_volume;
    double m;
    double b;
};

class Runtime::Impl {
public:
    Impl(int& argc, char**& argv)
        : mpi_initialized_by_runtime_(false), argc_(argc), argv_(argv) {}

    ~Impl() {
        release_tracked_field_events();
        reset_balance_window_events();

        for (auto& kv : kernels_) {
            detail::RegisteredKernel& rk = kv.second;
            for (cl_kernel k : rk.kernels_per_local_device) {
                if (k != nullptr) clReleaseKernel(k);
            }
            for (cl_program p : rk.programs_per_platform) {
                if (p != nullptr) clReleaseProgram(p);
            }
        }

        for (auto& kv : fields_) {
            detail::RegisteredField& rf = kv.second;
            for (cl_mem mem : rf.replicas) {
                if (mem != nullptr) clReleaseMemObject(mem);
            }
        }

        for (std::size_t i = 0; i < local_devices_.size(); ++i) {
            if (local_devices_[i].kernel_queue != nullptr) clReleaseCommandQueue(local_devices_[i].kernel_queue);
            if (local_devices_[i].transfer_queue != nullptr) clReleaseCommandQueue(local_devices_[i].transfer_queue);
        }

        for (std::size_t i = 0; i < platforms_.size(); ++i) {
            if (platforms_[i].context != nullptr) clReleaseContext(platforms_[i].context);
        }
    }

    void initialize_mpi_from_runtime() {
        int initialized = 0;
        MPI_Initialized(&initialized);
        if (!initialized) {
            int provided = 0;
            MPI_Init_thread(&argc_, &argv_, MPI_THREAD_SERIALIZED, &provided);
            mpi_initialized_by_runtime_ = true;
        } else {
            mpi_initialized_by_runtime_ = false;
        }
        detail::check_mpi(MPI_Comm_rank(MPI_COMM_WORLD, &rank_), "MPI_Comm_rank");
        detail::check_mpi(MPI_Comm_size(MPI_COMM_WORLD, &size_), "MPI_Comm_size");
        comm_ = MPI_COMM_WORLD;
    }

    void discover_devices(const DeviceSelection& selection) {
        clear_runtime_state();

        cl_uint n_platforms = 0;
        detail::check_cl(clGetPlatformIDs(0, nullptr, &n_platforms), "clGetPlatformIDs(count)");
        if (n_platforms == 0) {
            all_device_counts_.assign(size_, 0);
            return;
        }

        std::vector<cl_platform_id> platform_ids(n_platforms);
        detail::check_cl(clGetPlatformIDs(n_platforms, platform_ids.data(), nullptr), "clGetPlatformIDs(list)");

        int next_local_index = 0;
        const int max_per_rank = (selection.max_devices_per_rank <= 0)
            ? std::numeric_limits<int>::max()
            : selection.max_devices_per_rank;

        for (cl_uint p = 0; p < n_platforms; ++p) {
            cl_uint dev_count = 0;
            cl_int err = clGetDeviceIDs(platform_ids[p], CL_DEVICE_TYPE_ALL, 0, nullptr, &dev_count);
            if (err == CL_DEVICE_NOT_FOUND || dev_count == 0) continue;
            detail::check_cl(err, "clGetDeviceIDs(count)");

            std::vector<cl_device_id> platform_devices(dev_count);
            detail::check_cl(
                clGetDeviceIDs(platform_ids[p], CL_DEVICE_TYPE_ALL, dev_count, platform_devices.data(), nullptr),
                "clGetDeviceIDs(list)"
            );

            std::vector<cl_device_id> selected_devices;
            for (cl_uint i = 0; i < dev_count; ++i) {
                if (next_local_index >= max_per_rank) break;
                cl_device_type dtype = 0;
                detail::check_cl(
                    clGetDeviceInfo(platform_devices[i], CL_DEVICE_TYPE, sizeof(dtype), &dtype, nullptr),
                    "clGetDeviceInfo(CL_DEVICE_TYPE)"
                );
                const DeviceKind kind = detail::classify_device_kind(dtype);
                if (!detail::matches(selection.kind, kind)) continue;
                selected_devices.push_back(platform_devices[i]);
                ++next_local_index;
            }

            if (selected_devices.empty()) continue;

            detail::PlatformContext pc{};
            pc.platform = platform_ids[p];
            pc.devices = selected_devices;

            cl_context_properties props[] = {
                CL_CONTEXT_PLATFORM,
                reinterpret_cast<cl_context_properties>(platform_ids[p]),
                0
            };

            cl_int ctx_err = CL_SUCCESS;
            pc.context = clCreateContext(
                props,
                static_cast<cl_uint>(selected_devices.size()),
                selected_devices.data(),
                nullptr,
                nullptr,
                &ctx_err
            );
            detail::check_cl(ctx_err, "clCreateContext");

            platforms_.push_back(pc);
            const int platform_index = static_cast<int>(platforms_.size() - 1);

            for (std::size_t i = 0; i < selected_devices.size(); ++i) {
                detail::LocalDevice ld{};
                ld.platform_index = platform_index;
                ld.device_index_in_platform = static_cast<int>(i);
                ld.local_index = static_cast<int>(local_devices_.size());
                ld.device = selected_devices[i];
                ld.context = platforms_[platform_index].context;

                char name_buf[512] = {};
                cl_uint cus = 0;
                cl_device_type dtype = 0;

                detail::check_cl(clGetDeviceInfo(ld.device, CL_DEVICE_NAME, sizeof(name_buf), name_buf, nullptr), "clGetDeviceInfo(CL_DEVICE_NAME)");
                detail::check_cl(clGetDeviceInfo(ld.device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cus), &cus, nullptr), "clGetDeviceInfo(CL_DEVICE_MAX_COMPUTE_UNITS)");
                detail::check_cl(clGetDeviceInfo(ld.device, CL_DEVICE_TYPE, sizeof(dtype), &dtype, nullptr), "clGetDeviceInfo(CL_DEVICE_TYPE)");

                ld.name = std::string(name_buf);
                if (!ld.name.empty() && ld.name.back() == '\0') ld.name.pop_back();
                ld.compute_units = static_cast<unsigned>(cus);
                ld.kind = detail::classify_device_kind(dtype);

                cl_int qerr = CL_SUCCESS;
#if CL_TARGET_OPENCL_VERSION >= 200
                const cl_queue_properties qprops[] = {
                    CL_QUEUE_PROPERTIES,
                    static_cast<cl_queue_properties>(CL_QUEUE_PROFILING_ENABLE),
                    0
                };
                ld.kernel_queue = clCreateCommandQueueWithProperties(ld.context, ld.device, qprops, &qerr);
                detail::check_cl(qerr, "clCreateCommandQueueWithProperties(kernel_queue)");
                ld.transfer_queue = clCreateCommandQueueWithProperties(ld.context, ld.device, qprops, &qerr);
                detail::check_cl(qerr, "clCreateCommandQueueWithProperties(transfer_queue)");
#else
                ld.kernel_queue = clCreateCommandQueue(ld.context, ld.device, CL_QUEUE_PROFILING_ENABLE, &qerr);
                detail::check_cl(qerr, "clCreateCommandQueue(kernel_queue)");
                ld.transfer_queue = clCreateCommandQueue(ld.context, ld.device, CL_QUEUE_PROFILING_ENABLE, &qerr);
                detail::check_cl(qerr, "clCreateCommandQueue(transfer_queue)");
#endif

                local_devices_.push_back(ld);

                DeviceInfo info{};
                info.rank = rank_;
                info.local_index = ld.local_index;
                info.global_index = -1;
                info.name = ld.name;
                info.kind = ld.kind;
                info.compute_units = ld.compute_units;
                devices_.push_back(info);
            }

            if (next_local_index >= max_per_rank) break;
        }

        const int local_count = static_cast<int>(devices_.size());
        all_device_counts_.assign(size_, 0);
        detail::check_mpi(
            MPI_Allgather(&local_count, 1, MPI_INT, all_device_counts_.data(), 1, MPI_INT, comm_),
            "MPI_Allgather(device_counts)"
        );

        int global_base = 0;
        for (int r = 0; r < rank_; ++r) global_base += all_device_counts_[r];
        for (std::size_t i = 0; i < devices_.size(); ++i) {
            devices_[i].global_index = global_base + static_cast<int>(i);
            local_devices_[i].global_index = devices_[i].global_index;
        }

        device_timings_.assign(local_devices_.size(), DeviceTiming{});
        last_elapsed_local_.assign(local_devices_.size(), 0.0);
        balance_window_start_events_.assign(local_devices_.size(), nullptr);
        balance_window_end_events_.assign(local_devices_.size(), nullptr);
        balance_window_valid_.assign(local_devices_.size(), false);
        balance_window_kernel_events_.assign(local_devices_.size(), {});
        balance_window_begin_iteration_ = 0;
    }

    const std::vector<DeviceInfo>& devices() const noexcept { return devices_; }
    const std::vector<DevicePartition>& partitions() const noexcept { return partitions_; }
    const std::vector<DeviceTiming>& device_timings() const noexcept { return device_timings_; }
    int rank() const noexcept { return rank_; }
    int size() const noexcept { return size_; }
    MPI_Comm communicator() const noexcept { return comm_; }

    FieldHandle create_field(const FieldSpec& spec) {
        if (local_devices_.empty()) throw Error("discover_devices() must be called before create_field()");
        if (spec.global_elements == 0) throw Error("FieldSpec.global_elements must be > 0");
        if (spec.units_per_element == 0) throw Error("FieldSpec.units_per_element must be > 0");
        if (spec.bytes_per_unit == 0) throw Error("FieldSpec.bytes_per_unit must be > 0");

        FieldHandle h{next_field_id_++};
        detail::RegisteredField rf;
        rf.spec = spec;
        rf.replicas.resize(local_devices_.size(), nullptr);

        const std::size_t total_bytes = spec.global_elements * spec.units_per_element * spec.bytes_per_unit;
        const cl_mem_flags flags = detail::to_opencl_flags(spec.usage);

        for (std::size_t d = 0; d < local_devices_.size(); ++d) {
            cl_int err = CL_SUCCESS;
            rf.replicas[d] = clCreateBuffer(local_devices_[d].context, flags, total_bytes, nullptr, &err);
            detail::check_cl(err, "clCreateBuffer(create_field)");
        }

        fields_.insert(std::make_pair(h.value, rf));
        if (spec.host_ptr != nullptr) write_initial_field_data(h, spec.host_ptr);
        return h;
    }

    KernelHandle create_kernel(const KernelSpec& spec) {
        if (local_devices_.empty()) throw Error("discover_devices() must be called before create_kernel()");
        if (spec.source_file.empty()) throw Error("KernelSpec.source_file is empty");
        if (spec.entry_point.empty()) throw Error("KernelSpec.entry_point is empty");

        KernelHandle h{next_kernel_id_++};
        detail::RegisteredKernel rk;
        rk.spec = spec;
        rk.programs_per_platform.resize(platforms_.size(), nullptr);
        rk.kernels_per_local_device.resize(local_devices_.size(), nullptr);

        const std::string src = detail::slurp_file(spec.source_file);
        const char* src_ptr = src.c_str();
        const size_t src_len = src.size();

        for (std::size_t p = 0; p < platforms_.size(); ++p) {
            cl_int err = CL_SUCCESS;
            cl_program program = clCreateProgramWithSource(platforms_[p].context, 1, &src_ptr, &src_len, &err);
            detail::check_cl(err, "clCreateProgramWithSource");

            err = clBuildProgram(
                program,
                static_cast<cl_uint>(platforms_[p].devices.size()),
                platforms_[p].devices.data(),
                spec.build_options.empty() ? nullptr : spec.build_options.c_str(),
                nullptr,
                nullptr
            );

            if (err != CL_SUCCESS) {
                std::ostringstream oss;
                oss << "clBuildProgram failed for kernel " << spec.entry_point << "\n";
                for (std::size_t i = 0; i < platforms_[p].devices.size(); ++i) {
                    size_t log_size = 0;
                    clGetProgramBuildInfo(program, platforms_[p].devices[i], CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
                    std::vector<char> log(log_size + 1, '\0');
                    clGetProgramBuildInfo(program, platforms_[p].devices[i], CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
                    oss << "---- device " << i << " ----\n" << log.data() << "\n";
                }
                clReleaseProgram(program);
                throw Error(oss.str());
            }
            rk.programs_per_platform[p] = program;
        }

        for (std::size_t d = 0; d < local_devices_.size(); ++d) {
            const int p = local_devices_[d].platform_index;
            cl_int err = CL_SUCCESS;
            cl_kernel k = clCreateKernel(rk.programs_per_platform[p], spec.entry_point.c_str(), &err);
            detail::check_cl(err, "clCreateKernel");
            rk.kernels_per_local_device[d] = k;
        }

        kernels_.insert(std::make_pair(h.value, rk));
        active_kernel_ = h.value;
        return h;
    }

    void set_partition(const PartitionSpec& spec) {
        if (spec.global_elements == 0) throw Error("PartitionSpec.global_elements must be > 0");
        if (spec.units_per_element == 0) throw Error("PartitionSpec.units_per_element must be > 0");
        if (spec.bytes_per_unit == 0) throw Error("PartitionSpec.bytes_per_unit must be > 0");
        if (spec.granularity == 0) throw Error("PartitionSpec.granularity must be > 0");

        partition_ = spec;
        rebuild_partitions_equal_like_wrapper();

        current_loads_.assign(partitions_.size(), 0.0f);
        if (!partitions_.empty()) {
            const float step = 1.0f / static_cast<float>(partitions_.size());
            float acc = 0.0f;
            for (std::size_t i = 0; i < current_loads_.size(); ++i) {
                acc += step;
                current_loads_[i] = acc;
            }
            current_loads_.back() = 1.0f;
        }
        reset_balance_window_events();
    }

    void execute(const ExecutionStep& step) {
        if (step.invocations.empty()) return;
        // execute step profiling window update
        const bool collect_balance_window = (step.balance.mode != BalanceMode::off);

        for (std::size_t inv = 0; inv < step.invocations.size(); ++inv) {
            const KernelInvocation& ki = step.invocations[inv];

            auto kit = kernels_.find(ki.binding.kernel.value);
            if (kit == kernels_.end()) throw Error("Unknown kernel in execute()");

            detail::RegisteredKernel& rk = kit->second;
            std::vector<FieldHandle> output_fields =
                fields_with_role(step, StepFieldRole::write_target);
            if (output_fields.empty()) {
                output_fields = output_fields_from_binding(ki.binding);
            }

            if (step.halo.width_elements > 0 && !step.halo.fields.empty()) {
                std::vector<std::pair<std::size_t, cl_event>> interior_events;
                run_interior_phase(rk, ki, interior_events, step.halo.width_elements);
                if (collect_balance_window) update_balance_window_events(interior_events, {});
                if (collect_balance_window) append_balance_window_kernel_events(interior_events);
                record_field_write_events(output_fields, interior_events);

                // The halo source is the read buffer in double-buffered steps, so
                // border kernels can be submitted without a device-wide wait here.
                exchange_halo_set(step.halo);

                std::vector<std::pair<std::size_t, cl_event>> border_events;
                run_border_phase(rk, ki, border_events, step.halo.width_elements);
                if (collect_balance_window) update_balance_window_events({}, border_events);
                if (collect_balance_window) append_balance_window_kernel_events(border_events);
                record_field_write_events(output_fields, border_events);

                release_kernel_events(interior_events);
                release_kernel_events(border_events);
            } else {
                std::vector<std::pair<std::size_t, cl_event>> full_events;
                run_full_phase(rk, ki, full_events);
                if (collect_balance_window) update_balance_window_events(full_events, full_events);
                if (collect_balance_window) append_balance_window_kernel_events(full_events);
                record_field_write_events(output_fields, full_events);
                release_kernel_events(full_events);
            }

            last_input_fields_.clear();
            last_output_fields_.clear();
            extract_rw_fields(ki.binding);
        }

        ++iteration_counter_;
        if (step.balance.mode != BalanceMode::off) {
            const int every = (step.balance.interval <= 0) ? 1 : step.balance.interval;
            const bool interval_hit =
                (iteration_counter_ % static_cast<std::size_t>(every)) == 0;

            const std::optional<FieldHandle> tagged_target =
                this->first_field_with_role(step, StepFieldRole::rebalance_source);

            const std::optional<FieldHandle> fallback_target =
                !last_input_fields_.empty() ? std::optional<FieldHandle>(last_input_fields_.front())
                                            : (!last_output_fields_.empty() ? std::optional<FieldHandle>(last_output_fields_.front())
                                                                            : std::nullopt);

            const std::optional<FieldHandle> target_opt = tagged_target.has_value() ? tagged_target : fallback_target;

            bool should_try = false;
            if (target_opt.has_value()) {
                if (step.balance.mode == BalanceMode::dynamic_threshold ||
                    step.balance.mode == BalanceMode::dynamic_profiled) {
                    should_try = interval_hit;
                } else if (step.balance.mode == BalanceMode::static_threshold ||
                           step.balance.mode == BalanceMode::static_profiled) {
                    bool& attempted = static_balance_attempted_[step.name];
                    if (!attempted && interval_hit) {
                        should_try = true;
                        attempted = true;
                    }
                }
            }

            if (should_try && target_opt.has_value()) {
                const FieldHandle target = *target_opt;

                if (step.balance.mode == BalanceMode::dynamic_threshold ||
                    step.balance.mode == BalanceMode::static_threshold) {
                    this->maybe_rebalance_from_timings(target, step.balance.threshold);
                } else if (step.balance.mode == BalanceMode::dynamic_profiled ||
                           step.balance.mode == BalanceMode::static_profiled) {
                    this->maybe_rebalance_profiled(target, step.balance, step.balance.interval);
                }
            }
        }

        if (step.synchronize_at_end) {
            synchronize_all_local_devices(false);
        }
    }

    void rebalance_to(const std::vector<float>& loads) {
        if (!partition_.has_value()) return;
        if (partitions_.empty()) return;
        if (loads.size() != partitions_.size()) {
            throw Error("rebalance_to(): loads size must match number of partitions");
        }

        // From now on, loads is cumulative: [0.10, 0.20, ..., 1.00].
        std::vector<float> cumulative = loads;
        float prev = 0.0f;
        for (std::size_t i = 0; i < cumulative.size(); ++i) {
            if (!std::isfinite(cumulative[i])) {
                throw Error("rebalance_to(): non-finite cumulative load");
            }
            if (cumulative[i] < 0.0f) {
                throw Error("rebalance_to(): negative cumulative load not allowed");
            }
            if (cumulative[i] < prev) cumulative[i] = prev;
            if (cumulative[i] > 1.0f) cumulative[i] = 1.0f;
            prev = cumulative[i];
        }
        cumulative.back() = 1.0f;

        const std::vector<DevicePartition> old_parts = partitions_;
        const std::vector<DevicePartition> new_parts = partitions_from_loads(cumulative);
        if (new_parts.size() != old_parts.size()) {
            throw Error("rebalance_to(): rounded partition count mismatch");
        }

        bool same = (old_parts.size() == new_parts.size());
        if (same) {
            for (std::size_t i = 0; i < old_parts.size(); ++i) {
                if (old_parts[i].global_offset != new_parts[i].global_offset ||
                    old_parts[i].element_count != new_parts[i].element_count ||
                    old_parts[i].owning_rank   != new_parts[i].owning_rank ||
                    old_parts[i].local_index   != new_parts[i].local_index) {
                    same = false;
                    break;
                }
            }
        }
        if (same) {
            current_loads_ = loads_from_partitions(partitions_);
            reset_balance_window_events();
           // print_partition_loads(current_loads_);
            return;
        }

        this->synchronize(true);
        redistribute_all_registered_fields(old_parts, new_parts);
        partitions_ = new_parts;
        current_loads_ = loads_from_partitions(partitions_);
        this->synchronize(true);
        reset_balance_window_events();
      //  print_partition_loads(current_loads_);
    }

    // void gather(FieldHandle field, void* host_dst, std::size_t bytes) {
    //     auto fit = fields_.find(field.value);
    //     if (fit == fields_.end()) throw Error("Unknown field in gather()");
    //     if (host_dst == nullptr) throw Error("gather() host_dst is null");

    //     detail::RegisteredField& rf = fit->second;
    //     const std::size_t total_bytes = rf.spec.global_elements * rf.spec.units_per_element * rf.spec.bytes_per_unit;
    //     if (bytes < total_bytes) throw Error("gather() destination buffer too small");

    //     std::vector<unsigned char> local_image(total_bytes, 0);
    //     std::vector<cl_event> read_events;

    //     for (std::size_t p = 0; p < partitions_.size(); ++p) {
    //         const DevicePartition& dp = partitions_[p];
    //         if (dp.owning_rank != rank_ || dp.local_index < 0) continue;

    //         const std::size_t d = static_cast<std::size_t>(dp.local_index);
    //         const std::size_t off = dp.global_offset * rf.spec.units_per_element * rf.spec.bytes_per_unit;
    //         const std::size_t len = dp.element_count * rf.spec.units_per_element * rf.spec.bytes_per_unit;
    //         cl_event ev = nullptr;

    //         detail::check_cl(
    //             clEnqueueReadBuffer(
    //                 local_devices_[d].transfer_queue,
    //                 rf.replicas[d],
    //                 CL_FALSE,
    //                 off,
    //                 len,
    //                 local_image.data() + off,
    //                 0,
    //                 nullptr,
    //                 &ev
    //             ),
    //             "clEnqueueReadBuffer(gather)"
    //         );
    //         read_events.push_back(ev);
    //     }

    //     for (cl_event ev : read_events) {
    //         if (ev != nullptr) {
    //             detail::check_cl(clWaitForEvents(1, &ev), "clWaitForEvents(gather)");
    //             clReleaseEvent(ev);
    //         }
    //     }

    //     if (rank_ == 0) {
    //         std::vector<unsigned char> reduced(total_bytes, 0);
    //         reduce_bytes_bor(local_image.data(), reduced.data(), total_bytes, 0);
    //         std::memcpy(host_dst, reduced.data(), total_bytes);
    //     } else {
    //         reduce_bytes_bor(local_image.data(), nullptr, total_bytes, 0);
    //     }
    // }

    void gather(FieldHandle field, void* host_dst, std::size_t bytes) {
        auto fit = fields_.find(field.value);
        if (fit == fields_.end()) throw Error("Unknown field in gather()");
        if (host_dst == nullptr) throw Error("gather() host_dst is null");

        detail::RegisteredField& rf = fit->second;
        const std::size_t total_bytes =
            rf.spec.global_elements * rf.spec.units_per_element * rf.spec.bytes_per_unit;

        if (bytes < total_bytes) {
            throw Error("gather() destination buffer too small");
        }

        // Garante que toda computação/comunicação pendente terminou antes da leitura.
        this->synchronize(true);

        unsigned char* dst = static_cast<unsigned char*>(host_dst);

        // Apenas o root realmente precisa zerar o destino.
        if (rank_ == 0) {
            std::memset(dst, 0, total_bytes);
        }

        // Tamanho do staging host usado nos ranks != 0 para ler da GPU e enviar ao root.
        // 64 MiB costuma funcionar bem sem pressionar demais a memória.
        constexpr std::size_t STAGING_BYTES = 64ull * 1024ull * 1024ull;

        std::vector<unsigned char> staging;
        if (rank_ != 0) {
            staging.resize(STAGING_BYTES);
        }

        const int tag_ub = mpi_tag_upper_bound();

        constexpr int GATHER_TAG_BASE = 1000;
        if (tag_ub < GATHER_TAG_BASE ||
            partitions_.size() > static_cast<std::size_t>(tag_ub - GATHER_TAG_BASE + 1)) {
            throw Error("gather() has too many partitions for MPI_TAG_UB");
        }

        const std::size_t bytes_per_element =
            rf.spec.units_per_element * rf.spec.bytes_per_unit;

        std::vector<std::pair<std::size_t, std::size_t>> ranges;
        ranges.reserve(partitions_.size());
        for (const DevicePartition& dp : partitions_) {
            if (dp.element_count == 0) {
                continue;
            }
            if (dp.global_offset > rf.spec.global_elements ||
                dp.element_count > rf.spec.global_elements - dp.global_offset) {
                throw Error("gather() partition range outside field bounds");
            }
            ranges.push_back(std::make_pair(
                dp.global_offset,
                dp.global_offset + dp.element_count
            ));
        }
        std::sort(ranges.begin(), ranges.end());

        std::size_t cursor = 0;
        for (const std::pair<std::size_t, std::size_t>& range : ranges) {
            if (range.first != cursor) {
                std::ostringstream oss;
                oss << "gather() partition coverage gap/overlap at element "
                    << cursor << " next_range_begin=" << range.first;
                throw Error(oss.str());
            }
            cursor = range.second;
        }
        if (cursor != rf.spec.global_elements) {
            std::ostringstream oss;
            oss << "gather() partition coverage ends at element "
                << cursor << " expected=" << rf.spec.global_elements;
            throw Error(oss.str());
        }

        for (std::size_t p = 0; p < partitions_.size(); ++p) {
            const DevicePartition& dp = partitions_[p];
            const int gather_tag = GATHER_TAG_BASE + static_cast<int>(p);

            const std::size_t part_off_bytes = dp.global_offset * bytes_per_element;
            const std::size_t part_bytes     = dp.element_count * bytes_per_element;

            if (part_bytes == 0) continue;

            // --------------------------------------------------------------------
            // Caso 1: esta partição pertence ao rank atual -> precisamos lê-la
            // --------------------------------------------------------------------
            if (dp.owning_rank == rank_) {
                if (dp.local_index < 0 ||
                    static_cast<std::size_t>(dp.local_index) >= local_devices_.size()) {
                    throw Error("Invalid local_index in gather()");
                }

                const std::size_t d = static_cast<std::size_t>(dp.local_index);

                if (rf.replicas[d] == nullptr) {
                    throw Error("Null field replica in gather()");
                }

                std::size_t done = 0;
                while (done < part_bytes) {
                    const std::size_t chunk = std::min<std::size_t>(part_bytes - done, STAGING_BYTES);

                    if (rank_ == 0) {
                        // Root lê direto da GPU para a posição final do buffer de saída.
                        detail::check_cl(
                            clEnqueueReadBuffer(
                                local_devices_[d].transfer_queue,
                                rf.replicas[d],
                                CL_TRUE,
                                part_off_bytes + done,
                                chunk,
                                dst + part_off_bytes + done,
                                0,
                                nullptr,
                                nullptr
                            ),
                            "clEnqueueReadBuffer(gather root direct)"
                        );
                    } else {
                        // Ranks remotos leem para staging e enviam ao root.
                        detail::check_cl(
                            clEnqueueReadBuffer(
                                local_devices_[d].transfer_queue,
                                rf.replicas[d],
                                CL_TRUE,
                                part_off_bytes + done,
                                chunk,
                                staging.data(),
                                0,
                                nullptr,
                                nullptr
                            ),
                            "clEnqueueReadBuffer(gather staging)"
                        );

                        std::size_t sent = 0;
                        while (sent < chunk) {
                            const int mpi_chunk = static_cast<int>(
                                std::min<std::size_t>(chunk - sent,
                                                    static_cast<std::size_t>(INT_MAX))
                            );

                            detail::check_mpi(
                                MPI_Send(
                                    staging.data() + sent,
                                    mpi_chunk,
                                    MPI_BYTE,
                                    0,
                                    gather_tag,
                                    comm_
                                ),
                                "MPI_Send(gather)"
                            );

                            sent += static_cast<std::size_t>(mpi_chunk);
                        }
                    }

                    done += chunk;
                }
            }

            // --------------------------------------------------------------------
            // Caso 2: esta partição pertence a outro rank e eu sou o root -> recebo
            // --------------------------------------------------------------------
            else if (rank_ == 0) {
                std::size_t recvd = 0;
                while (recvd < part_bytes) {
                    const std::size_t chunk =
                        std::min<std::size_t>(part_bytes - recvd, STAGING_BYTES);
                    std::size_t chunk_recvd = 0;

                    while (chunk_recvd < chunk) {
                        const int mpi_chunk = static_cast<int>(
                            std::min<std::size_t>(
                                chunk - chunk_recvd,
                                static_cast<std::size_t>(INT_MAX)
                            )
                        );

                        detail::check_mpi(
                            MPI_Recv(
                                dst + part_off_bytes + recvd + chunk_recvd,
                                mpi_chunk,
                                MPI_BYTE,
                                dp.owning_rank,
                                gather_tag,
                                comm_,
                                MPI_STATUS_IGNORE
                            ),
                            "MPI_Recv(gather)"
                        );

                        chunk_recvd += static_cast<std::size_t>(mpi_chunk);
                    }

                    recvd += chunk;
                }
            }
        }
    }







    void synchronize(bool force_finish) {
        synchronize_all_local_devices(force_finish);
        detail::check_mpi(MPI_Barrier(comm_), "MPI_Barrier(synchronize)");
    }

private:
    void reduce_bytes_bor(const unsigned char* sendbuf,
                          unsigned char* recvbuf,
                          std::size_t total_bytes,
                          int root) {
        std::size_t done = 0;
        while (done < total_bytes) {
            const std::size_t rem = total_bytes - done;
            const int chunk = static_cast<int>(std::min<std::size_t>(rem, static_cast<std::size_t>(INT_MAX)));
            detail::check_mpi(
                MPI_Reduce(sendbuf + done,
                           recvbuf ? (recvbuf + done) : nullptr,
                           chunk,
                           MPI_BYTE,
                           MPI_BOR,
                           root,
                           comm_),
                "MPI_Reduce(bytes)"
            );
            done += static_cast<std::size_t>(chunk);
        }
    }

    int mpi_tag_upper_bound() const {
        int tag_flag = 0;
        void* tag_attr = nullptr;
        detail::check_mpi(
            MPI_Comm_get_attr(comm_, MPI_TAG_UB, &tag_attr, &tag_flag),
            "MPI_Comm_get_attr(MPI_TAG_UB)"
        );
        return (tag_flag != 0 && tag_attr != nullptr)
            ? *static_cast<int*>(tag_attr)
            : 32767;
    }

    void check_mpi_tag_base(int tag_base, std::size_t total_bytes, const char* context) const {
        if (tag_base < 0) {
            throw Error(std::string(context) + ": negative MPI tag");
        }
        const std::size_t chunks =
            (total_bytes + static_cast<std::size_t>(INT_MAX) - 1) /
            static_cast<std::size_t>(INT_MAX);
        const int last_tag = tag_base + static_cast<int>(chunks == 0 ? 0 : chunks - 1);
        if (last_tag > mpi_tag_upper_bound()) {
            throw Error(std::string(context) + ": MPI tag exceeds MPI_TAG_UB");
        }
    }

    void mpi_transfer_bytes_chunked(const unsigned char* sendbuf,
                                    unsigned char* recvbuf,
                                    std::size_t total_bytes,
                                    int peer_rank,
                                    int tag_base,
                                    bool do_send,
                                    bool do_recv) {
        if (do_send && sendbuf == nullptr) {
            throw Error("mpi_transfer_bytes_chunked(): send buffer is null");
        }
        if (do_recv && recvbuf == nullptr) {
            throw Error("mpi_transfer_bytes_chunked(): recv buffer is null");
        }
        check_mpi_tag_base(tag_base, total_bytes, "mpi_transfer_bytes_chunked()");

        std::size_t offset = 0;
        int chunk_id = 0;
        while (offset < total_bytes) {
            const std::size_t remaining = total_bytes - offset;
            const int chunk = static_cast<int>(std::min<std::size_t>(remaining, static_cast<std::size_t>(INT_MAX)));
            MPI_Request reqs[2];
            int req_count = 0;
            if (do_recv) {
                detail::check_mpi(
                    MPI_Irecv(recvbuf + offset, chunk, MPI_BYTE, peer_rank, tag_base + chunk_id, comm_, &reqs[req_count++]),
                    "MPI_Irecv(chunked transfer)"
                );
            }
            if (do_send) {
                detail::check_mpi(
                    MPI_Isend(sendbuf + offset, chunk, MPI_BYTE, peer_rank, tag_base + chunk_id, comm_, &reqs[req_count++]),
                    "MPI_Isend(chunked transfer)"
                );
            }
            if (req_count > 0) {
                detail::check_mpi(MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE), "MPI_Waitall(chunked transfer)");
            }
            offset += static_cast<std::size_t>(chunk);
            ++chunk_id;
        }
    }

    void update_device_timings(const std::vector<double>& elapsed_local) {
        if (device_timings_.size() != elapsed_local.size()) {
            device_timings_.assign(elapsed_local.size(), DeviceTiming{});
        }
        for (std::size_t d = 0; d < elapsed_local.size(); ++d) {
            DeviceTiming& dt = device_timings_[d];
            dt.kernel_seconds_last = elapsed_local[d];
            dt.kernel_seconds_avg =
                (dt.kernel_seconds_avg * static_cast<double>(dt.samples) + elapsed_local[d]) /
                static_cast<double>(dt.samples + 1);
            ++dt.samples;
        }
    }

    void print_partition_loads(const std::vector<float>& loads) const {
        if (rank_ != 0) return;

        const std::vector<float> individual = detail::cumulative_to_individual(loads);

        std::cout << "DCL loads by partition:\n";
        float total = 0.0f;
        for (std::size_t i = 0; i < loads.size(); ++i) {
            const DevicePartition& dp = partitions_[i];
            const float share = (i < individual.size()) ? individual[i] : 0.0f;
            total += share;
            std::cout << "  part[" << i << "]"
                      << " device_global=" << dp.device_global_index
                      << " rank=" << dp.owning_rank
                      << " local_index=" << dp.local_index
                      << " offset=" << dp.global_offset
                      << " count=" << dp.element_count
                      << " cum=" << (100.0f * loads[i]) << "%"
                      << " share=" << (100.0f * share) << "%\n";
        }
        std::cout << "  total_share=" << (100.0f * total) << "%\n";
        std::cout << std::flush;
    }


    void clear_runtime_state() {
        release_tracked_field_events();
        reset_balance_window_events();
        platforms_.clear();
        local_devices_.clear();
        devices_.clear();
        partitions_.clear();
        all_device_counts_.clear();
        fields_.clear();
        kernels_.clear();
        current_loads_.clear();
        device_timings_.clear();
        last_elapsed_local_.clear();
        balance_window_start_events_.clear();
        balance_window_end_events_.clear();
        balance_window_valid_.clear();
        balance_window_kernel_events_.clear();
        last_input_fields_.clear();
        last_output_fields_.clear();
        field_write_events_.clear();

        next_field_id_ = 0;
        next_kernel_id_ = 0;
        active_kernel_ = -1;
        iteration_counter_ = 0;
    }

    int owner_rank_of_global_device(int g) const {
        int acc = 0;
        for (int r = 0; r < size_; ++r) {
            if (g < acc + all_device_counts_[r]) return r;
            acc += all_device_counts_[r];
        }
        return size_ - 1;
    }

    int local_index_of_global_device(int g) const {
        int acc = 0;
        for (int r = 0; r < size_; ++r) {
            if (g < acc + all_device_counts_[r]) {
                return (r == rank_) ? (g - acc) : -1;
            }
            acc += all_device_counts_[r];
        }
        return -1;
    }

    void rebuild_partitions_equal_like_wrapper() {
        partitions_.clear();
        if (!partition_.has_value()) return;

        int total_devices = 0;
        for (int c : all_device_counts_) total_devices += c;
        if (total_devices <= 0) return;

        const std::size_t n = partition_->global_elements;
        const std::size_t gran = std::max<std::size_t>(1, partition_->granularity);

        std::vector<std::size_t> cuts(static_cast<std::size_t>(total_devices) + 1, 0);
        cuts[0] = 0;

        const std::size_t base = n / static_cast<std::size_t>(total_devices);
        const std::size_t rem = n % static_cast<std::size_t>(total_devices);
        std::size_t acc = 0;
        for (int g = 0; g < total_devices; ++g) {
            acc += base + (static_cast<std::size_t>(g) < rem ? 1u : 0u);
            if (g == total_devices - 1) {
                cuts[static_cast<std::size_t>(g) + 1] = n;
            } else {
                cuts[static_cast<std::size_t>(g) + 1] = (acc / gran) * gran;
            }
        }
        cuts.back() = n;

        for (int g = 0; g < total_devices; ++g) {
            DevicePartition dp;
            dp.device_global_index = g;
            dp.owning_rank = owner_rank_of_global_device(g);
            dp.local_index = local_index_of_global_device(g);
            dp.global_offset = cuts[static_cast<std::size_t>(g)];
            dp.element_count = cuts[static_cast<std::size_t>(g) + 1] - cuts[static_cast<std::size_t>(g)];
            partitions_.push_back(dp);
        }
    }

std::vector<DevicePartition> partitions_from_loads(const std::vector<float>& loads) const {
    std::vector<DevicePartition> out;
    if (!partition_.has_value()) return out;
    if (loads.empty()) return out;

    const std::size_t n = partition_->global_elements;
    const std::size_t gran =
        std::max<std::size_t>(static_cast<std::size_t>(1), partition_->granularity);

    // loads is cumulative: [0.10, 0.20, ..., 1.00].
    std::vector<float> cumulative = loads;
    float prev = 0.0f;
    for (std::size_t i = 0; i < cumulative.size(); ++i) {
        if (!std::isfinite(cumulative[i])) return out;
        if (cumulative[i] < 0.0f) cumulative[i] = 0.0f;
        if (cumulative[i] < prev) cumulative[i] = prev;
        if (cumulative[i] > 1.0f) cumulative[i] = 1.0f;
        prev = cumulative[i];
    }
    cumulative.back() = 1.0f;

    std::vector<std::size_t> cuts(cumulative.size() + 1, 0);
    cuts[0] = 0;

    if ((n % gran) == 0 && (n / gran) >= cumulative.size()) {
        const std::size_t total_units = n / gran;
        std::vector<std::size_t> unit_cuts(cumulative.size() + 1, 0);
        unit_cuts[0] = 0;

        for (std::size_t i = 0; i + 1 < cumulative.size(); ++i) {
            const double raw =
                static_cast<double>(cumulative[i]) * static_cast<double>(total_units);
            std::size_t cut = static_cast<std::size_t>(std::llround(raw));

            const std::size_t min_cut = unit_cuts[i] + 1;
            const std::size_t remaining_parts = cumulative.size() - (i + 1);
            const std::size_t max_cut = total_units - remaining_parts;

            if (cut < min_cut) cut = min_cut;
            if (cut > max_cut) cut = max_cut;
            unit_cuts[i + 1] = cut;
        }

        unit_cuts.back() = total_units;
        for (std::size_t i = 0; i < unit_cuts.size(); ++i) {
            cuts[i] = unit_cuts[i] * gran;
        }
    } else {
        for (std::size_t i = 0; i < cumulative.size(); ++i) {
            if (i + 1 == cumulative.size()) {
                cuts[i + 1] = n;
            } else {
                const double raw = static_cast<double>(cumulative[i]) * static_cast<double>(n);
                std::size_t cut = static_cast<std::size_t>(std::llround(raw));
                cut = (cut / gran) * gran;
                if (cut > n) cut = n;
                if (cut < cuts[i]) cut = cuts[i];
                cuts[i + 1] = cut;
            }
        }

        cuts.back() = n;
        for (std::size_t i = 1; i < cuts.size(); ++i) {
            if (cuts[i] < cuts[i - 1]) cuts[i] = cuts[i - 1];
            if (cuts[i] > n) cuts[i] = n;
        }
        cuts.back() = n;
    }

    for (std::size_t g = 0; g < cumulative.size(); ++g) {
        DevicePartition dp;
        dp.device_global_index = static_cast<int>(g);
        dp.owning_rank = owner_rank_of_global_device(static_cast<int>(g));
        dp.local_index = (dp.owning_rank == rank_)
            ? local_index_of_global_device(static_cast<int>(g))
            : -1;
        dp.global_offset = cuts[g];
        dp.element_count = cuts[g + 1] - cuts[g];
        out.push_back(dp);
    }

    return out;
}

std::vector<float> loads_from_partitions(const std::vector<DevicePartition>& parts) const {
    std::vector<float> loads(parts.size(), 1.0f);
    if (!partition_.has_value() || partition_->global_elements == 0 || parts.empty()) {
        return loads;
    }

    const double total = static_cast<double>(partition_->global_elements);
    for (std::size_t i = 0; i < parts.size(); ++i) {
        const std::size_t end = parts[i].global_offset + parts[i].element_count;
        loads[i] = static_cast<float>(static_cast<double>(end) / total);
        if (i > 0 && loads[i] < loads[i - 1]) {
            loads[i] = loads[i - 1];
        }
        if (loads[i] > 1.0f) loads[i] = 1.0f;
    }
    loads.back() = 1.0f;
    return loads;
}

    void write_initial_field_data(FieldHandle h, const void* host_ptr) {
        auto it = fields_.find(h.value);
        if (it == fields_.end()) throw Error("write_initial_field_data(): unknown field");

        detail::RegisteredField& rf = it->second;
        const std::size_t total_bytes = rf.spec.global_elements * rf.spec.units_per_element * rf.spec.bytes_per_unit;
        std::vector<cl_event> write_events;

        for (std::size_t d = 0; d < local_devices_.size(); ++d) {
            cl_event ev = nullptr;
            detail::check_cl(
                clEnqueueWriteBuffer(
                    local_devices_[d].transfer_queue,
                    rf.replicas[d],
                    CL_FALSE,
                    0,
                    total_bytes,
                    host_ptr,
                    0,
                    nullptr,
                    &ev
                ),
                "clEnqueueWriteBuffer(write_initial_field_data)"
            );
            write_events.push_back(ev);
        }

        for (cl_event ev : write_events) {
            if (ev != nullptr) {
                detail::check_cl(clWaitForEvents(1, &ev), "clWaitForEvents(write_initial_field_data)");
                clReleaseEvent(ev);
            }
        }
    }

    cl_mem resolve_mem_for_local_device(const KernelArg& arg, std::size_t local_device) {
        if (const FieldHandle* fh = std::get_if<FieldHandle>(&arg)) {
            auto it = fields_.find(fh->value);
            if (it == fields_.end()) throw Error("Unknown field handle");
            return it->second.replicas[local_device];
        }
        throw Error("KernelArg memory resolution failed");
    }

    void bind_kernel_args(std::size_t local_device, cl_kernel kernel, const KernelBinding& binding) {
        for (std::size_t i = 0; i < binding.args.size(); ++i) {
            const unsigned arg_index = binding.args[i].first;
            const KernelArg& arg = binding.args[i].second;

            if (const ScalarArg* sa = std::get_if<ScalarArg>(&arg)) {
                detail::check_cl(clSetKernelArg(kernel, arg_index, sa->bytes.size(), sa->bytes.data()), "clSetKernelArg(scalar)");
                continue;
            }

            cl_mem mem = resolve_mem_for_local_device(arg, local_device);
            detail::check_cl(clSetKernelArg(kernel, arg_index, sizeof(cl_mem), &mem), "clSetKernelArg(mem)");
        }
    }


    std::vector<FieldHandle> fields_with_role(const ExecutionStep& step, StepFieldRole role) {
        std::vector<FieldHandle> out;
        for (const StepFieldTag& tag : step.field_tags) {
            if (tag.role == role) out.push_back(tag.field);
        }
        return out;
    }

    std::optional<FieldHandle> first_field_with_role(const ExecutionStep& step, StepFieldRole role) {
        for (const StepFieldTag& tag : step.field_tags) {
            if (tag.role == role) return tag.field;
        }
        return std::nullopt;
    }

    void extract_rw_fields(const KernelBinding& binding) {
        for (std::size_t i = 0; i < binding.args.size(); ++i) {
            const unsigned idx = binding.args[i].first;
            const KernelArg& arg = binding.args[i].second;
            if (const FieldHandle* fh = std::get_if<FieldHandle>(&arg)) {
                if (idx == 0) last_output_fields_.push_back(*fh);
                if (idx == 1) last_input_fields_.push_back(*fh);
            }
        }
    }

    std::vector<FieldHandle> output_fields_from_binding(const KernelBinding& binding) const {
        std::vector<FieldHandle> out;
        for (std::size_t i = 0; i < binding.args.size(); ++i) {
            const unsigned idx = binding.args[i].first;
            const KernelArg& arg = binding.args[i].second;
            if (idx == 0) {
                if (const FieldHandle* fh = std::get_if<FieldHandle>(&arg)) {
                    out.push_back(*fh);
                }
            }
        }
        return out;
    }

    std::vector<cl_event> field_dependencies_for_binding(
        const KernelBinding& binding,
        std::size_t local_device
    ) const {
        std::vector<cl_event> deps;
        for (std::size_t i = 0; i < binding.args.size(); ++i) {
            const unsigned idx = binding.args[i].first;
            if (idx == 0) continue;

            const KernelArg& arg = binding.args[i].second;
            const FieldHandle* fh = std::get_if<FieldHandle>(&arg);
            if (fh == nullptr) continue;

            std::unordered_map<int, std::vector<cl_event>>::const_iterator it =
                field_write_events_.find(fh->value);
            if (it == field_write_events_.end()) continue;
            if (local_device >= it->second.size()) continue;

            cl_event ev = it->second[local_device];
            if (ev == nullptr) continue;

            bool seen = false;
            for (cl_event existing : deps) {
                if (existing == ev) {
                    seen = true;
                    break;
                }
            }
            if (!seen) deps.push_back(ev);
        }
        return deps;
    }

    std::vector<cl_event> field_write_dependency(FieldHandle field, std::size_t local_device) const {
        std::vector<cl_event> deps;
        std::unordered_map<int, std::vector<cl_event>>::const_iterator it =
            field_write_events_.find(field.value);
        if (it != field_write_events_.end() && local_device < it->second.size()) {
            cl_event ev = it->second[local_device];
            if (ev != nullptr) deps.push_back(ev);
        }
        return deps;
    }

    void set_field_write_event(FieldHandle field, std::size_t local_device, cl_event ev) {
        if (ev == nullptr) return;
        std::vector<cl_event>& events = field_write_events_[field.value];
        if (events.size() < local_devices_.size()) {
            events.resize(local_devices_.size(), nullptr);
        }
        release_event_if_needed(events[local_device]);
        detail::check_cl(clRetainEvent(ev), "clRetainEvent(field write)");
        events[local_device] = ev;
    }

    void record_field_write_events(
        const std::vector<FieldHandle>& fields,
        const std::vector<std::pair<std::size_t, cl_event>>& write_events
    ) {
        if (fields.empty()) return;
        for (const auto& item : write_events) {
            const std::size_t d = item.first;
            cl_event ev = item.second;
            if (ev == nullptr || d >= local_devices_.size()) continue;
            for (FieldHandle field : fields) {
                set_field_write_event(field, d, ev);
            }
        }
    }

    void release_kernel_events(std::vector<std::pair<std::size_t, cl_event>>& kernel_events) {
        for (std::size_t i = 0; i < kernel_events.size(); ++i) {
            cl_event ev = kernel_events[i].second;
            if (ev != nullptr) {
                clReleaseEvent(ev);
                kernel_events[i].second = nullptr;
            }
        }
        kernel_events.clear();
    }

    void release_tracked_field_events() {
        for (auto& kv : field_write_events_) {
            for (cl_event& ev : kv.second) {
                release_event_if_needed(ev);
            }
        }
        field_write_events_.clear();
    }

    void finalize_kernel_events(std::vector<std::pair<std::size_t, cl_event>>& kernel_events,
                                std::vector<double>& elapsed_local) {
        for (std::size_t i = 0; i < kernel_events.size(); ++i) {
            const std::size_t d = kernel_events[i].first;
            cl_event ev = kernel_events[i].second;
            if (ev == nullptr) continue;

            detail::check_cl(clWaitForEvents(1, &ev), "clWaitForEvents(kernel profiling)");

            cl_ulong t0 = 0;
            cl_ulong t1 = 0;
            const cl_int e0 = clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &t0, nullptr);
            const cl_int e1 = clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &t1, nullptr);

            if (e0 == CL_SUCCESS && e1 == CL_SUCCESS && t1 >= t0) {
                elapsed_local[d] += static_cast<double>(t1 - t0) * 1.0e-9;
            }

            clReleaseEvent(ev);
        }
        kernel_events.clear();
    }

    void release_event_if_needed(cl_event& ev) {
        if (ev != nullptr) {
            clReleaseEvent(ev);
            ev = nullptr;
        }
    }

    void reset_balance_window_events() {
        for (std::size_t d = 0; d < balance_window_start_events_.size(); ++d) {
            release_event_if_needed(balance_window_start_events_[d]);
            release_event_if_needed(balance_window_end_events_[d]);
            balance_window_valid_[d] = false;
        }
        for (std::size_t d = 0; d < balance_window_kernel_events_.size(); ++d) {
            for (cl_event& ev : balance_window_kernel_events_[d]) {
                release_event_if_needed(ev);
            }
            balance_window_kernel_events_[d].clear();
        }
        balance_window_begin_iteration_ = iteration_counter_;
    }

    void append_balance_window_kernel_events(
        const std::vector<std::pair<std::size_t, cl_event>>& kernel_events
    ) {
        if (balance_window_kernel_events_.size() != local_devices_.size()) {
            balance_window_kernel_events_.assign(local_devices_.size(), {});
        }

        for (const auto& item : kernel_events) {
            const std::size_t d = item.first;
            cl_event ev = item.second;
            if (ev == nullptr || d >= balance_window_kernel_events_.size()) continue;
            detail::check_cl(clRetainEvent(ev), "clRetainEvent(balance window kernel)");
            balance_window_kernel_events_[d].push_back(ev);
            balance_window_valid_[d] = true;
        }
    }

    void update_balance_window_events(
        const std::vector<std::pair<std::size_t, cl_event>>& start_events,
        const std::vector<std::pair<std::size_t, cl_event>>& end_events) {
        for (const auto& item : start_events) {
            const std::size_t d = item.first;
            cl_event ev = item.second;
            if (ev == nullptr || d >= balance_window_start_events_.size()) continue;

            if (balance_window_start_events_[d] == nullptr) {
                detail::check_cl(clRetainEvent(ev), "clRetainEvent(balance window start)");
                balance_window_start_events_[d] = ev;
                balance_window_valid_[d] = true;
            }
        }

        for (const auto& item : end_events) {
            const std::size_t d = item.first;
            cl_event ev = item.second;
            if (ev == nullptr || d >= balance_window_end_events_.size()) continue;

            if (balance_window_start_events_[d] == nullptr) {
                detail::check_cl(clRetainEvent(ev), "clRetainEvent(balance window fallback start)");
                balance_window_start_events_[d] = ev;
            }

            release_event_if_needed(balance_window_end_events_[d]);
            detail::check_cl(clRetainEvent(ev), "clRetainEvent(balance window end)");
            balance_window_end_events_[d] = ev;
            balance_window_valid_[d] = true;
        }
    }

    std::vector<double> compute_balance_window_elapsed_local() {
        std::vector<double> elapsed(local_devices_.size(), 0.0);

        for (std::size_t d = 0; d < local_devices_.size(); ++d) {
            if (d >= balance_window_kernel_events_.size()) continue;

            for (cl_event ev : balance_window_kernel_events_[d]) {
                if (ev == nullptr) continue;

                detail::check_cl(
                    clWaitForEvents(1, &ev),
                    "clWaitForEvents(balance window kernel)"
                );

                cl_ulong t0 = 0;
                cl_ulong t1 = 0;
                const cl_int e0 = clGetEventProfilingInfo(
                    ev,
                    CL_PROFILING_COMMAND_START,
                    sizeof(cl_ulong),
                    &t0,
                    nullptr
                );
                const cl_int e1 = clGetEventProfilingInfo(
                    ev,
                    CL_PROFILING_COMMAND_END,
                    sizeof(cl_ulong),
                    &t1,
                    nullptr
                );

                if (e0 == CL_SUCCESS && e1 == CL_SUCCESS && t1 >= t0) {
                    elapsed[d] += static_cast<double>(t1 - t0) * 1.0e-9;
                }
            }
        }

        last_elapsed_local_ = elapsed;
        update_device_timings(elapsed);
        return elapsed;
    }

    void synchronize_all_local_devices(bool force_finish) {
        if (force_finish) {
            for (std::size_t d = 0; d < local_devices_.size(); ++d) {
                if (local_devices_[d].kernel_queue != nullptr) {
                    detail::check_cl(clFinish(local_devices_[d].kernel_queue), "clFinish(kernel queue)");
                }
                if (local_devices_[d].transfer_queue != nullptr) {
                    detail::check_cl(clFinish(local_devices_[d].transfer_queue), "clFinish(transfer queue)");
                }
            }
            release_tracked_field_events();
            return;
        }

        std::vector<cl_event> marker_events;
        marker_events.reserve(local_devices_.size() * 2);

        for (std::size_t d = 0; d < local_devices_.size(); ++d) {
            if (local_devices_[d].kernel_queue != nullptr) {
                cl_event ev = nullptr;
#if CL_TARGET_OPENCL_VERSION >= 120
                detail::check_cl(clEnqueueMarkerWithWaitList(local_devices_[d].kernel_queue, 0, nullptr, &ev), "clEnqueueMarkerWithWaitList(kernel queue)");
#else
                detail::check_cl(clEnqueueMarker(local_devices_[d].kernel_queue, &ev), "clEnqueueMarker(kernel queue)");
#endif
                marker_events.push_back(ev);
            }

            if (local_devices_[d].transfer_queue != nullptr) {
                cl_event ev = nullptr;
#if CL_TARGET_OPENCL_VERSION >= 120
                detail::check_cl(clEnqueueMarkerWithWaitList(local_devices_[d].transfer_queue, 0, nullptr, &ev), "clEnqueueMarkerWithWaitList(transfer queue)");
#else
                detail::check_cl(clEnqueueMarker(local_devices_[d].transfer_queue, &ev), "clEnqueueMarker(transfer queue)");
#endif
                marker_events.push_back(ev);
            }
        }

        for (cl_event ev : marker_events) {
            if (ev != nullptr) {
                detail::check_cl(clWaitForEvents(1, &ev), "clWaitForEvents(queue marker)");
                clReleaseEvent(ev);
            }
        }
        release_tracked_field_events();
    }

    void run_full_phase(detail::RegisteredKernel& rk,
                        const KernelInvocation& ki,
                        std::vector<std::pair<std::size_t, cl_event>>& kernel_events) {
        for (std::size_t p = 0; p < partitions_.size(); ++p) {
            const DevicePartition& dp = partitions_[p];
            if (dp.owning_rank != rank_ || dp.local_index < 0) continue;

            const std::size_t d = static_cast<std::size_t>(dp.local_index);
            cl_kernel kernel = rk.kernels_per_local_device[d];
            bind_kernel_args(d, kernel, ki.binding);

            const std::size_t gwo = dp.global_offset + ki.geometry.global_offset;
            const std::size_t gws = dp.element_count;

            const std::size_t* lws_ptr = nullptr;
            std::size_t lws_value = 0;
            if (ki.geometry.local_size.has_value()) {
                lws_value = *ki.geometry.local_size;
                lws_ptr = &lws_value;
            }

            const std::vector<cl_event> deps =
                field_dependencies_for_binding(ki.binding, d);
            cl_event kernel_ev = nullptr;
            detail::check_cl(
                clEnqueueNDRangeKernel(
                    local_devices_[d].kernel_queue,
                    kernel,
                    1,
                    &gwo,
                    &gws,
                    lws_ptr,
                    static_cast<cl_uint>(deps.size()),
                    deps.empty() ? nullptr : deps.data(),
                    &kernel_ev
                ),
                "clEnqueueNDRangeKernel(full)"
            );
            kernel_events.push_back(std::make_pair(d, kernel_ev));
        }
    }

    void run_interior_phase(detail::RegisteredKernel& rk,
                            const KernelInvocation& ki,
                            std::vector<std::pair<std::size_t, cl_event>>& kernel_events,
                            std::size_t halo) {
        for (std::size_t p = 0; p < partitions_.size(); ++p) {
            const DevicePartition& dp = partitions_[p];
            if (dp.owning_rank != rank_ || dp.local_index < 0) continue;
            if (dp.element_count <= 2 * halo) continue;

            const std::size_t d = static_cast<std::size_t>(dp.local_index);
            cl_kernel kernel = rk.kernels_per_local_device[d];
            bind_kernel_args(d, kernel, ki.binding);

            const std::size_t gwo = dp.global_offset + halo + ki.geometry.global_offset;
            const std::size_t gws = dp.element_count - 2 * halo;

            const std::size_t* lws_ptr = nullptr;
            std::size_t lws_value = 0;
            if (ki.geometry.local_size.has_value()) {
                lws_value = *ki.geometry.local_size;
                lws_ptr = &lws_value;
            }

            const std::vector<cl_event> deps =
                field_dependencies_for_binding(ki.binding, d);
            cl_event kernel_ev = nullptr;
            detail::check_cl(
                clEnqueueNDRangeKernel(
                    local_devices_[d].kernel_queue,
                    kernel,
                    1,
                    &gwo,
                    &gws,
                    lws_ptr,
                    static_cast<cl_uint>(deps.size()),
                    deps.empty() ? nullptr : deps.data(),
                    &kernel_ev
                ),
                "clEnqueueNDRangeKernel(interior)"
            );
            kernel_events.push_back(std::make_pair(d, kernel_ev));
        }
    }

    void run_border_phase(detail::RegisteredKernel& rk,
                          const KernelInvocation& ki,
                          std::vector<std::pair<std::size_t, cl_event>>& kernel_events,
                          std::size_t halo) {
        for (std::size_t p = 0; p < partitions_.size(); ++p) {
            const DevicePartition& dp = partitions_[p];
            if (dp.owning_rank != rank_ || dp.local_index < 0) continue;

            const std::size_t d = static_cast<std::size_t>(dp.local_index);
            cl_kernel kernel = rk.kernels_per_local_device[d];
            bind_kernel_args(d, kernel, ki.binding);

            const std::size_t* lws_ptr = nullptr;
            std::size_t lws_value = 0;
            if (ki.geometry.local_size.has_value()) {
                lws_value = *ki.geometry.local_size;
                lws_ptr = &lws_value;
            }

            const std::vector<cl_event> deps =
                field_dependencies_for_binding(ki.binding, d);
            const std::size_t left_count = std::min(halo, dp.element_count);
            const std::size_t right_count =
                (dp.element_count > halo) ? std::min(halo, dp.element_count - left_count) : 0;

            if (left_count > 0) {
                const std::size_t gwo = dp.global_offset + ki.geometry.global_offset;
                const std::size_t gws = left_count;
                cl_event kernel_ev = nullptr;
                detail::check_cl(
                    clEnqueueNDRangeKernel(
                        local_devices_[d].kernel_queue,
                        kernel,
                        1,
                        &gwo,
                        &gws,
                        lws_ptr,
                        static_cast<cl_uint>(deps.size()),
                        deps.empty() ? nullptr : deps.data(),
                        &kernel_ev
                    ),
                    "clEnqueueNDRangeKernel(border-left)"
                );
                kernel_events.push_back(std::make_pair(d, kernel_ev));
            }

            if (right_count > 0 && dp.element_count > left_count) {
                const std::size_t gwo = dp.global_offset + dp.element_count - right_count + ki.geometry.global_offset;
                const std::size_t gws = right_count;
                cl_event kernel_ev = nullptr;
                detail::check_cl(
                    clEnqueueNDRangeKernel(
                        local_devices_[d].kernel_queue,
                        kernel,
                        1,
                        &gwo,
                        &gws,
                        lws_ptr,
                        static_cast<cl_uint>(deps.size()),
                        deps.empty() ? nullptr : deps.data(),
                        &kernel_ev
                    ),
                    "clEnqueueNDRangeKernel(border-right)"
                );
                kernel_events.push_back(std::make_pair(d, kernel_ev));
            }
        }
    }

    void exchange_halo_set(const HaloSpec& hs) {
        if (hs.width_elements == 0) return;
        for (std::size_t i = 0; i < hs.fields.size(); ++i) {
            exchange_halos_for_field(hs.fields[i], hs.width_elements);
        }
    }

inline void exchange_halos_for_field(FieldHandle fh, std::size_t halo) {
    if (halo == 0 || partitions_.size() < 2) {
        return;
    }

    std::unordered_map<int, detail::RegisteredField>::iterator it = fields_.find(fh.value);
    if (it == fields_.end()) {
        throw Error("Unknown field in halo exchange");
    }

    detail::RegisteredField& rf = it->second;
    const std::size_t elem_bytes = rf.spec.units_per_element * rf.spec.bytes_per_unit;
    const std::size_t halo_bytes = halo * elem_bytes;

    std::vector<unsigned char> send_left(halo_bytes);
    std::vector<unsigned char> send_right(halo_bytes);
    std::vector<unsigned char> recv_left(halo_bytes);
    std::vector<unsigned char> recv_right(halo_bytes);

    std::vector<std::size_t> nonempty_parts;
    nonempty_parts.reserve(partitions_.size());
    for (std::size_t p = 0; p < partitions_.size(); ++p) {
        if (partitions_[p].element_count >= halo) {
            nonempty_parts.push_back(p);
        }
    }

    for (std::size_t pair_index = 0; pair_index + 1 < nonempty_parts.size(); ++pair_index) {
        const std::size_t left_index = nonempty_parts[pair_index];
        const std::size_t right_index = nonempty_parts[pair_index + 1];
        const DevicePartition& left = partitions_[left_index];
        const DevicePartition& right = partitions_[right_index];

        const std::size_t left_border_off =
            (left.global_offset + left.element_count - halo) * elem_bytes;
        const std::size_t left_halo_off =
            (left.global_offset + left.element_count) * elem_bytes;
        const std::size_t right_border_off =
            right.global_offset * elem_bytes;
        const std::size_t right_halo_off =
            (right.global_offset - halo) * elem_bytes;

        cl_event read_left_to_right_ev = nullptr;
        cl_event read_right_to_left_ev = nullptr;
        cl_event write_left_ev = nullptr;
        cl_event write_right_ev = nullptr;

        if (left.owning_rank == rank_) {
            const std::size_t dl = static_cast<std::size_t>(left.local_index);
            const std::vector<cl_event> deps = field_write_dependency(fh, dl);
            detail::check_cl(
                clEnqueueReadBuffer(
                    local_devices_[dl].transfer_queue,
                    rf.replicas[dl],
                    CL_FALSE,
                    left_border_off,
                    halo_bytes,
                    send_right.data(),
                    static_cast<cl_uint>(deps.size()),
                    deps.empty() ? nullptr : deps.data(),
                    &read_left_to_right_ev
                ),
                "clEnqueueReadBuffer(halo left->right send)"
            );
        }

        if (right.owning_rank == rank_) {
            const std::size_t dr = static_cast<std::size_t>(right.local_index);
            const std::vector<cl_event> deps = field_write_dependency(fh, dr);
            detail::check_cl(
                clEnqueueReadBuffer(
                    local_devices_[dr].transfer_queue,
                    rf.replicas[dr],
                    CL_FALSE,
                    right_border_off,
                    halo_bytes,
                    send_left.data(),
                    static_cast<cl_uint>(deps.size()),
                    deps.empty() ? nullptr : deps.data(),
                    &read_right_to_left_ev
                ),
                "clEnqueueReadBuffer(halo right->left send)"
            );
        }

        // Caso local-local: os dispositivos podem estar em contextos distintos.
        // Então a dependência precisa ser resolvida no host, e não via event_wait_list.
        if (left.owning_rank == rank_ && right.owning_rank == rank_) {
            const std::size_t dl = static_cast<std::size_t>(left.local_index);
            const std::size_t dr = static_cast<std::size_t>(right.local_index);

            if (read_left_to_right_ev != nullptr) {
                detail::check_cl(
                    clWaitForEvents(1, &read_left_to_right_ev),
                    "clWaitForEvents(halo left->right read ready)"
                );
                clReleaseEvent(read_left_to_right_ev);
                read_left_to_right_ev = nullptr;

                detail::check_cl(
                    clEnqueueWriteBuffer(
                        local_devices_[dr].transfer_queue,
                        rf.replicas[dr],
                        CL_FALSE,
                        right_halo_off,
                        halo_bytes,
                        send_right.data(),
                        0,
                        nullptr,
                        &write_right_ev
                    ),
                    "clEnqueueWriteBuffer(halo left->right local)"
                );
                set_field_write_event(fh, dr, write_right_ev);
            }

            if (read_right_to_left_ev != nullptr) {
                detail::check_cl(
                    clWaitForEvents(1, &read_right_to_left_ev),
                    "clWaitForEvents(halo right->left read ready)"
                );
                clReleaseEvent(read_right_to_left_ev);
                read_right_to_left_ev = nullptr;

                detail::check_cl(
                    clEnqueueWriteBuffer(
                        local_devices_[dl].transfer_queue,
                        rf.replicas[dl],
                        CL_FALSE,
                        left_halo_off,
                        halo_bytes,
                        send_left.data(),
                        0,
                        nullptr,
                        &write_left_ev
                    ),
                    "clEnqueueWriteBuffer(halo right->left local)"
                );
                set_field_write_event(fh, dl, write_left_ev);
            }

            if (write_right_ev != nullptr) {
                detail::check_cl(
                    clWaitForEvents(1, &write_right_ev),
                    "clWaitForEvents(halo left->right write done)"
                );
                clReleaseEvent(write_right_ev);
                write_right_ev = nullptr;
            }

            if (write_left_ev != nullptr) {
                detail::check_cl(
                    clWaitForEvents(1, &write_left_ev),
                    "clWaitForEvents(halo right->left write done)"
                );
                clReleaseEvent(write_left_ev);
                write_left_ev = nullptr;
            }

            continue;
        }

        if (left.owning_rank == rank_ && read_left_to_right_ev != nullptr) {
            detail::check_cl(
                clWaitForEvents(1, &read_left_to_right_ev),
                "clWaitForEvents(halo left->right send ready)"
            );
            clReleaseEvent(read_left_to_right_ev);
            read_left_to_right_ev = nullptr;
        }

        if (right.owning_rank == rank_ && read_right_to_left_ev != nullptr) {
            detail::check_cl(
                clWaitForEvents(1, &read_right_to_left_ev),
                "clWaitForEvents(halo right->left send ready)"
            );
            clReleaseEvent(read_right_to_left_ev);
            read_right_to_left_ev = nullptr;
        }

        const int halo_tag_base = 2000 + static_cast<int>(pair_index) * 4;
        check_mpi_tag_base(halo_tag_base + 1, halo_bytes, "halo exchange");

        // left -> right (unidirecional)
        {
            const int tag_base = halo_tag_base;
            if (left.owning_rank == rank_) {
                this->mpi_transfer_bytes_chunked(
                    send_right.data(),
                    recv_right.data(),
                    halo_bytes,
                    right.owning_rank,
                    tag_base,
                    true,
                    false
                );
            } else if (right.owning_rank == rank_) {
                this->mpi_transfer_bytes_chunked(
                    recv_right.data(),
                    recv_left.data(),
                    halo_bytes,
                    left.owning_rank,
                    tag_base,
                    false,
                    true
                );
            }
        }

        // right -> left (unidirecional)
        {
            const int tag_base = halo_tag_base + 1;
            if (right.owning_rank == rank_) {
                this->mpi_transfer_bytes_chunked(
                    send_left.data(),
                    recv_left.data(),
                    halo_bytes,
                    left.owning_rank,
                    tag_base,
                    true,
                    false
                );
            } else if (left.owning_rank == rank_) {
                this->mpi_transfer_bytes_chunked(
                    recv_left.data(),
                    recv_right.data(),
                    halo_bytes,
                    right.owning_rank,
                    tag_base,
                    false,
                    true
                );
            }
        }

        if (right.owning_rank == rank_) {
            const std::size_t dr = static_cast<std::size_t>(right.local_index);
            detail::check_cl(
                clEnqueueWriteBuffer(
                    local_devices_[dr].transfer_queue,
                    rf.replicas[dr],
                    CL_FALSE,
                    right_halo_off,
                    halo_bytes,
                    recv_left.data(),
                    0,
                    nullptr,
                    &write_right_ev
                ),
                "clEnqueueWriteBuffer(halo left->right remote)"
            );
            set_field_write_event(fh, dr, write_right_ev);
        }

        if (left.owning_rank == rank_) {
            const std::size_t dl = static_cast<std::size_t>(left.local_index);
            detail::check_cl(
                clEnqueueWriteBuffer(
                    local_devices_[dl].transfer_queue,
                    rf.replicas[dl],
                    CL_FALSE,
                    left_halo_off,
                    halo_bytes,
                    recv_right.data(),
                    0,
                    nullptr,
                    &write_left_ev
                ),
                "clEnqueueWriteBuffer(halo right->left remote)"
            );
            set_field_write_event(fh, dl, write_left_ev);
        }

        if (write_right_ev != nullptr) {
            detail::check_cl(
                clWaitForEvents(1, &write_right_ev),
                "clWaitForEvents(halo left->right remote write done)"
            );
            clReleaseEvent(write_right_ev);
            write_right_ev = nullptr;
        }

        if (write_left_ev != nullptr) {
            detail::check_cl(
                clWaitForEvents(1, &write_left_ev),
                "clWaitForEvents(halo right->left remote write done)"
            );
            clReleaseEvent(write_left_ev);
            write_left_ev = nullptr;
        }
    }
}
inline void redistribute_field_intersection(
    FieldHandle fh,
    const std::vector<DevicePartition>& old_parts,
    const std::vector<DevicePartition>& new_parts
) {
    std::unordered_map<int, detail::RegisteredField>::iterator fit = fields_.find(fh.value);
    if (fit == fields_.end()) {
        throw Error("redistribute_field_intersection(): unknown field");
    }

    detail::RegisteredField& rf = fit->second;
    const std::size_t elem_bytes =
        rf.spec.units_per_element * rf.spec.bytes_per_unit;
    const std::size_t total_bytes =
        rf.spec.global_elements * elem_bytes;

    for (std::size_t src = 0; src < old_parts.size(); ++src) {
        for (std::size_t dst = 0; dst < new_parts.size(); ++dst) {
            std::size_t inter_off = 0;
            std::size_t inter_len = 0;

            if (!detail::intersect_1d(
                    old_parts[src].global_offset,
                    old_parts[src].element_count,
                    new_parts[dst].global_offset,
                    new_parts[dst].element_count,
                    inter_off,
                    inter_len)) {
                continue;
            }

            if (inter_len == 0) continue;

            const std::size_t byte_off = inter_off * elem_bytes;
            const std::size_t byte_len = inter_len * elem_bytes;

            if (byte_off + byte_len > total_bytes) {
                throw Error("redistribute_field_intersection(): byte range out of bounds");
            }

            const int rank_src = old_parts[src].owning_rank;
            const int rank_dst = new_parts[dst].owning_rank;

            // se origem e destino são exatamente o mesmo device local, não há nada a copiar
            if (rank_src == rank_ && rank_dst == rank_ &&
                old_parts[src].local_index >= 0 &&
                new_parts[dst].local_index >= 0 &&
                old_parts[src].local_index == new_parts[dst].local_index) {
                continue;
            }

            if (rank_src != rank_ && rank_dst != rank_) {
                continue;
            }

            constexpr std::size_t REDISTRIBUTE_STAGING_BYTES = 64ull * 1024ull * 1024ull;
            std::vector<unsigned char> chunk(
                std::min<std::size_t>(byte_len, REDISTRIBUTE_STAGING_BYTES)
            );

            for (std::size_t done = 0; done < byte_len; done += chunk.size()) {
                const std::size_t chunk_len =
                    std::min<std::size_t>(chunk.size(), byte_len - done);
                const std::size_t chunk_off = byte_off + done;

                cl_event read_ev = nullptr;
                if (rank_src == rank_ && old_parts[src].local_index >= 0) {
                    const std::size_t dsrc =
                        static_cast<std::size_t>(old_parts[src].local_index);

                    if (dsrc >= rf.replicas.size() || rf.replicas[dsrc] == nullptr) {
                        throw Error("redistribute_field_intersection(): invalid source replica");
                    }

                    detail::check_cl(
                        clEnqueueReadBuffer(
                            local_devices_[dsrc].transfer_queue,
                            rf.replicas[dsrc],
                            CL_FALSE,
                            chunk_off,
                            chunk_len,
                            chunk.data(),
                            0,
                            nullptr,
                            &read_ev
                        ),
                        "clEnqueueReadBuffer(redistribute read)"
                    );

                    detail::check_cl(
                        clWaitForEvents(1, &read_ev),
                        "clWaitForEvents(redistribute read)"
                    );
                    clReleaseEvent(read_ev);
                }

                if (rank_src == rank_ && rank_dst == rank_) {
                    if (new_parts[dst].local_index >= 0) {
                        const std::size_t ddst =
                            static_cast<std::size_t>(new_parts[dst].local_index);

                        if (ddst >= rf.replicas.size() || rf.replicas[ddst] == nullptr) {
                            throw Error("redistribute_field_intersection(): invalid destination replica");
                        }

                        detail::check_cl(
                            clEnqueueWriteBuffer(
                                local_devices_[ddst].transfer_queue,
                                rf.replicas[ddst],
                                CL_TRUE,
                                chunk_off,
                                chunk_len,
                                chunk.data(),
                                0,
                                nullptr,
                                nullptr
                            ),
                            "clEnqueueWriteBuffer(redistribute local)"
                        );
                    }
                } else {
                    constexpr int REDISTRIBUTE_TAG_BASE = 3000;
                    constexpr int REDISTRIBUTE_FIELD_TAG_STRIDE = 4096;
                    const int tag_base =
                        REDISTRIBUTE_TAG_BASE +
                        fh.value * REDISTRIBUTE_FIELD_TAG_STRIDE;

                    this->mpi_transfer_bytes_chunked(
                        rank_src == rank_ ? chunk.data() : nullptr,
                        rank_dst == rank_ ? chunk.data() : nullptr,
                        chunk_len,
                        rank_src == rank_ ? rank_dst : rank_src,
                        tag_base + static_cast<int>(done / chunk.size()),
                        rank_src == rank_,
                        rank_dst == rank_
                    );

                    if (rank_dst == rank_ && new_parts[dst].local_index >= 0) {
                        const std::size_t ddst =
                            static_cast<std::size_t>(new_parts[dst].local_index);

                        if (ddst >= rf.replicas.size() || rf.replicas[ddst] == nullptr) {
                            throw Error("redistribute_field_intersection(): invalid remote destination replica");
                        }

                        detail::check_cl(
                            clEnqueueWriteBuffer(
                                local_devices_[ddst].transfer_queue,
                                rf.replicas[ddst],
                                CL_TRUE,
                                chunk_off,
                                chunk_len,
                                chunk.data(),
                                0,
                                nullptr,
                                nullptr
                            ),
                            "clEnqueueWriteBuffer(redistribute remote)"
                        );
                    }
                }
            }
        }
    }

    synchronize_all_local_devices(false);
}

inline void rebalance(FieldHandle target_field) {
    if (!partition_.has_value()) return;
    if (partitions_.empty()) return;

    std::unordered_map<int, detail::RegisteredField>::iterator fit = fields_.find(target_field.value);
    if (fit == fields_.end()) {
        throw Error("rebalance() received unknown field");
    }

    std::vector<double> local_times(partitions_.size(), 0.0);
    for (std::size_t i = 0; i < partitions_.size(); ++i) {
        if (partitions_[i].owning_rank == rank_ && partitions_[i].local_index >= 0) {
            const int li = partitions_[i].local_index;
            if (li >= 0 && static_cast<std::size_t>(li) < last_elapsed_local_.size()) {
                local_times[i] = last_elapsed_local_[li];
            }
        }
    }

    std::vector<double> global_times(partitions_.size(), 0.0);
    detail::check_mpi(
        MPI_Allreduce(
            local_times.data(),
            global_times.data(),
            static_cast<int>(global_times.size()),
            MPI_DOUBLE,
            MPI_SUM,
            comm_
        ),
        "MPI_Allreduce(rebalance times)"
    );

    const std::vector<float> proposed_loads =
        detail::compute_loads_from_partition_throughput(global_times, partitions_);
    if (proposed_loads.empty()) return;

    const std::vector<DevicePartition> old_parts = partitions_;
    const std::vector<DevicePartition> new_parts = partitions_from_loads(proposed_loads);
    if (new_parts.size() != old_parts.size()) {
        return;
    }
    const std::vector<float> effective_new_loads = loads_from_partitions(new_parts);

    if (!current_loads_.empty()) {
        const float n = detail::l2_norm_diff(current_loads_, effective_new_loads);
        if (n <= 0.000025f) {
            return;
        }
    }

    bool same = (old_parts.size() == new_parts.size());
    if (same) {
        for (std::size_t i = 0; i < old_parts.size(); ++i) {
            if (old_parts[i].global_offset != new_parts[i].global_offset ||
                old_parts[i].element_count != new_parts[i].element_count ||
                old_parts[i].owning_rank   != new_parts[i].owning_rank ||
                old_parts[i].local_index   != new_parts[i].local_index) {
                same = false;
                break;
            }
        }
    }
    if (same) {
        current_loads_ = loads_from_partitions(partitions_);
        return;
    }

    // Igual ao rebalance_to(): todos os campos proporcionais precisam acompanhar
    // a nova particao. Migrar so o campo usado para medir tempos deixa o outro
    // buffer do ping-pong em particoes antigas e quebra o gather final.
    this->synchronize(true);

    this->redistribute_all_registered_fields(old_parts, new_parts);

    partitions_ = new_parts;
    current_loads_ = loads_from_partitions(partitions_);

    // Igual ao rebalance_to(): sincroniza depois
    this->synchronize(true);
}



inline bool maybe_rebalance_from_timings(FieldHandle target_field, float threshold) {
    if (!partition_.has_value()) return false;
    if (partitions_.empty()) return false;

    std::unordered_map<int, detail::RegisteredField>::iterator fit =
        fields_.find(target_field.value);
    if (fit == fields_.end()) {
        throw Error("maybe_rebalance_from_timings(): unknown field");
    }

    const std::vector<double> measured_elapsed = compute_balance_window_elapsed_local();

    std::vector<double> local_times(partitions_.size(), 0.0);
    for (std::size_t i = 0; i < partitions_.size(); ++i) {
        if (partitions_[i].owning_rank == rank_ && partitions_[i].local_index >= 0) {
            const int li = partitions_[i].local_index;
            if (li >= 0 && static_cast<std::size_t>(li) < measured_elapsed.size()) {
                local_times[i] = measured_elapsed[static_cast<std::size_t>(li)];
            }
        }
    }

    std::vector<double> global_times(partitions_.size(), 0.0);
    detail::check_mpi(
        MPI_Allreduce(
            local_times.data(),
            global_times.data(),
            static_cast<int>(global_times.size()),
            MPI_DOUBLE,
            MPI_SUM,
            comm_
        ),
        "MPI_Allreduce(balance times)"
    );

    const std::vector<float> proposed_loads =
        detail::compute_loads_from_partition_throughput(global_times, partitions_);

    if (proposed_loads.empty()) return false;

    const std::vector<DevicePartition> old_parts = partitions_;
    const std::vector<DevicePartition> new_parts = partitions_from_loads(proposed_loads);
    if (new_parts.size() != old_parts.size()) {
        reset_balance_window_events();
        return false;
    }
    const std::vector<float> effective_new_loads = loads_from_partitions(new_parts);

    bool same = (old_parts.size() == new_parts.size());
    if (same) {
        for (std::size_t i = 0; i < old_parts.size(); ++i) {
            if (old_parts[i].global_offset != new_parts[i].global_offset ||
                old_parts[i].element_count != new_parts[i].element_count ||
                old_parts[i].owning_rank   != new_parts[i].owning_rank ||
                old_parts[i].local_index   != new_parts[i].local_index) {
                same = false;
                break;
            }
        }
    }

    if (same) {
        current_loads_ = loads_from_partitions(partitions_);
        reset_balance_window_events();
        return false;
    }

    double diff_l2 = std::numeric_limits<double>::infinity();
    if (!current_loads_.empty() && current_loads_.size() == effective_new_loads.size()) {
        diff_l2 = 0.0;
        for (std::size_t i = 0; i < effective_new_loads.size(); ++i) {
            const double d =
                static_cast<double>(current_loads_[i]) -
                static_cast<double>(effective_new_loads[i]);
            diff_l2 += d * d;
        }
        diff_l2 = std::sqrt(diff_l2);
    }
/*
    if (rank_ == 0) {
        
        std::cout << "[DCL][threshold] balance attempt:\n";
        std::cout << "  OLD loads:\n";
        detail::print_loads_debug(current_loads_);

        std::cout << "  PROPOSED loads:\n";
        detail::print_loads_debug(proposed_loads);
        std::cout << "  EFFECTIVE loads:\n";
        detail::print_loads_debug(effective_new_loads);
        std::cout
            << "  l2_diff=" << (100.0 * diff_l2) << "% "
            << "threshold=" << (100.0f * threshold) << "%\n";
    }
*/
    if (diff_l2 < static_cast<double>(threshold)) {
       // if (rank_ == 0) {
          //  std::cout << "  action=skip\n";
       // }
        reset_balance_window_events();
        return false;
    }

  //  if (rank_ == 0) {
   //     std::cout << "  action=rebalance\n";
  //  }

    this->rebalance_to(effective_new_loads);
    return true;
}

inline void load_profiling_data(const std::string& profiling_file) {
    int n_points = 0;

    if (rank_ == 0) {
        profiling_data_.clear();

        std::ifstream file(profiling_file);
        if (file.is_open()) {
            std::string header;
            std::getline(file, header);

            std::size_t v = 0;
            double m = 0.0;
            double b = 0.0;

            while (file >> v >> m >> b) {
                profiling_data_.push_back(ProfileSegment{v, m, b});
            }

            file.close();
        } else {
            std::cerr
                << "[DCL Warning] Rank 0 nao encontrou '"
                << profiling_file
                << "'. Usando fallback seguro."
                << std::endl;

            profiling_data_.push_back(ProfileSegment{16777216ull, 1.0e-9, 1.0e-6});
        }

        std::sort(
            profiling_data_.begin(),
            profiling_data_.end(),
            [](const ProfileSegment& a, const ProfileSegment& b) {
                return a.max_volume < b.max_volume;
            }
        );

        n_points = static_cast<int>(profiling_data_.size());
    }

    detail::check_mpi(
        MPI_Bcast(&n_points, 1, MPI_INT, 0, comm_),
        "MPI_Bcast(n_points)"
    );

    if (rank_ != 0) {
        profiling_data_.resize(static_cast<std::size_t>(n_points));
    }

    if (n_points > 0) {
        detail::check_mpi(
            MPI_Bcast(
                profiling_data_.data(),
                static_cast<int>(n_points * sizeof(ProfileSegment)),
                MPI_BYTE,
                0,
                comm_
            ),
            "MPI_Bcast(profiling_data)"
        );
    }

    profiling_loaded_ = true;
    profiling_file_loaded_ = profiling_file;
/* 
    if (rank_ == 0) {
        std::cout
            << "[DCL][profiled] loaded "
            << n_points
            << " profiling segments from '"
            << profiling_file
            << "'\n";
    }
            */
}

double get_migration_overhead(std::size_t volume) const {
    if (volume == 0 || profiling_data_.empty()) return 0.0;

    auto it = std::lower_bound(
        profiling_data_.begin(), profiling_data_.end(), volume,
        [](const ProfileSegment& p, std::size_t v) { return p.max_volume < v; }
    );

    if (it == profiling_data_.end()) {
        it = std::prev(profiling_data_.end());
    }

    const double overhead = it->m * static_cast<double>(volume) + it->b;
    return (overhead > 0.0) ? overhead : 0.0;
}



inline bool maybe_rebalance_profiled(FieldHandle target_field, const AutoBalancePolicy& policy, int interval) {
    if (!partition_.has_value()) return false;
    if (partitions_.empty()) return false;
    if (policy.total_iterations <= 0) return false;

    if (!profiling_loaded_ || profiling_file_loaded_ != policy.profiling_file) {
        load_profiling_data(policy.profiling_file);
    }

    // ---------------------------------------------------------------------
    // 1. Coleta tempos locais
    // ---------------------------------------------------------------------
    const std::vector<double> measured_elapsed = compute_balance_window_elapsed_local();

    std::vector<double> local_times(partitions_.size(), 0.0);
    for (std::size_t i = 0; i < partitions_.size(); ++i) {
        if (partitions_[i].owning_rank == rank_ && partitions_[i].local_index >= 0) {
            const int li = partitions_[i].local_index;
            if (li >= 0 && static_cast<std::size_t>(li) < measured_elapsed.size()) {
                local_times[i] = measured_elapsed[static_cast<std::size_t>(li)];
            }
        }
    }

    // ---------------------------------------------------------------------
    // 2. Redução global
    // ---------------------------------------------------------------------
    std::vector<double> global_times(partitions_.size(), 0.0);
    detail::check_mpi(
        MPI_Allreduce(
            local_times.data(),
            global_times.data(),
            static_cast<int>(global_times.size()),
            MPI_DOUBLE,
            MPI_SUM,
            comm_
        ),
        "MPI_Allreduce(profiled balance times)"
    );

    const std::vector<float> proposed_loads =
        detail::compute_loads_from_partition_throughput(global_times, partitions_);

    if (proposed_loads.empty()) return false;

    const std::vector<DevicePartition> old_parts = partitions_;
    const std::vector<DevicePartition> new_parts = partitions_from_loads(proposed_loads);

    if (new_parts.size() != old_parts.size()) {
        reset_balance_window_events();
        return false;
    }

    const std::vector<float> effective_new_loads = loads_from_partitions(new_parts);

    // ---------------------------------------------------------------------
    // 3. Verifica se mudou algo
    // ---------------------------------------------------------------------
    bool same = (old_parts.size() == new_parts.size());
    if (same) {
        for (std::size_t i = 0; i < old_parts.size(); ++i) {
            if (old_parts[i].global_offset != new_parts[i].global_offset ||
                old_parts[i].element_count != new_parts[i].element_count ||
                old_parts[i].owning_rank   != new_parts[i].owning_rank ||
                old_parts[i].local_index   != new_parts[i].local_index) {
                same = false;
                break;
            }
        }
    }

    if (same) {
        current_loads_ = loads_from_partitions(partitions_);
        reset_balance_window_events();
        return false;
    }

    // ---------------------------------------------------------------------
    // 4. Estima tempos
    // ---------------------------------------------------------------------
    double T_comp_int = 0.0;
    double C_total = 0.0;
    std::vector<double> capacity(partitions_.size(), 0.0);

    for (std::size_t i = 0; i < partitions_.size(); ++i) {
        if (global_times[i] > T_comp_int) {
            T_comp_int = global_times[i];
        }

        if (global_times[i] > 1.0e-12 && partitions_[i].element_count > 0) {
            capacity[i] = static_cast<double>(partitions_[i].element_count) / global_times[i];
            C_total += capacity[i];
        }
    }

    double T_comp_bal = T_comp_int;
    if (C_total > 0.0) {
        T_comp_bal = 0.0;

        for (std::size_t i = 0; i < new_parts.size(); ++i) {
            if (new_parts[i].element_count == 0) {
                continue;
            }

            if (i >= capacity.size() || capacity[i] <= 0.0) {
                T_comp_bal = T_comp_int;
                break;
            }

            const double projected =
                static_cast<double>(new_parts[i].element_count) / capacity[i];

            if (projected > T_comp_bal) {
                T_comp_bal = projected;
            }
        }
    }

    if (interval > 0) {
        T_comp_int /= static_cast<double>(interval);
        T_comp_bal /= static_cast<double>(interval);
    }

    // ---------------------------------------------------------------------
    // 5. Estima bytes migrados
    // ---------------------------------------------------------------------
    std::vector<std::size_t> rank_recv_bytes(static_cast<std::size_t>(size_), 0ull);

    auto count_migration_bytes = [&](FieldHandle fh) {
        auto fit = fields_.find(fh.value);
        if (fit == fields_.end()) return;

        const std::size_t elem_bytes =
            fit->second.spec.units_per_element * fit->second.spec.bytes_per_unit;

        std::size_t src_idx = 0;
        std::size_t dst_idx = 0;

        // Percorre as partições em tempo linear O(N)
        while (src_idx < old_parts.size() && dst_idx < new_parts.size()) {
            const auto& src = old_parts[src_idx];
            const auto& dst = new_parts[dst_idx];

            const std::size_t src_end = src.global_offset + src.element_count;
            const std::size_t dst_end = dst.global_offset + dst.element_count;

            const std::size_t start = std::max(src.global_offset, dst.global_offset);
            const std::size_t end = std::min(src_end, dst_end);

            // Se houver sobreposição espacial
            if (start < end) {
                const std::size_t len = end - start;

                // Contabiliza rede apenas se a fatia cruzar a fronteira entre ranks
                if (src.owning_rank != dst.owning_rank) {
                    rank_recv_bytes[static_cast<std::size_t>(dst.owning_rank)] +=
                        len * elem_bytes;
                }
            }

            // Avança o ponteiro da partição que termina primeiro
            if (src_end < dst_end) {
                ++src_idx;
            } else if (dst_end < src_end) {
                ++dst_idx;
            } else {
                ++src_idx;
                ++dst_idx;
            }
        }
    };

    // CORREÇÃO: o lambda recebe FieldHandle, então usamos a chave do mapa.
    for (const auto& kv : fields_) {
        count_migration_bytes(FieldHandle{kv.first});
    }

    std::size_t max_v_migrado = 0;
    for (std::size_t bytes : rank_recv_bytes) {
        if (bytes > max_v_migrado) {
            max_v_migrado = bytes;
        }
    }

    // ---------------------------------------------------------------------
    // 6. Modelo de custo
    // ---------------------------------------------------------------------
    const double custo_migracao = get_migration_overhead(max_v_migrado);

    const std::size_t remaining_iterations =
        (iteration_counter_ < static_cast<std::size_t>(policy.total_iterations))
            ? (static_cast<std::size_t>(policy.total_iterations) - iteration_counter_)
            : 0ull;

    const double ganho_por_iter = std::max(0.0, T_comp_int - T_comp_bal);
    const double ganho_total_estimado =
        ganho_por_iter * static_cast<double>(remaining_iterations);

    // ---------------------------------------------------------------------
    // 7. Logs opcionais
    // ---------------------------------------------------------------------
    /*
    if (rank_ == 0) {
        std::cout
            << "[DCL][profiled] iteration " << iteration_counter_ << ":\n"
            << "  remaining_iterations=" << remaining_iterations << "\n"
            << "  T_comp_int=" << T_comp_int << "\n"
            << "  T_comp_bal=" << T_comp_bal << "\n"
            << "  ganho_por_iter=" << ganho_por_iter << "\n"
            << "  ganho_total=" << ganho_total_estimado << "\n"
            << "  max_v_migrado=" << max_v_migrado << " bytes\n"
            << "  custo_migracao=" << custo_migracao << "\n";

        std::cout << "  OLD loads:\n";
        detail::print_loads_debug(current_loads_);

        std::cout << "  PROPOSED loads:\n";
        detail::print_loads_debug(proposed_loads);

        std::cout << "  EFFECTIVE loads:\n";
        detail::print_loads_debug(effective_new_loads);
    }
    */

    // ---------------------------------------------------------------------
    // 8. Decisão
    // ---------------------------------------------------------------------
    const int local_should_rebalance =
        (remaining_iterations != 0 && custo_migracao < ganho_total_estimado) ? 1 : 0;

    int min_should_rebalance = 0;
    int max_should_rebalance = 0;

    detail::check_mpi(
        MPI_Allreduce(
            &local_should_rebalance,
            &min_should_rebalance,
            1,
            MPI_INT,
            MPI_MIN,
            comm_
        ),
        "MPI_Allreduce(profiled rebalance decision min)"
    );

    detail::check_mpi(
        MPI_Allreduce(
            &local_should_rebalance,
            &max_should_rebalance,
            1,
            MPI_INT,
            MPI_MAX,
            comm_
        ),
        "MPI_Allreduce(profiled rebalance decision max)"
    );

    if (min_should_rebalance != max_should_rebalance) {
        throw Error("maybe_rebalance_profiled(): inconsistent rebalance decision across ranks");
    }

    if (local_should_rebalance == 0) {
        reset_balance_window_events();
        return false;
    }

    // ---------------------------------------------------------------------
    // 9. Aplica rebalanceamento
    // ---------------------------------------------------------------------
    this->synchronize(true);
    this->redistribute_all_registered_fields(old_parts, new_parts);
    partitions_ = new_parts;
    current_loads_ = loads_from_partitions(partitions_);
    this->synchronize(true);
    reset_balance_window_events();

    return true;
}



inline void transfer_field_range(
    FieldHandle fh,
    const DevicePartition& src_part,
    const DevicePartition& dst_part,
    std::size_t global_off,
    std::size_t len
) {
    if (len == 0) return;

    std::unordered_map<int, detail::RegisteredField>::iterator fit = fields_.find(fh.value);
    if (fit == fields_.end()) {
        throw Error("transfer_field_range(): unknown field");
    }

    detail::RegisteredField& rf = fit->second;
    const std::size_t elem_bytes =
        rf.spec.units_per_element * rf.spec.bytes_per_unit;

    const std::size_t byte_off = global_off * elem_bytes;
    const std::size_t byte_len = len * elem_bytes;
    const std::size_t total_bytes =
        rf.spec.global_elements * elem_bytes;

    if (byte_off + byte_len > total_bytes) {
        throw Error("transfer_field_range(): byte range out of bounds");
    }

    // Se origem e destino forem exatamente o mesmo device local, nada a fazer.
    if (src_part.owning_rank == dst_part.owning_rank &&
        src_part.owning_rank == rank_ &&
        src_part.local_index >= 0 &&
        dst_part.local_index >= 0 &&
        src_part.local_index == dst_part.local_index) {
        return;
    }

    if (src_part.owning_rank != rank_ && dst_part.owning_rank != rank_) {
        return;
    }

    constexpr std::size_t TRANSFER_STAGING_BYTES = 64ull * 1024ull * 1024ull;
    std::vector<unsigned char> chunk(
        std::min<std::size_t>(byte_len, TRANSFER_STAGING_BYTES)
    );

    const int tag_base = 4000;

    for (std::size_t done = 0; done < byte_len; done += chunk.size()) {
        const std::size_t chunk_len =
            std::min<std::size_t>(chunk.size(), byte_len - done);
        const std::size_t chunk_off = byte_off + done;

        // Fonte local: lê do device origem
        if (src_part.owning_rank == rank_ && src_part.local_index >= 0) {
            const std::size_t dsrc = static_cast<std::size_t>(src_part.local_index);

            cl_event read_ev = nullptr;
            detail::check_cl(
                clEnqueueReadBuffer(
                    local_devices_[dsrc].transfer_queue,
                    rf.replicas[dsrc],
                    CL_FALSE,
                    chunk_off,
                    chunk_len,
                    chunk.data(),
                    0,
                    nullptr,
                    &read_ev
                ),
                "clEnqueueReadBuffer(transfer_field_range read)"
            );

            detail::check_cl(
                clWaitForEvents(1, &read_ev),
                "clWaitForEvents(transfer_field_range read)"
            );
            clReleaseEvent(read_ev);
        }

        // Caso remoto: envio unidirecional por chunks
        if (src_part.owning_rank != dst_part.owning_rank) {
            this->mpi_transfer_bytes_chunked(
                src_part.owning_rank == rank_ ? chunk.data() : nullptr,
                dst_part.owning_rank == rank_ ? chunk.data() : nullptr,
                chunk_len,
                src_part.owning_rank == rank_ ? dst_part.owning_rank : src_part.owning_rank,
                tag_base + static_cast<int>(done / chunk.size()),
                src_part.owning_rank == rank_,
                dst_part.owning_rank == rank_
            );
        }

        // Destino local: escreve no device destino
        if (dst_part.owning_rank == rank_ && dst_part.local_index >= 0) {
            const std::size_t ddst = static_cast<std::size_t>(dst_part.local_index);

            cl_event write_ev = nullptr;
            detail::check_cl(
                clEnqueueWriteBuffer(
                    local_devices_[ddst].transfer_queue,
                    rf.replicas[ddst],
                    CL_FALSE,
                    chunk_off,
                    chunk_len,
                    chunk.data(),
                    0,
                    nullptr,
                    &write_ev
                ),
                src_part.owning_rank == rank_
                    ? "clEnqueueWriteBuffer(transfer_field_range local)"
                    : "clEnqueueWriteBuffer(transfer_field_range remote)"
            );

            detail::check_cl(
                clWaitForEvents(1, &write_ev),
                src_part.owning_rank == rank_
                    ? "clWaitForEvents(transfer_field_range local)"
                    : "clWaitForEvents(transfer_field_range remote)"
            );
            clReleaseEvent(write_ev);
        }
    }
}

inline void redistribute_field_proportional_delta(
    FieldHandle fh,
    const std::vector<DevicePartition>& old_parts,
    const std::vector<DevicePartition>& new_parts
) {
    std::unordered_map<int, detail::RegisteredField>::iterator fit = fields_.find(fh.value);
    if (fit == fields_.end()) {
        throw Error("redistribute_field_proportional_delta(): unknown field");
    }

    if (old_parts.size() != new_parts.size()) {
        throw Error("redistribute_field_proportional_delta(): mismatched partition counts");
    }

    // Garante que nada pendente ainda está usando os buffers
    this->synchronize(true);

    for (std::size_t i = 0; i < new_parts.size(); ++i) {
        const DevicePartition& oldp = old_parts[i];
        const DevicePartition& newp = new_parts[i];

        const std::size_t old_begin = oldp.global_offset;
        const std::size_t old_end   = oldp.global_offset + oldp.element_count;

        const std::size_t new_begin = newp.global_offset;
        const std::size_t new_end   = newp.global_offset + newp.element_count;

        // 1) Prefixo novo à esquerda: [new_begin, old_begin)
        if (new_begin < old_begin) {
            std::size_t need_begin = new_begin;
            const std::size_t need_end = std::min(old_begin, new_end);

            for (std::size_t src = 0; src < old_parts.size() && need_begin < need_end; ++src) {
                const DevicePartition& srcp = old_parts[src];
                const std::size_t src_begin = srcp.global_offset;
                const std::size_t src_end   = srcp.global_offset + srcp.element_count;

                std::size_t inter_off = 0;
                std::size_t inter_len = 0;
                if (!detail::intersect_1d(
                        need_begin,
                        need_end - need_begin,
                        src_begin,
                        src_end - src_begin,
                        inter_off,
                        inter_len)) {
                    continue;
                }

                this->transfer_field_range(fh, srcp, newp, inter_off, inter_len);
                need_begin = inter_off + inter_len;
            }
        }

        // 2) Sufixo novo à direita: [old_end, new_end)
        if (new_end > old_end) {
            std::size_t need_begin = std::max(old_end, new_begin);
            const std::size_t need_end = new_end;

            for (std::size_t src = 0; src < old_parts.size() && need_begin < need_end; ++src) {
                const DevicePartition& srcp = old_parts[src];
                const std::size_t src_begin = srcp.global_offset;
                const std::size_t src_end   = srcp.global_offset + srcp.element_count;

                std::size_t inter_off = 0;
                std::size_t inter_len = 0;
                if (!detail::intersect_1d(
                        need_begin,
                        need_end - need_begin,
                        src_begin,
                        src_end - src_begin,
                        inter_off,
                        inter_len)) {
                    continue;
                }

                this->transfer_field_range(fh, srcp, newp, inter_off, inter_len);
                need_begin = inter_off + inter_len;
            }
        }
    }

    this->synchronize_all_local_devices(false);
}

inline void redistribute_all_registered_fields(
    const std::vector<DevicePartition>& old_parts,
    const std::vector<DevicePartition>& new_parts
) {
    std::vector<int> field_ids;
    field_ids.reserve(fields_.size());
    for (std::unordered_map<int, detail::RegisteredField>::const_iterator it = fields_.begin();
         it != fields_.end(); ++it) {
        field_ids.push_back(it->first);
    }
    std::sort(field_ids.begin(), field_ids.end());

    for (int field_id : field_ids) {
        std::unordered_map<int, detail::RegisteredField>::iterator it = fields_.find(field_id);
        if (it == fields_.end()) {
            continue;
        }

        const FieldHandle fh{field_id};

        if (it->second.spec.redistribution == RedistributionDependency::none) {
            continue;
        }

        if (it->second.spec.redistribution == RedistributionDependency::total) {
            //this->redistribute_field_total(fh);
            continue;
        }

        if (it->second.spec.redistribution == RedistributionDependency::proportional) {
            this->redistribute_field_intersection(fh, old_parts, new_parts);
            detail::check_mpi(
                MPI_Barrier(comm_),
                "MPI_Barrier(redistribute proportional field)"
            );
            continue;
        }
    }
}

private:
    bool mpi_initialized_by_runtime_{false};

    int& argc_;
    char**& argv_;
    int rank_{0};
    int size_{1};
    MPI_Comm comm_{MPI_COMM_WORLD};

    std::vector<detail::PlatformContext> platforms_;
    std::vector<detail::LocalDevice> local_devices_;
    std::vector<DeviceInfo> devices_;
    std::vector<int> all_device_counts_;
    std::vector<DevicePartition> partitions_;

    std::unordered_map<int, detail::RegisteredField> fields_;
    std::unordered_map<int, detail::RegisteredKernel> kernels_;
    std::unordered_map<std::string, bool> static_balance_attempted_;

    std::optional<PartitionSpec> partition_;
    int active_kernel_{-1};

    int next_field_id_{0};
    int next_kernel_id_{0};

    std::vector<float> current_loads_;
    std::vector<DeviceTiming> device_timings_;
    std::vector<double> last_elapsed_local_;
    std::vector<cl_event> balance_window_start_events_;
    std::vector<cl_event> balance_window_end_events_;
    std::vector<bool> balance_window_valid_;
    std::vector<std::vector<cl_event>> balance_window_kernel_events_;
    std::unordered_map<int, std::vector<cl_event>> field_write_events_;
    std::size_t balance_window_begin_iteration_{0};
    std::vector<FieldHandle> last_input_fields_;
    std::vector<FieldHandle> last_output_fields_;
    std::size_t iteration_counter_{0};

    std::vector<ProfileSegment> profiling_data_;
    bool profiling_loaded_{false};
    std::string profiling_file_loaded_;
};

} // namespace dcl

#endif
