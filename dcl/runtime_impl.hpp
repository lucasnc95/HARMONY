
#ifndef DCL_RUNTIME_IMPL_HPP
#define DCL_RUNTIME_IMPL_HPP

#include "runtime.hpp"
#include "topo_metrics_io.hpp"
#include "algorithms.hpp"
#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 300
#endif
#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <mpi.h>

#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
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
#include <iomanip>

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

static inline std::string device_string(cl_device_id device, cl_device_info field) {
    std::size_t size = 0;
    check_cl(clGetDeviceInfo(device, field, 0, nullptr, &size), "clGetDeviceInfo(string size)");
    std::string value(size, '\0');
    check_cl(clGetDeviceInfo(device, field, size, value.data(), nullptr), "clGetDeviceInfo(string)");
    if (!value.empty() && value.back() == '\0') value.pop_back();
    return value;
}

static inline std::string platform_string(cl_platform_id platform, cl_platform_info field) {
    std::size_t size = 0;
    check_cl(clGetPlatformInfo(platform, field, 0, nullptr, &size), "clGetPlatformInfo(string size)");
    std::string value(size, '\0');
    check_cl(clGetPlatformInfo(platform, field, size, value.data(), nullptr), "clGetPlatformInfo(string)");
    if (!value.empty() && value.back() == '\0') value.pop_back();
    return value;
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
        const double t = tempos_medidos[static_cast<std::size_t>(i)];
        // A zero or negative time means the participant has no devices / zero capacity.
        // Do NOT clamp to 1e-9: that would assign enormous (1/1e-9) capacity to
        // device-less ranks, causing their allocated elements to vanish from the partition.
        if (t > 0.0) {
            capacidade[static_cast<std::size_t>(i)] = 1.0 / t;
            capacidade_total += capacidade[static_cast<std::size_t>(i)];
        }
        // else capacidade[i] stays 0.0 — this rank gets zero load share
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
    return ::dcl::erad_loads(times, partitions);
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

template<typename T, typename Deleter>
class AutoCLHandle {
    T handle_{nullptr};
public:
    AutoCLHandle() = default;
    AutoCLHandle(T h) : handle_(h) {}
    ~AutoCLHandle() { if (handle_) Deleter()(handle_); }
    AutoCLHandle(const AutoCLHandle&) = delete;
    AutoCLHandle& operator=(const AutoCLHandle&) = delete;
    AutoCLHandle(AutoCLHandle&& other) noexcept : handle_(other.handle_) {
        other.handle_ = nullptr;
    }
    AutoCLHandle& operator=(AutoCLHandle&& other) noexcept {
        if (this != &other) {
            if (handle_) Deleter()(handle_);
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }
    AutoCLHandle& operator=(T h) {
        if (handle_) Deleter()(handle_);
        handle_ = h;
        return *this;
    }
    operator T() const { return handle_; }
    T get() const { return handle_; }
    T* ptr() { return &handle_; }
    bool operator==(T h) const { return handle_ == h; }
    bool operator!=(T h) const { return handle_ != h; }
};

struct CLMemDeleter { void operator()(cl_mem m) const { if(m) clReleaseMemObject(m); } };
struct CLContextDeleter { void operator()(cl_context m) const { if(m) clReleaseContext(m); } };
struct CLQueueDeleter { void operator()(cl_command_queue m) const { if(m) clReleaseCommandQueue(m); } };
struct CLProgramDeleter { void operator()(cl_program m) const { if(m) clReleaseProgram(m); } };
struct CLKernelDeleter { void operator()(cl_kernel m) const { if(m) clReleaseKernel(m); } };
struct EventGuard {
    cl_event& event;
    ~EventGuard() { if (event != nullptr) clReleaseEvent(event); }
};

using AutoMem = AutoCLHandle<cl_mem, CLMemDeleter>;
using AutoContext = AutoCLHandle<cl_context, CLContextDeleter>;
using AutoQueue = AutoCLHandle<cl_command_queue, CLQueueDeleter>;
using AutoProgram = AutoCLHandle<cl_program, CLProgramDeleter>;
using AutoKernel = AutoCLHandle<cl_kernel, CLKernelDeleter>;

struct PlatformContext {
    cl_platform_id platform{nullptr};
    AutoContext context;
    std::vector<cl_device_id> devices;
};

struct LocalDevice {
    int platform_index{-1};
    int device_index_in_platform{-1};
    int local_index{-1};
    int global_index{-1};

    cl_device_id device{nullptr};
    cl_context context{nullptr};
    AutoQueue kernel_queue;
    AutoQueue transfer_queue;

    cl_uint compute_units{0};
    DeviceKind kind{DeviceKind::all};
    std::string name;
};

struct RegisteredField {
    FieldSpec spec;
    std::vector<AutoMem> replicas;
};

struct RegisteredKernel {
    KernelSpec spec;
    std::vector<AutoProgram> programs_per_platform;
    std::vector<AutoKernel> kernels_per_local_device;
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
#ifdef __CPPCHECK__
        cppcheck_anchor_unused();
#endif
        reset_balance_window_events();
        clear_runtime_state();
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

        try {
            std::exception_ptr local_discovery_error;
            try {
            cl_uint n_platforms = 0;
            const cl_int platform_status = clGetPlatformIDs(0, nullptr, &n_platforms);
            if (platform_status != CL_PLATFORM_NOT_FOUND_KHR) {
                detail::check_cl(platform_status, "clGetPlatformIDs(count)");
            }

            std::vector<cl_platform_id> platform_ids(n_platforms);
            if (n_platforms > 0) {
                detail::check_cl(clGetPlatformIDs(n_platforms, platform_ids.data(), nullptr),
                                 "clGetPlatformIDs(list)");
            }

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

                platforms_.push_back(std::move(pc));
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

                    local_devices_.push_back(std::move(ld));
                    const auto& owned_device = local_devices_.back();

                    DeviceInfo info{};
                    info.rank = rank_;
                    info.local_index = owned_device.local_index;
                    info.global_index = -1;
                    info.name = owned_device.name;
                    info.kind = owned_device.kind;
                    info.compute_units = owned_device.compute_units;
                    info.platform_name = detail::platform_string(platform_ids[p], CL_PLATFORM_NAME);
                    info.platform_version = detail::platform_string(platform_ids[p], CL_PLATFORM_VERSION);
                    info.device_version = detail::device_string(owned_device.device, CL_DEVICE_VERSION);
                    info.driver_version = detail::device_string(owned_device.device, CL_DRIVER_VERSION);
                    const std::string extensions = detail::device_string(owned_device.device, CL_DEVICE_EXTENSIONS);
                    const auto has_extension = [&extensions](const std::string& name) {
                        return (" " + extensions + " ").find(" " + name + " ") != std::string::npos;
                    };
                    if (has_extension("cl_khr_device_uuid")) {
                        unsigned char uuid[CL_UUID_SIZE_KHR]{};
                        detail::check_cl(clGetDeviceInfo(owned_device.device, CL_DEVICE_UUID_KHR, sizeof(uuid), uuid, nullptr),
                                         "clGetDeviceInfo(CL_DEVICE_UUID_KHR)");
                        std::ostringstream token;
                        for (unsigned char byte : uuid) {
                            token << std::hex << std::setfill('0') << std::setw(2)
                                  << static_cast<unsigned>(byte);
                        }
                        info.identity_kind = "uuid";
                        info.identity = token.str();
                    } else if (has_extension("cl_khr_pci_bus_info")) {
                        cl_device_pci_bus_info_khr pci{};
                        detail::check_cl(clGetDeviceInfo(owned_device.device, CL_DEVICE_PCI_BUS_INFO_KHR, sizeof(pci), &pci, nullptr),
                                         "clGetDeviceInfo(CL_DEVICE_PCI_BUS_INFO_KHR)");
                        std::ostringstream token;
                        token << std::hex << std::setfill('0') << std::setw(4) << pci.pci_domain << ':'
                              << std::setw(2) << pci.pci_bus << ':' << std::setw(2) << pci.pci_device
                              << '.' << pci.pci_function;
                        info.identity_kind = "pci";
                        info.identity = token.str();
                    }
                    devices_.push_back(info);
                }

                if (next_local_index >= max_per_rank) break;
            }
            } catch (...) {
                local_discovery_error = std::current_exception();
            }

            int local_ok = local_discovery_error ? 0 : 1;
            int all_ok = 0;
            detail::check_mpi(
                MPI_Allreduce(&local_ok, &all_ok, 1, MPI_INT, MPI_MIN, comm_),
                "MPI_Allreduce(device discovery status)"
            );
            if (all_ok == 0) {
                if (local_discovery_error) std::rethrow_exception(local_discovery_error);
                throw Error("Device discovery failed on another MPI rank");
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
        } catch (...) {
            clear_runtime_state();
            throw;
        }
    }

    const std::vector<DeviceInfo>& devices() const noexcept { return devices_; }
    const std::vector<DevicePartition>& partitions() const noexcept { return partitions_; }
    const std::vector<DeviceTiming>& device_timings() const noexcept { return device_timings_; }
    int rank() const noexcept { return rank_; }
    int size() const noexcept { return size_; }
    MPI_Comm communicator() const noexcept { return comm_; }
    void set_verbose_metrics(bool enabled) noexcept { verbose_metrics_ = enabled; }

    FieldHandle create_field(const FieldSpec& spec) {
        const int global_devices =
            std::accumulate(all_device_counts_.begin(), all_device_counts_.end(), 0);
        if (local_devices_.empty() && simulated_total_devices_ <= 0 &&
            global_devices <= 0) {
            throw Error("discover_devices() must be called before create_field()");
        }
        if (spec.global_elements == 0) throw Error("FieldSpec.global_elements must be > 0");
        if (spec.units_per_element == 0) throw Error("FieldSpec.units_per_element must be > 0");
        if (spec.bytes_per_unit == 0) throw Error("FieldSpec.bytes_per_unit must be > 0");

        FieldHandle h{next_field_id_++};
        detail::RegisteredField rf;
        rf.spec = spec;
        rf.replicas.resize(local_devices_.empty() && simulated_total_devices_ > 0
            ? static_cast<std::size_t>(simulated_total_devices_)
            : local_devices_.size());

        const std::size_t total_bytes = spec.global_elements * spec.units_per_element * spec.bytes_per_unit;
        const cl_mem_flags flags = detail::to_opencl_flags(spec.usage);

        for (std::size_t d = 0; d < local_devices_.size(); ++d) {
            cl_int err = CL_SUCCESS;
            rf.replicas[d] = clCreateBuffer(local_devices_[d].context, flags, total_bytes, nullptr, &err);
            detail::check_cl(err, "clCreateBuffer(create_field)");
        }

        fields_.insert(std::make_pair(h.value, std::move(rf)));
        try {
            if (spec.host_ptr != nullptr && !local_devices_.empty()) {
                write_initial_field_data(h, spec.host_ptr);
            }
        } catch (...) {
            fields_.erase(h.value);
            throw;
        }
        return h;
    }

    KernelHandle create_kernel(const KernelSpec& spec) {
        if (local_devices_.empty() &&
            std::accumulate(all_device_counts_.begin(), all_device_counts_.end(), 0) <= 0) {
            throw Error("discover_devices() must be called before create_kernel()");
        }
        if (spec.source_file.empty()) throw Error("KernelSpec.source_file is empty");
        if (spec.entry_point.empty()) throw Error("KernelSpec.entry_point is empty");

        KernelHandle h{next_kernel_id_++};
        detail::RegisteredKernel rk;
        rk.spec = spec;
        rk.programs_per_platform.resize(platforms_.size());
        rk.kernels_per_local_device.resize(local_devices_.size());

        const std::string src = detail::slurp_file(spec.source_file);
        const char* src_ptr = src.c_str();
        const size_t src_len = src.size();

        for (std::size_t p = 0; p < platforms_.size(); ++p) {
            cl_int err = CL_SUCCESS;
            detail::AutoProgram program = clCreateProgramWithSource(platforms_[p].context, 1, &src_ptr, &src_len, &err);
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
                throw Error(oss.str());
            }
            rk.programs_per_platform[p] = std::move(program);
        }

        for (std::size_t d = 0; d < local_devices_.size(); ++d) {
            const int p = local_devices_[d].platform_index;
            cl_int err = CL_SUCCESS;
            rk.kernels_per_local_device[d] = clCreateKernel(rk.programs_per_platform[p], spec.entry_point.c_str(), &err);
            detail::check_cl(err, "clCreateKernel");
        }

        kernels_.insert(std::make_pair(h.value, std::move(rk)));
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

    if (step.halo.width_elements > 0 && !step.halo.fields.empty() &&
        partitions_.size() >= 2) {
        auto has_unsafe_micro = [&](const std::vector<DevicePartition>& parts) {
            std::size_t active = 0;
            for (const auto& part : parts) active += part.element_count > 0 ? 1u : 0u;
            if (active < 2) return false;
            for (const auto& part : parts) {
                if (part.element_count > 0 && part.element_count < step.halo.width_elements) {
                    return true;
                }
            }
            return false;
        };

        std::vector<DevicePartition> projected = partitions_;
        std::vector<float> new_cum;
        for (std::size_t pass = 0;
             pass < partitions_.size() && has_unsafe_micro(projected);
             ++pass) {
            std::vector<std::size_t> counts;
            counts.reserve(projected.size());
            for (const auto& part : projected) counts.push_back(part.element_count);

            for (std::size_t p = 0; p < counts.size(); ++p) {
                if (counts[p] == 0 || counts[p] >= step.halo.width_elements) continue;
                std::size_t recipient = counts.size();
                for (std::size_t q = p; q > 0; --q) {
                    if (counts[q - 1] > 0) {
                        recipient = q - 1;
                        break;
                    }
                }
                if (recipient == counts.size()) {
                    for (std::size_t q = p + 1; q < counts.size(); ++q) {
                        if (counts[q] > 0) {
                            recipient = q;
                            break;
                        }
                    }
                }
                if (recipient == counts.size()) break;
                counts[recipient] += counts[p];
                counts[p] = 0;
            }

            new_cum.resize(counts.size());
            std::size_t covered = 0;
            for (std::size_t p = 0; p < counts.size(); ++p) {
                covered += counts[p];
                new_cum[p] = static_cast<float>(
                    static_cast<double>(covered) /
                    static_cast<double>(partition_->global_elements)
                );
            }
            new_cum.back() = 1.0f;
            projected = partitions_from_loads(new_cum);
        }
        const int local_invalid =
            (projected.size() != partitions_.size() || has_unsafe_micro(projected)) ? 1 : 0;
        int local_rebalance = 0;
        if (!local_invalid) {
            for (std::size_t p = 0; p < projected.size(); ++p) {
                if (projected[p].global_offset != partitions_[p].global_offset ||
                    projected[p].element_count != partitions_[p].element_count) {
                    local_rebalance = 1;
                    break;
                }
            }
        }
        const int local_state[2] = {local_invalid, local_rebalance};
        int minimum_state[2] = {};
        int maximum_state[2] = {};
        detail::check_mpi(
            MPI_Allreduce(local_state, minimum_state, 2, MPI_INT, MPI_MIN, comm_),
            "MPI_Allreduce(halo state minimum)"
        );
        detail::check_mpi(
            MPI_Allreduce(local_state, maximum_state, 2, MPI_INT, MPI_MAX, comm_),
            "MPI_Allreduce(halo state maximum)"
        );
        if (maximum_state[0]) {
            throw Error("execute(): halo partitions cannot satisfy the minimum width");
        }
        if (minimum_state[1] != maximum_state[1]) {
            throw Error("execute(): inconsistent halo partitions across ranks");
        }
        if (local_rebalance) {
            rebalance_to(new_cum);
        }
    }

    const int every = (step.balance.interval <= 0) ? 1 : step.balance.interval;

    // Coleta eventos se o balanceamento está ativo OU se o usuário configurou
    // um intervalo. Isso permite imprimir métricas mesmo com balance-mode off.
    const bool collect_balance_window =
        (step.balance.mode != BalanceMode::off) || (step.balance.interval > 0);

    for (std::size_t inv = 0; inv < step.invocations.size(); ++inv) {
        const KernelInvocation& ki = step.invocations[inv];

        auto kit = kernels_.find(ki.binding.kernel.value);
        if (kit == kernels_.end()) {
            throw Error("Unknown kernel in execute()");
        }

        detail::RegisteredKernel& rk = kit->second;

        std::vector<FieldHandle> output_fields =
            fields_with_role(step, StepFieldRole::write_target);

        if (output_fields.empty()) {
            output_fields = output_fields_from_binding(ki.binding);
        }

        if (step.halo.width_elements > 0 && !step.halo.fields.empty()) {
            std::vector<std::pair<std::size_t, cl_event>> interior_events;
            std::vector<std::pair<std::size_t, cl_event>> border_events;
            try {
                run_interior_phase(rk, ki, interior_events, step.halo.width_elements);
                if (collect_balance_window) {
                    update_balance_window_events(interior_events, {});
                    append_balance_window_kernel_events(interior_events);
                }
                record_field_write_events(output_fields, interior_events);
                exchange_halo_set(step.halo);
                run_border_phase(rk, ki, border_events, step.halo.width_elements);
                if (collect_balance_window) {
                    update_balance_window_events({}, border_events);
                    append_balance_window_kernel_events(border_events);
                }
                record_field_write_events(output_fields, border_events);
            } catch (...) {
                release_kernel_events(interior_events);
                release_kernel_events(border_events);
                throw;
            }
            release_kernel_events(interior_events);
            release_kernel_events(border_events);
        } else {
            std::vector<std::pair<std::size_t, cl_event>> full_events;
            try {
                run_full_phase(rk, ki, full_events);
                if (collect_balance_window) {
                    update_balance_window_events(full_events, full_events);
                    append_balance_window_kernel_events(full_events);
                }
                record_field_write_events(output_fields, full_events);
            } catch (...) {
                release_kernel_events(full_events);
                throw;
            }
            release_kernel_events(full_events);
        }

        last_input_fields_.clear();
        last_output_fields_.clear();

        extract_rw_fields(step, ki.binding);
    }

    ++iteration_counter_;

    const int first_every = first_rebalance_interval_from_env(every);
    bool interval_hit = false;

    if (first_every > 0 && iteration_counter_ < static_cast<std::size_t>(first_every)) {
        interval_hit = false;
    } else if (first_every > 0 && iteration_counter_ == static_cast<std::size_t>(first_every)) {
        interval_hit = true;
    } else {
        const std::size_t base = (first_every > 0) ? static_cast<std::size_t>(first_every) : 0u;
        const std::size_t delta = iteration_counter_ - base;
        interval_hit = (delta % static_cast<std::size_t>(every)) == 0;
    }

    if (!interval_hit) {
        if (step.synchronize_at_end) {
            synchronize_all_local_devices(false);
        }
        return;
    }

    // Pega TODOS os campos marcados como rebalance_source.
    std::vector<FieldHandle> rebalance_fields =
        fields_with_role(step, StepFieldRole::rebalance_source);

    // Fallback seguro para códigos antigos.
    if (rebalance_fields.empty()) {
        rebalance_fields = fields_with_role(step, StepFieldRole::read_source);
    }

    if (rebalance_fields.empty()) {
        rebalance_fields = last_input_fields_;
    }

    if (rebalance_fields.empty()) {
        rebalance_fields = last_output_fields_;
    }

    rebalance_fields = unique_existing_proportional_fields(rebalance_fields);

    if (step.balance.mode == BalanceMode::off) {
        const std::vector<double> global_times =
            collect_global_balance_times_from_window();

        print_balance_interval_metrics(
            "off",
            "skip",
            "balance disabled",
            global_times,
            current_loads_,
            std::vector<float>(),
            current_loads_,
            0.0,
            true
        );

        if (step.synchronize_at_end) {
            synchronize_all_local_devices(false);
        }

        return;
    }

    bool should_try = false;
    const char* no_try_reason = "not scheduled";

    if (rebalance_fields.empty() && step.balance.mode != BalanceMode::hierarchical) {
        should_try = false;
        no_try_reason = "no proportional rebalance_source field";
    } else if (step.balance.mode == BalanceMode::dynamic_threshold ||
               step.balance.mode == BalanceMode::dynamic_profiled ||
               step.balance.mode == BalanceMode::hierarchical) {
        should_try = true;
        no_try_reason = (step.balance.mode == BalanceMode::hierarchical)
            ? "hierarchical interval hit"
            : "dynamic interval hit";
    } else if (step.balance.mode == BalanceMode::static_threshold ||
               step.balance.mode == BalanceMode::static_profiled) {
        bool& attempted = static_balance_attempted_[step.name];

        if (!attempted) {
            should_try = true;
            attempted = true;
            no_try_reason = "static first attempt";
        } else {
            should_try = false;
            no_try_reason = "static balance already attempted";
        }
    }

    if (should_try) {
        if (step.balance.mode == BalanceMode::dynamic_threshold ||
            step.balance.mode == BalanceMode::static_threshold) {
            this->maybe_rebalance_from_timings(
                rebalance_fields,
                step.balance
            );
        } else if (step.balance.mode == BalanceMode::dynamic_profiled ||
                   step.balance.mode == BalanceMode::static_profiled) {
            this->maybe_rebalance_profiled(
                rebalance_fields,
                step.balance,
                step.balance.interval
            );
        } else if (step.balance.mode == BalanceMode::hierarchical) {
            this->maybe_rebalance_hierarchical(rebalance_fields);
        }
    } else {
        const std::vector<double> global_times =
            collect_global_balance_times_from_window();

        print_balance_interval_metrics(
            "scheduled",
            "skip",
            no_try_reason,
            global_times,
            current_loads_,
            std::vector<float>(),
            current_loads_,
            0.0,
            true
        );
    }

    if (step.synchronize_at_end) {
        synchronize_all_local_devices(false);
    }
}
    void rebalance_to(const std::vector<float>& loads) {
    int local_invalid = (!partition_.has_value() || partitions_.empty() ||
        (rank_ == 0 && loads.empty()) ||
        (!loads.empty() && loads.size() != partitions_.size())) ? 1 : 0;
    int any_invalid = 0;
    detail::check_mpi(
        MPI_Allreduce(&local_invalid, &any_invalid, 1, MPI_INT, MPI_MAX, comm_),
        "MPI_Allreduce(rebalance input status)"
    );
    if (any_invalid) throw Error("rebalance_to(): inconsistent partition or load input");

    unsigned long long local_shape[2] = {
        static_cast<unsigned long long>(partitions_.size()),
        static_cast<unsigned long long>(partition_->global_elements)
    };
    unsigned long long min_shape[2] = {};
    unsigned long long max_shape[2] = {};
    detail::check_mpi(
        MPI_Allreduce(local_shape, min_shape, 2, MPI_UNSIGNED_LONG_LONG, MPI_MIN, comm_),
        "MPI_Allreduce(rebalance minimum shape)"
    );
    detail::check_mpi(
        MPI_Allreduce(local_shape, max_shape, 2, MPI_UNSIGNED_LONG_LONG, MPI_MAX, comm_),
        "MPI_Allreduce(rebalance maximum shape)"
    );
    if (min_shape[0] != max_shape[0] || min_shape[1] != max_shape[1]) {
        throw Error("rebalance_to(): partition shape differs across ranks");
    }

    std::vector<float> synced_loads = loads;
    if (rank_ != 0) {
        synced_loads.resize(partitions_.size(), 0.0f);
    }
    
    detail::check_mpi(
        MPI_Bcast(synced_loads.data(), static_cast<int>(partitions_.size()), MPI_FLOAT, 0, comm_),
        "MPI_Bcast(rebalance loads)"
    );

    // From now on, loads is cumulative: [0.10, 0.20, ..., 1.00].
    std::vector<float> cumulative = synced_loads;

    float prev = 0.0f;
    for (std::size_t i = 0; i < cumulative.size(); ++i) {
        if (!std::isfinite(cumulative[i])) {
            throw Error("rebalance_to(): non-finite cumulative load");
        }

        if (cumulative[i] < 0.0f) {
            throw Error("rebalance_to(): negative cumulative load not allowed");
        }

        if (cumulative[i] < prev) {
            cumulative[i] = prev;
        }

        if (cumulative[i] > 1.0f) {
            cumulative[i] = 1.0f;
        }

        prev = cumulative[i];
    }

    cumulative.back() = 1.0f;

    const std::vector<DevicePartition> old_parts = partitions_;
    const std::vector<DevicePartition> new_parts =
        partitions_from_loads(cumulative);

    if (new_parts.size() != old_parts.size()) {
        throw Error("rebalance_to(): rounded partition count mismatch");
    }
    auto partition_signature = [](const std::vector<DevicePartition>& parts) {
        unsigned long long hash = 1469598103934665603ULL;
        for (const auto& part : parts) {
            for (unsigned long long value : {
                    static_cast<unsigned long long>(part.global_offset),
                    static_cast<unsigned long long>(part.element_count),
                    static_cast<unsigned long long>(part.owning_rank),
                    static_cast<unsigned long long>(part.device_global_index)}) {
                hash = (hash ^ value) * 1099511628211ULL;
            }
        }
        return hash;
    };
    unsigned long long local_hash[2] = {
        partition_signature(old_parts), partition_signature(new_parts)
    };
    unsigned long long min_hash[2] = {};
    unsigned long long max_hash[2] = {};
    detail::check_mpi(
        MPI_Allreduce(local_hash, min_hash, 2, MPI_UNSIGNED_LONG_LONG, MPI_MIN, comm_),
        "MPI_Allreduce(rebalance minimum partition signature)"
    );
    detail::check_mpi(
        MPI_Allreduce(local_hash, max_hash, 2, MPI_UNSIGNED_LONG_LONG, MPI_MAX, comm_),
        "MPI_Allreduce(rebalance maximum partition signature)"
    );
    if (min_hash[0] != max_hash[0] || min_hash[1] != max_hash[1]) {
        throw Error("rebalance_to(): partition map differs across ranks");
    }

    bool same = true;
    for (std::size_t i = 0; i < old_parts.size(); ++i) {
        if (old_parts[i].global_offset != new_parts[i].global_offset ||
            old_parts[i].element_count != new_parts[i].element_count ||
            old_parts[i].owning_rank   != new_parts[i].owning_rank ||
            old_parts[i].local_index   != new_parts[i].local_index) {
            same = false;
            break;
        }
    }

    if (same) {
        current_loads_ = loads_from_partitions(partitions_);
        reset_balance_window_events();
        interval_comm_seconds_local_ = 0.0;
        print_partition_loads(current_loads_);
        return;
    }

    this->synchronize(true);

    // Manual rebalance: move all proportional fields.
    // Automatic per-step balancing should use redistribute_selected_registered_fields().
    redistribute_all_registered_fields(old_parts, new_parts);

    partitions_ = new_parts;
    current_loads_ = loads_from_partitions(partitions_);

    this->synchronize(true);

    reset_balance_window_events();
    interval_comm_seconds_local_ = 0.0;

    print_partition_loads(current_loads_);
}

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

public:
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

  void mpi_transfer_bytes_chunked(
    const unsigned char* sendbuf,
    unsigned char* recvbuf,
    std::size_t total_bytes,
    int peer_rank,
    int tag_base,
    bool do_send,
    bool do_recv
) {
    if (do_send && sendbuf == nullptr) {
        throw Error("mpi_transfer_bytes_chunked(): send buffer is null");
    }

    if (do_recv && recvbuf == nullptr) {
        throw Error("mpi_transfer_bytes_chunked(): recv buffer is null");
    }

    check_mpi_tag_base(
        tag_base,
        total_bytes,
        "mpi_transfer_bytes_chunked()"
    );

    std::size_t offset = 0;
    int chunk_id = 0;

    while (offset < total_bytes) {
        const std::size_t remaining = total_bytes - offset;

        const int chunk = static_cast<int>(
            std::min<std::size_t>(
                remaining,
                static_cast<std::size_t>(INT_MAX)
            )
        );

        MPI_Request reqs[2];
        int req_count = 0;

        const double t0 = MPI_Wtime();

        if (do_recv) {
            detail::check_mpi(
                MPI_Irecv(
                    recvbuf + offset,
                    chunk,
                    MPI_BYTE,
                    peer_rank,
                    tag_base + chunk_id,
                    comm_,
                    &reqs[req_count++]
                ),
                "MPI_Irecv(chunked transfer)"
            );
        }

        if (do_send) {
            detail::check_mpi(
                MPI_Isend(
                    sendbuf + offset,
                    chunk,
                    MPI_BYTE,
                    peer_rank,
                    tag_base + chunk_id,
                    comm_,
                    &reqs[req_count++]
                ),
                "MPI_Isend(chunked transfer)"
            );
        }

        if (req_count > 0) {
            detail::check_mpi(
                MPI_Waitall(
                    req_count,
                    reqs,
                    MPI_STATUSES_IGNORE
                ),
                "MPI_Waitall(chunked transfer)"
            );
        }

        const double t1 = MPI_Wtime();

        interval_comm_seconds_local_ += (t1 - t0);

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
        if (!verbose_metrics_ || rank_ != 0) return;

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
        // Fields and kernels release their device resources before queues and contexts.
        fields_.clear();
        kernels_.clear();
        local_devices_.clear();
        platforms_.clear();
        devices_.clear();
        partitions_.clear();
        all_device_counts_.clear();
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
        interval_comm_seconds_local_ = 0.0;
        accumulated_compute_seconds_ = 0.0;
        accumulated_comm_seconds_ = 0.0;
        accumulated_balance_seconds_ = 0.0;
        accumulated_rebalance_apply_seconds_ = 0.0;
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
        total_devices = std::accumulate(all_device_counts_.begin(), all_device_counts_.end(), 0);
        if (total_devices <= 0 && simulated_total_devices_ > 0) {
            total_devices = simulated_total_devices_;
        }
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

            const std::size_t min_cut = unit_cuts[i];
            const std::size_t max_cut = total_units;

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
        struct EventOwner {
            std::vector<cl_event> events;
            ~EventOwner() {
                for (cl_event event : events) {
                    if (event != nullptr) clReleaseEvent(event);
                }
            }
        } write_events;

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
            write_events.events.push_back(ev);
            detail::check_cl(clFlush(local_devices_[d].transfer_queue),
                             "clFlush(initial field write)");
        }

        for (cl_event ev : write_events.events) {
            if (ev != nullptr) {
                detail::check_cl(clWaitForEvents(1, &ev), "clWaitForEvents(write_initial_field_data)");
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

    std::vector<FieldHandle> unique_existing_proportional_fields(
        const std::vector<FieldHandle>& input
    ) const {
        std::vector<int> ids;
        ids.reserve(input.size());

        for (FieldHandle fh : input) {
            std::unordered_map<int, detail::RegisteredField>::const_iterator it =
                fields_.find(fh.value);
            if (it == fields_.end()) {
                continue;
            }
            if (it->second.spec.redistribution != RedistributionDependency::proportional) {
                continue;
            }
            ids.push_back(fh.value);
        }

        std::sort(ids.begin(), ids.end());
        ids.erase(std::unique(ids.begin(), ids.end()), ids.end());

        std::vector<FieldHandle> out;
        out.reserve(ids.size());
        std::transform(ids.begin(), ids.end(), std::back_inserter(out),
                       [](int id) { return FieldHandle{id}; });
        return out;
    }

    std::optional<FieldHandle> first_field_with_role(const ExecutionStep& step, StepFieldRole role) {
        const auto it = std::find_if(step.field_tags.begin(), step.field_tags.end(),
                                     [role](const StepFieldTag& tag) { return tag.role == role; });
        if (it != step.field_tags.end()) return it->field;
        return std::nullopt;
    }

void extract_rw_fields(
    const ExecutionStep& step,
    const KernelBinding& binding
) {
    last_input_fields_.clear();
    last_output_fields_.clear();

    for (const StepFieldTag& tag : step.field_tags) {
        if (tag.role == StepFieldRole::read_source) {
            last_input_fields_.push_back(tag.field);
        } else if (tag.role == StepFieldRole::write_target) {
            last_output_fields_.push_back(tag.field);
        }
    }

    if (!last_input_fields_.empty() || !last_output_fields_.empty()) {
        return;
    }

    for (std::size_t i = 0; i < binding.args.size(); ++i) {
        const unsigned idx = binding.args[i].first;
        const KernelArg& arg = binding.args[i].second;

        if (const FieldHandle* fh = std::get_if<FieldHandle>(&arg)) {
            if (idx == 0) {
                last_output_fields_.push_back(*fh);
            } else if (idx == 1) {
                last_input_fields_.push_back(*fh);
            }
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
            const KernelArg& arg = binding.args[i].second;
            const FieldHandle* fh = std::get_if<FieldHandle>(&arg);
            if (fh == nullptr) continue;

            std::unordered_map<int, std::vector<cl_event>>::const_iterator it =
                field_write_events_.find(fh->value);
            if (it == field_write_events_.end()) continue;
            if (local_device >= it->second.size()) continue;

            cl_event ev = it->second[local_device];
            if (ev == nullptr) continue;

            const bool seen = std::any_of(deps.begin(), deps.end(),
                                          [ev](cl_event existing) { return existing == ev; });
            if (!seen) deps.push_back(ev);
        }
        if (!deps.empty() && local_device < local_devices_.size()) {
            detail::check_cl(clFlush(local_devices_[local_device].kernel_queue),
                             "clFlush(kernel dependency producer)");
            detail::check_cl(clFlush(local_devices_[local_device].transfer_queue),
                             "clFlush(transfer dependency producer)");
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

            detail::check_cl(clFlush(local_devices_[d].kernel_queue),
                             "clFlush(kernel profiling)");
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

            detail::check_cl(clFlush(local_devices_[d].kernel_queue), "clFlush(kernel queue)");
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
        struct MarkerOwner {
            std::vector<cl_event>& events;
            ~MarkerOwner() {
                for (cl_event event : events) {
                    if (event != nullptr) clReleaseEvent(event);
                }
            }
        } owned_markers{marker_events};

        for (std::size_t d = 0; d < local_devices_.size(); ++d) {
            if (local_devices_[d].kernel_queue != nullptr) {
                cl_event ev = nullptr;
#if CL_TARGET_OPENCL_VERSION >= 120
                detail::check_cl(clEnqueueMarkerWithWaitList(local_devices_[d].kernel_queue, 0, nullptr, &ev), "clEnqueueMarkerWithWaitList(kernel queue)");
#else
                detail::check_cl(clEnqueueMarker(local_devices_[d].kernel_queue, &ev), "clEnqueueMarker(kernel queue)");
#endif
                marker_events.push_back(ev);
                detail::check_cl(clFlush(local_devices_[d].kernel_queue),
                                 "clFlush(kernel queue marker)");
            }

            if (local_devices_[d].transfer_queue != nullptr) {
                cl_event ev = nullptr;
#if CL_TARGET_OPENCL_VERSION >= 120
                detail::check_cl(clEnqueueMarkerWithWaitList(local_devices_[d].transfer_queue, 0, nullptr, &ev), "clEnqueueMarkerWithWaitList(transfer queue)");
#else
                detail::check_cl(clEnqueueMarker(local_devices_[d].transfer_queue, &ev), "clEnqueueMarker(transfer queue)");
#endif
                marker_events.push_back(ev);
                detail::check_cl(clFlush(local_devices_[d].transfer_queue),
                                 "clFlush(transfer queue marker)");
            }
        }

        for (cl_event ev : marker_events) {
            if (ev != nullptr) {
                detail::check_cl(clWaitForEvents(1, &ev), "clWaitForEvents(queue marker)");
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
            if (dp.element_count == 0) continue;

            const std::size_t d = static_cast<std::size_t>(dp.local_index);
            cl_kernel kernel = rk.kernels_per_local_device[d];
            bind_kernel_args(d, kernel, ki.binding);

            const std::size_t gwo = dp.global_offset + ki.geometry.global_offset;
            const std::size_t gws = dp.element_count;

            const std::size_t* lws_ptr = ki.geometry.local_size.has_value() ? &ki.geometry.local_size.value() : nullptr;

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

            const std::size_t* lws_ptr = ki.geometry.local_size.has_value() ? &ki.geometry.local_size.value() : nullptr;

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

            const std::size_t* lws_ptr = ki.geometry.local_size.has_value() ? &ki.geometry.local_size.value() : nullptr;

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
    if (hs.fields.empty()) return;

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
        struct HaloEventOwner {
            cl_event& read_left;
            cl_event& read_right;
            cl_event& write_left;
            cl_event& write_right;
            ~HaloEventOwner() {
                for (cl_event event : {read_left, read_right, write_left, write_right}) {
                    if (event != nullptr) clReleaseEvent(event);
                }
            }
        } owned_events{
            read_left_to_right_ev, read_right_to_left_ev,
            write_left_ev, write_right_ev
        };

        if (left.owning_rank == rank_) {
            const std::size_t dl = static_cast<std::size_t>(left.local_index);
            const std::vector<cl_event> deps = field_write_dependency(fh, dl);
            if (!deps.empty()) detail::check_cl(clFlush(local_devices_[dl].kernel_queue), "clFlush(kernel queue)");
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
            if (!deps.empty()) detail::check_cl(clFlush(local_devices_[dr].kernel_queue), "clFlush(kernel queue)");
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
                detail::check_cl(clFlush(local_devices_[dl].transfer_queue), "clFlush(transfer queue local)");
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
                detail::check_cl(clFlush(local_devices_[dr].transfer_queue), "clFlush(transfer queue local)");
            }

            if (read_right_to_left_ev != nullptr) {
                detail::check_cl(clFlush(local_devices_[dr].transfer_queue), "clFlush(transfer queue local)");
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
                detail::check_cl(clFlush(local_devices_[dl].transfer_queue), "clFlush(transfer queue local)");
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
            const std::size_t dl = static_cast<std::size_t>(left.local_index);
            detail::check_cl(clFlush(local_devices_[dl].transfer_queue), "clFlush(transfer queue)");
            detail::check_cl(
                clWaitForEvents(1, &read_left_to_right_ev),
                "clWaitForEvents(halo left->right send ready)"
            );
            clReleaseEvent(read_left_to_right_ev);
            read_left_to_right_ev = nullptr;
        }

        if (right.owning_rank == rank_ && read_right_to_left_ev != nullptr) {
            const std::size_t dr = static_cast<std::size_t>(right.local_index);
            detail::check_cl(clFlush(local_devices_[dr].transfer_queue), "clFlush(transfer queue)");
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
            detail::check_cl(clFlush(local_devices_[dr].transfer_queue), "clFlush(transfer queue)");
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
            detail::check_cl(clFlush(local_devices_[dl].transfer_queue), "clFlush(transfer queue)");
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
    if (local_devices_.empty()) {
        return;
    }
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
                detail::EventGuard read_guard{read_ev};
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
                        clFlush(local_devices_[dsrc].transfer_queue),
                        "clFlush(redistribute read)"
                    );
                    detail::check_cl(
                        clWaitForEvents(1, &read_ev),
                        "clWaitForEvents(redistribute read)"
                    );
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
            if (static_cast<std::size_t>(li) < last_elapsed_local_.size()) {
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

    bool same = true;
    for (std::size_t i = 0; i < old_parts.size(); ++i) {
        if (old_parts[i].global_offset != new_parts[i].global_offset ||
            old_parts[i].element_count != new_parts[i].element_count ||
            old_parts[i].owning_rank   != new_parts[i].owning_rank ||
            old_parts[i].local_index   != new_parts[i].local_index) {
            same = false;
            break;
        }
    }
    if (same) {
        current_loads_ = loads_from_partitions(partitions_);
        return;
    }

    this->synchronize(true);
    this->redistribute_selected_registered_fields(
        std::vector<FieldHandle>{target_field},
        old_parts,
        new_parts
    );
    partitions_ = new_parts;
    current_loads_ = loads_from_partitions(partitions_);
    this->synchronize(true);
}

static double max_value(const std::vector<double>& values) {
    double out = 0.0;
    if (!values.empty()) {
        out = *std::max_element(values.begin(), values.end());
    }
    return out;
}

double mpi_max_double(double local_value, const char* what) const {
    double global_value = 0.0;
    detail::check_mpi(
        MPI_Allreduce(
            &local_value,
            &global_value,
            1,
            MPI_DOUBLE,
            MPI_MAX,
            comm_
        ),
        what
    );
    return global_value;
}

inline void reset_interval_comm_stats() {
    interval_comm_seconds_local_ = 0.0;
}

std::vector<double> collect_global_balance_times_from_window() {
    if (!simulated_times_.empty()) {
        return simulated_times_;
    }
    const std::vector<double> measured_elapsed =
        compute_balance_window_elapsed_local();

    std::vector<double> local_times(partitions_.size(), 0.0);

    for (std::size_t i = 0; i < partitions_.size(); ++i) {
        if (partitions_[i].owning_rank == rank_ && partitions_[i].local_index >= 0) {
            const int li = partitions_[i].local_index;

            if (static_cast<std::size_t>(li) < measured_elapsed.size()) {
                local_times[i] = measured_elapsed[static_cast<std::size_t>(li)];
            }
        }
    }

    std::vector<double> global_times(partitions_.size(), 0.0);

    if (!global_times.empty()) {
        detail::check_mpi(
            MPI_Allreduce(
                local_times.data(),
                global_times.data(),
                static_cast<int>(global_times.size()),
                MPI_DOUBLE,
                MPI_SUM,
                comm_
            ),
            "MPI_Allreduce(balance interval global times)"
        );
    }

    return global_times;
}

inline void print_named_loads(
    const char* title,
    const std::vector<float>& loads
) const {
    if (rank_ != 0) return;

    std::cout << "  " << title << ":\n";

    if (loads.empty()) {
        std::cout << "    <empty>\n";
        return;
    }

    float prev = 0.0f;
    float total = 0.0f;

    for (std::size_t i = 0; i < loads.size(); ++i) {
        const float cum = loads[i];
        const float share = cum - prev;

        prev = cum;
        total += share;

        std::cout
            << "    part[" << i << "]"
            << " cum=" << (100.0f * cum) << "%"
            << " share=" << (100.0f * share) << "%";

        if (i < partitions_.size()) {
            const DevicePartition& p = partitions_[i];

            std::cout
                << " device_global=" << p.device_global_index
                << " rank=" << p.owning_rank
                << " local_index=" << p.local_index
                << " offset=" << p.global_offset
                << " count=" << p.element_count;
        }

        std::cout << "\n";
    }

    std::cout << "    total=" << (100.0f * total) << "%\n";
}


int first_rebalance_interval_from_env(int fallback) const {
    const char* raw = std::getenv("DCL_FIRST_REBALANCE_INTERVAL");
    if (raw == nullptr || raw[0] == '\0') return fallback;

    const int value = std::atoi(raw);
    return (value > 0) ? value : fallback;
}

std::string metrics_file_from_env() const {
    const char* raw = std::getenv("DCL_METRICS_FILE");
    if (raw == nullptr) return std::string();
    return std::string(raw);
}

std::string metrics_run_id_from_env() const {
    const char* raw = std::getenv("DCL_METRICS_RUN_ID");
    if (raw == nullptr || raw[0] == '\0') return std::string("0");
    return std::string(raw);
}

std::string csv_sanitize(const std::string& in) const {
    std::string out;
    out.reserve(in.size());

    for (char c : in) {
        if (c == ',' || c == '\n' || c == '\r' || c == '\t') {
            out.push_back(';');
        } else {
            out.push_back(c);
        }
    }

    return out;
}

std::string format_float_vector_for_csv(const std::vector<float>& values) const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(9);
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) oss << "|";
        oss << values[i];
    }
    return oss.str();
}

std::string format_double_vector_for_csv(const std::vector<double>& values) const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(9);
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) oss << "|";
        oss << values[i];
    }
    return oss.str();
}

bool file_is_empty_or_missing(const std::string& path) const {
    std::ifstream in(path.c_str(), std::ios::binary);
    if (!in.good()) return true;
    return in.peek() == std::ifstream::traits_type::eof();
}

void append_balance_metrics_csv(
    const char* policy_name,
    const char* action,
    const char* reason,
    const std::vector<double>& global_times,
    const std::vector<float>& old_loads,
    const std::vector<float>& proposed_loads,
    const std::vector<float>& current_loads_after,
    double T_compute_interval,
    double T_comm_interval,
    double T_balance_total,
    double T_rebalance_apply
) {
    if (rank_ != 0) return;

    const std::string path = metrics_file_from_env();
    if (path.empty()) return;

    const bool need_header = file_is_empty_or_missing(path);

    std::ofstream out(path.c_str(), std::ios::app);
    if (!out.is_open()) {
        std::cerr << "[DCL Warning] Could not open metrics file: " << path << "\n";
        return;
    }

    if (need_header) {
        out << "run_id,iteration,policy,action,reason,"
            << "T_compute_interval_s,T_comm_interval_s,T_balance_total_s,T_rebalance_apply_s,"
            << "T_compute_accum_s,T_comm_accum_s,T_balance_accum_s,T_rebalance_apply_accum_s,"
            << "old_loads,proposed_loads,current_loads,device_times\n";
    }

    out << metrics_run_id_from_env() << ","
        << iteration_counter_ << ","
        << csv_sanitize(policy_name ? std::string(policy_name) : std::string()) << ","
        << csv_sanitize(action ? std::string(action) : std::string()) << ","
        << csv_sanitize(reason ? std::string(reason) : std::string()) << ","
        << std::fixed << std::setprecision(9)
        << T_compute_interval << ","
        << T_comm_interval << ","
        << T_balance_total << ","
        << T_rebalance_apply << ","
        << accumulated_compute_seconds_ << ","
        << accumulated_comm_seconds_ << ","
        << accumulated_balance_seconds_ << ","
        << accumulated_rebalance_apply_seconds_ << ","
        << format_float_vector_for_csv(old_loads) << ","
        << format_float_vector_for_csv(proposed_loads) << ","
        << format_float_vector_for_csv(current_loads_after) << ","
        << format_double_vector_for_csv(global_times) << "\n";
}

void print_balance_interval_metrics(
    const char* policy_name,
    const char* action,
    const char* reason,
    const std::vector<double>& global_times,
    const std::vector<float>& old_loads,
    const std::vector<float>& proposed_loads,
    const std::vector<float>& current_loads_after,
    double local_balance_seconds,
    bool reset_after_print,
    double local_rebalance_apply_seconds = 0.0
) {
    const double T_compute_interval = max_value(global_times);

    const double T_comm_interval =
        mpi_max_double(
            interval_comm_seconds_local_,
            "MPI_Allreduce(balance interval communication time)"
        );

    const double T_balance_total =
        mpi_max_double(
            local_balance_seconds,
            "MPI_Allreduce(balance total time)"
        );

    const double T_rebalance_apply =
        mpi_max_double(
            local_rebalance_apply_seconds,
            "MPI_Allreduce(rebalance apply time)"
        );

    accumulated_compute_seconds_ += T_compute_interval;
    accumulated_comm_seconds_ += T_comm_interval;
    accumulated_balance_seconds_ += T_balance_total;
    accumulated_rebalance_apply_seconds_ += T_rebalance_apply;

    if (verbose_metrics_) append_balance_metrics_csv(
        policy_name,
        action,
        reason,
        global_times,
        old_loads,
        proposed_loads,
        current_loads_after,
        T_compute_interval,
        T_comm_interval,
        T_balance_total,
        T_rebalance_apply
    );

    if (verbose_metrics_ && rank_ == 0) {
        std::cout << "[DCL][balance-metrics] iteration=" << iteration_counter_ << "\n";
        std::cout << "  policy=" << policy_name << "\n";
        std::cout << "  action=" << action << "\n";
        std::cout << "  reason=" << reason << "\n";
        std::cout << "  T_compute_interval_s=" << T_compute_interval << "\n";
        std::cout << "  T_comm_interval_s=" << T_comm_interval << "\n";
        std::cout << "  T_balance_total_s=" << T_balance_total << "\n";
        std::cout << "  T_rebalance_apply_s=" << T_rebalance_apply << "\n";
        std::cout << "  T_compute_accum_s=" << accumulated_compute_seconds_ << "\n";
        std::cout << "  T_comm_accum_s=" << accumulated_comm_seconds_ << "\n";
        std::cout << "  T_balance_accum_s=" << accumulated_balance_seconds_ << "\n";
        std::cout << "  T_rebalance_apply_accum_s=" << accumulated_rebalance_apply_seconds_ << "\n";

        std::cout << "  device_times_s:\n";
        for (std::size_t i = 0; i < global_times.size(); ++i) {
            std::cout << "    part[" << i << "]=" << global_times[i] << "\n";
        }
    }

    if (verbose_metrics_) {
        print_named_loads("OLD loads", old_loads);
        if (!proposed_loads.empty()) {
            print_named_loads("PROPOSED loads", proposed_loads);
        }
        print_named_loads("CURRENT loads", current_loads_after);
        if (rank_ == 0) std::cout << std::flush;
    }

    if (reset_after_print) {
        reset_interval_comm_stats();
        reset_balance_window_events();
    }
}

inline bool maybe_rebalance_from_timings(
    const std::vector<FieldHandle>& rebalance_fields,
    const AutoBalancePolicy& policy
) {
    const float threshold = policy.threshold;
    const double policy_numa_threshold = policy.numa_cost_gain_ratio_threshold;
    const double balance_t0 = MPI_Wtime();

    const std::vector<float> old_loads = current_loads_;
    const std::vector<FieldHandle> fields_to_move =
        unique_existing_proportional_fields(rebalance_fields);
    unsigned long long field_signature = 1469598103934665603ULL;
    for (FieldHandle field : fields_to_move) {
        field_signature ^= static_cast<unsigned long long>(field.value);
        field_signature *= 1099511628211ULL;
    }

    // Every rank must enter the same collective path, including early exits.
    // A mixed timing source or field list would otherwise make one rank enter
    // a reduction while another rank skips directly to metrics or migration.
    const unsigned long long local_state[6] = {
        partition_.has_value() ? 1ULL : 0ULL,
        static_cast<unsigned long long>(partitions_.size()),
        static_cast<unsigned long long>(simulated_times_.size()),
        static_cast<unsigned long long>(fields_to_move.size()),
        static_cast<unsigned long long>(current_loads_.size()),
        field_signature
    };
    unsigned long long minimum_state[6] = {};
    unsigned long long maximum_state[6] = {};
    detail::check_mpi(
        MPI_Allreduce(local_state, minimum_state, 6, MPI_UNSIGNED_LONG_LONG, MPI_MIN, comm_),
        "MPI_Allreduce(balance input minima)"
    );
    detail::check_mpi(
        MPI_Allreduce(local_state, maximum_state, 6, MPI_UNSIGNED_LONG_LONG, MPI_MAX, comm_),
        "MPI_Allreduce(balance input maxima)"
    );
    for (int i = 0; i < 6; ++i) {
        if (minimum_state[i] != maximum_state[i]) {
            throw Error("maybe_rebalance_from_timings(): inconsistent inputs across ranks");
        }
    }

    if (!partition_.has_value()) {
        const std::vector<double> global_times =
            collect_global_balance_times_from_window();

        print_balance_interval_metrics(
            "threshold",
            "skip",
            "no partition",
            global_times,
            old_loads,
            std::vector<float>(),
            current_loads_,
            MPI_Wtime() - balance_t0,
            true
        );

        return false;
    }

    if (partitions_.empty()) {
        const std::vector<double> global_times =
            collect_global_balance_times_from_window();

        print_balance_interval_metrics(
            "threshold",
            "skip",
            "empty partition list",
            global_times,
            old_loads,
            std::vector<float>(),
            current_loads_,
            MPI_Wtime() - balance_t0,
            true
        );

        return false;
    }

    if (fields_to_move.empty()) {
        const std::vector<double> global_times =
            collect_global_balance_times_from_window();

        print_balance_interval_metrics(
            "threshold",
            "skip",
            "empty proportional rebalance field list",
            global_times,
            old_loads,
            std::vector<float>(),
            current_loads_,
            MPI_Wtime() - balance_t0,
            true
        );

        return false;
    }

    const std::vector<double> global_times =
        collect_global_balance_times_from_window();

    std::vector<float> proposed_loads;
    if (policy.use_contention_adjustment && topo_metrics_.has_value() &&
        !topo_metrics_->memory_contention_factor.empty()) {
        std::vector<double> capacity(global_times.size(), 0.0);
        double total_capacity = 0.0;

        for (std::size_t i = 0; i < global_times.size(); ++i) {
            if (partitions_[i].element_count == 0 || global_times[i] <= 1.0e-12) {
                continue;
            }

            const double raw_throughput =
                static_cast<double>(partitions_[i].element_count) / global_times[i];
            const int dev_idx = (partitions_[i].device_global_index >= 0)
                ? partitions_[i].device_global_index
                : static_cast<int>(i);

            capacity[i] = adjusted_capacity(*topo_metrics_, dev_idx, raw_throughput);
            total_capacity += capacity[i];
        }

        proposed_loads.resize(global_times.size(), 0.0f);
        if (total_capacity <= 0.0) {
            const float step = 1.0f / static_cast<float>(global_times.size());
            float acc = 0.0f;
            for (std::size_t i = 0; i < proposed_loads.size(); ++i) {
                acc += step;
                proposed_loads[i] = acc;
            }
            proposed_loads.back() = 1.0f;
        } else {
            double cumulative = 0.0;
            for (std::size_t i = 0; i < capacity.size(); ++i) {
                cumulative += capacity[i] / total_capacity;
                proposed_loads[i] = static_cast<float>(cumulative);
            }
            proposed_loads.back() = 1.0f;
        }
    } else {
        proposed_loads = detail::compute_loads_from_partition_throughput(
            global_times,
            partitions_
        );
    }

    int power_cap_error = 0;
    if (rank_ == 0 && !proposed_loads.empty() && policy.use_power_cap) {
        try {
            if (!topo_metrics_.has_value()) {
                throw Error("Power cap requires topology power measurements");
            }
            proposed_loads = apply_power_cap(
                proposed_loads, *topo_metrics_, policy.power_budget_watts,
                nullptr, &current_loads_
            );
        } catch (...) {
            power_cap_error = 1;
        }
    }
    detail::check_mpi(
        MPI_Bcast(&power_cap_error, 1, MPI_INT, 0, comm_),
        "MPI_Bcast(power cap status)"
    );
    if (power_cap_error != 0) {
        throw Error("Power cap infeasible or measurements unavailable");
    }

    // The proposal must be identical before any rank can take a local early
    // return or enter the NUMA consensus collective. Topology estimates may
    // differ across ranks, so rank zero chooses the global proposal.
    int proposed_count = (rank_ == 0) ? static_cast<int>(proposed_loads.size()) : 0;
    detail::check_mpi(
        MPI_Bcast(&proposed_count, 1, MPI_INT, 0, comm_),
        "MPI_Bcast(balance proposal size)"
    );
    if (rank_ != 0) {
        proposed_loads.resize(static_cast<std::size_t>(proposed_count));
    }
    if (proposed_count > 0) {
        detail::check_mpi(
            MPI_Bcast(proposed_loads.data(), proposed_count, MPI_FLOAT, 0, comm_),
            "MPI_Bcast(balance proposal)"
        );
    }

    if (proposed_loads.empty()) {
        print_balance_interval_metrics(
            "threshold",
            "skip",
            "empty proposed loads",
            global_times,
            old_loads,
            std::vector<float>(),
            current_loads_,
            MPI_Wtime() - balance_t0,
            true
        );

        return false;
    }

    const std::vector<DevicePartition> old_parts = partitions_;
    const std::vector<DevicePartition> new_parts =
        partitions_from_loads(proposed_loads);

    if (new_parts.size() != old_parts.size()) {
        print_balance_interval_metrics(
            "threshold",
            "skip",
            "rounded partition count mismatch",
            global_times,
            old_loads,
            proposed_loads,
            current_loads_,
            MPI_Wtime() - balance_t0,
            true
        );

        return false;
    }

    const std::vector<float> effective_new_loads =
        loads_from_partitions(new_parts);

    float diff = std::numeric_limits<float>::infinity();

    if (!current_loads_.empty() &&
        current_loads_.size() == effective_new_loads.size()) {
        diff = detail::l2_norm_diff(current_loads_, effective_new_loads);
    }

    bool same = true;
    for (std::size_t i = 0; i < old_parts.size(); ++i) {
        if (old_parts[i].global_offset != new_parts[i].global_offset ||
            old_parts[i].element_count != new_parts[i].element_count ||
            old_parts[i].owning_rank   != new_parts[i].owning_rank ||
            old_parts[i].local_index   != new_parts[i].local_index) {
            same = false;
            break;
        }
    }

    if (same) {
        current_loads_ = loads_from_partitions(partitions_);

        print_balance_interval_metrics(
            "threshold",
            "skip",
            "new partition identical",
            global_times,
            old_loads,
            effective_new_loads,
            current_loads_,
            MPI_Wtime() - balance_t0,
            true
        );

        return false;
    }

    if (diff < threshold) {
        current_loads_ = loads_from_partitions(partitions_);

        std::ostringstream oss;
        oss
            << "load difference below fixed threshold; "
            << "diff=" << diff
            << ", threshold=" << threshold;

        const std::string reason = oss.str();

        print_balance_interval_metrics(
            "threshold",
            "skip",
            reason.c_str(),
            global_times,
            old_loads,
            effective_new_loads,
            current_loads_,
            MPI_Wtime() - balance_t0,
            true
        );

        return false;
    }

    int local_skip = 0;

    // NUMA-Aware Cost Function in Load Balancer (R2.4)
    // When power capping is active, thermal protection takes priority over throughput gain
    if (topo_metrics_.has_value() && !policy.use_power_cap) {
        const double numa_ratio_limit = (policy_numa_threshold >= 0.0)
            ? policy_numa_threshold
            : numa_cost_gain_ratio_threshold_;

        double total_migration_cost_s = 0.0;

        for (FieldHandle fh : fields_to_move) {
            auto fit = fields_.find(fh.value);
            if (fit == fields_.end()) continue;

            const std::size_t elem_bytes =
                fit->second.spec.units_per_element *
                fit->second.spec.bytes_per_unit;

            std::size_t src_idx = 0;
            std::size_t dst_idx = 0;

            while (src_idx < old_parts.size() && dst_idx < new_parts.size()) {
                const auto& src = old_parts[src_idx];
                const auto& dst = new_parts[dst_idx];

                const std::size_t src_end = src.global_offset + src.element_count;
                const std::size_t dst_end = dst.global_offset + dst.element_count;

                const std::size_t start = std::max(src.global_offset, dst.global_offset);
                const std::size_t end = std::min(src_end, dst_end);

                if (start < end) {
                    const int src_dev = (src.device_global_index >= 0)
                        ? src.device_global_index
                        : static_cast<int>(src_idx);
                    const int dst_dev = (dst.device_global_index >= 0)
                        ? dst.device_global_index
                        : static_cast<int>(dst_idx);

                    if (src_dev != dst_dev) {
                        const std::size_t bytes = (end - start) * elem_bytes;
                        total_migration_cost_s += estimate_migration_cost_bytes(
                            topo_metrics_.value(),
                            src_dev,
                            dst_dev,
                            src.owning_rank,
                            dst.owning_rank,
                            bytes
                        );
                    }
                }

                if (src_end < dst_end) {
                    ++src_idx;
                } else if (dst_end < src_end) {
                    ++dst_idx;
                } else {
                    ++src_idx;
                    ++dst_idx;
                }
            }
        }

        double t_max_current = 0.0;
        double total_cap = 0.0;
        std::vector<double> capacity(partitions_.size(), 0.0);

        for (std::size_t i = 0; i < partitions_.size(); ++i) {
            if (global_times[i] > t_max_current) {
                t_max_current = global_times[i];
            }
            if (global_times[i] > 1.0e-12 && partitions_[i].element_count > 0) {
                double cap = static_cast<double>(partitions_[i].element_count) / global_times[i];
                if (policy.use_contention_adjustment && topo_metrics_.has_value() &&
                    !topo_metrics_->memory_contention_factor.empty()) {
                    const int dev_idx = (partitions_[i].device_global_index >= 0)
                        ? partitions_[i].device_global_index
                        : static_cast<int>(i);
                    cap = adjusted_capacity(*topo_metrics_, dev_idx, cap);
                }
                capacity[i] = cap;
                total_cap += capacity[i];
            }
        }

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

        const double expected_gain_s = std::max(0.0, t_max_current - t_max_projected);

        local_skip = (total_migration_cost_s > numa_ratio_limit * expected_gain_s) ? 1 : 0;
    }

    int global_skip = 0;
    
    detail::check_mpi(
        MPI_Allreduce(&local_skip, &global_skip, 1, MPI_INT, MPI_LOR, comm_),
        "MPI_Allreduce(NUMA decision to skip rebalance)"
    );

    if (global_skip != 0) {
        if (rank_ == 0) {
            std::cout << "[NUMA] migration cost exceeds gain threshold, skipping rebalance" << std::endl;
        }

        current_loads_ = loads_from_partitions(partitions_);

        print_balance_interval_metrics(
            "threshold",
            "skip",
            "[NUMA] migration cost exceeds gain threshold",
            global_times,
            old_loads,
            effective_new_loads,
            current_loads_,
            MPI_Wtime() - balance_t0,
            true
        );

        return false;
    }

    const double apply_t0 = MPI_Wtime();

    this->synchronize(true);

    this->redistribute_selected_registered_fields(
        fields_to_move,
        old_parts,
        new_parts
    );

    partitions_ = new_parts;
    current_loads_ = loads_from_partitions(partitions_);

    this->synchronize(true);

    const double apply_elapsed = MPI_Wtime() - apply_t0;

    std::ostringstream oss;
    oss
        << "accepted by fixed threshold; "
        << "diff=" << diff
        << ", threshold=" << threshold;

    const std::string reason = oss.str();

    print_balance_interval_metrics(
        "threshold",
        "rebalance",
        reason.c_str(),
        global_times,
        old_loads,
        effective_new_loads,
        current_loads_,
        MPI_Wtime() - balance_t0,
        true,
        apply_elapsed
    );

    return true;
}

inline bool maybe_rebalance_from_timings(
    const std::vector<FieldHandle>& rebalance_fields,
    float threshold,
    double numa_cost_gain_ratio_threshold = -1.0,
    bool use_contention_adjustment = false,
    bool use_power_cap = false,
    double power_budget_watts = 0.0
) {
    AutoBalancePolicy policy;
    policy.threshold = threshold;
    policy.numa_cost_gain_ratio_threshold = numa_cost_gain_ratio_threshold;
    policy.use_contention_adjustment = use_contention_adjustment;
    policy.use_power_cap = use_power_cap;
    policy.power_budget_watts = power_budget_watts;
    return maybe_rebalance_from_timings(rebalance_fields, policy);
}

inline bool maybe_rebalance_hierarchical(
    const std::vector<FieldHandle>& rebalance_fields = {}
) {
    if (!partition_.has_value() || partitions_.empty()) {
        return false;
    }

    const double balance_t0 = MPI_Wtime();
    const std::vector<float> old_loads = current_loads_;
    const std::vector<DevicePartition> old_parts = partitions_;

    // Identify local devices owned by this rank
    std::vector<int> local_global_indices;
    for (const auto& dp : partitions_) {
        if (dp.owning_rank == rank_) {
            local_global_indices.push_back(dp.device_global_index);
        }
    }

    const std::size_t L_r = local_global_indices.size();

    // R3.1: Intra-node balancing phase (within each rank, no MPI communication)
    // Measure local device times
    std::vector<double> local_times(L_r, 0.0);
    const std::vector<double> window_elapsed = compute_balance_window_elapsed_local();

    for (std::size_t li = 0; li < L_r; ++li) {
        const int g = local_global_indices[li];
        if (!simulated_times_.empty()) {
            if (g >= 0 && static_cast<std::size_t>(g) < simulated_times_.size()) {
                local_times[li] = simulated_times_[static_cast<std::size_t>(g)];
            } else if (li < simulated_times_.size()) {
                local_times[li] = simulated_times_[li];
            }
        } else if (li < window_elapsed.size() && window_elapsed[li] > 1.0e-12) {
            local_times[li] = window_elapsed[li];
        } else if (li < device_timings_.size()) {
            if (device_timings_[li].kernel_seconds_last > 1.0e-12) {
                local_times[li] = device_timings_[li].kernel_seconds_last;
            } else if (device_timings_[li].kernel_seconds_avg > 1.0e-12) {
                local_times[li] = device_timings_[li].kernel_seconds_avg;
            }
        }
        if (local_times[li] <= 1.0e-12) {
            local_times[li] = 1.0;
        }
        local_times[li] = std::max(local_times[li], 1.0e-9);
    }

    // Compute per-device throughput from work completed in the measured interval.
    std::vector<double> p_local(L_r, 0.0);
    double local_capacity = 0.0;
    if (L_r > 0) {
        for (std::size_t li = 0; li < L_r; ++li) {
            const int g = local_global_indices[li];
            const double elements =
                static_cast<double>(old_parts[static_cast<std::size_t>(g)].element_count);
            p_local[li] = elements / local_times[li];
            local_capacity += p_local[li];
        }
        if (local_capacity > 0.0) {
            for (std::size_t li = 0; li < L_r; ++li) {
                p_local[li] /= local_capacity;
            }
        } else {
            for (std::size_t li = 0; li < L_r; ++li) {
                p_local[li] = 1.0 / static_cast<double>(L_r);
            }
        }
    }

    // R3.1 & R3.4: Print on rank 0: [HIER] intra-node loads: [...]
    // Uses NO MPI communication
    if (rank_ == 0) {
        std::cout << "[HIER] intra-node loads: [";
        for (std::size_t li = 0; li < p_local.size(); ++li) {
            if (li > 0) std::cout << ", ";
            std::cout << p_local[li];
        }
        std::cout << "]" << std::endl;
    }

    // Finish local work before entering the rank-capacity collective.
    this->synchronize_all_local_devices(false);

    // R3.2: Inter-node Balancing Phase
    std::size_t rank_elements_count = 0;
    for (std::size_t li = 0; li < L_r; ++li) {
        int g = local_global_indices[li];
        rank_elements_count += old_parts[static_cast<std::size_t>(g)].element_count;
    }

    // Aggregate time per rank: max of local device times on that rank.
    // If this rank has no devices (L_r == 0), use 0.0 to signal zero capacity.
    double local_max_time = (L_r > 0) ? max_value(local_times) : 0.0;
    if (L_r > 0 && local_max_time <= 0.0) {
        // Devices exist but reported zero time — use a small positive sentinel so
        // they still participate with finite (but very high) capacity.
        local_max_time = 1.0e-9;
    }
    // Ranks with L_r == 0 keep local_max_time = 0.0 → zero capacity → zero elements.

    double local_data[3] = {
        static_cast<double>(rank_elements_count), local_max_time, local_capacity
    };
    std::vector<double> rank_data(static_cast<std::size_t>(size_) * 3, 0.0);
    detail::check_mpi(
        MPI_Gather(
            local_data,
            3,
            MPI_DOUBLE,
            rank_data.data(),
            3,
            MPI_DOUBLE,
            0,
            comm_
        ),
        "MPI_Gather(hierarchical rank capacities)"
    );

    std::vector<float> inter_cum_loads;
    std::vector<float> inter_rank_loads(static_cast<std::size_t>(size_), 0.0f);
    int abort_migration_flag = 0;
    if (rank_ == 0) {
        double total_capacity = 0.0;
        double global_max_time = 0.0;
        std::vector<double> capacities(static_cast<std::size_t>(size_), 0.0);

        for (int r = 0; r < size_; ++r) {
            double r_max_time = rank_data[static_cast<std::size_t>(r * 3 + 1)];
            double r_capacity = rank_data[static_cast<std::size_t>(r * 3 + 2)];
            total_capacity += r_capacity;
            global_max_time = std::max(global_max_time, r_max_time);
            capacities[static_cast<std::size_t>(r)] = r_capacity;
        }

        double projected_time = 0.0;
        if (total_capacity > 0.0) {
            projected_time = static_cast<double>(partition_->global_elements) / total_capacity;
        }

        bool abort_migration = false;
        if (global_max_time > 0.0 && projected_time > 0.0) {
            if ((global_max_time - projected_time) < 0.05 * global_max_time) {
                abort_migration = true;
                abort_migration_flag = 1;
                std::cout << "[HIER] Inter-node migration aborted due to hysteresis (gain < 5%). projected_time="
                          << projected_time << " global_max_time=" << global_max_time << std::endl;
            }
        }

        inter_cum_loads.assign(static_cast<std::size_t>(size_), 1.0f);
        if (abort_migration) {
            double elements_accum = 0.0;
            for (int r = 0; r < size_; ++r) {
                elements_accum += rank_data[static_cast<std::size_t>(r * 3)];
                inter_cum_loads[static_cast<std::size_t>(r)] = static_cast<float>(elements_accum / static_cast<double>(partition_->global_elements));
            }
            inter_cum_loads.back() = 1.0f;
        } else {
            if (total_capacity <= 0.0) {
                const float passo = 1.0f / static_cast<float>(size_);
                float acumulada = 0.0f;
                for (int r = 0; r < size_; ++r) {
                    acumulada += passo;
                    inter_cum_loads[static_cast<std::size_t>(r)] = acumulada;
                }
                inter_cum_loads.back() = 1.0f;
            } else {
                double carga_acumulada = 0.0;
                for (int r = 0; r < size_; ++r) {
                    const double fatia = capacities[static_cast<std::size_t>(r)] / total_capacity;
                    carga_acumulada += fatia;
                    inter_cum_loads[static_cast<std::size_t>(r)] = static_cast<float>(carga_acumulada);
                }
                inter_cum_loads.back() = 1.0f;
            }
        }

        float prev = 0.0f;
        for (int r = 0; r < size_; ++r) {
            const float cum = inter_cum_loads[static_cast<std::size_t>(r)];
            inter_rank_loads[static_cast<std::size_t>(r)] = cum - prev;
            prev = cum;
        }

        // R3.4: Print on rank 0: [HIER] inter-node loads: [...]
        std::cout << "[HIER] inter-node loads: [";
        for (int r = 0; r < size_; ++r) {
            if (r > 0) std::cout << ", ";
            std::cout << inter_rank_loads[static_cast<std::size_t>(r)];
        }
        std::cout << "]" << std::endl;
    } else {
        inter_cum_loads.assign(static_cast<std::size_t>(size_), 0.0f);
    }

    // Broadcast inter-node partition / loads to all ranks
    detail::check_mpi(
        MPI_Bcast(
            inter_cum_loads.data(),
            size_,
            MPI_FLOAT,
            0,
            comm_
        ),
        "MPI_Bcast(hierarchical inter_cum_loads)"
    );

    detail::check_mpi(
        MPI_Bcast(
            &abort_migration_flag,
            1,
            MPI_INT,
            0,
            comm_
        ),
        "MPI_Bcast(hierarchical abort_migration_flag)"
    );

    // Compute inter-node partition (rank element slices)
    const std::size_t N = partition_->global_elements;
    const std::size_t gran = std::max<std::size_t>(1, partition_->granularity);
    const std::size_t total_units = N / gran;

    std::vector<std::size_t> rank_cuts(static_cast<std::size_t>(size_) + 1, 0);
    rank_cuts[0] = 0;
    for (int r = 0; r + 1 < size_; ++r) {
        const double raw = static_cast<double>(inter_cum_loads[static_cast<std::size_t>(r)]) *
                           static_cast<double>(total_units);
        std::size_t cut = static_cast<std::size_t>(std::llround(raw)) * gran;
        if (cut < rank_cuts[static_cast<std::size_t>(r)]) {
            cut = rank_cuts[static_cast<std::size_t>(r)];
        }
        if (cut > N) {
            cut = N;
        }
        rank_cuts[static_cast<std::size_t>(r) + 1] = cut;
    }
    rank_cuts.back() = N;

    // Each rank maps its inter-node allocation to its local devices
    std::size_t rank_start = 0;
    std::size_t rank_end = 0;
    if (abort_migration_flag && L_r > 0) {
        rank_start = old_parts[local_global_indices.front()].global_offset;
        rank_end = old_parts[local_global_indices.back()].global_offset + old_parts[local_global_indices.back()].element_count;
    } else {
        rank_start = rank_cuts[static_cast<std::size_t>(rank_)];
        rank_end = rank_cuts[static_cast<std::size_t>(rank_) + 1];
    }
    const std::size_t rank_elements = rank_end - rank_start;
    const std::size_t U_r = rank_elements / gran;

    std::vector<std::size_t> local_dev_cuts(L_r + 1, rank_start);
    double cum_p = 0.0;
    for (std::size_t li = 0; li + 1 < L_r; ++li) {
        cum_p += p_local[li];
        const double raw = cum_p * static_cast<double>(U_r);
        std::size_t cut = rank_start + static_cast<std::size_t>(std::llround(raw)) * gran;
        if (cut < local_dev_cuts[li]) {
            cut = local_dev_cuts[li];
        }
        if (cut > rank_end) {
            cut = rank_end;
        }
        local_dev_cuts[li + 1] = cut;
    }
    local_dev_cuts.back() = rank_end;

    std::vector<unsigned long long> local_offsets(L_r, 0ULL);
    std::vector<unsigned long long> local_counts(L_r, 0ULL);
    for (std::size_t li = 0; li < L_r; ++li) {
        local_offsets[li] = static_cast<unsigned long long>(local_dev_cuts[li]);
        local_counts[li] = static_cast<unsigned long long>(local_dev_cuts[li + 1] - local_dev_cuts[li]);
    }

    std::vector<DevicePartition> new_parts = partitions_;

    if (!abort_migration_flag) {
        // Exchange partitions across all ranks using MPI_Allgatherv
        const std::size_t total_devices = partitions_.size();
        std::vector<int> recvcounts(static_cast<std::size_t>(size_), 0);
        std::vector<int> displs(static_cast<std::size_t>(size_), 0);

        if (all_device_counts_.empty() ||
            std::accumulate(all_device_counts_.begin(), all_device_counts_.end(), 0) == 0) {
            if (size_ > 0) {
                all_device_counts_.assign(static_cast<std::size_t>(size_), static_cast<int>(total_devices / size_));
                const int rem = static_cast<int>(total_devices % size_);
                for (int i = 0; i < rem; ++i) {
                    all_device_counts_[static_cast<std::size_t>(i)]++;
                }
            } else {
                all_device_counts_ = {static_cast<int>(total_devices)};
            }
        }

        for (int r = 0; r < size_; ++r) {
            recvcounts[static_cast<std::size_t>(r)] = all_device_counts_[static_cast<std::size_t>(r)];
        }
        for (int r = 1; r < size_; ++r) {
            displs[static_cast<std::size_t>(r)] =
                displs[static_cast<std::size_t>(r) - 1] + recvcounts[static_cast<std::size_t>(r) - 1];
        }

        std::vector<unsigned long long> all_offsets(total_devices, 0ULL);
        std::vector<unsigned long long> all_counts(total_devices, 0ULL);

        detail::check_mpi(
            MPI_Allgatherv(
                local_offsets.data(),
                static_cast<int>(L_r),
                MPI_UNSIGNED_LONG_LONG,
                all_offsets.data(),
                recvcounts.data(),
                displs.data(),
                MPI_UNSIGNED_LONG_LONG,
                comm_
            ),
            "MPI_Allgatherv(hierarchical all_offsets)"
        );

        detail::check_mpi(
            MPI_Allgatherv(
                local_counts.data(),
                static_cast<int>(L_r),
                MPI_UNSIGNED_LONG_LONG,
                all_counts.data(),
                recvcounts.data(),
                displs.data(),
                MPI_UNSIGNED_LONG_LONG,
                comm_
            ),
            "MPI_Allgatherv(hierarchical all_counts)"
        );

        new_parts.clear();
        new_parts.reserve(total_devices);
        for (std::size_t g = 0; g < total_devices; ++g) {
            DevicePartition dp;
            dp.device_global_index = static_cast<int>(g);
            dp.owning_rank = owner_rank_of_global_device(static_cast<int>(g));
            dp.local_index = (dp.owning_rank == rank_)
                ? local_index_of_global_device(static_cast<int>(g))
                : -1;
            dp.global_offset = static_cast<std::size_t>(all_offsets[g]);
            dp.element_count = static_cast<std::size_t>(all_counts[g]);
            new_parts.push_back(dp);
        }

        // C1 fix: Collective invariant check — sum(element_count) must equal global_elements.
        // Every rank performs this check so that a silent loss of elements is caught immediately
        // rather than producing incorrect results without any diagnostic.
        {
            const std::size_t expected = partition_->global_elements;
            std::size_t covered = 0;
            for (const auto& dp : new_parts) {
                covered += dp.element_count;
            }
            // Use MPI_Allreduce with MPI_MIN to verify that ALL ranks compute the same covered
            // count (they should, since new_parts is replicated). Any discrepancy also fires here.
            unsigned long long covered_ull = static_cast<unsigned long long>(covered);
            unsigned long long min_covered = 0;
            unsigned long long max_covered = 0;
            detail::check_mpi(
                MPI_Allreduce(&covered_ull, &min_covered, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, comm_),
                "MPI_Allreduce(C1 partition conservation min)"
            );
            detail::check_mpi(
                MPI_Allreduce(&covered_ull, &max_covered, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, comm_),
                "MPI_Allreduce(C1 partition conservation max)"
            );
            if (min_covered != static_cast<unsigned long long>(expected) ||
                max_covered != static_cast<unsigned long long>(expected)) {
                // Abort with a descriptive message rather than silently accepting a broken partition.
                if (rank_ == 0) {
                    std::cerr << "[HIER][C1 ERROR] Partition invariant violated: "
                              << "covered=" << min_covered << ".." << max_covered
                              << " expected=" << expected
                              << ". Partition not applied.\n";
                }
                // Return false: caller may retry or fall back. Do NOT update partitions_.
                return false;
            }
        }
    } else {
        // Every rank owns disjoint entries. One reduction publishes the local
        // cuts to all ranks without an Allgatherv or a global barrier.
        const std::size_t total_devices = new_parts.size();
        std::vector<unsigned long long> local_metadata(total_devices * 2, 0ULL);
        std::vector<unsigned long long> global_metadata(total_devices * 2, 0ULL);
        for (std::size_t li = 0; li < L_r; ++li) {
            const std::size_t g = static_cast<std::size_t>(local_global_indices[li]);
            local_metadata[g * 2] = local_offsets[li];
            local_metadata[g * 2 + 1] = local_counts[li];
        }
        detail::check_mpi(
            MPI_Allreduce(
                local_metadata.data(), global_metadata.data(),
                static_cast<int>(global_metadata.size()), MPI_UNSIGNED_LONG_LONG,
                MPI_SUM, comm_
            ),
            "MPI_Allreduce(hierarchical local partition metadata)"
        );
        for (std::size_t g = 0; g < total_devices; ++g) {
            new_parts[g].global_offset = static_cast<std::size_t>(global_metadata[g * 2]);
            new_parts[g].element_count = static_cast<std::size_t>(global_metadata[g * 2 + 1]);
        }
    }

    int local_invalid_partition = 0;
    std::size_t covered_elements = 0;
    for (const auto& part : new_parts) {
        if (part.global_offset != covered_elements ||
            part.element_count > partition_->global_elements - covered_elements) {
            local_invalid_partition = 1;
            break;
        }
        covered_elements += part.element_count;
    }
    if (covered_elements != partition_->global_elements) local_invalid_partition = 1;
    int any_invalid_partition = 0;
    detail::check_mpi(
        MPI_Allreduce(&local_invalid_partition, &any_invalid_partition, 1,
                      MPI_INT, MPI_MAX, comm_),
        "MPI_Allreduce(hierarchical partition coverage)"
    );
    if (any_invalid_partition != 0) {
        throw Error("Hierarchical partition has a gap, overlap, or missing elements");
    }

    // Apply partition update
    const std::vector<FieldHandle> fields_to_move =
        unique_existing_proportional_fields(rebalance_fields);

    if (abort_migration_flag) {
        this->synchronize_all_local_devices(true);
    } else {
        this->synchronize(true);
    }

    if (!fields_to_move.empty()) {
        this->redistribute_selected_registered_fields(
            fields_to_move,
            old_parts,
            new_parts,
            !abort_migration_flag
        );
    }

    partitions_ = new_parts;
    current_loads_ = loads_from_partitions(partitions_);

    if (abort_migration_flag) {
        this->synchronize_all_local_devices(true);
    } else {
        this->synchronize(true);
    }

    // R3.4: Print on rank 0: [HIER] final partition: [...]
    if (rank_ == 0) {
        std::cout << "[HIER] final partition: [";
        for (std::size_t i = 0; i < partitions_.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << "dev " << partitions_[i].device_global_index
                      << ": offset=" << partitions_[i].global_offset
                      << " count=" << partitions_[i].element_count;
        }
        std::cout << "]" << std::endl;
    }

    reset_balance_window_events();
    (void)balance_t0;
    (void)old_loads;
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



inline bool maybe_rebalance_profiled(
    const std::vector<FieldHandle>& rebalance_fields,
    const AutoBalancePolicy& policy,
    int interval
) {
    const double balance_t0 = MPI_Wtime();

    const std::vector<float> old_loads = current_loads_;

    if (!partition_.has_value()) {
        const std::vector<double> global_times =
            collect_global_balance_times_from_window();

        print_balance_interval_metrics(
            "profiled",
            "skip",
            "no partition",
            global_times,
            old_loads,
            std::vector<float>(),
            current_loads_,
            MPI_Wtime() - balance_t0,
            true
        );

        return false;
    }

    if (partitions_.empty()) {
        const std::vector<double> global_times =
            collect_global_balance_times_from_window();

        print_balance_interval_metrics(
            "profiled",
            "skip",
            "empty partition list",
            global_times,
            old_loads,
            std::vector<float>(),
            current_loads_,
            MPI_Wtime() - balance_t0,
            true
        );

        return false;
    }

    if (policy.total_iterations <= 0) {
        const std::vector<double> global_times =
            collect_global_balance_times_from_window();

        print_balance_interval_metrics(
            "profiled",
            "skip",
            "total_iterations not set",
            global_times,
            old_loads,
            std::vector<float>(),
            current_loads_,
            MPI_Wtime() - balance_t0,
            true
        );

        return false;
    }

    const std::vector<FieldHandle> fields_to_move =
        unique_existing_proportional_fields(rebalance_fields);

    if (fields_to_move.empty()) {
        const std::vector<double> global_times =
            collect_global_balance_times_from_window();

        print_balance_interval_metrics(
            "profiled",
            "skip",
            "empty proportional rebalance field list",
            global_times,
            old_loads,
            std::vector<float>(),
            current_loads_,
            MPI_Wtime() - balance_t0,
            true
        );

        return false;
    }

    if (!profiling_loaded_ ||
        profiling_file_loaded_ != policy.profiling_file) {
        load_profiling_data(policy.profiling_file);
    }

    const std::vector<double> global_times =
        collect_global_balance_times_from_window();

    const std::vector<float> proposed_loads =
        detail::compute_loads_from_partition_throughput(
            global_times,
            partitions_
        );

    if (proposed_loads.empty()) {
        print_balance_interval_metrics(
            "profiled",
            "skip",
            "empty proposed loads",
            global_times,
            old_loads,
            std::vector<float>(),
            current_loads_,
            MPI_Wtime() - balance_t0,
            true
        );

        return false;
    }

    const std::vector<DevicePartition> old_parts = partitions_;
    const std::vector<DevicePartition> new_parts =
        partitions_from_loads(proposed_loads);

    if (new_parts.size() != old_parts.size()) {
        print_balance_interval_metrics(
            "profiled",
            "skip",
            "rounded partition count mismatch",
            global_times,
            old_loads,
            proposed_loads,
            current_loads_,
            MPI_Wtime() - balance_t0,
            true
        );

        return false;
    }

    const std::vector<float> effective_new_loads =
        loads_from_partitions(new_parts);

    bool same = true;
    for (std::size_t i = 0; i < old_parts.size(); ++i) {
        if (old_parts[i].global_offset != new_parts[i].global_offset ||
            old_parts[i].element_count != new_parts[i].element_count ||
            old_parts[i].owning_rank   != new_parts[i].owning_rank ||
            old_parts[i].local_index   != new_parts[i].local_index) {
            same = false;
            break;
        }
    }

    if (same) {
        current_loads_ = loads_from_partitions(partitions_);

        print_balance_interval_metrics(
            "profiled",
            "skip",
            "new partition identical",
            global_times,
            old_loads,
            effective_new_loads,
            current_loads_,
            MPI_Wtime() - balance_t0,
            true
        );

        return false;
    }

    double T_comp_int = 0.0;
    double C_total = 0.0;

    std::vector<double> capacity(partitions_.size(), 0.0);

    for (std::size_t i = 0; i < partitions_.size(); ++i) {
        if (global_times[i] > T_comp_int) {
            T_comp_int = global_times[i];
        }

        if (global_times[i] > 1.0e-12 &&
            partitions_[i].element_count > 0) {
            capacity[i] =
                static_cast<double>(partitions_[i].element_count) /
                global_times[i];

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
                static_cast<double>(new_parts[i].element_count) /
                capacity[i];

            if (projected > T_comp_bal) {
                T_comp_bal = projected;
            }
        }
    }

    if (interval > 0) {
        T_comp_int /= static_cast<double>(interval);
        T_comp_bal /= static_cast<double>(interval);
    }

    std::vector<std::size_t> rank_recv_bytes(
        static_cast<std::size_t>(size_),
        0ull
    );

    auto count_migration_bytes = [&](FieldHandle fh) {
        auto fit = fields_.find(fh.value);
        if (fit == fields_.end()) return;

        const std::size_t elem_bytes =
            fit->second.spec.units_per_element *
            fit->second.spec.bytes_per_unit;

        std::size_t src_idx = 0;
        std::size_t dst_idx = 0;

        while (src_idx < old_parts.size() &&
               dst_idx < new_parts.size()) {
            const auto& src = old_parts[src_idx];
            const auto& dst = new_parts[dst_idx];

            const std::size_t src_end =
                src.global_offset + src.element_count;

            const std::size_t dst_end =
                dst.global_offset + dst.element_count;

            const std::size_t start =
                std::max(src.global_offset, dst.global_offset);

            const std::size_t end =
                std::min(src_end, dst_end);

            if (start < end) {
                const std::size_t len = end - start;

                if (src.owning_rank != dst.owning_rank) {
                    rank_recv_bytes[
                        static_cast<std::size_t>(dst.owning_rank)
                    ] += len * elem_bytes;
                }
            }

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

    for (FieldHandle fh : fields_to_move) {
        count_migration_bytes(fh);
    }

    std::size_t max_v_migrado = 0;

    if (!rank_recv_bytes.empty()) {
        max_v_migrado = *std::max_element(rank_recv_bytes.begin(), rank_recv_bytes.end());
    }

    const double custo_migracao =
        get_migration_overhead(max_v_migrado);

    const std::size_t remaining_iterations =
        (iteration_counter_ <
         static_cast<std::size_t>(policy.total_iterations))
            ? (static_cast<std::size_t>(policy.total_iterations) -
               iteration_counter_)
            : 0ull;

    const double ganho_por_iter =
        std::max(0.0, T_comp_int - T_comp_bal);

    const double ganho_total_estimado =
        ganho_por_iter *
        static_cast<double>(remaining_iterations);

    const int local_should_rebalance =
        (remaining_iterations != 0 &&
         custo_migracao < ganho_total_estimado)
            ? 1
            : 0;

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
        throw Error(
            "maybe_rebalance_profiled(): inconsistent rebalance decision across ranks"
        );
    }

    if (local_should_rebalance == 0) {
        current_loads_ = loads_from_partitions(partitions_);

        std::ostringstream oss;
        oss
            << "adaptive decision rejected; "
            << "remaining_iterations=" << remaining_iterations
            << ", gain_total=" << ganho_total_estimado
            << ", migration_cost=" << custo_migracao
            << ", max_migrated_bytes=" << max_v_migrado
            << ", T_comp_int_per_iter=" << T_comp_int
            << ", T_comp_bal_per_iter=" << T_comp_bal;

        const std::string reason = oss.str();

        print_balance_interval_metrics(
            "profiled",
            "skip",
            reason.c_str(),
            global_times,
            old_loads,
            effective_new_loads,
            current_loads_,
            MPI_Wtime() - balance_t0,
            true
        );

        return false;
    }

    const double apply_t0 = MPI_Wtime();

    this->synchronize(true);

    this->redistribute_selected_registered_fields(
        fields_to_move,
        old_parts,
        new_parts
    );

    partitions_ = new_parts;
    current_loads_ = loads_from_partitions(partitions_);

    this->synchronize(true);

    const double apply_elapsed = MPI_Wtime() - apply_t0;

    std::ostringstream oss;
    oss
        << "adaptive decision accepted; "
        << "remaining_iterations=" << remaining_iterations
        << ", gain_total=" << ganho_total_estimado
        << ", migration_cost=" << custo_migracao
        << ", max_migrated_bytes=" << max_v_migrado
        << ", T_comp_int_per_iter=" << T_comp_int
        << ", T_comp_bal_per_iter=" << T_comp_bal;

    const std::string reason = oss.str();

    print_balance_interval_metrics(
        "profiled",
        "rebalance",
        reason.c_str(),
        global_times,
        old_loads,
        effective_new_loads,
        current_loads_,
        MPI_Wtime() - balance_t0,
        true,
        apply_elapsed
    );

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
            detail::EventGuard read_guard{read_ev};
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
                clFlush(local_devices_[dsrc].transfer_queue),
                "clFlush(transfer_field_range read)"
            );
            detail::check_cl(
                clWaitForEvents(1, &read_ev),
                "clWaitForEvents(transfer_field_range read)"
            );
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
            detail::EventGuard write_guard{write_ev};
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
                clFlush(local_devices_[ddst].transfer_queue),
                "clFlush(transfer_field_range write)"
            );
            detail::check_cl(
                clWaitForEvents(1, &write_ev),
                src_part.owning_rank == rank_
                    ? "clWaitForEvents(transfer_field_range local)"
                    : "clWaitForEvents(transfer_field_range remote)"
            );
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

inline void redistribute_selected_registered_fields(
    const std::vector<FieldHandle>& selected_fields,
    const std::vector<DevicePartition>& old_parts,
    const std::vector<DevicePartition>& new_parts,
    bool synchronize_ranks = true
) {
    const std::vector<FieldHandle> fields_to_move =
        unique_existing_proportional_fields(selected_fields);

    for (FieldHandle fh : fields_to_move) {
        this->redistribute_field_intersection(
            fh,
            old_parts,
            new_parts
        );

        if (synchronize_ranks) {
            detail::check_mpi(
                MPI_Barrier(comm_),
                "MPI_Barrier(redistribute selected proportional field)"
            );
        }
    }
}

inline void redistribute_all_registered_fields(
    const std::vector<DevicePartition>& old_parts,
    const std::vector<DevicePartition>& new_parts
) {
    std::vector<int> field_ids;
    field_ids.reserve(fields_.size());

    std::transform(fields_.begin(), fields_.end(), std::back_inserter(field_ids),
                   [](const auto& kv) { return kv.first; });

    std::sort(field_ids.begin(), field_ids.end());

    for (int field_id : field_ids) {
        std::unordered_map<int, detail::RegisteredField>::iterator it =
            fields_.find(field_id);

        if (it == fields_.end()) {
            continue;
        }

        if (it->second.spec.redistribution != RedistributionDependency::proportional) {
            continue;
        }

        FieldHandle fh{field_id};

        this->redistribute_field_intersection(
            fh,
            old_parts,
            new_parts
        );

        detail::check_mpi(
            MPI_Barrier(comm_),
            "MPI_Barrier(redistribute proportional field)"
        );
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
    double interval_comm_seconds_local_{0.0};
    double accumulated_compute_seconds_{0.0};
    double accumulated_comm_seconds_{0.0};
    double accumulated_balance_seconds_{0.0};
    double accumulated_rebalance_apply_seconds_{0.0};
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
    bool verbose_metrics_{true};

    std::vector<ProfileSegment> profiling_data_;
    bool profiling_loaded_{false};
    std::string profiling_file_loaded_;

    std::optional<TopoMetrics> topo_metrics_{std::nullopt};
    int simulated_total_devices_{-1};
    double numa_cost_gain_ratio_threshold_{0.50};
    std::vector<double> simulated_times_;

public:
    void set_simulated_devices_count(int count) noexcept {
        simulated_total_devices_ = count;
        if (all_device_counts_.empty() ||
            std::accumulate(all_device_counts_.begin(), all_device_counts_.end(), 0) == 0) {
            if (size_ > 0) {
                all_device_counts_.assign(static_cast<std::size_t>(size_), count / size_);
                const int rem = count % size_;
                for (int i = 0; i < rem; ++i) {
                    all_device_counts_[static_cast<std::size_t>(i)]++;
                }
            } else {
                all_device_counts_ = {count};
            }
        }
    }

    void set_simulated_ranks_for_testing(int ranks) noexcept {
        size_ = ranks;
        all_device_counts_.clear();
    }

    void set_numa_cost_gain_ratio_threshold(double threshold) noexcept {
        numa_cost_gain_ratio_threshold_ = threshold;
    }

    double numa_cost_gain_ratio_threshold() const noexcept {
        return numa_cost_gain_ratio_threshold_;
    }

    void set_simulated_times(const std::vector<double>& times) noexcept {
        simulated_times_ = times;
    }

    void clear_simulated_times() noexcept {
        simulated_times_.clear();
    }

    const std::optional<TopoMetrics>& topo_metrics() const noexcept {
        return topo_metrics_;
    }

    void set_topo_metrics(const TopoMetrics& metrics) {
        std::size_t global_device_count = 0;
        for (int count : all_device_counts_) {
            global_device_count += count;
        }

        std::size_t total_devices = (simulated_total_devices_ >= 0)
            ? static_cast<std::size_t>(simulated_total_devices_)
            : global_device_count;

        if (total_devices == 0 && !metrics.pcie_latency_ns.empty()) {
            total_devices = metrics.pcie_latency_ns.size();
        }

        const std::size_t expected_matrix_size =
            static_cast<std::size_t>(size_) * static_cast<std::size_t>(size_);
        if (metrics.mpi_latency_ns.size() != expected_matrix_size ||
            metrics.mpi_bandwidth_gbps.size() != expected_matrix_size) {
            throw dcl::Error("Invalid matrix dimensions");
        }

        if (!metrics.pcie_latency_ns.empty() && metrics.pcie_latency_ns.size() != total_devices) {
            throw dcl::Error("Invalid pcie_latency_ns dimensions");
        }
        if (!metrics.pcie_bandwidth_gbps.empty() && metrics.pcie_bandwidth_gbps.size() != total_devices) {
            throw dcl::Error("Invalid pcie_bandwidth_gbps dimensions");
        }
        if (!metrics.device_numa_node.empty() && metrics.device_numa_node.size() != total_devices) {
            throw dcl::Error("Invalid device_numa_node dimensions");
        }
        if (!metrics.memory_contention_factor.empty() && metrics.memory_contention_factor.size() != total_devices) {
            throw dcl::Error("Invalid memory_contention_factor dimensions");
        }
        if (!metrics.thermal_tdp_watts.empty() && metrics.thermal_tdp_watts.size() != total_devices) {
            throw dcl::Error("Invalid thermal_tdp_watts dimensions");
        }
        if (!metrics.current_power_watts.empty() && metrics.current_power_watts.size() != total_devices) {
            throw dcl::Error("Invalid current_power_watts dimensions");
        }
        auto require_finite_range = [](const std::vector<double>& values,
                                       double minimum, bool strict,
                                       const char* name) {
            for (double value : values) {
                if (!std::isfinite(value) ||
                    (strict ? value <= minimum : value < minimum)) {
                    throw dcl::Error(std::string("Invalid ") + name + " value");
                }
            }
        };
        require_finite_range(metrics.pcie_latency_ns, 0.0, false, "pcie_latency_ns");
        require_finite_range(metrics.pcie_bandwidth_gbps, 0.0, true, "pcie_bandwidth_gbps");
        require_finite_range(metrics.mpi_latency_ns, 0.0, false, "mpi_latency_ns");
        require_finite_range(metrics.mpi_bandwidth_gbps, 0.0, false, "mpi_bandwidth_gbps");
        require_finite_range(metrics.memory_contention_factor, 1.0, false,
                             "memory_contention_factor");
        require_finite_range(metrics.thermal_tdp_watts, 0.0, true, "thermal_tdp_watts");
        require_finite_range(metrics.current_power_watts, 0.0, false,
                             "current_power_watts");
        for (int source = 0; source < size_; ++source) {
            for (int target = 0; target < size_; ++target) {
                if (source != target &&
                    metrics.mpi_bandwidth_gbps[static_cast<std::size_t>(source * size_ + target)] <= 0.0) {
                    throw dcl::Error("Off-diagonal MPI bandwidth must be positive");
                }
            }
        }

        if (std::any_of(metrics.memory_contention_factor.begin(), metrics.memory_contention_factor.end(),
                        [](double factor) { return factor < 1.0; })) {
            throw dcl::Error("memory_contention_factor must be >= 1.0");
        }

        if (!metrics.numa_distance.empty()) {
            const std::size_t n_dist = metrics.numa_distance.size();
            const std::size_t num_nodes = static_cast<std::size_t>(std::round(std::sqrt(static_cast<double>(n_dist))));
            if (num_nodes == 0 || num_nodes * num_nodes != n_dist) {
                throw dcl::Error("numa_distance must be an N x N matrix");
            }
            for (std::size_t i = 0; i < num_nodes; ++i) {
                const int diag = metrics.numa_distance[i * num_nodes + i];
                if (diag <= 0) {
                    throw dcl::Error("numa_distance diagonal entries must be positive integers");
                }
                for (std::size_t j = 0; j < num_nodes; ++j) {
                    const int val = metrics.numa_distance[i * num_nodes + j];
                    if (val <= 0) {
                        throw dcl::Error("numa_distance values must be positive integers");
                    }
                    if (val < diag) {
                        throw dcl::Error("numa_distance diagonal must be the minimum distance");
                    }
                }
            }
        }

        if (!metrics.device_numa_node.empty()) {
            if (metrics.numa_distance.empty()) {
                throw dcl::Error("device_numa_node provided without numa_distance matrix");
            }
            const std::size_t n_dist = metrics.numa_distance.size();
            const std::size_t num_nodes = static_cast<std::size_t>(std::round(std::sqrt(static_cast<double>(n_dist))));
            if (std::any_of(metrics.device_numa_node.begin(), metrics.device_numa_node.end(),
                            [num_nodes](int node) { return node < 0 || static_cast<std::size_t>(node) >= num_nodes; })) {
                throw dcl::Error("device_numa_node out of range");
            }
        }

        topo_metrics_ = metrics;
    }

#ifdef __CPPCHECK__
    static void cppcheck_anchor_unused() {
        (void)&Impl::reduce_bytes_bor;
        (void)&Impl::first_field_with_role;
        (void)&Impl::finalize_kernel_events;
        (void)&Impl::rebalance;
        (void)&Impl::redistribute_field_proportional_delta;
        (void)&Impl::set_simulated_times;
        (void)&Impl::clear_simulated_times;
        (void)&Impl::set_numa_cost_gain_ratio_threshold;
        (void)&Impl::numa_cost_gain_ratio_threshold;
        (void)(bool (Impl::*)(const std::vector<FieldHandle>&, const AutoBalancePolicy&))&Impl::maybe_rebalance_from_timings;
        (void)(bool (Impl::*)(const std::vector<FieldHandle>&, float, double, bool, bool, double))&Impl::maybe_rebalance_from_timings;
        (void)&Impl::maybe_rebalance_hierarchical;
        (void)&::dcl::erad_loads;
        (void)&::dcl::hwtopolb_loads;
    }
#endif
};

} // namespace dcl

#endif
