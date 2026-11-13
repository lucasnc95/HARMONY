// MPI start-up/shutdown, OpenCL device discovery and teardown.

#include "runtime_impl.hpp"
#include "detail/format.hpp"

#include <algorithm>
#include <cstring>
#include <exception>
#include <iostream>
#include <limits>
#include <sstream>

namespace harmony {

Runtime::Impl::Impl(int& argc, char**& argv) : argc_(argc), argv_(argv) {}

Runtime::Impl::~Impl() {
    release_opencl_objects();

    int finalized = 0;
    MPI_Finalized(&finalized);
    if (halo_comm_ != MPI_COMM_NULL && !finalized) MPI_Comm_free(&halo_comm_);

    // While an exception is propagating, other ranks may still be inside a
    // collective call; MPI_Finalize would then hang, so the application is
    // expected to call MPI_Abort instead.
    if (mpi_initialized_by_runtime_ && std::uncaught_exceptions() == 0) {
        if (!finalized) MPI_Finalize();
    }
}

void Runtime::Impl::initialize_mpi_from_runtime() {
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized) {
        int provided = 0;
        MPI_Init_thread(&argc_, &argv_, MPI_THREAD_SERIALIZED, &provided);
        mpi_initialized_by_runtime_ = true;
    }
    detail::check_mpi(MPI_Comm_rank(MPI_COMM_WORLD, &rank_), "MPI_Comm_rank");
    detail::check_mpi(MPI_Comm_size(MPI_COMM_WORLD, &size_), "MPI_Comm_size");
    comm_ = MPI_COMM_WORLD;

    // Halo messages travel on their own communicator, so their tags can never
    // match a gather or redistribution message.
    detail::check_mpi(MPI_Comm_dup(comm_, &halo_comm_), "MPI_Comm_dup(halo)");
    timing_start_ = MPI_Wtime();
}

void Runtime::Impl::release_opencl_objects() {
    release_tracked_field_events();
    for (std::deque<cl_event>& pending : pending_transfer_events_) {
        for (cl_event ev : pending) clReleaseEvent(ev);
        pending.clear();
    }
    for (std::deque<cl_event>& pending : pending_kernel_events_) {
        for (cl_event ev : pending) clReleaseEvent(ev);
        pending.clear();
    }

    for (auto& kv : kernels_) {
        for (cl_kernel k : kv.second.kernels_per_local_device) {
            if (k != nullptr) clReleaseKernel(k);
        }
        for (cl_program p : kv.second.programs_per_platform) {
            if (p != nullptr) clReleaseProgram(p);
        }
    }
    kernels_.clear();

    for (auto& kv : fields_) {
        for (cl_mem mem : kv.second.replicas) {
            if (mem != nullptr) clReleaseMemObject(mem);
        }
    }
    fields_.clear();

    for (detail::LocalDevice& ld : local_devices_) {
        for (cl_command_queue q : ld.kernel_queues) {
            if (q != nullptr) clReleaseCommandQueue(q);
        }
        if (ld.transfer_queue != nullptr) clReleaseCommandQueue(ld.transfer_queue);
    }
    local_devices_.clear();

    for (detail::PlatformContext& pc : platforms_) {
        if (pc.context != nullptr) clReleaseContext(pc.context);
#if CL_TARGET_OPENCL_VERSION >= 120
        for (cl_device_id sub : pc.sub_devices) clReleaseDevice(sub);
#endif
    }
    platforms_.clear();
}

void Runtime::Impl::clear_runtime_state() {
    release_opencl_objects();

    devices_.clear();
    partitions_.clear();
    all_device_counts_.clear();
    current_loads_.clear();
    device_timings_.clear();
    global_devices_.clear();
    pending_kernel_events_.clear();
    pending_transfer_events_.clear();
    kernel_busy_total_.clear();
    kernel_busy_window_.clear();
    kernel_busy_until_.clear();
    active_busy_total_.clear();
    active_busy_until_.clear();
    halo_buffers_.clear();

    next_field_id_ = 0;
    next_kernel_id_ = 0;
    iteration_counter_ = 0;
    balance_window_start_iteration_ = 0;
    static_balance_attempted_ = false;
    balance_policy_announced_ = false;
    balance_policy_.reset();
}

namespace {

cl_command_queue create_profiling_queue(cl_context context, cl_device_id device, const char* what) {
    cl_int err = CL_SUCCESS;
#if CL_TARGET_OPENCL_VERSION >= 200
    const cl_queue_properties props[] = {
        CL_QUEUE_PROPERTIES,
        static_cast<cl_queue_properties>(CL_QUEUE_PROFILING_ENABLE),
        0
    };
    cl_command_queue q = clCreateCommandQueueWithProperties(context, device, props, &err);
#else
    cl_command_queue q = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
#endif
    detail::check_cl(err, what);
    return q;
}

// Returns a sub-device of a CPU device with `reserve` fewer compute units, or
// the device itself when it is not a CPU or cannot be partitioned.
cl_device_id reserve_host_cores(cl_device_id dev, int reserve, detail::PlatformContext& pc) {
#if CL_TARGET_OPENCL_VERSION >= 120
    cl_device_type type = 0;
    cl_uint units = 0;
    detail::check_cl(clGetDeviceInfo(dev, CL_DEVICE_TYPE, sizeof(type), &type, nullptr),
                     "clGetDeviceInfo(CL_DEVICE_TYPE)");
    detail::check_cl(clGetDeviceInfo(dev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(units), &units, nullptr),
                     "clGetDeviceInfo(CL_DEVICE_MAX_COMPUTE_UNITS)");
    if (!(type & CL_DEVICE_TYPE_CPU) || units <= static_cast<cl_uint>(reserve)) return dev;

    const cl_device_partition_property props[] = {
        CL_DEVICE_PARTITION_BY_COUNTS,
        static_cast<cl_device_partition_property>(units - static_cast<cl_uint>(reserve)),
        CL_DEVICE_PARTITION_BY_COUNTS_LIST_END,
        0
    };
    cl_device_id sub = nullptr;
    cl_uint created = 0;
    const cl_int err = clCreateSubDevices(dev, props, 1, &sub, &created);
    if (err != CL_SUCCESS || created != 1 || sub == nullptr) {
        std::cerr << "[HARMONY Warning] the OpenCL driver cannot split the CPU device (error " << err
                  << "); using all its compute units\n";
        return dev;
    }
    pc.sub_devices.push_back(sub);
    return sub;
#else
    (void)reserve;
    (void)pc;
    return dev;
#endif
}

} // namespace

void Runtime::Impl::discover_devices(const DeviceSelection& selection) {
    clear_runtime_state();

    // A node without any OpenCL platform (CL_PLATFORM_NOT_FOUND_KHR = -1001)
    // just contributes no device; it must still join the collectives below.
    constexpr cl_int kPlatformNotFound = -1001;
    cl_uint n_platforms = 0;
    const cl_int count_err = clGetPlatformIDs(0, nullptr, &n_platforms);
    if (count_err == kPlatformNotFound) n_platforms = 0;
    else detail::check_cl(count_err, "clGetPlatformIDs(count)");

    std::vector<cl_platform_id> platform_ids(n_platforms);
    if (n_platforms > 0) {
        detail::check_cl(clGetPlatformIDs(n_platforms, platform_ids.data(), nullptr), "clGetPlatformIDs(list)");
    }

    const int max_per_rank = (selection.max_devices_per_rank <= 0)
        ? std::numeric_limits<int>::max()
        : selection.max_devices_per_rank;
    int selected_count = 0;

    for (cl_uint p = 0; p < n_platforms && selected_count < max_per_rank; ++p) {
        cl_uint dev_count = 0;
        const cl_int err = clGetDeviceIDs(platform_ids[p], CL_DEVICE_TYPE_ALL, 0, nullptr, &dev_count);
        if (err == CL_DEVICE_NOT_FOUND || dev_count == 0) continue;
        detail::check_cl(err, "clGetDeviceIDs(count)");

        std::vector<cl_device_id> platform_devices(dev_count);
        detail::check_cl(
            clGetDeviceIDs(platform_ids[p], CL_DEVICE_TYPE_ALL, dev_count, platform_devices.data(), nullptr),
            "clGetDeviceIDs(list)");

        // Filter by device kind and per-rank limit.
        std::vector<cl_device_id> selected;
        for (cl_device_id dev : platform_devices) {
            if (selected_count >= max_per_rank) break;
            cl_device_type dtype = 0;
            detail::check_cl(clGetDeviceInfo(dev, CL_DEVICE_TYPE, sizeof(dtype), &dtype, nullptr),
                             "clGetDeviceInfo(CL_DEVICE_TYPE)");
            if (!detail::matches(selection.kind, detail::classify_device_kind(dtype))) continue;
            selected.push_back(dev);
            ++selected_count;
        }
        if (selected.empty()) continue;

        // One context per platform.
        detail::PlatformContext pc{};
        pc.platform = platform_ids[p];

        // CPU devices: leave `cpu_host_cores` compute units to this process.
        if (selection.cpu_host_cores > 0) {
            for (cl_device_id& dev : selected) dev = reserve_host_cores(dev, selection.cpu_host_cores, pc);
        }
        pc.devices = selected;

        cl_context_properties props[] = {
            CL_CONTEXT_PLATFORM,
            reinterpret_cast<cl_context_properties>(platform_ids[p]),
            0
        };
        cl_int ctx_err = CL_SUCCESS;
        pc.context = clCreateContext(props, static_cast<cl_uint>(selected.size()), selected.data(),
                                     nullptr, nullptr, &ctx_err);
        detail::check_cl(ctx_err, "clCreateContext");

        platforms_.push_back(pc);
        const int platform_index = static_cast<int>(platforms_.size() - 1);

        for (std::size_t i = 0; i < selected.size(); ++i) {
            detail::LocalDevice ld{};
            ld.platform_index = platform_index;
            ld.device_index_in_platform = static_cast<int>(i);
            ld.local_index = static_cast<int>(local_devices_.size());
            ld.device = selected[i];
            ld.context = pc.context;

            char name_buf[512] = {};
            cl_uint cus = 0;
            cl_device_type dtype = 0;
            detail::check_cl(clGetDeviceInfo(ld.device, CL_DEVICE_NAME, sizeof(name_buf), name_buf, nullptr),
                             "clGetDeviceInfo(CL_DEVICE_NAME)");
            detail::check_cl(clGetDeviceInfo(ld.device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cus), &cus, nullptr),
                             "clGetDeviceInfo(CL_DEVICE_MAX_COMPUTE_UNITS)");
            detail::check_cl(clGetDeviceInfo(ld.device, CL_DEVICE_TYPE, sizeof(dtype), &dtype, nullptr),
                             "clGetDeviceInfo(CL_DEVICE_TYPE)");

            ld.name = detail::collapse_spaces(name_buf);  // "CPU      E5620" -> "CPU E5620"
            detail::check_cl(clGetDeviceInfo(ld.device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(ld.global_memory),
                                             &ld.global_memory, nullptr),
                             "clGetDeviceInfo(CL_DEVICE_GLOBAL_MEM_SIZE)");
            detail::check_cl(clGetDeviceInfo(ld.device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(ld.max_buffer),
                                             &ld.max_buffer, nullptr),
                             "clGetDeviceInfo(CL_DEVICE_MAX_MEM_ALLOC_SIZE)");
            ld.compute_units = static_cast<unsigned>(cus);
            ld.kind = detail::classify_device_kind(dtype);
            const int queues = std::max(1, selection.kernel_queues);
            for (int q = 0; q < queues; ++q) {
                ld.kernel_queues.push_back(create_profiling_queue(ld.context, ld.device, "create kernel queue"));
            }
            ld.transfer_queue = create_profiling_queue(ld.context, ld.device, "create transfer_queue");
            local_devices_.push_back(ld);

            DeviceInfo info{};
            info.rank = rank_;
            info.local_index = ld.local_index;
            info.global_index = -1;  // assigned below, once all ranks are known
            info.name = ld.name;
            info.kind = ld.kind;
            info.compute_units = ld.compute_units;
            info.global_memory = ld.global_memory;
            info.max_buffer = ld.max_buffer;
            devices_.push_back(info);
        }
    }

    // Global device numbering: rank 0 devices first, then rank 1, ...
    const int local_count = static_cast<int>(devices_.size());
    all_device_counts_.assign(size_, 0);
    detail::check_mpi(
        MPI_Allgather(&local_count, 1, MPI_INT, all_device_counts_.data(), 1, MPI_INT, comm_),
        "MPI_Allgather(device_counts)");

    int global_base = 0;
    for (int r = 0; r < rank_; ++r) global_base += all_device_counts_[r];
    for (std::size_t i = 0; i < devices_.size(); ++i) {
        devices_[i].global_index = global_base + static_cast<int>(i);
        local_devices_[i].global_index = devices_[i].global_index;
    }

    const std::size_t n = local_devices_.size();
    device_timings_.assign(n, DeviceTiming{});
    pending_kernel_events_.assign(n, {});
    pending_transfer_events_.assign(n, {});
    kernel_busy_total_.assign(n, 0.0);
    kernel_busy_window_.assign(n, 0.0);
    kernel_busy_until_.assign(n, 0.0);
    active_busy_total_.assign(n, 0.0);
    active_busy_until_.assign(n, 0.0);

    gather_global_device_info();
    devices_discovered_ = true;

    // A rank without devices is allowed (it simply owns no partition), but
    // the whole job needs at least one.
    if (global_devices_.empty()) {
        const char* kind = selection.kind == DeviceKind::gpu   ? "GPU"
                           : selection.kind == DeviceKind::cpu ? "CPU"
                           : selection.kind == DeviceKind::accelerator ? "accelerator"
                                                                        : "OpenCL";
        throw Error(std::string("discover_devices(): no ") + kind +
                    " device found on any rank (check the OpenCL drivers/ICDs installed on the nodes)");
    }
}

namespace {

// Fixed-size record so device descriptions can travel with MPI_BYTE.
struct DeviceRecord {
    int global_index;
    int rank;
    int local_index;
    int kind;
    unsigned compute_units;
    std::uint64_t global_memory;
    std::uint64_t max_buffer;
    char name[128];
};

} // namespace

// Every rank learns the devices of every other rank (used by the tables).
void Runtime::Impl::gather_global_device_info() {
    std::vector<DeviceRecord> local(devices_.size());
    for (std::size_t i = 0; i < devices_.size(); ++i) {
        DeviceRecord& r = local[i];
        std::memset(&r, 0, sizeof(r));
        r.global_index = devices_[i].global_index;
        r.rank = rank_;
        r.local_index = devices_[i].local_index;
        r.kind = static_cast<int>(devices_[i].kind);
        r.compute_units = devices_[i].compute_units;
        r.global_memory = devices_[i].global_memory;
        r.max_buffer = devices_[i].max_buffer;
        std::strncpy(r.name, devices_[i].name.c_str(), sizeof(r.name) - 1);
    }

    std::vector<int> counts(size_), displs(size_);
    int total = 0;
    for (int r = 0; r < size_; ++r) {
        counts[r] = all_device_counts_[r] * static_cast<int>(sizeof(DeviceRecord));
        displs[r] = total;
        total += counts[r];
    }
    std::vector<DeviceRecord> all(static_cast<std::size_t>(total) / sizeof(DeviceRecord));
    detail::check_mpi(MPI_Allgatherv(local.data(), counts[rank_], MPI_BYTE, all.data(), counts.data(),
                                     displs.data(), MPI_BYTE, comm_),
                      "MPI_Allgatherv(device info)");

    global_devices_.clear();
    for (const DeviceRecord& r : all) {
        DeviceInfo info;
        info.rank = r.rank;
        info.local_index = r.local_index;
        info.global_index = r.global_index;
        info.name = r.name;
        info.kind = static_cast<DeviceKind>(r.kind);
        info.compute_units = r.compute_units;
        info.global_memory = r.global_memory;
        info.max_buffer = r.max_buffer;
        global_devices_.push_back(info);
    }
}

void Runtime::Impl::print_devices(std::ostream& os) const {
    if (rank_ != 0) return;
    detail::Table table({{"Dev", false}, {"Rank", false}, {"Type", false}, {"Compute units", true},
                         {"Memory", true}, {"Largest buffer", true}, {"Name", false}});
    for (const DeviceInfo& d : global_devices_) {
        table.add_row({std::to_string(d.global_index), std::to_string(d.rank), detail::device_kind_name(d.kind),
                       std::to_string(d.compute_units), detail::format_bytes(d.global_memory),
                       detail::format_bytes(d.max_buffer), d.name});
    }
    std::ostringstream out;
    out << "Devices: " << global_devices_.size() << " on " << size_
        << (size_ == 1 ? " MPI process" : " MPI processes") << "\n";
    table.print(out, "  ");
    os << out.str() << std::flush;
}

} // namespace harmony
