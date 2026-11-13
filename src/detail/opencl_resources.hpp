// OpenCL objects owned by the runtime and small OpenCL helpers.

#ifndef HARMONY_DETAIL_OPENCL_RESOURCES_HPP
#define HARMONY_DETAIL_OPENCL_RESOURCES_HPP

#include "harmony/types.hpp"

#include <CL/cl.h>

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace harmony::detail {

// One OpenCL context per platform, shared by the selected devices of it.
struct PlatformContext {
    cl_platform_id platform{nullptr};
    cl_context context{nullptr};
    std::vector<cl_device_id> devices;
    std::vector<cl_device_id> sub_devices;  // created by the runtime, released at teardown
};

// A device visible to this rank. Independent kernels go to different kernel
// queues so they can run at the same time; transfers use their own queue so
// halo exchanges overlap with computation.
struct LocalDevice {
    int platform_index{-1};
    int device_index_in_platform{-1};
    int local_index{-1};
    int global_index{-1};

    cl_device_id device{nullptr};
    cl_context context{nullptr};
    std::vector<cl_command_queue> kernel_queues;
    cl_command_queue transfer_queue{nullptr};

    cl_uint compute_units{0};
    DeviceKind kind{DeviceKind::all};
    std::string name;
    cl_ulong global_memory{0};
    cl_ulong max_buffer{0};
};

// Pending OpenCL commands touching one region of a field on one device.
// Readers wait for the writers; a writer waits for both and replaces them.
struct AccessTracker {
    std::vector<cl_event> writers;
    std::vector<cl_event> readers;
};

// A field keeps one full-size replica per local device; each device only
// keeps up to date the slice of its partition ("core") plus the elements
// copied from its neighbours ("ghost", the halo).
struct RegisteredField {
    FieldSpec spec;
    std::vector<cl_mem> replicas;  // indexed by local device
    std::vector<AccessTracker> core;
    std::vector<AccessTracker> ghost;

    // Halo width (elements) whose ghost data is up to date on every device.
    // Every replica starts with the full initial data, so it starts "infinite".
    std::size_t valid_halo{static_cast<std::size_t>(-1)};

    std::size_t element_bytes() const { return spec.units_per_element * spec.bytes_per_unit; }
    std::size_t total_bytes() const { return spec.global_elements * element_bytes(); }
};

struct RegisteredKernel {
    KernelSpec spec;
    std::vector<cl_program> programs_per_platform;
    std::vector<cl_kernel> kernels_per_local_device;
};

// Linear segment of the migration-cost model: cost = m * bytes + b,
// valid for volumes up to max_volume bytes.
struct ProfileSegment {
    std::size_t max_volume;
    double m;
    double b;
};

inline DeviceKind classify_device_kind(cl_device_type type) {
    if (type & CL_DEVICE_TYPE_GPU) return DeviceKind::gpu;
    if (type & CL_DEVICE_TYPE_CPU) return DeviceKind::cpu;
    if (type & CL_DEVICE_TYPE_ACCELERATOR) return DeviceKind::accelerator;
    return DeviceKind::all;
}

inline bool matches(DeviceKind wanted, DeviceKind actual) {
    return wanted == DeviceKind::all || wanted == actual;
}

inline cl_mem_flags to_opencl_flags(BufferUsage usage) {
    switch (usage) {
        case BufferUsage::read_only:  return CL_MEM_READ_ONLY;
        case BufferUsage::write_only: return CL_MEM_WRITE_ONLY;
        case BufferUsage::read_write: return CL_MEM_READ_WRITE;
    }
    return CL_MEM_READ_WRITE;
}

inline void release_event_if_needed(cl_event& ev) {
    if (ev != nullptr) {
        clReleaseEvent(ev);
        ev = nullptr;
    }
}

inline void release_events(std::vector<cl_event>& events) {
    for (cl_event& ev : events) release_event_if_needed(ev);
    events.clear();
}

// Profiling timestamps (seconds) of a finished command; false if unavailable.
inline bool event_interval(cl_event ev, double& start, double& end) {
    cl_ulong t0 = 0;
    cl_ulong t1 = 0;
    const cl_int e0 = clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &t0, nullptr);
    const cl_int e1 = clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &t1, nullptr);
    if (e0 != CL_SUCCESS || e1 != CL_SUCCESS || t1 < t0) return false;
    start = static_cast<double>(t0) * 1.0e-9;
    end = static_cast<double>(t1) * 1.0e-9;
    return true;
}

// Reads a whole text file (kernel sources). Throws harmony::Error on failure.
std::string slurp_file(const std::string& path);

} // namespace harmony::detail

#endif // HARMONY_DETAIL_OPENCL_RESOURCES_HPP
