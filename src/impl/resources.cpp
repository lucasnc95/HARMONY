// Fields (distributed buffers) and kernels.

#include "runtime_impl.hpp"
#include "detail/format.hpp"

#include <iostream>

#include <sstream>

namespace harmony {

detail::RegisteredField& Runtime::Impl::field_or_throw(FieldHandle h, const char* context) {
    auto it = fields_.find(h.value);
    if (it == fields_.end()) {
        throw Error(std::string(context) + ": unknown field");
    }
    return it->second;
}

FieldHandle Runtime::Impl::create_field(const FieldSpec& spec) {
    if (!devices_discovered_) throw Error("discover_devices() must be called before create_field()");
    if (spec.global_elements == 0) throw Error("FieldSpec.global_elements must be > 0");
    if (spec.units_per_element == 0) throw Error("FieldSpec.units_per_element must be > 0");
    if (spec.bytes_per_unit == 0) throw Error("FieldSpec.bytes_per_unit must be > 0");

    const FieldHandle h{next_field_id_++};
    detail::RegisteredField rf;
    rf.spec = spec;
    rf.replicas.resize(local_devices_.size(), nullptr);
    rf.core.resize(local_devices_.size());
    rf.ghost.resize(local_devices_.size());

    const std::size_t total_bytes = rf.total_bytes();
    const cl_mem_flags flags = detail::to_opencl_flags(spec.usage);

    for (std::size_t d = 0; d < local_devices_.size(); ++d) {
        // Every device keeps the whole field in ONE buffer, so the field must
        // fit in the device's largest buffer (on NVIDIA usually 1/4 of the memory).
        const detail::LocalDevice& ld = local_devices_[d];
        if (ld.max_buffer > 0 && total_bytes > ld.max_buffer) {
            std::cerr << "[HARMONY Warning] rank " << rank_ << ": field '" << spec.name << "' needs "
                      << detail::format_bytes(total_bytes) << " but the largest buffer of " << ld.name << " is "
                      << detail::format_bytes(static_cast<std::size_t>(ld.max_buffer))
                      << "; the allocation may fail (reduce the mesh)" << std::endl;
        }
        cl_int err = CL_SUCCESS;
        rf.replicas[d] = clCreateBuffer(ld.context, flags, total_bytes, nullptr, &err);
        if (err != CL_SUCCESS) {
            throw Error("clCreateBuffer(create_field): field '" + spec.name + "' of " +
                        detail::format_bytes(total_bytes) + " on " + ld.name + " failed with OpenCL error " +
                        std::to_string(err) + " (device memory " +
                        detail::format_bytes(static_cast<std::size_t>(ld.global_memory)) + ", largest buffer " +
                        detail::format_bytes(static_cast<std::size_t>(ld.max_buffer)) + ")");
        }
    }

    fields_.emplace(h.value, std::move(rf));
    if (spec.host_ptr != nullptr) write_initial_field_data(h, spec.host_ptr);
    return h;
}

// Uploads the full initial content to every local replica.
void Runtime::Impl::write_initial_field_data(FieldHandle h, const void* host_ptr) {
    detail::RegisteredField& rf = field_or_throw(h, "write_initial_field_data()");
    const std::size_t total_bytes = rf.total_bytes();

    std::vector<cl_event> write_events;
    for (std::size_t d = 0; d < local_devices_.size(); ++d) {
        cl_event ev = nullptr;
        detail::check_cl(
            clEnqueueWriteBuffer(local_devices_[d].transfer_queue, rf.replicas[d], CL_FALSE,
                                 0, total_bytes, host_ptr, 0, nullptr, &ev),
            "clEnqueueWriteBuffer(write_initial_field_data)");
        write_events.push_back(ev);
    }

    for (cl_event ev : write_events) {
        if (ev == nullptr) continue;
        detail::check_cl(clWaitForEvents(1, &ev), "clWaitForEvents(write_initial_field_data)");
        clReleaseEvent(ev);
    }
}

KernelHandle Runtime::Impl::create_kernel(const KernelSpec& spec) {
    if (!devices_discovered_) throw Error("discover_devices() must be called before create_kernel()");
    if (spec.source_file.empty()) throw Error("KernelSpec.source_file is empty");
    if (spec.entry_point.empty()) throw Error("KernelSpec.entry_point is empty");

    const KernelHandle h{next_kernel_id_++};
    detail::RegisteredKernel rk;
    rk.spec = spec;
    rk.programs_per_platform.resize(platforms_.size(), nullptr);
    rk.kernels_per_local_device.resize(local_devices_.size(), nullptr);

    const std::string src = detail::slurp_file(spec.source_file);
    const char* src_ptr = src.c_str();
    const std::size_t src_len = src.size();

    // Build one program per platform context.
    for (std::size_t p = 0; p < platforms_.size(); ++p) {
        const detail::PlatformContext& pc = platforms_[p];
        cl_int err = CL_SUCCESS;
        cl_program program = clCreateProgramWithSource(pc.context, 1, &src_ptr, &src_len, &err);
        detail::check_cl(err, "clCreateProgramWithSource");

        err = clBuildProgram(program, static_cast<cl_uint>(pc.devices.size()), pc.devices.data(),
                             spec.build_options.empty() ? nullptr : spec.build_options.c_str(),
                             nullptr, nullptr);

        if (err != CL_SUCCESS) {
            std::ostringstream oss;
            oss << "clBuildProgram failed for kernel " << spec.entry_point << "\n";
            for (std::size_t i = 0; i < pc.devices.size(); ++i) {
                std::size_t log_size = 0;
                clGetProgramBuildInfo(program, pc.devices[i], CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
                std::vector<char> log(log_size + 1, '\0');
                clGetProgramBuildInfo(program, pc.devices[i], CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
                oss << "---- device " << i << " ----\n" << log.data() << "\n";
            }
            clReleaseProgram(program);
            for (cl_program built : rk.programs_per_platform) {
                if (built != nullptr) clReleaseProgram(built);
            }
            throw Error(oss.str());
        }
        rk.programs_per_platform[p] = program;
    }

    // One kernel object per local device (arguments are set per device).
    for (std::size_t d = 0; d < local_devices_.size(); ++d) {
        const int p = local_devices_[d].platform_index;
        cl_int err = CL_SUCCESS;
        rk.kernels_per_local_device[d] =
            clCreateKernel(rk.programs_per_platform[p], spec.entry_point.c_str(), &err);
        detail::check_cl(err, "clCreateKernel");
    }

    kernels_.emplace(h.value, std::move(rk));
    return h;
}

} // namespace harmony
