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

    detail::ScopedBufferList guarded_replicas;
    guarded_replicas.reserve(local_devices_.size());

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
        detail::ScopedBuffer buf(clCreateBuffer(ld.context, flags, total_bytes, nullptr, &err));
        if (err != CL_SUCCESS || !buf) {
            throw Error("clCreateBuffer(create_field): field '" + spec.name + "' of " +
                        detail::format_bytes(total_bytes) + " on " + ld.name + " failed with OpenCL error " +
                        std::to_string(err) + " (device memory " +
                        detail::format_bytes(static_cast<std::size_t>(ld.global_memory)) + ", largest buffer " +
                        detail::format_bytes(static_cast<std::size_t>(ld.max_buffer)) + ")");
        }
        guarded_replicas.push_back(std::move(buf));
    }

    auto [it, inserted] = fields_.emplace(h.value, std::move(rf));
    (void)inserted;
    auto rollback = detail::make_scope_exit([&]() {
        auto fit = fields_.find(h.value);
        if (fit != fields_.end()) {
            for (cl_mem mem : fit->second.replicas) {
                if (mem != nullptr) clReleaseMemObject(mem);
            }
            fields_.erase(fit);
        }
    });

    for (std::size_t d = 0; d < local_devices_.size(); ++d) {
        it->second.replicas[d] = guarded_replicas[d].release();
    }

    if (spec.host_ptr != nullptr) write_initial_field_data(h, spec.host_ptr);
    rollback.dismiss();
    return h;
}

// Uploads the full initial content to every local replica.
void Runtime::Impl::write_initial_field_data(FieldHandle h, const void* host_ptr) {
    detail::RegisteredField& rf = field_or_throw(h, "write_initial_field_data()");
    const std::size_t total_bytes = rf.total_bytes();

    detail::ScopedEventList write_events;
    write_events.reserve(local_devices_.size());
    for (std::size_t d = 0; d < local_devices_.size(); ++d) {
        detail::ScopedEvent ev;
        detail::check_cl(
            clEnqueueWriteBuffer(local_devices_[d].transfer_queue, rf.replicas[d], CL_FALSE,
                                 0, total_bytes, host_ptr, 0, nullptr, ev.out_ptr()),
            "clEnqueueWriteBuffer(write_initial_field_data)");
        write_events.push_back(std::move(ev));
        detail::check_cl(clFlush(local_devices_[d].transfer_queue), "clFlush(write_initial_field_data)");
    }

    for (detail::ScopedEvent& ev : write_events) {
        if (!ev) continue;
        cl_event raw = ev.get();
        detail::check_cl(clWaitForEvents(1, &raw), "clWaitForEvents(write_initial_field_data)");
        ev.reset();
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

    std::vector<detail::ScopedProgram> programs;
    programs.reserve(platforms_.size());

    // Build one program per platform context.
    for (std::size_t p = 0; p < platforms_.size(); ++p) {
        const detail::PlatformContext& pc = platforms_[p];
        cl_int err = CL_SUCCESS;
        detail::ScopedProgram program(
            clCreateProgramWithSource(pc.context, 1, &src_ptr, &src_len, &err));
        detail::check_cl(err, "clCreateProgramWithSource");

        err = clBuildProgram(program.get(), static_cast<cl_uint>(pc.devices.size()), pc.devices.data(),
                             spec.build_options.empty() ? nullptr : spec.build_options.c_str(),
                             nullptr, nullptr);

        if (err != CL_SUCCESS) {
            std::ostringstream oss;
            oss << "clBuildProgram failed for kernel " << spec.entry_point << "\n";
            for (std::size_t i = 0; i < pc.devices.size(); ++i) {
                std::size_t log_size = 0;
                clGetProgramBuildInfo(program.get(), pc.devices[i], CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
                std::vector<char> log(log_size + 1, '\0');
                clGetProgramBuildInfo(program.get(), pc.devices[i], CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
                oss << "---- device " << i << " ----\n" << log.data() << "\n";
            }
            throw Error(oss.str());
        }
        programs.push_back(std::move(program));
    }

    // One kernel object per local device (arguments are set per device).
    std::vector<detail::ScopedKernel> kernels;
    kernels.reserve(local_devices_.size());
    for (std::size_t d = 0; d < local_devices_.size(); ++d) {
        const int p = local_devices_[d].platform_index;
        cl_int err = CL_SUCCESS;
        detail::ScopedKernel kernel(
            clCreateKernel(programs[static_cast<std::size_t>(p)].get(), spec.entry_point.c_str(), &err));
        detail::check_cl(err, "clCreateKernel");
        kernels.push_back(std::move(kernel));
    }

    auto [it, inserted] = kernels_.emplace(h.value, std::move(rk));
    (void)inserted;
    for (std::size_t p = 0; p < programs.size(); ++p) {
        it->second.programs_per_platform[p] = programs[p].release();
    }
    for (std::size_t d = 0; d < kernels.size(); ++d) {
        it->second.kernels_per_local_device[d] = kernels[d].release();
    }
    return h;
}

} // namespace harmony
