// stencil2d written directly with MPI + OpenCL, without HARMONY.
//
// It mirrors what the runtime does for this case (equal 1D partitions, one
// partition per device, interior/halo/border phases, gather to rank 0) and
// serves as a baseline to measure the runtime overhead.

#include "stencil2d.hpp"

#include <CL/cl.h>
#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace stencil2d {

namespace {

void check_cl(cl_int err, const char* what) {
    if (err != CL_SUCCESS) {
        std::ostringstream oss;
        oss << what << " failed with OpenCL error " << err;
        throw std::runtime_error(oss.str());
    }
}

void check_mpi(int err, const char* what) {
    if (err != MPI_SUCCESS) {
        char buf[MPI_MAX_ERROR_STRING];
        int len = 0;
        MPI_Error_string(err, buf, &len);
        throw std::runtime_error(std::string(what) + " failed with MPI error: " + std::string(buf, len));
    }
}

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open kernel file: " + path);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

// One context, queue and program per device.
struct OclDevice {
    cl_device_id id = nullptr;
    cl_context ctx = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;
    cl_mem in = nullptr;
    cl_mem out = nullptr;
    std::string name;
    cl_uint compute_units = 0;
};

struct Devices {
    std::vector<OclDevice> list;

    ~Devices() {
        for (OclDevice& d : list) {
            if (d.in) clReleaseMemObject(d.in);
            if (d.out) clReleaseMemObject(d.out);
            if (d.kernel) clReleaseKernel(d.kernel);
            if (d.program) clReleaseProgram(d.program);
            if (d.queue) clReleaseCommandQueue(d.queue);
            if (d.ctx) clReleaseContext(d.ctx);
        }
    }
};

void discover(Devices& out, Device filter, const std::string& src, const char* entry_point,
              const std::string& build_options) {
    cl_uint nplat = 0;
    check_cl(clGetPlatformIDs(0, nullptr, &nplat), "clGetPlatformIDs(count)");
    std::vector<cl_platform_id> plats(nplat);
    check_cl(clGetPlatformIDs(nplat, plats.data(), nullptr), "clGetPlatformIDs(list)");

    const cl_device_type dtype = (filter == Device::cpu)   ? CL_DEVICE_TYPE_CPU
                                 : (filter == Device::gpu) ? CL_DEVICE_TYPE_GPU
                                                                      : CL_DEVICE_TYPE_ALL;

    for (cl_platform_id p : plats) {
        cl_uint ndev = 0;
        if (clGetDeviceIDs(p, dtype, 0, nullptr, &ndev) != CL_SUCCESS || ndev == 0) continue;
        std::vector<cl_device_id> ids(ndev);
        check_cl(clGetDeviceIDs(p, dtype, ndev, ids.data(), nullptr), "clGetDeviceIDs");

        for (cl_device_id id : ids) {
            out.list.emplace_back();
            OclDevice& d = out.list.back();
            d.id = id;

            char name[512] = {};
            check_cl(clGetDeviceInfo(id, CL_DEVICE_NAME, sizeof(name), name, nullptr), "clGetDeviceInfo(name)");
            d.name = name;
            check_cl(clGetDeviceInfo(id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(d.compute_units), &d.compute_units,
                                     nullptr),
                     "clGetDeviceInfo(compute units)");

            cl_int err = CL_SUCCESS;
            d.ctx = clCreateContext(nullptr, 1, &id, nullptr, nullptr, &err);
            check_cl(err, "clCreateContext");
#if CL_TARGET_OPENCL_VERSION >= 200
            d.queue = clCreateCommandQueueWithProperties(d.ctx, id, nullptr, &err);
#else
            d.queue = clCreateCommandQueue(d.ctx, id, 0, &err);
#endif
            check_cl(err, "clCreateCommandQueue");

            const char* csrc = src.c_str();
            const std::size_t len = src.size();
            d.program = clCreateProgramWithSource(d.ctx, 1, &csrc, &len, &err);
            check_cl(err, "clCreateProgramWithSource");
            err = clBuildProgram(d.program, 1, &id, build_options.c_str(), nullptr, nullptr);
            if (err != CL_SUCCESS) {
                std::size_t log_sz = 0;
                clGetProgramBuildInfo(d.program, id, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_sz);
                std::string log(log_sz, '\0');
                clGetProgramBuildInfo(d.program, id, CL_PROGRAM_BUILD_LOG, log_sz, log.data(), nullptr);
                std::cerr << log << std::endl;
                check_cl(err, "clBuildProgram");
            }
            d.kernel = clCreateKernel(d.program, entry_point, &err);
            check_cl(err, "clCreateKernel");
        }
    }
    if (out.list.empty()) throw std::runtime_error("No OpenCL devices found for the selected filter");
}

struct Partition {
    int owning_rank = -1;
    int local_index = -1;
    std::size_t offset = 0;
    std::size_t count = 0;
};

// Equal split over all devices of all ranks; global device order is rank by rank.
std::vector<Partition> equal_partitions(std::size_t total, int local_devices, int rank, MPI_Comm comm) {
    int size = 1;
    MPI_Comm_size(comm, &size);
    std::vector<int> counts(size, 0);
    check_mpi(MPI_Allgather(&local_devices, 1, MPI_INT, counts.data(), 1, MPI_INT, comm), "MPI_Allgather");

    std::vector<Partition> parts;
    for (int r = 0; r < size; ++r) {
        for (int i = 0; i < counts[r]; ++i) {
            parts.push_back(Partition{r, r == rank ? i : -1, 0, 0});
        }
    }

    const std::size_t base = total / parts.size();
    const std::size_t rem = total % parts.size();
    std::size_t cur = 0;
    for (std::size_t g = 0; g < parts.size(); ++g) {
        parts[g].offset = cur;
        parts[g].count = base + (g < rem ? 1 : 0);
        cur += parts[g].count;
    }
    return parts;
}

void wait_and_release(std::vector<cl_event>& events) {
    for (cl_event ev : events) {
        check_cl(clWaitForEvents(1, &ev), "clWaitForEvents");
        clReleaseEvent(ev);
    }
    events.clear();
}

void enqueue(OclDevice& d, std::size_t offset, std::size_t count, std::vector<cl_event>& events) {
    if (count == 0) return;
    cl_event ev = nullptr;
    check_cl(clEnqueueNDRangeKernel(d.queue, d.kernel, 1, &offset, &count, nullptr, 0, nullptr, &ev),
             "clEnqueueNDRangeKernel");
    events.push_back(ev);
}

// Copies the last `halo` elements of each partition after it in the right
// neighbour, and the first `halo` elements before it in the left neighbour.
template <typename T>
void exchange_halos(const std::vector<Partition>& parts, int rank, MPI_Comm comm, Devices& devs, bool use_in,
                    std::size_t halo) {
    const std::size_t bytes = halo * sizeof(T);
    std::vector<T> send_lr(halo), send_rl(halo), recv_lr(halo), recv_rl(halo);

    for (std::size_t b = 0; b + 1 < parts.size(); ++b) {
        const Partition& left = parts[b];
        const Partition& right = parts[b + 1];
        if (left.count < halo || right.count < halo) continue;

        OclDevice* dl = left.owning_rank == rank ? &devs.list[left.local_index] : nullptr;
        OclDevice* dr = right.owning_rank == rank ? &devs.list[right.local_index] : nullptr;
        auto buf = [&](OclDevice* d) { return use_in ? d->in : d->out; };

        const std::size_t left_border = (left.offset + left.count - halo) * sizeof(T);
        const std::size_t left_halo = (left.offset + left.count) * sizeof(T);
        const std::size_t right_border = right.offset * sizeof(T);
        const std::size_t right_halo = (right.offset - halo) * sizeof(T);

        if (dl) check_cl(clEnqueueReadBuffer(dl->queue, buf(dl), CL_TRUE, left_border, bytes, send_lr.data(), 0,
                                             nullptr, nullptr), "read halo lr");
        if (dr) check_cl(clEnqueueReadBuffer(dr->queue, buf(dr), CL_TRUE, right_border, bytes, send_rl.data(), 0,
                                             nullptr, nullptr), "read halo rl");

        const T* to_right = send_lr.data();
        const T* to_left = send_rl.data();

        if (!(dl && dr)) {
            const int tag_lr = 10000 + static_cast<int>(b);
            const int tag_rl = 20000 + static_cast<int>(b);
            MPI_Request reqs[2];
            if (dl) {
                check_mpi(MPI_Isend(send_lr.data(), static_cast<int>(bytes), MPI_BYTE, right.owning_rank, tag_lr,
                                    comm, &reqs[0]), "MPI_Isend(lr)");
                check_mpi(MPI_Irecv(recv_rl.data(), static_cast<int>(bytes), MPI_BYTE, right.owning_rank, tag_rl,
                                    comm, &reqs[1]), "MPI_Irecv(rl)");
                check_mpi(MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE), "MPI_Waitall(left)");
                to_left = recv_rl.data();
            }
            if (dr) {
                check_mpi(MPI_Irecv(recv_lr.data(), static_cast<int>(bytes), MPI_BYTE, left.owning_rank, tag_lr,
                                    comm, &reqs[0]), "MPI_Irecv(lr)");
                check_mpi(MPI_Isend(send_rl.data(), static_cast<int>(bytes), MPI_BYTE, left.owning_rank, tag_rl,
                                    comm, &reqs[1]), "MPI_Isend(rl)");
                check_mpi(MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE), "MPI_Waitall(right)");
                to_right = recv_lr.data();
            }
        }

        if (dr) check_cl(clEnqueueWriteBuffer(dr->queue, buf(dr), CL_TRUE, right_halo, bytes, to_right, 0, nullptr,
                                              nullptr), "write halo lr");
        if (dl) check_cl(clEnqueueWriteBuffer(dl->queue, buf(dl), CL_TRUE, left_halo, bytes, to_left, 0, nullptr,
                                              nullptr), "write halo rl");
    }
}

// Every rank fills its own slices of a zeroed image; a bitwise-OR reduction
// assembles the full array on rank 0.
template <typename T>
std::vector<T> gather(const std::vector<Partition>& parts, int rank, MPI_Comm comm, Devices& devs, bool use_in,
                      std::size_t total) {
    std::vector<T> local(total, T(0));
    for (const Partition& p : parts) {
        if (p.owning_rank != rank || p.count == 0) continue;
        OclDevice& d = devs.list[p.local_index];
        check_cl(clEnqueueReadBuffer(d.queue, use_in ? d.in : d.out, CL_TRUE, p.offset * sizeof(T),
                                     p.count * sizeof(T), local.data() + p.offset, 0, nullptr, nullptr),
                 "read gather");
    }

    std::vector<T> result(rank == 0 ? total : 0);
    unsigned char* send = reinterpret_cast<unsigned char*>(local.data());
    unsigned char* recv = reinterpret_cast<unsigned char*>(result.data());
    const std::size_t bytes = total * sizeof(T);
    for (std::size_t off = 0; off < bytes;) {
        const int chunk = static_cast<int>(std::min<std::size_t>(bytes - off, INT_MAX));
        check_mpi(MPI_Reduce(send + off, rank == 0 ? recv + off : nullptr, chunk, MPI_BYTE, MPI_BOR, 0, comm),
                  "MPI_Reduce(gather)");
        off += static_cast<std::size_t>(chunk);
    }
    return result;
}

template <typename T>
RunResult run_typed(const Options& o) {
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    Devices devs;
    discover(devs, o.device, read_file(o.kernel_file), kernel_entry_point(o.kernel), o.build_options());

    const std::size_t total = o.elements();
    const std::vector<Partition> parts = equal_partitions(total, static_cast<int>(devs.list.size()), rank, comm);
    const std::size_t halo = o.is_stencil() ? static_cast<std::size_t>(o.nx) : 0;
    const bool use_halo = halo > 0 && parts.size() > 1;

    // Every device holds a full-size copy; it only updates its own slice.
    const std::vector<T> init = initial_data<T>(o);
    const std::vector<T> zeros(o.is_stencil() ? 0 : total, T(0));  // only the map needs it
    for (OclDevice& d : devs.list) {
        cl_int err = CL_SUCCESS;
        d.in = clCreateBuffer(d.ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, total * sizeof(T),
                              const_cast<T*>(init.data()), &err);
        check_cl(err, "clCreateBuffer(in)");
        d.out = clCreateBuffer(d.ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, total * sizeof(T),
                               const_cast<T*>(o.is_stencil() ? init.data() : zeros.data()), &err);
        check_cl(err, "clCreateBuffer(out)");
    }

    auto set_args = [&](OclDevice& d, bool in_is_a) {
        cl_mem in = in_is_a ? d.in : d.out;
        cl_mem out = in_is_a ? d.out : d.in;
        const int n = static_cast<int>(total);
        check_cl(clSetKernelArg(d.kernel, 0, sizeof(cl_mem), &in), "arg0");
        check_cl(clSetKernelArg(d.kernel, 1, sizeof(cl_mem), &out), "arg1");
        switch (o.kernel) {
            case Kernel::ocean:
                check_cl(clSetKernelArg(d.kernel, 2, sizeof(int), &o.nx), "arg2");
                check_cl(clSetKernelArg(d.kernel, 3, sizeof(int), &o.ny), "arg3");
                break;
            case Kernel::lu:
                check_cl(clSetKernelArg(d.kernel, 2, sizeof(int), &o.nx), "arg2");
                check_cl(clSetKernelArg(d.kernel, 3, sizeof(int), &o.ny), "arg3");
                check_cl(clSetKernelArg(d.kernel, 4, sizeof(int), &o.flop_repeat), "arg4");
                break;
            case Kernel::map:
                check_cl(clSetKernelArg(d.kernel, 2, sizeof(int), &n), "arg2");
                check_cl(clSetKernelArg(d.kernel, 3, sizeof(int), &o.flop_repeat), "arg3");
                break;
        }
    };

    // Make sure the initial data is on the devices before the clock starts.
    for (OclDevice& d : devs.list) {
        cl_mem objs[2] = {d.in, d.out};
        check_cl(clEnqueueMigrateMemObjects(d.queue, 2, objs, 0, 0, nullptr, nullptr), "clEnqueueMigrateMemObjects");
        check_cl(clFinish(d.queue), "clFinish");
    }

    MPI_Barrier(comm);
    const auto t0 = std::chrono::steady_clock::now();

    // Stencils ping-pong between the buffers; the map always reads "in".
    bool in_is_a = true;
    std::vector<cl_event> events;
    for (int it = 0; it < o.iterations; ++it) {
        for (const Partition& p : parts) {
            if (p.owning_rank == rank) set_args(devs.list[p.local_index], in_is_a);
        }

        if (!use_halo) {
            for (const Partition& p : parts) {
                if (p.owning_rank == rank) enqueue(devs.list[p.local_index], p.offset, p.count, events);
            }
            wait_and_release(events);
        } else {
            // Interior while the halos travel, then the borders.
            for (const Partition& p : parts) {
                if (p.owning_rank != rank || p.count <= 2 * halo) continue;
                enqueue(devs.list[p.local_index], p.offset + halo, p.count - 2 * halo, events);
            }
            exchange_halos<T>(parts, rank, comm, devs, in_is_a, halo);
            wait_and_release(events);

            for (const Partition& p : parts) {
                if (p.owning_rank != rank) continue;
                const std::size_t left = std::min(halo, p.count);
                const std::size_t right = (p.count > halo) ? std::min(halo, p.count - left) : 0;
                enqueue(devs.list[p.local_index], p.offset, left, events);
                enqueue(devs.list[p.local_index], p.offset + p.count - right, right, events);
            }
            wait_and_release(events);
        }

        if (o.is_stencil()) in_is_a = !in_is_a;
    }

    MPI_Barrier(comm);
    const auto t1 = std::chrono::steady_clock::now();

    // Same measure as HARMONY's stop_timing(): the slowest rank.
    const double local_seconds = std::chrono::duration<double>(t1 - t0).count();
    double seconds = local_seconds;
    check_mpi(MPI_Allreduce(&local_seconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, comm), "MPI_Allreduce(time)");

    // After the loop the latest data is in the buffer that would be read next.
    const bool result_in_a = o.is_stencil() ? in_is_a : false;
    RunResult result;
    result.seconds = seconds;
    const std::vector<T> output = gather<T>(parts, rank, comm, devs, result_in_a, total);
    result.output.assign(output.begin(), output.end());
    return result;
}

} // namespace

RunResult run_mpiocl(const Options& o) {
    return o.is_double() ? run_typed<double>(o) : run_typed<float>(o);
}

} // namespace stencil2d
