// Standalone MPI + OpenCL topology probe emitting cluster PCIe and MPI
// latency/bandwidth metrics in key=value stdout format and optional JSON.

#include "harmony/topo_metrics_io.hpp"
#include "harmony/types.hpp"

#include <CL/cl.h>
#include <mpi.h>

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

template <typename Handle, typename ReleaseFn>
class ScopedClHandle {
public:
    ScopedClHandle() = default;
    explicit ScopedClHandle(Handle handle, ReleaseFn release = ReleaseFn{}) noexcept
        : handle_(handle), release_(std::move(release)) {}
    ~ScopedClHandle() { reset(); }

    ScopedClHandle(const ScopedClHandle&) = delete;
    ScopedClHandle& operator=(const ScopedClHandle&) = delete;

    ScopedClHandle(ScopedClHandle&& other) noexcept
        : handle_(std::exchange(other.handle_, Handle{})),
          release_(std::move(other.release_)) {}

    ScopedClHandle& operator=(ScopedClHandle&& other) noexcept {
        if (this != &other) {
            reset();
            handle_ = std::exchange(other.handle_, Handle{});
            release_ = std::move(other.release_);
        }
        return *this;
    }

    Handle get() const noexcept { return handle_; }
    explicit operator bool() const noexcept { return handle_ != Handle{}; }

    void reset(Handle next = Handle{}) noexcept {
        if (handle_ != Handle{}) {
            release_(handle_);
        }
        handle_ = next;
    }

private:
    Handle handle_{};
    ReleaseFn release_{};
};

struct ClContextRelease {
    void operator()(cl_context ctx) const noexcept { clReleaseContext(ctx); }
};

struct ClQueueRelease {
    void operator()(cl_command_queue queue) const noexcept { clReleaseCommandQueue(queue); }
};

struct ClMemRelease {
    void operator()(cl_mem mem) const noexcept { clReleaseMemObject(mem); }
};

using ScopedClContext = ScopedClHandle<cl_context, ClContextRelease>;
using ScopedClQueue = ScopedClHandle<cl_command_queue, ClQueueRelease>;
using ScopedClMem = ScopedClHandle<cl_mem, ClMemRelease>;

struct LocalDeviceProbe {
    double pcie_latency_ns{-1.0};
    double pcie_bandwidth_gbps{-1.0};
};

void checked_mpi(int result, const char* operation) {
    if (result == MPI_SUCCESS) return;
    char message[MPI_MAX_ERROR_STRING] = {};
    int length = 0;
    MPI_Error_string(result, message, &length);
    std::cerr << operation << " failed: " << std::string(message, static_cast<std::size_t>(length)) << "\n";
    MPI_Abort(MPI_COMM_WORLD, result);
}

[[noreturn]] void abort_probe(const char* reason) {
    std::cerr << "Error: " << reason << "\n";
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::exit(1);
}

ScopedClQueue create_device_queue(cl_context ctx, cl_device_id dev, cl_int& err) {
#if CL_TARGET_OPENCL_VERSION >= 200
    const cl_queue_properties qprops[] = {0};
    return ScopedClQueue(clCreateCommandQueueWithProperties(ctx, dev, qprops, &err));
#else
    return ScopedClQueue(clCreateCommandQueue(ctx, dev, 0, &err));
#endif
}

LocalDeviceProbe measure_device_pcie(cl_platform_id plat, cl_device_id dev, bool allow_synthetic, bool& created) {
    created = false;
    LocalDeviceProbe probe;

    cl_context_properties props[] = {
        CL_CONTEXT_PLATFORM,
        reinterpret_cast<cl_context_properties>(plat),
        0
    };
    cl_int c_err = CL_SUCCESS;
    ScopedClContext ctx(clCreateContext(props, 1, &dev, nullptr, nullptr, &c_err));
    if (c_err != CL_SUCCESS || !ctx) {
        return probe;
    }

    ScopedClQueue queue = create_device_queue(ctx.get(), dev, c_err);
    if (c_err != CL_SUCCESS || !queue) {
        return probe;
    }
    created = true;

    // Benchmark small buffer (512 B) round-trip latency.
    const std::size_t small_size = 512;
    std::vector<char> small_host_src(small_size, 1);
    std::vector<char> small_host_dst(small_size, 0);

    ScopedClMem small_buf(clCreateBuffer(ctx.get(), CL_MEM_READ_WRITE, small_size, nullptr, &c_err));
    if (c_err == CL_SUCCESS && small_buf) {
        bool measured = true;
        for (int w = 0; w < 3; ++w) {
            measured = measured &&
                (clEnqueueWriteBuffer(queue.get(), small_buf.get(), CL_TRUE, 0, small_size,
                                      small_host_src.data(), 0, nullptr, nullptr) == CL_SUCCESS);
            measured = measured &&
                (clEnqueueReadBuffer(queue.get(), small_buf.get(), CL_TRUE, 0, small_size,
                                     small_host_dst.data(), 0, nullptr, nullptr) == CL_SUCCESS);
        }
        const int iters = 20;
        const auto t0 = std::chrono::high_resolution_clock::now();
        for (int it = 0; it < iters; ++it) {
            measured = measured &&
                (clEnqueueWriteBuffer(queue.get(), small_buf.get(), CL_FALSE, 0, small_size,
                                      small_host_src.data(), 0, nullptr, nullptr) == CL_SUCCESS);
            measured = measured &&
                (clEnqueueReadBuffer(queue.get(), small_buf.get(), CL_FALSE, 0, small_size,
                                     small_host_dst.data(), 0, nullptr, nullptr) == CL_SUCCESS);
        }
        measured = measured && (clFinish(queue.get()) == CL_SUCCESS);
        const auto t1 = std::chrono::high_resolution_clock::now();
        const double total_ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
        if (measured && total_ns > 0.0) {
            probe.pcie_latency_ns = total_ns / static_cast<double>(iters);
        }
    }

    // Benchmark large buffer (64 MiB) round-trip bandwidth.
    const std::size_t large_size = 64u * 1024u * 1024u;
    std::vector<char> large_host(large_size, 2);
    ScopedClMem large_buf(clCreateBuffer(ctx.get(), CL_MEM_READ_WRITE, large_size, nullptr, &c_err));
    if (c_err == CL_SUCCESS && large_buf) {
        bool measured = true;
        measured = measured &&
            (clEnqueueWriteBuffer(queue.get(), large_buf.get(), CL_TRUE, 0, large_size,
                                  large_host.data(), 0, nullptr, nullptr) == CL_SUCCESS);
        measured = measured &&
            (clEnqueueReadBuffer(queue.get(), large_buf.get(), CL_TRUE, 0, large_size,
                                 large_host.data(), 0, nullptr, nullptr) == CL_SUCCESS);

        const int bw_iters = 3;
        const auto t0 = std::chrono::high_resolution_clock::now();
        for (int it = 0; it < bw_iters; ++it) {
            measured = measured &&
                (clEnqueueWriteBuffer(queue.get(), large_buf.get(), CL_FALSE, 0, large_size,
                                      large_host.data(), 0, nullptr, nullptr) == CL_SUCCESS);
            measured = measured &&
                (clEnqueueReadBuffer(queue.get(), large_buf.get(), CL_FALSE, 0, large_size,
                                     large_host.data(), 0, nullptr, nullptr) == CL_SUCCESS);
        }
        measured = measured && (clFinish(queue.get()) == CL_SUCCESS);
        const auto t1 = std::chrono::high_resolution_clock::now();
        const double total_sec = std::chrono::duration<double>(t1 - t0).count();
        if (measured && total_sec > 0.0) {
            const double bytes_transferred = 2.0 * static_cast<double>(large_size) * static_cast<double>(bw_iters);
            probe.pcie_bandwidth_gbps = bytes_transferred / (total_sec * 1e9);
        }
    }

    if (probe.pcie_latency_ns < 0.0 || probe.pcie_bandwidth_gbps < 0.0) {
        if (!allow_synthetic) {
            abort_probe("Measurement failed for a real GPU.");
        }
        probe.pcie_latency_ns = 4500.0;
        probe.pcie_bandwidth_gbps = 15.75;
    }

    return probe;
}

std::vector<LocalDeviceProbe> probe_opencl_devices(bool allow_synthetic) {
    std::vector<LocalDeviceProbe> devices;

    cl_uint num_platforms = 0;
    cl_int err = clGetPlatformIDs(0, nullptr, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) {
        if (!allow_synthetic) {
            abort_probe("clGetPlatformIDs failed or returned 0 platforms.");
        }
        devices.push_back(LocalDeviceProbe{4500.0, 15.75});
        return devices;
    }

    std::vector<cl_platform_id> platforms(num_platforms);
    err = clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
    if (err != CL_SUCCESS) {
        if (!allow_synthetic) {
            abort_probe("clGetPlatformIDs failed to get platform IDs.");
        }
        devices.push_back(LocalDeviceProbe{4500.0, 15.75});
        return devices;
    }

    for (cl_platform_id plat : platforms) {
        cl_uint num_devs = 0;
        err = clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devs);
        if (err == CL_DEVICE_NOT_FOUND || num_devs == 0) continue;
        if (err != CL_SUCCESS) {
            if (!allow_synthetic) {
                abort_probe("GPU discovery failed.");
            }
            continue;
        }

        std::vector<cl_device_id> devs(num_devs);
        err = clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, num_devs, devs.data(), nullptr);
        if (err != CL_SUCCESS) {
            if (!allow_synthetic) {
                abort_probe("GPU device listing failed.");
            }
            continue;
        }

        for (cl_device_id dev : devs) {
            bool created = false;
            const LocalDeviceProbe probe = measure_device_pcie(plat, dev, allow_synthetic, created);
            if (created) {
                devices.push_back(probe);
            }
        }
    }

    if (devices.empty()) {
        if (!allow_synthetic) {
            abort_probe("No OpenCL devices successfully probed.");
        }
        devices.push_back(LocalDeviceProbe{4500.0, 15.75});
    }
    return devices;
}

} // namespace

int main(int argc, char** argv) {
    int mpi_init = 0;
    MPI_Initialized(&mpi_init);
    if (!mpi_init) {
        const int init_rc = MPI_Init(&argc, &argv);
        if (init_rc != MPI_SUCCESS) {
            std::cerr << "Error: MPI_Init failed\n";
            return 1;
        }
    }

    int rank = 0;
    int size = 1;
    checked_mpi(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "MPI_Comm_rank");
    checked_mpi(MPI_Comm_size(MPI_COMM_WORLD, &size), "MPI_Comm_size");

    std::string output_json_path;
    bool allow_synthetic = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            if (rank == 0) {
                std::cout << "Usage: topo_probe [--output-json <path> | <path>] [--synthetic]\n";
            }
            MPI_Finalize();
            return 0;
        }
        if (arg == "--output-json") {
            if (i + 1 >= argc) {
                if (rank == 0) {
                    std::cerr << "Error: --output-json requires a file path argument\n";
                }
                MPI_Finalize();
                return 1;
            }
            output_json_path = argv[++i];
        } else if (arg == "--synthetic") {
            allow_synthetic = true;
        } else if (!arg.empty() && arg[0] != '-' && output_json_path.empty()) {
            output_json_path = arg;
        } else {
            if (rank == 0) {
                std::cerr << "Error: unknown or unexpected argument '" << arg << "'\n";
            }
            MPI_Finalize();
            return 1;
        }
    }

    // 1. Measure PCIe metrics for local OpenCL devices.
    const std::vector<LocalDeviceProbe> local_probes = probe_opencl_devices(allow_synthetic);
    const int local_dev_count = static_cast<int>(local_probes.size());

    std::vector<int> all_dev_counts(static_cast<std::size_t>(size), 0);
    checked_mpi(MPI_Allgather(&local_dev_count, 1, MPI_INT, all_dev_counts.data(), 1, MPI_INT, MPI_COMM_WORLD),
                "MPI_Allgather");

    int total_devices = 0;
    std::vector<int> dev_displs(static_cast<std::size_t>(size), 0);
    for (int r = 0; r < size; ++r) {
        dev_displs[static_cast<std::size_t>(r)] = total_devices;
        total_devices += all_dev_counts[static_cast<std::size_t>(r)];
    }

    std::vector<double> local_pcie_lat(static_cast<std::size_t>(local_dev_count));
    std::vector<double> local_pcie_bw(static_cast<std::size_t>(local_dev_count));
    for (int d = 0; d < local_dev_count; ++d) {
        local_pcie_lat[static_cast<std::size_t>(d)] = local_probes[static_cast<std::size_t>(d)].pcie_latency_ns;
        local_pcie_bw[static_cast<std::size_t>(d)] = local_probes[static_cast<std::size_t>(d)].pcie_bandwidth_gbps;
    }

    std::vector<double> all_pcie_lat(static_cast<std::size_t>(total_devices), 0.0);
    std::vector<double> all_pcie_bw(static_cast<std::size_t>(total_devices), 0.0);

    checked_mpi(MPI_Allgatherv(local_pcie_lat.data(), local_dev_count, MPI_DOUBLE,
                               all_pcie_lat.data(), all_dev_counts.data(), dev_displs.data(),
                               MPI_DOUBLE, MPI_COMM_WORLD),
                "MPI_Allgatherv(pcie_latency_ns)");
    checked_mpi(MPI_Allgatherv(local_pcie_bw.data(), local_dev_count, MPI_DOUBLE,
                               all_pcie_bw.data(), all_dev_counts.data(), dev_displs.data(),
                               MPI_DOUBLE, MPI_COMM_WORLD),
                "MPI_Allgatherv(pcie_bandwidth_gbps)");

    // 2. Measure all-to-all MPI point-to-point latency & bandwidth.
    std::vector<double> rank_to_rank_lat(static_cast<std::size_t>(size) * static_cast<std::size_t>(size), 0.0);
    std::vector<double> rank_to_rank_bw(static_cast<std::size_t>(size) * static_cast<std::size_t>(size), 0.0);

    if (size > 1) {
        std::vector<double> local_pairs_lat(static_cast<std::size_t>(size), 0.0);
        std::vector<double> local_pairs_bw(static_cast<std::size_t>(size), 0.0);

        const int small_bytes = 64;
        std::vector<char> s_send(static_cast<std::size_t>(small_bytes), 1);
        std::vector<char> s_recv(static_cast<std::size_t>(small_bytes), 0);

        const int large_bytes = 1024 * 1024;
        std::vector<char> l_send(static_cast<std::size_t>(large_bytes), 2);
        std::vector<char> l_recv(static_cast<std::size_t>(large_bytes), 0);

        for (int step = 1; step < size; ++step) {
            const int next_rank = (rank + step) % size;
            const int prev_rank = (rank - step + size) % size;

            for (int w = 0; w < 5; ++w) {
                checked_mpi(MPI_Sendrecv(s_send.data(), small_bytes, MPI_CHAR, next_rank, 100,
                                         s_recv.data(), small_bytes, MPI_CHAR, prev_rank, 100,
                                         MPI_COMM_WORLD, MPI_STATUS_IGNORE),
                            "MPI_Sendrecv(lat warmup)");
            }

            const int mpi_lat_iters = 30;
            checked_mpi(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier");
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int it = 0; it < mpi_lat_iters; ++it) {
                checked_mpi(MPI_Sendrecv(s_send.data(), small_bytes, MPI_CHAR, next_rank, 101,
                                         s_recv.data(), small_bytes, MPI_CHAR, prev_rank, 101,
                                         MPI_COMM_WORLD, MPI_STATUS_IGNORE),
                            "MPI_Sendrecv(lat)");
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            const double elapsed_ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
            const double neighbor_lat_ns = elapsed_ns / static_cast<double>(mpi_lat_iters);

            checked_mpi(MPI_Sendrecv(l_send.data(), large_bytes, MPI_CHAR, next_rank, 200,
                                     l_recv.data(), large_bytes, MPI_CHAR, prev_rank, 200,
                                     MPI_COMM_WORLD, MPI_STATUS_IGNORE),
                        "MPI_Sendrecv(bw warmup)");

            const int mpi_bw_iters = 5;
            checked_mpi(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier");
            t0 = std::chrono::high_resolution_clock::now();
            for (int it = 0; it < mpi_bw_iters; ++it) {
                checked_mpi(MPI_Sendrecv(l_send.data(), large_bytes, MPI_CHAR, next_rank, 201,
                                         l_recv.data(), large_bytes, MPI_CHAR, prev_rank, 201,
                                         MPI_COMM_WORLD, MPI_STATUS_IGNORE),
                            "MPI_Sendrecv(bw)");
            }
            t1 = std::chrono::high_resolution_clock::now();
            const double elapsed_sec = std::chrono::duration<double>(t1 - t0).count();
            double neighbor_bw_gbps = 0.0;
            if (elapsed_sec > 0.0) {
                neighbor_bw_gbps =
                    (static_cast<double>(large_bytes) * static_cast<double>(mpi_bw_iters)) / (elapsed_sec * 1e9);
            }

            local_pairs_lat[static_cast<std::size_t>(next_rank)] = neighbor_lat_ns;
            local_pairs_bw[static_cast<std::size_t>(next_rank)] = neighbor_bw_gbps;
        }

        checked_mpi(MPI_Allgather(local_pairs_lat.data(), size, MPI_DOUBLE,
                                  rank_to_rank_lat.data(), size, MPI_DOUBLE, MPI_COMM_WORLD),
                    "MPI_Allgather(mpi_latency_ns)");
        checked_mpi(MPI_Allgather(local_pairs_bw.data(), size, MPI_DOUBLE,
                                  rank_to_rank_bw.data(), size, MPI_DOUBLE, MPI_COMM_WORLD),
                    "MPI_Allgather(mpi_bandwidth_gbps)");
    }

    // 3. Populate harmony::TopoMetrics.
    harmony::TopoMetrics metrics;
    metrics.pcie_latency_ns = std::move(all_pcie_lat);
    metrics.pcie_bandwidth_gbps = std::move(all_pcie_bw);
    metrics.mpi_latency_ns = std::move(rank_to_rank_lat);
    metrics.mpi_bandwidth_gbps = std::move(rank_to_rank_bw);
    metrics.synthetic = allow_synthetic;

    for (int src = 0; src < size; ++src) {
        for (int dst = 0; dst < size; ++dst) {
            if (src == dst) continue;
            const std::size_t idx =
                static_cast<std::size_t>(src) * static_cast<std::size_t>(size) + static_cast<std::size_t>(dst);
            if (metrics.mpi_latency_ns[idx] <= 0.0 || metrics.mpi_bandwidth_gbps[idx] <= 0.0) {
                if (!allow_synthetic) {
                    abort_probe("MPI route measurement is missing.");
                }
                metrics.mpi_latency_ns[idx] = 1500.0;
                metrics.mpi_bandwidth_gbps[idx] = 3.5;
            }
        }
    }

    if (allow_synthetic) {
        metrics.numa_distance = {10};
        metrics.device_numa_node.assign(static_cast<std::size_t>(total_devices), 0);
        metrics.memory_contention_factor.assign(static_cast<std::size_t>(total_devices), 1.0);
        metrics.thermal_tdp_watts.assign(static_cast<std::size_t>(total_devices), 250.0);
        metrics.current_power_watts.assign(static_cast<std::size_t>(total_devices), 120.0);
    }

    // 4. Emit machine-readable key=value output and optional JSON on rank 0.
    if (rank == 0) {
        std::cout << "# Topology Probe Measurements\n";
        std::cout << "mode=" << (metrics.synthetic ? "synthetic" : "measured") << "\n";
        std::cout << "total_devices=" << total_devices << "\n";
        std::cout << "mpi_ranks=" << size << "\n";
        std::cout << "rank_device_counts=";
        for (int r = 0; r < size; ++r) {
            std::cout << (r == 0 ? "" : ",") << all_dev_counts[static_cast<std::size_t>(r)];
        }
        std::cout << "\n";
        std::cout << std::fixed << std::setprecision(3);
        for (int d = 0; d < total_devices; ++d) {
            const std::size_t di = static_cast<std::size_t>(d);
            std::cout << "pcie_latency_ns[" << d << "]=" << metrics.pcie_latency_ns[di] << "\n";
            std::cout << "pcie_bandwidth_gbps[" << d << "]=" << metrics.pcie_bandwidth_gbps[di] << "\n";
        }
        for (int s = 0; s < size; ++s) {
            for (int d = 0; d < size; ++d) {
                const std::size_t idx =
                    static_cast<std::size_t>(s) * static_cast<std::size_t>(size) + static_cast<std::size_t>(d);
                std::cout << "mpi_latency_ns[" << s << "][" << d << "]=" << metrics.mpi_latency_ns[idx] << "\n";
                std::cout << "mpi_bandwidth_gbps[" << s << "][" << d << "]=" << metrics.mpi_bandwidth_gbps[idx] << "\n";
            }
        }

        if (!output_json_path.empty()) {
            try {
                harmony::save_topo_metrics(metrics, output_json_path);
                std::cout << "saved_json=" << output_json_path << "\n";
            } catch (const harmony::Error& e) {
                std::cerr << "Error: " << e.what() << "\n";
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
        }
        std::cout.flush();
    }

    checked_mpi(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier");
    MPI_Finalize();
    return 0;
}
