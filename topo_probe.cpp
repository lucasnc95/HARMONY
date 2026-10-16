#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 300
#endif
#ifndef OMPI_SKIP_MPICXX
#define OMPI_SKIP_MPICXX 1
#endif
#ifndef MPICH_SKIP_MPICXX
#define MPICH_SKIP_MPICXX 1
#endif

#include "dcl/types.hpp"
#include "dcl/topo_metrics_io.hpp"
#include <CL/cl.h>
#include <mpi.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct LocalDeviceProbe {
    double pcie_latency_ns{-1.0};
    double pcie_bandwidth_gbps{-1.0};
    int numa_node{-1};
};

std::vector<LocalDeviceProbe> probe_opencl_devices(bool allow_synthetic) {
    std::vector<LocalDeviceProbe> devices;

    cl_uint num_platforms = 0;
    cl_int err = clGetPlatformIDs(0, nullptr, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) {
        if (!allow_synthetic) {
            std::cerr << "Error: clGetPlatformIDs failed or returned 0 platforms." << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
            exit(1);
        }
        // Fallback to 1 synthetic device
        devices.push_back(LocalDeviceProbe{4500.0, 15.75, 0});
        return devices;
    }

    std::vector<cl_platform_id> platforms(num_platforms);
    err = clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
    if (err != CL_SUCCESS) {
        if (!allow_synthetic) {
            std::cerr << "Error: clGetPlatformIDs failed to get platform IDs." << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
            exit(1);
        }
        devices.push_back(LocalDeviceProbe{4500.0, 15.75, 0});
        return devices;
    }

    for (cl_platform_id plat : platforms) {
        cl_uint num_devs = 0;
        err = clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devs);
        if (err == CL_DEVICE_NOT_FOUND) continue;
        if (err != CL_SUCCESS) {
            std::cerr << "Error: GPU discovery failed." << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
            exit(1);
        }
        if (num_devs == 0) continue;

        std::vector<cl_device_id> devs(num_devs);
        err = clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, num_devs, devs.data(), nullptr);
        if (err != CL_SUCCESS) {
            std::cerr << "Error: GPU device listing failed." << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
            exit(1);
        }

        for (cl_device_id dev : devs) {
            LocalDeviceProbe probe;
            cl_context_properties props[] = {
                CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(plat),
                0
            };
            cl_int c_err = CL_SUCCESS;
            cl_context ctx = clCreateContext(props, 1, &dev, nullptr, nullptr, &c_err);
            if (c_err != CL_SUCCESS || ctx == nullptr) {
                continue;
            }

            cl_queue_properties qprops[] = {0};
            cl_command_queue queue = clCreateCommandQueueWithProperties(ctx, dev, qprops, &c_err);
            if (c_err != CL_SUCCESS || queue == nullptr) {
                clReleaseContext(ctx);
                continue;
            }

            // Benchmark small buffer (512 B) round-trip latency
            const std::size_t small_size = 512;
            std::vector<char> small_host_src(small_size, 1);
            std::vector<char> small_host_dst(small_size, 0);

            cl_mem small_buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE, small_size, nullptr, &c_err);
            if (c_err == CL_SUCCESS && small_buf != nullptr) {
                bool measured = true;
                // Warmup
                for (int w = 0; w < 3; ++w) {
                    measured &= clEnqueueWriteBuffer(queue, small_buf, CL_TRUE, 0, small_size, small_host_src.data(), 0, nullptr, nullptr) == CL_SUCCESS;
                    measured &= clEnqueueReadBuffer(queue, small_buf, CL_TRUE, 0, small_size, small_host_dst.data(), 0, nullptr, nullptr) == CL_SUCCESS;
                }
                const int iters = 20;
                auto t0 = std::chrono::high_resolution_clock::now();
                for (int it = 0; it < iters; ++it) {
                    measured &= clEnqueueWriteBuffer(queue, small_buf, CL_FALSE, 0, small_size, small_host_src.data(), 0, nullptr, nullptr) == CL_SUCCESS;
                    measured &= clEnqueueReadBuffer(queue, small_buf, CL_FALSE, 0, small_size, small_host_dst.data(), 0, nullptr, nullptr) == CL_SUCCESS;
                }
                measured &= clFinish(queue) == CL_SUCCESS;
                auto t1 = std::chrono::high_resolution_clock::now();
                double total_ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
                if (measured && total_ns > 0.0) {
                    probe.pcie_latency_ns = total_ns / static_cast<double>(iters);
                }
                clReleaseMemObject(small_buf);
            }

            // Benchmark large buffer (64 MiB) round-trip bandwidth
            const std::size_t large_size = 64 * 1024 * 1024;
            std::vector<char> large_host(large_size, 2);
            cl_mem large_buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE, large_size, nullptr, &c_err);
            if (c_err == CL_SUCCESS && large_buf != nullptr) {
                bool measured = true;
                // Warmup
                measured &= clEnqueueWriteBuffer(queue, large_buf, CL_TRUE, 0, large_size, large_host.data(), 0, nullptr, nullptr) == CL_SUCCESS;
                measured &= clEnqueueReadBuffer(queue, large_buf, CL_TRUE, 0, large_size, large_host.data(), 0, nullptr, nullptr) == CL_SUCCESS;

                const int bw_iters = 3;
                auto t0 = std::chrono::high_resolution_clock::now();
                for (int it = 0; it < bw_iters; ++it) {
                    measured &= clEnqueueWriteBuffer(queue, large_buf, CL_FALSE, 0, large_size, large_host.data(), 0, nullptr, nullptr) == CL_SUCCESS;
                    measured &= clEnqueueReadBuffer(queue, large_buf, CL_FALSE, 0, large_size, large_host.data(), 0, nullptr, nullptr) == CL_SUCCESS;
                }
                measured &= clFinish(queue) == CL_SUCCESS;
                auto t1 = std::chrono::high_resolution_clock::now();
                double total_sec = std::chrono::duration<double>(t1 - t0).count();
                if (measured && total_sec > 0.0) {
                    double bytes_transferred = 2.0 * static_cast<double>(large_size) * static_cast<double>(bw_iters);
                    probe.pcie_bandwidth_gbps = bytes_transferred / (total_sec * 1e9);
                }
                clReleaseMemObject(large_buf);
            }

            clReleaseCommandQueue(queue);
            clReleaseContext(ctx);

            if (probe.pcie_latency_ns < 0.0 || probe.pcie_bandwidth_gbps < 0.0) {
                if (!allow_synthetic) {
                    std::cerr << "Error: Measurement failed for a real GPU." << std::endl;
                    MPI_Abort(MPI_COMM_WORLD, 1);
                    exit(1);
                }
                probe.pcie_latency_ns = 4500.0;
                probe.pcie_bandwidth_gbps = 15.75;
            }
            if (allow_synthetic) probe.numa_node = 0;

            devices.push_back(probe);
        }
    }

    if (devices.empty()) {
        if (!allow_synthetic) {
            std::cerr << "Error: No OpenCL devices successfully probed." << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
            exit(1);
        }
        devices.push_back(LocalDeviceProbe{4500.0, 15.75, 0});
    }
    return devices;
}

} // namespace

void checked_mpi(int result, const char* operation) {
    if (result == MPI_SUCCESS) return;
    char message[MPI_MAX_ERROR_STRING] = {};
    int length = 0;
    MPI_Error_string(result, message, &length);
    std::cerr << operation << " failed: " << std::string(message, length) << std::endl;
    MPI_Abort(MPI_COMM_WORLD, result);
}

int main(int argc, char** argv) {
    int mpi_init = 0;
    MPI_Initialized(&mpi_init);
    if (!mpi_init) {
        MPI_Init(&argc, &argv);
    }

    int rank = 0;
    int size = 1;
    checked_mpi(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "MPI_Comm_rank");
    checked_mpi(MPI_Comm_size(MPI_COMM_WORLD, &size), "MPI_Comm_size");

    std::string output_json_path;
    bool allow_synthetic = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--output-json" && i + 1 < argc) {
            output_json_path = argv[i + 1];
            ++i;
        } else if (std::string(argv[i]) == "--synthetic") {
            allow_synthetic = true;
        }
    }

    // 1. Measure PCIe metrics for local OpenCL devices
    std::vector<LocalDeviceProbe> local_probes = probe_opencl_devices(allow_synthetic);
    int local_dev_count = static_cast<int>(local_probes.size());

    std::vector<int> all_dev_counts(static_cast<std::size_t>(size), 0);
    checked_mpi(MPI_Allgather(&local_dev_count, 1, MPI_INT, all_dev_counts.data(), 1, MPI_INT, MPI_COMM_WORLD), "MPI_Allgather");

    int total_devices = 0;
    std::vector<int> dev_displs(static_cast<std::size_t>(size), 0);
    for (int r = 0; r < size; ++r) {
        dev_displs[static_cast<std::size_t>(r)] = total_devices;
        total_devices += all_dev_counts[static_cast<std::size_t>(r)];
    }

    // Gather PCIe metrics across all ranks
    std::vector<double> local_pcie_lat(static_cast<std::size_t>(local_dev_count));
    std::vector<double> local_pcie_bw(static_cast<std::size_t>(local_dev_count));
    for (int d = 0; d < local_dev_count; ++d) {
        local_pcie_lat[static_cast<std::size_t>(d)] = local_probes[static_cast<std::size_t>(d)].pcie_latency_ns;
        local_pcie_bw[static_cast<std::size_t>(d)] = local_probes[static_cast<std::size_t>(d)].pcie_bandwidth_gbps;
    }

    std::vector<double> all_pcie_lat(static_cast<std::size_t>(total_devices), 0.0);
    std::vector<double> all_pcie_bw(static_cast<std::size_t>(total_devices), 0.0);

    checked_mpi(MPI_Allgatherv(local_pcie_lat.data(), local_dev_count, MPI_DOUBLE,
                   all_pcie_lat.data(), all_dev_counts.data(), dev_displs.data(), MPI_DOUBLE, MPI_COMM_WORLD), "MPI_Allgatherv");
    checked_mpi(MPI_Allgatherv(local_pcie_bw.data(), local_dev_count, MPI_DOUBLE,
                   all_pcie_bw.data(), all_dev_counts.data(), dev_displs.data(), MPI_DOUBLE, MPI_COMM_WORLD), "MPI_Allgatherv");

    // 2. Measure MPI point-to-point latency & bandwidth
    std::vector<double> rank_to_rank_lat(static_cast<std::size_t>(size * size), 0.0);
    std::vector<double> rank_to_rank_bw(static_cast<std::size_t>(size * size), 0.0);

    if (size > 1) {
        std::vector<double> local_pairs_lat(static_cast<std::size_t>(size), 0.0);
        std::vector<double> local_pairs_bw(static_cast<std::size_t>(size), 0.0);

        // Small message (64 B) latency measurement
        const int small_bytes = 64;
        std::vector<char> s_send(small_bytes, 1);
        std::vector<char> s_recv(small_bytes, 0);

        // Large message (1 MiB) bandwidth measurement
        const int large_bytes = 1024 * 1024;
        std::vector<char> l_send(large_bytes, 2);
        std::vector<char> l_recv(large_bytes, 0);

        for (int step = 1; step < size; ++step) {
            int next_rank = (rank + step) % size;
            int prev_rank = (rank - step + size) % size;

            // Warmup
            for (int w = 0; w < 5; ++w) {
                checked_mpi(MPI_Sendrecv(s_send.data(), small_bytes, MPI_CHAR, next_rank, 100,
                             s_recv.data(), small_bytes, MPI_CHAR, prev_rank, 100,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE), "MPI_Sendrecv");
            }

            const int mpi_lat_iters = 30;
            checked_mpi(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier");
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int it = 0; it < mpi_lat_iters; ++it) {
                checked_mpi(MPI_Sendrecv(s_send.data(), small_bytes, MPI_CHAR, next_rank, 101,
                             s_recv.data(), small_bytes, MPI_CHAR, prev_rank, 101,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE), "MPI_Sendrecv");
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            double elapsed_ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
            double neighbor_lat_ns = (elapsed_ns / static_cast<double>(mpi_lat_iters));

            // Warmup
            checked_mpi(MPI_Sendrecv(l_send.data(), large_bytes, MPI_CHAR, next_rank, 200,
                         l_recv.data(), large_bytes, MPI_CHAR, prev_rank, 200,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE), "MPI_Sendrecv");

            const int mpi_bw_iters = 5;
            checked_mpi(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier");
            t0 = std::chrono::high_resolution_clock::now();
            for (int it = 0; it < mpi_bw_iters; ++it) {
                checked_mpi(MPI_Sendrecv(l_send.data(), large_bytes, MPI_CHAR, next_rank, 201,
                             l_recv.data(), large_bytes, MPI_CHAR, prev_rank, 201,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE), "MPI_Sendrecv");
            }
            t1 = std::chrono::high_resolution_clock::now();
            double elapsed_sec = std::chrono::duration<double>(t1 - t0).count();
            double neighbor_bw_gbps = 0.0;
            if (elapsed_sec > 0.0) {
                neighbor_bw_gbps = (static_cast<double>(large_bytes) * static_cast<double>(mpi_bw_iters)) / (elapsed_sec * 1e9);
            }

            local_pairs_lat[static_cast<std::size_t>(next_rank)] = neighbor_lat_ns;
            local_pairs_bw[static_cast<std::size_t>(next_rank)] = neighbor_bw_gbps;
        }

        checked_mpi(MPI_Allgather(local_pairs_lat.data(), size, MPI_DOUBLE,
                      rank_to_rank_lat.data(), size, MPI_DOUBLE, MPI_COMM_WORLD), "MPI_Allgather");
        checked_mpi(MPI_Allgather(local_pairs_bw.data(), size, MPI_DOUBLE,
                      rank_to_rank_bw.data(), size, MPI_DOUBLE, MPI_COMM_WORLD), "MPI_Allgather");
    }

    // 3. Store the measured rank-to-rank matrix directly.
    dcl::TopoMetrics metrics;
    metrics.pcie_latency_ns = all_pcie_lat;
    metrics.pcie_bandwidth_gbps = all_pcie_bw;
    metrics.mpi_latency_ns = rank_to_rank_lat;
    metrics.mpi_bandwidth_gbps = rank_to_rank_bw;
    metrics.synthetic = allow_synthetic;
    for (int src = 0; src < size; ++src) {
        for (int dst = 0; dst < size; ++dst) {
            if (src == dst) continue;
            const std::size_t idx = static_cast<std::size_t>(src * size + dst);
            if (metrics.mpi_latency_ns[idx] <= 0.0 ||
                metrics.mpi_bandwidth_gbps[idx] <= 0.0) {
                if (!allow_synthetic) {
                    std::cerr << "Error: MPI route measurement is missing." << std::endl;
                    MPI_Abort(MPI_COMM_WORLD, 1);
                    return 1;
                }
                metrics.mpi_latency_ns[idx] = 1500.0;
                metrics.mpi_bandwidth_gbps[idx] = 3.5;
            }
        }
    }

    // Forward-compatible fields for R2/R5
    if (allow_synthetic) {
        metrics.numa_distance = {10};
        metrics.device_numa_node.assign(static_cast<std::size_t>(total_devices), 0);
        metrics.memory_contention_factor.assign(static_cast<std::size_t>(total_devices), 1.0);
        metrics.thermal_tdp_watts.assign(static_cast<std::size_t>(total_devices), 250.0);
        metrics.current_power_watts.assign(static_cast<std::size_t>(total_devices), 120.0);
    }

    // 4. Output machine-readable key=value format on rank 0
    if (rank == 0) {
        std::cout << "# Topology Probe Measurements" << std::endl;
        std::cout << "mode=" << (metrics.synthetic ? "synthetic" : "measured") << std::endl;
        std::cout << "total_devices=" << total_devices << std::endl;
        std::cout << "mpi_ranks=" << size << std::endl;
        for (int d = 0; d < total_devices; ++d) {
            std::cout << "pcie_latency_ns[" << d << "]="
                      << std::fixed << std::setprecision(3) << metrics.pcie_latency_ns[static_cast<std::size_t>(d)]
                      << std::endl;
            std::cout << "pcie_bandwidth_gbps[" << d << "]="
                      << std::fixed << std::setprecision(3) << metrics.pcie_bandwidth_gbps[static_cast<std::size_t>(d)]
                      << std::endl;
        }
        for (int s = 0; s < size; ++s) {
            for (int d = 0; d < size; ++d) {
                std::size_t idx = static_cast<std::size_t>(s * size + d);
                std::cout << "mpi_latency_ns[" << s << "][" << d << "]="
                          << std::fixed << std::setprecision(3) << metrics.mpi_latency_ns[idx]
                          << std::endl;
                std::cout << "mpi_bandwidth_gbps[" << s << "][" << d << "]="
                          << std::fixed << std::setprecision(3) << metrics.mpi_bandwidth_gbps[idx]
                          << std::endl;
            }
        }

        if (!output_json_path.empty()) {
            dcl::save_topo_metrics(metrics, output_json_path);
            std::cout << "saved_json=" << output_json_path << std::endl;
        }
    }

    checked_mpi(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier");
    MPI_Finalize();
    return 0;
}
