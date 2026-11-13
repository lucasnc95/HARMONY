#include <mpi.h>
#include <CL/cl.h>

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

struct RawPoint {
    std::size_t volume = 0;
    double migration_time = 0.0;
};

struct OclDeviceContext {
    bool valid = false;
    int local_device_index = -1;
    int global_device_index = -1;
    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    std::string platform_name;
    std::string device_name;
};

struct GlobalDevice {
    int global_index = -1;
    int rank = -1;
    int local_index = -1;
};

static std::string Get_Platform_String(cl_platform_id platform, cl_platform_info param) {
    std::size_t size = 0;
    clGetPlatformInfo(platform, param, 0, nullptr, &size);
    std::string out(size, '\0');
    if (size > 0) {
        clGetPlatformInfo(platform, param, size, out.data(), nullptr);
        if (!out.empty() && out.back() == '\0') out.pop_back();
    }
    return out;
}

static std::string Get_Device_String(cl_device_id device, cl_device_info param) {
    std::size_t size = 0;
    clGetDeviceInfo(device, param, 0, nullptr, &size);
    std::string out(size, '\0');
    if (size > 0) {
        clGetDeviceInfo(device, param, size, out.data(), nullptr);
        if (!out.empty() && out.back() == '\0') out.pop_back();
    }
    return out;
}

static void Check_CL(cl_int err, const char* what) {
    if (err != CL_SUCCESS) {
        std::cerr << "[OpenCL error] " << what << " failed with code " << err << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
}

static void Check_MPI(int err, const char* what) {
    if (err != MPI_SUCCESS) {
        std::cerr << "[MPI error] " << what << " failed" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 3);
    }
}

std::vector<OclDeviceContext> Discover_OpenCL_Devices(int rank) {
    std::vector<OclDeviceContext> out;

    cl_uint num_platforms = 0;
    cl_int err = clGetPlatformIDs(0, nullptr, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) {
        std::cerr << "[rank " << rank << "] No OpenCL platforms found." << std::endl;
        return out;
    }

    std::vector<cl_platform_id> platforms(num_platforms);
    Check_CL(clGetPlatformIDs(num_platforms, platforms.data(), nullptr), "clGetPlatformIDs");

    int local_index = 0;

    for (cl_platform_id platform : platforms) {
        cl_uint num_devices = 0;
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &num_devices);
        if (err != CL_SUCCESS || num_devices == 0) continue;

        std::vector<cl_device_id> devices(num_devices);
        Check_CL(
            clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices, devices.data(), nullptr),
            "clGetDeviceIDs"
        );

        const std::string platform_name = Get_Platform_String(platform, CL_PLATFORM_NAME);

        for (cl_device_id device : devices) {
            OclDeviceContext ctx;
            ctx.local_device_index = local_index++;
            ctx.platform = platform;
            ctx.device = device;
            ctx.platform_name = platform_name;
            ctx.device_name = Get_Device_String(device, CL_DEVICE_NAME);

            cl_int create_err = CL_SUCCESS;
            ctx.context = clCreateContext(nullptr, 1, &ctx.device, nullptr, nullptr, &create_err);
            if (create_err != CL_SUCCESS || ctx.context == nullptr) {
                std::cerr << "[rank " << rank << "] Could not create context for device "
                          << ctx.device_name << std::endl;
                continue;
            }

#if CL_TARGET_OPENCL_VERSION >= 200
            ctx.queue = clCreateCommandQueueWithProperties(ctx.context, ctx.device, nullptr, &create_err);
#else
            ctx.queue = clCreateCommandQueue(ctx.context, ctx.device, 0, &create_err);
#endif
            if (create_err != CL_SUCCESS || ctx.queue == nullptr) {
                clReleaseContext(ctx.context);
                ctx.context = nullptr;
                std::cerr << "[rank " << rank << "] Could not create queue for device "
                          << ctx.device_name << std::endl;
                continue;
            }

            ctx.valid = true;
            out.push_back(ctx);
        }
    }

    return out;
}

void Release_OpenCL_Devices(std::vector<OclDeviceContext>& devices) {
    for (auto& d : devices) {
        if (d.queue) {
            clReleaseCommandQueue(d.queue);
            d.queue = nullptr;
        }
        if (d.context) {
            clReleaseContext(d.context);
            d.context = nullptr;
        }
        d.valid = false;
    }
}

std::vector<int> Gather_Device_Counts(int local_count, int world_size) {
    std::vector<int> counts(static_cast<std::size_t>(world_size), 0);
    Check_MPI(
        MPI_Allgather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD),
        "MPI_Allgather(device counts)"
    );
    return counts;
}

std::vector<GlobalDevice> Build_Global_Device_List(
    const std::vector<int>& device_counts
) {
    std::vector<GlobalDevice> devices;
    int global = 0;

    for (int rank = 0; rank < static_cast<int>(device_counts.size()); ++rank) {
        for (int local = 0; local < device_counts[static_cast<std::size_t>(rank)]; ++local) {
            devices.push_back(GlobalDevice{global, rank, local});
            ++global;
        }
    }

    return devices;
}

void Assign_Global_Indices(
    std::vector<OclDeviceContext>& local_devices,
    const std::vector<int>& device_counts,
    int rank
) {
    int offset = 0;
    for (int r = 0; r < rank; ++r) {
        offset += device_counts[static_cast<std::size_t>(r)];
    }

    for (auto& dev : local_devices) {
        dev.global_device_index = offset + dev.local_device_index;
    }
}

std::vector<std::size_t> Generate_Test_Volumes() {
    std::vector<std::size_t> volumes = {
        512,
        1024,
        1500,
        2048,
        4096,
        8192,
        16384,
        32768,
        65536,
        131072,
        262144,
        524288,
        1048576,
        4194304,
        16777216
    };

    std::sort(volumes.begin(), volumes.end());
    volumes.erase(std::unique(volumes.begin(), volumes.end()), volumes.end());
    return volumes;
}

static int Iterations_For_Volume(std::size_t volume) {
    return (volume < 1048576) ? 500 : 50;
}

static double Median(std::vector<double>& values) {
    if (values.empty()) return 0.0;

    std::sort(values.begin(), values.end());
    const std::size_t n = values.size();

    if ((n % 2) == 0) {
        return 0.5 * (values[n / 2 - 1] + values[n / 2]);
    }

    return values[n / 2];
}

// Same-rank neighbor migration:
// source OpenCL device -> host -> destination OpenCL device.
// No MPI is included because both devices are visible in the same MPI process.
double Benchmark_OpenCL_Local_Device_To_Device(
    std::size_t volume,
    int rank,
    const OclDeviceContext& src,
    const OclDeviceContext& dst
) {
    if (!src.valid || !dst.valid || volume == 0) return 0.0;

    std::vector<char> host_buffer(volume, 'A');

    cl_int err = CL_SUCCESS;

    cl_mem src_buffer = clCreateBuffer(src.context, CL_MEM_READ_WRITE, volume, nullptr, &err);
    Check_CL(err, "clCreateBuffer(src_buffer local)");

    cl_mem dst_buffer = clCreateBuffer(dst.context, CL_MEM_READ_WRITE, volume, nullptr, &err);
    Check_CL(err, "clCreateBuffer(dst_buffer local)");

    Check_CL(
        clEnqueueWriteBuffer(src.queue, src_buffer, CL_TRUE, 0, volume, host_buffer.data(), 0, nullptr, nullptr),
        "clEnqueueWriteBuffer(local init src)"
    );

    const int iterations = Iterations_For_Volume(volume);
    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(iterations));

    MPI_Barrier(MPI_COMM_WORLD);

    for (int i = 0; i < iterations; ++i) {
        const double t0 = MPI_Wtime();

        Check_CL(
            clEnqueueReadBuffer(src.queue, src_buffer, CL_TRUE, 0, volume, host_buffer.data(), 0, nullptr, nullptr),
            "clEnqueueReadBuffer(local src)"
        );

        Check_CL(
            clEnqueueWriteBuffer(dst.queue, dst_buffer, CL_TRUE, 0, volume, host_buffer.data(), 0, nullptr, nullptr),
            "clEnqueueWriteBuffer(local dst)"
        );

        const double t1 = MPI_Wtime();
        samples.push_back(t1 - t0);
    }

    clReleaseMemObject(src_buffer);
    clReleaseMemObject(dst_buffer);

    const double median = Median(samples);

    std::cout << "[local-neighbor] volume=" << volume
              << " rank=" << rank
              << " src_dev=" << src.local_device_index
              << " dst_dev=" << dst.local_device_index
              << " median=" << median << " s" << std::endl;

    return median;
}

// Cross-rank neighbor migration:
// source OpenCL device -> host -> MPI -> host -> destination OpenCL device.
// The measured time on the source includes source OpenCL read plus MPI_Send.
// The destination performs MPI_Recv plus OpenCL write.
double Benchmark_MPI_OpenCL_Neighbor(
    std::size_t volume,
    int rank,
    const GlobalDevice& src_global,
    const GlobalDevice& dst_global,
    const std::vector<OclDeviceContext>& local_devices
) {
    const bool is_src = (rank == src_global.rank);
    const bool is_dst = (rank == dst_global.rank);

    if (!is_src && !is_dst) {
        MPI_Barrier(MPI_COMM_WORLD);
        return 0.0;
    }

    const OclDeviceContext* ocl = nullptr;
    if (is_src) {
        if (src_global.local_index < 0 ||
            static_cast<std::size_t>(src_global.local_index) >= local_devices.size()) {
            MPI_Barrier(MPI_COMM_WORLD);
            return 0.0;
        }
        ocl = &local_devices[static_cast<std::size_t>(src_global.local_index)];
    } else {
        if (dst_global.local_index < 0 ||
            static_cast<std::size_t>(dst_global.local_index) >= local_devices.size()) {
            MPI_Barrier(MPI_COMM_WORLD);
            return 0.0;
        }
        ocl = &local_devices[static_cast<std::size_t>(dst_global.local_index)];
    }

    if (ocl == nullptr || !ocl->valid || volume == 0) {
        MPI_Barrier(MPI_COMM_WORLD);
        return 0.0;
    }

    std::vector<char> host_buffer(volume, is_src ? 'A' : 'B');

    cl_int err = CL_SUCCESS;
    cl_mem dev_buffer = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, volume, nullptr, &err);
    Check_CL(err, "clCreateBuffer(cross-rank dev_buffer)");

    Check_CL(
        clEnqueueWriteBuffer(ocl->queue, dev_buffer, CL_TRUE, 0, volume, host_buffer.data(), 0, nullptr, nullptr),
        "clEnqueueWriteBuffer(cross-rank init)"
    );

    const int iterations = Iterations_For_Volume(volume);
    std::vector<double> samples;
    if (is_src) samples.reserve(static_cast<std::size_t>(iterations));

    const int tag = 2000 + src_global.global_index * 1000 + dst_global.global_index;

    MPI_Barrier(MPI_COMM_WORLD);

    for (int i = 0; i < iterations; ++i) {
        if (is_src) {
            const double t0 = MPI_Wtime();

            Check_CL(
                clEnqueueReadBuffer(ocl->queue, dev_buffer, CL_TRUE, 0, volume, host_buffer.data(), 0, nullptr, nullptr),
                "clEnqueueReadBuffer(cross-rank src)"
            );

            Check_MPI(
                MPI_Send(host_buffer.data(), static_cast<int>(volume), MPI_BYTE,
                         dst_global.rank, tag, MPI_COMM_WORLD),
                "MPI_Send(cross-rank neighbor)"
            );

            const double t1 = MPI_Wtime();
            samples.push_back(t1 - t0);
        }

        if (is_dst) {
            Check_MPI(
                MPI_Recv(host_buffer.data(), static_cast<int>(volume), MPI_BYTE,
                         src_global.rank, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE),
                "MPI_Recv(cross-rank neighbor)"
            );

            Check_CL(
                clEnqueueWriteBuffer(ocl->queue, dev_buffer, CL_TRUE, 0, volume, host_buffer.data(), 0, nullptr, nullptr),
                "clEnqueueWriteBuffer(cross-rank dst)"
            );
        }
    }

    clReleaseMemObject(dev_buffer);

    double median = 0.0;
    if (is_src) {
        median = Median(samples);

        std::cout << "[mpi-neighbor] volume=" << volume
                  << " src_global=" << src_global.global_index
                  << "(rank=" << src_global.rank << ",dev=" << src_global.local_index << ")"
                  << " dst_global=" << dst_global.global_index
                  << "(rank=" << dst_global.rank << ",dev=" << dst_global.local_index << ")"
                  << " median=" << median << " s" << std::endl;
    }

    return median;
}

double Benchmark_Neighbor_Direction(
    std::size_t volume,
    int rank,
    const GlobalDevice& src,
    const GlobalDevice& dst,
    const std::vector<OclDeviceContext>& local_devices
) {
    if (src.rank == dst.rank) {
        double result = 0.0;

        if (rank == src.rank) {
            result = Benchmark_OpenCL_Local_Device_To_Device(
                volume,
                rank,
                local_devices[static_cast<std::size_t>(src.local_index)],
                local_devices[static_cast<std::size_t>(dst.local_index)]
            );
        } else {
            MPI_Barrier(MPI_COMM_WORLD);
        }

        return result;
    }

    return Benchmark_MPI_OpenCL_Neighbor(volume, rank, src, dst, local_devices);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int world_size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    std::string output_filename = "profiling_results.txt";

    if (argc >= 2) {
        output_filename = argv[1];
    }


    std::vector<OclDeviceContext> local_devices = Discover_OpenCL_Devices(rank);
    const int local_device_count = static_cast<int>(local_devices.size());

    std::vector<int> device_counts = Gather_Device_Counts(local_device_count, world_size);
    Assign_Global_Indices(local_devices, device_counts, rank);

    const std::vector<GlobalDevice> global_devices = Build_Global_Device_List(device_counts);
    const int global_device_count = static_cast<int>(global_devices.size());

    MPI_Barrier(MPI_COMM_WORLD);

    for (int r = 0; r < world_size; ++r) {
        if (rank == r) {
            std::cout << "\n[rank " << rank << "] OpenCL devices found: "
                      << local_devices.size() << std::endl;
            for (const auto& d : local_devices) {
                std::cout << "  global_device=" << d.global_device_index
                          << " local_device=" << d.local_device_index
                          << " platform=\"" << d.platform_name << "\""
                          << " device=\"" << d.device_name << "\""
                          << std::endl;
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    if (global_device_count < 1) {
        if (rank == 0) {
            std::cerr << "No usable OpenCL devices found. Aborting." << std::endl;
        }
        Release_OpenCL_Devices(local_devices);
        MPI_Finalize();
        return 1;
    }

    if (global_device_count == 1 && rank == 0) {
        std::cout << "\nOnly one global OpenCL device found. "
                  << "Profiling will use local OpenCL self-migration only."
                  << std::endl;
    }

    if (rank == 0) {
        std::cout << "\nGlobal linear device order:" << std::endl;
        for (const auto& gd : global_devices) {
            std::cout << "  global_device=" << gd.global_index
                      << " rank=" << gd.rank
                      << " local_device=" << gd.local_index
                      << std::endl;
        }

        std::cout << "\nNeighbor directions to be measured:" << std::endl;
        if (global_device_count == 1) {
            std::cout << "  0 -> 0 (local OpenCL only)" << std::endl;
        } else {
            for (int i = 0; i + 1 < global_device_count; ++i) {
                std::cout << "  " << i << " -> " << (i + 1) << std::endl;
                std::cout << "  " << (i + 1) << " -> " << i << std::endl;
            }
        }
    }

    std::vector<std::size_t> volumes;
    int volume_count = 0;

    if (rank == 0) {
        volumes = Generate_Test_Volumes();
        volume_count = static_cast<int>(volumes.size());
    }

    Check_MPI(MPI_Bcast(&volume_count, 1, MPI_INT, 0, MPI_COMM_WORLD), "MPI_Bcast(volume_count)");

    if (rank != 0) {
        volumes.resize(static_cast<std::size_t>(volume_count));
    }

    Check_MPI(
        MPI_Bcast(volumes.data(), volume_count * static_cast<int>(sizeof(std::size_t)),
                  MPI_BYTE, 0, MPI_COMM_WORLD),
        "MPI_Bcast(volumes)"
    );

    std::vector<RawPoint> raw_points;
    raw_points.reserve(volumes.size());

    for (std::size_t volume : volumes) {
        double local_worst = 0.0;

        if (global_device_count == 1) {
            if (!local_devices.empty()) {
                local_worst = Benchmark_OpenCL_Local_Device_To_Device(
                    volume,
                    rank,
                    local_devices[0],
                    local_devices[0]
                );
            }
        } else {
            for (int i = 0; i + 1 < global_device_count; ++i) {
                const GlobalDevice& left = global_devices[static_cast<std::size_t>(i)];
                const GlobalDevice& right = global_devices[static_cast<std::size_t>(i + 1)];

                const double lr = Benchmark_Neighbor_Direction(
                    volume,
                    rank,
                    left,
                    right,
                    local_devices
                );

                if (rank == left.rank && lr > local_worst) {
                    local_worst = lr;
                }

                MPI_Barrier(MPI_COMM_WORLD);

                const double rl = Benchmark_Neighbor_Direction(
                    volume,
                    rank,
                    right,
                    left,
                    local_devices
                );

                if (rank == right.rank && rl > local_worst) {
                    local_worst = rl;
                }

                MPI_Barrier(MPI_COMM_WORLD);
            }
        }

        double global_worst = 0.0;
        Check_MPI(
            MPI_Allreduce(&local_worst, &global_worst, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD),
            "MPI_Allreduce(global worst neighbor time)"
        );

        if (rank == 0) {
            raw_points.push_back(RawPoint{volume, global_worst});
            std::cout << "[profile] selected volume " << volume
                      << " bytes: worst_neighbor_time=" << global_worst
                      << " s" << std::endl;
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    if (rank == 0) {
        std::ofstream output(output_filename);
        if (!output.is_open()) {
            std::cerr << "Could not create " << output_filename << std::endl;
            Release_OpenCL_Devices(local_devices);
            MPI_Finalize();
            return 1;
        }

        output << "Max_Volume_Bytes\tm_TempoPorByte\tb_Latencia\n";

        for (std::size_t i = 0; i + 1 < raw_points.size(); ++i) {
            const RawPoint& p1 = raw_points[i];
            const RawPoint& p2 = raw_points[i + 1];

            double m = 0.0;
            const double delta_v = static_cast<double>(p2.volume - p1.volume);

            if (delta_v > 0.0) {
                m = (p2.migration_time - p1.migration_time) / delta_v;
                if (m < 0.0) m = 0.0;
            }

            double b = p1.migration_time - m * static_cast<double>(p1.volume);
            if (b < 0.0) b = 0.0;

            output << p2.volume << "\t" << m << "\t" << b << "\n";
        }

        output.close();
        std::cout << "\nProfiling completed. Linear segments saved to " << output_filename << std::endl;
    }

    Release_OpenCL_Devices(local_devices);
    MPI_Finalize();

    return 0;
}
