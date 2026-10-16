#ifndef HARMONY_BENCHMARK_SAMPLE_ADMISSION_HPP
#define HARMONY_BENCHMARK_SAMPLE_ADMISSION_HPP

#include <mpi.h>
#ifdef __linux__
#include <sched.h>
#include <cerrno>
#include <cstring>
#endif

#include <array>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace benchmark {

inline std::string json_string(const std::string& text) {
    std::ostringstream out;
    out << '"';
    for (unsigned char ch : text) {
        switch (ch) {
        case '"': out << "\\\""; break;
        case '\\': out << "\\\\"; break;
        case '\n': out << "\\n"; break;
        case '\r': out << "\\r"; break;
        case '\t': out << "\\t"; break;
        default:
            if (ch < 0x20) {
                const char hex[] = "0123456789abcdef";
                out << "\\u00" << hex[ch >> 4] << hex[ch & 15];
            } else {
                out << static_cast<char>(ch);
            }
        }
    }
    out << '"';
    return out.str();
}

inline std::string cpu_affinity_json() {
#ifdef __linux__
    cpu_set_t mask;
    CPU_ZERO(&mask);
    if (sched_getaffinity(0, sizeof(mask), &mask) == 0) {
        std::ostringstream cpus;
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
            if (CPU_ISSET(cpu, &mask)) {
                if (cpus.tellp() > 0) cpus << ',';
                cpus << cpu;
            }
        }
        return "{\"value\":" + json_string(cpus.str()) + ",\"unavailable_reason\":null}";
    }
    return "{\"value\":null,\"unavailable_reason\":" +
           json_string(std::strerror(errno)) + '}';
#else
    return "{\"value\":null,\"unavailable_reason\":\"unsupported platform\"}";
#endif
}

inline bool admit_sample(const dcl::Runtime& runtime, bool strict,
                         const std::string& inventory_path) {
    const MPI_Comm comm = runtime.communicator();
    const int rank = runtime.rank();
    const auto& devices = runtime.devices();
    const int local_strict = strict ? 1 : 0;
    int strict_min = 0;
    int strict_max = 0;
    dcl::detail::check_mpi(MPI_Allreduce(&local_strict, &strict_min, 1, MPI_INT, MPI_MIN, comm),
                           "MPI_Allreduce(strict mode minimum)");
    dcl::detail::check_mpi(MPI_Allreduce(&local_strict, &strict_max, 1, MPI_INT, MPI_MAX, comm),
                           "MPI_Allreduce(strict mode maximum)");
    if (strict_min != strict_max) {
        if (rank == 0) std::cerr << "Benchmark ranks disagree on cluster sample mode\n";
        return false;
    }
    char host[MPI_MAX_PROCESSOR_NAME]{};
    int host_length = 0;
    dcl::detail::check_mpi(MPI_Get_processor_name(host, &host_length), "MPI_Get_processor_name");
    const dcl::DeviceInfo* device = devices.empty() ? nullptr : &devices.front();

    // Every rank contributes exactly one inventory record, even on admission failure.
    std::ostringstream record;
    record << "{\"rank\":" << rank << ",\"host\":" << json_string(host)
           << ",\"gpu_count\":" << devices.size()
           << ",\"local_index\":" << (device ? device->local_index : -1)
           << ",\"global_index\":" << (device ? device->global_index : -1)
           << ",\"name\":" << json_string(device ? device->name : "")
           << ",\"platform_name\":" << json_string(device ? device->platform_name : "")
           << ",\"platform_version\":" << json_string(device ? device->platform_version : "")
           << ",\"device_version\":" << json_string(device ? device->device_version : "")
           << ",\"driver_version\":" << json_string(device ? device->driver_version : "")
           << ",\"identity_kind\":" << json_string(device ? device->identity_kind : "unavailable")
           << ",\"identity\":" << json_string(device ? device->identity : "")
           << ",\"cpu_affinity\":" << cpu_affinity_json() << '}';
    const std::string local_record = record.str();
    int local_size = static_cast<int>(local_record.size());
    std::vector<int> sizes(runtime.size(), 0);
    dcl::detail::check_mpi(MPI_Allgather(&local_size, 1, MPI_INT, sizes.data(), 1, MPI_INT, comm),
                           "MPI_Allgather(inventory lengths)");
    std::vector<int> offsets(runtime.size(), 0);
    int all_size = 0;
    for (int r = 0; r < runtime.size(); ++r) {
        offsets[r] = all_size;
        all_size += sizes[r];
    }
    std::string records(static_cast<std::size_t>(all_size), '\0');
    dcl::detail::check_mpi(MPI_Allgatherv(local_record.data(), local_size, MPI_CHAR,
                           records.data(), sizes.data(), offsets.data(), MPI_CHAR, comm),
                           "MPI_Allgatherv(inventory)");

    int inventory_ok = 1;
    if (rank == 0 && !inventory_path.empty()) {
        std::ofstream out(inventory_path);
        if (!out) {
            std::cerr << "Cannot open device inventory JSON: " << inventory_path << '\n';
            inventory_ok = 0;
        } else {
            out << "{\"schema_version\":1,\"ranks\":[";
            for (int r = 0; r < runtime.size(); ++r) {
                if (r) out << ',';
                out.write(records.data() + offsets[r], sizes[r]);
            }
            out << "]}\n";
            out.close();
            if (!out) {
                std::cerr << "Cannot write device inventory JSON: " << inventory_path << '\n';
                inventory_ok = 0;
            }
        }
    }
    dcl::detail::check_mpi(MPI_Bcast(&inventory_ok, 1, MPI_INT, 0, comm),
                           "MPI_Bcast(inventory write status)");
    if (!inventory_ok) return false;
    if (!strict) return true;

    const int local_path_ok = inventory_path.empty() ? 0 : 1;
    int all_path_ok = 0;
    dcl::detail::check_mpi(MPI_Allreduce(&local_path_ok, &all_path_ok, 1, MPI_INT, MPI_MIN, comm),
                           "MPI_Allreduce(inventory path availability)");
    if (!all_path_ok) {
        if (rank == 0) std::cerr << "Cluster sample requires --device-inventory-json PATH\n";
        return false;
    }

    const int local_gpu_ok = devices.size() == 1 && devices.front().kind == dcl::DeviceKind::gpu ? 1 : 0;
    int all_gpu_ok = 0;
    dcl::detail::check_mpi(MPI_Allreduce(&local_gpu_ok, &all_gpu_ok, 1, MPI_INT, MPI_MIN, comm),
                           "MPI_Allreduce(GPU admission)");
    if (runtime.size() != 2 || !all_gpu_ok) {
        if (rank == 0) std::cerr << "Cluster sample requires exactly two MPI ranks and exactly one OpenCL GPU per rank\n";
        return false;
    }

    // A UUID is globally meaningful. A PCI address is meaningful only with its host.
    const std::string local_key = device->identity_kind == "uuid"
        ? "uuid:" + device->identity
        : device->identity_kind == "pci" ? "pci:" + std::string(host) + ':' + device->identity : "";
    int local_identity_ok = local_key.empty() ? 0 : 1;
    int all_identity_ok = 0;
    dcl::detail::check_mpi(MPI_Allreduce(&local_identity_ok, &all_identity_ok, 1, MPI_INT, MPI_MIN, comm),
                           "MPI_Allreduce(physical identity availability)");
    if (!all_identity_ok) {
        if (rank == 0) std::cerr << "Cluster sample requires verifiable GPU UUID or host-qualified PCI identity on each rank\n";
        return false;
    }
    // Fixed buffer is sufficient for UUID and host-qualified PCI tokens; reject truncation.
    std::array<char, MPI_MAX_PROCESSOR_NAME + 128> key{};
    int local_length_ok = local_key.size() < key.size() ? 1 : 0;
    int all_length_ok = 0;
    dcl::detail::check_mpi(MPI_Allreduce(&local_length_ok, &all_length_ok, 1, MPI_INT, MPI_MIN, comm),
                           "MPI_Allreduce(identity length)");
    if (!all_length_ok) {
        if (rank == 0) std::cerr << "GPU identity token exceeds admission limit\n";
        return false;
    }
    local_key.copy(key.data(), local_key.size());
    std::array<char, 2 * (MPI_MAX_PROCESSOR_NAME + 128)> keys{};
    dcl::detail::check_mpi(MPI_Allgather(key.data(), static_cast<int>(key.size()), MPI_CHAR,
                           keys.data(), static_cast<int>(key.size()), MPI_CHAR, comm),
                           "MPI_Allgather(physical identities)");
    const std::string first(keys.data());
    const std::string second(keys.data() + key.size());
    if (first.substr(0, first.find(':')) != second.substr(0, second.find(':'))) {
        if (rank == 0) std::cerr << "Cluster sample GPU identity kinds differ across ranks\n";
        return false;
    }
    if (first == second) {
        if (rank == 0) std::cerr << "Cluster sample ranks resolve to the same physical GPU\n";
        return false;
    }
    return true;
}

} // namespace benchmark

#endif
