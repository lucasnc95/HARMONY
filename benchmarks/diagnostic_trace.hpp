#pragma once

#include <mpi.h>
#include "../dcl/runtime.hpp"

#include <chrono>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <iostream>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

namespace benchmark {

class DiagnosticTrace {
public:
    using Clock = std::chrono::steady_clock;
    static constexpr const char* header = "schema_version,rank,seq,iteration,seed,algorithm,record_type,phase,part,device_global,owner_rank,local_index,offset,count,duration_s,device_event_s,boundary_shift_elements_estimate,transfer_bytes_measured,device_name,device_kind,device_identity\n";

    DiagnosticTrace(const std::string& prefix, int rank, unsigned int seed, const std::string& algorithm)
        : enabled_(!prefix.empty()), rank_(rank), seed_(seed), algorithm_(std::move(algorithm)),
          path_(enabled_ ? prefix + ".rank" + std::to_string(rank) + ".csv" : "") {}

    DiagnosticTrace(const DiagnosticTrace&) = delete;
    DiagnosticTrace& operator=(const DiagnosticTrace&) = delete;
    ~DiagnosticTrace() { if (fd_ >= 0) ::close(fd_); }
    bool enabled() const { return enabled_; }

    bool prepare(MPI_Comm comm) {
        int enabled_count = 0;
        int own_enabled = enabled_ ? 1 : 0;
        dcl::detail::check_mpi(MPI_Allreduce(&own_enabled, &enabled_count, 1, MPI_INT, MPI_SUM, comm),
                               "MPI_Allreduce(trace enabled)");
        int ranks = 0;
        dcl::detail::check_mpi(MPI_Comm_size(comm, &ranks), "MPI_Comm_size(trace)");
        if (enabled_count != 0 && enabled_count != ranks) {
            if (rank_ == 0) std::cerr << "Diagnostic trace must be enabled on every rank\n";
            return false;
        }
        if (!enabled_) return true;

        // The rank suffix makes paths distinct for a common prefix. Verify that all
        // ranks received that same prefix before any file is created.
        std::error_code path_error;
        const auto absolute_path = std::filesystem::absolute(path_, path_error);
        int own_path_ok = path_error ? 0 : 1;
        int all_path_ok = 0;
        dcl::detail::check_mpi(MPI_Allreduce(&own_path_ok, &all_path_ok, 1, MPI_INT, MPI_MIN, comm),
                               "MPI_Allreduce(trace path)");
        if (!all_path_ok) {
            if (path_error) std::cerr << "rank " << rank_ << " invalid diagnostic trace path: "
                                      << path_error.message() << "\n";
            return false;
        }
        std::string parent = absolute_path.lexically_normal().string();
        const auto suffix = ".rank" + std::to_string(rank_) + ".csv";
        parent.resize(parent.size() - suffix.size());
        int length = static_cast<int>(parent.size());
        std::vector<int> lengths(ranks);
        dcl::detail::check_mpi(MPI_Allgather(&length, 1, MPI_INT, lengths.data(), 1, MPI_INT, comm),
                               "MPI_Allgather(trace prefix lengths)");
        bool matching = true;
        for (int other : lengths) matching = matching && other == length;
        int own_length_match = matching ? 1 : 0;
        int all_length_match = 0;
        dcl::detail::check_mpi(MPI_Allreduce(&own_length_match, &all_length_match, 1, MPI_INT, MPI_MIN, comm),
                               "MPI_Allreduce(trace prefix length match)");
        if (all_length_match) {
            std::vector<char> prefixes(static_cast<std::size_t>(ranks) * length);
            dcl::detail::check_mpi(MPI_Allgather(parent.data(), length, MPI_CHAR, prefixes.data(), length,
                                                  MPI_CHAR, comm), "MPI_Allgather(trace prefixes)");
            for (int r = 0; r < ranks; ++r) {
                matching = matching && std::string(prefixes.data() + static_cast<std::size_t>(r) * length,
                                                   static_cast<std::size_t>(length)) == parent;
            }
        } else matching = false;
        int own_match = matching ? 1 : 0;
        int all_match = 0;
        dcl::detail::check_mpi(MPI_Allreduce(&own_match, &all_match, 1, MPI_INT, MPI_MIN, comm),
                               "MPI_Allreduce(trace prefix match)");
        if (!all_match) {
            if (rank_ == 0) std::cerr << "Diagnostic trace prefixes differ across ranks\n";
            return false;
        }

        fd_ = ::open(path_.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW, 0644);
        const int local_ok = fd_ >= 0 ? 1 : 0;
        if (!local_ok) std::cerr << "rank " << rank_ << " cannot create diagnostic trace " << path_
                                 << ": " << std::strerror(errno) << "\n";
        int all_ok = 0;
        dcl::detail::check_mpi(MPI_Allreduce(&local_ok, &all_ok, 1, MPI_INT, MPI_MIN, comm),
                               "MPI_Allreduce(trace open)");
        if (!all_ok) {
            if (fd_ >= 0) { ::close(fd_); fd_ = -1; ::unlink(path_.c_str()); }
            return false;
        }
        return true;
    }

    static double seconds(Clock::time_point start, Clock::time_point end) {
        return std::chrono::duration<double>(end - start).count();
    }

    void phase(int iteration, const char* name, Clock::time_point start, Clock::time_point end) {
        if (!enabled_) return;
        row(iteration, "phase", name, "", "", "", "", "", "",
            number(seconds(start, end)), "", "", "", "", "", "");
    }

    void partitions(int iteration, const char* state, const std::vector<dcl::DevicePartition>& parts,
                    const std::vector<dcl::DevicePartition>* old = nullptr) {
        if (!enabled_) return;
        for (std::size_t i = 0; i < parts.size(); ++i) {
            const auto& part = parts[i];
            std::string shift;
            if (old && i < old->size()) {
                const auto a = part.global_offset;
                const auto b = (*old)[i].global_offset;
                shift = std::to_string(a > b ? a - b : b - a);
            }
            row(iteration, "partition", state, std::to_string(i),
                std::to_string(part.device_global_index), std::to_string(part.owning_rank),
                std::to_string(part.local_index), std::to_string(part.global_offset),
                std::to_string(part.element_count), "", "", shift, "", "", "", "");
        }
    }

    void inventory(const std::vector<dcl::DeviceInfo>& devices) {
        if (!enabled_) return;
        for (const auto& device : devices) {
            row(-1, "inventory", "discovered", "", std::to_string(device.global_index),
                std::to_string(device.rank), std::to_string(device.local_index), "", "", "", "", "", "",
                quoted(device.name), "gpu", quoted(device.identity_kind + ":" + device.identity));
        }
    }

    void device_events(int iteration, const std::vector<dcl::DevicePartition>& parts,
                       const std::vector<double>& seconds_by_part, bool opencl) {
        if (!enabled_ || !opencl) return;
        for (std::size_t i = 0; i < parts.size(); ++i) {
            const auto& part = parts[i];
            if (part.owning_rank != rank_) continue;
            row(iteration, "device_event", "kernel", std::to_string(i),
                std::to_string(part.device_global_index), std::to_string(part.owning_rank),
                std::to_string(part.local_index), std::to_string(part.global_offset),
                std::to_string(part.element_count), "", number(seconds_by_part[i]), "", "", "", "", "");
        }
    }

    bool finish(MPI_Comm comm) {
        if (!enabled_) return true;
        const std::string output = std::string(header) + buffer_.str();
        std::size_t written = 0;
        bool local_ok = true;
        while (written < output.size()) {
            const ssize_t count = ::write(fd_, output.data() + written, output.size() - written);
            if (count < 0 && errno == EINTR) continue;
            if (count <= 0) { local_ok = false; break; }
            written += static_cast<std::size_t>(count);
        }
        if (::close(fd_) != 0) local_ok = false;
        fd_ = -1;
        if (!local_ok) std::cerr << "rank " << rank_ << " failed to write diagnostic trace " << path_ << "\n";
        int own_ok = local_ok ? 1 : 0;
        int all_ok = 0;
        dcl::detail::check_mpi(MPI_Allreduce(&own_ok, &all_ok, 1, MPI_INT, MPI_MIN, comm),
                               "MPI_Allreduce(trace write)");
        return all_ok != 0;
    }

private:
    static std::string number(double value) {
        std::ostringstream out;
        out.precision(17);
        out << value;
        return out.str();
    }
    static std::string quoted(const std::string& value) {
        std::string out = "\"";
        for (char c : value) {
            if (c == '"') out += "\"\"";
            else if (c == '\n' || c == '\r') out += ' ';
            else out += c;
        }
        return out + '"';
    }
    void row(int iteration, const char* type, const char* phase, const std::string& part,
             const std::string& global, const std::string& owner, const std::string& local,
             const std::string& offset, const std::string& count, const std::string& duration,
             const std::string& event, const std::string& shift, const std::string& transfer,
             const std::string& name, const std::string& kind, const std::string& identity) {
        buffer_ << "1," << rank_ << ',' << seq_++ << ',' << iteration << ',' << seed_ << ',' << algorithm_
                << ',' << type << ',' << phase << ',' << part << ',' << global << ',' << owner << ','
                << local << ',' << offset << ',' << count << ',' << duration << ',' << event << ','
                << shift << ',' << transfer << ',' << name << ',' << kind << ',' << identity << '\n';
    }
    bool enabled_;
    int rank_;
    unsigned int seed_;
    std::string algorithm_;
    std::string path_;
    int fd_{-1};
    std::size_t seq_{0};
    std::ostringstream buffer_;
};

} // namespace benchmark
