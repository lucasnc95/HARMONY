// Automatic load-balancing policies.
//
// Both policies measure the kernel time of every partition since the last
// decision (the "balance window") and propose new cumulative loads
// proportional to each device's throughput. They differ in how they accept
// the proposal:
//
//   threshold: accept if the L2 distance between the current and the
//              proposed loads is at least `threshold`.
//   profiled:  accept if the estimated time saved over the remaining
//              iterations exceeds the estimated migration cost, given by a
//              piecewise-linear profile (see tools/migration_profiler).

#include "runtime_impl.hpp"
#include "detail/format.hpp"
#include "detail/load_math.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>

namespace harmony {

// Kernel seconds of every partition in the current window, known by all ranks.
std::vector<double> Runtime::Impl::collect_global_balance_times_from_window(const char* context) {
    const std::vector<double> measured = compute_balance_window_elapsed_local();

    std::vector<double> local_times(partitions_.size(), 0.0);
    for (std::size_t i = 0; i < partitions_.size(); ++i) {
        const DevicePartition& dp = partitions_[i];
        if (dp.owning_rank == rank_ && dp.local_index >= 0 &&
            static_cast<std::size_t>(dp.local_index) < measured.size()) {
            local_times[i] = measured[static_cast<std::size_t>(dp.local_index)];
        }
    }

    std::vector<double> global_times(partitions_.size(), 0.0);
    PhaseTimer timer(*this, Phase::mpi_balance);
    detail::check_mpi(MPI_Allreduce(local_times.data(), global_times.data(),
                                    static_cast<int>(global_times.size()), MPI_DOUBLE, MPI_SUM, comm_),
                      context);
    return global_times;
}

namespace {

// Time per iteration of the slowest device now (measured) and after moving
// to `new_parts` (expected, assuming every device keeps the throughput
// measured in the window: elements per second).
void estimate_iteration_times(const std::vector<double>& times, const std::vector<DevicePartition>& old_parts,
                              const std::vector<DevicePartition>& new_parts, std::size_t window_iterations,
                              double& now, double& after) {
    now = 0.0;
    std::vector<double> capacity(old_parts.size(), 0.0);
    bool all_measured = true;
    for (std::size_t i = 0; i < old_parts.size() && i < times.size(); ++i) {
        now = std::max(now, times[i]);
        if (times[i] > 1.0e-12 && old_parts[i].element_count > 0) {
            capacity[i] = static_cast<double>(old_parts[i].element_count) / times[i];
        }
    }

    after = 0.0;
    for (std::size_t i = 0; i < new_parts.size(); ++i) {
        if (new_parts[i].element_count == 0) continue;
        if (i >= capacity.size() || capacity[i] <= 0.0) {
            all_measured = false;
            break;
        }
        after = std::max(after, static_cast<double>(new_parts[i].element_count) / capacity[i]);
    }
    if (!all_measured) after = now;

    const double n = static_cast<double>(std::max<std::size_t>(window_iterations, 1));
    now /= n;
    after /= n;
}

} // namespace

// Measures the window, proposes a division proportional to each device's
// throughput and fills the measured part of the record. Returns false (and
// the reason) when there is nothing to decide.
bool Runtime::Impl::propose_division(std::vector<DevicePartition>& new_parts, BalanceRecord& record,
                                     const char* context) {
    record.window_iterations = iteration_counter_ - balance_window_start_iteration_;
    const std::vector<double> times = collect_global_balance_times_from_window(context);
    record.window_seconds = times;

    const std::vector<float> proposed = detail::compute_loads_from_partition_throughput(times, partitions_);
    new_parts = partitions_from_loads(proposed);
    if (proposed.empty() || new_parts.size() != partitions_.size()) {
        reset_balance_window_events();
        record.reason = "could not compute a new division";
        return false;
    }

    estimate_iteration_times(times, partitions_, new_parts, record.window_iterations,
                             record.iteration_seconds_before, record.iteration_seconds_after);

    if (detail::same_partition_layout(partitions_, new_parts)) {
        current_loads_ = loads_from_partitions(partitions_);
        reset_balance_window_events();
        record.iteration_seconds_after = record.iteration_seconds_before;
        record.reason = "the division is already balanced";
        return false;
    }
    return true;
}

bool Runtime::Impl::maybe_rebalance_from_timings(const std::vector<FieldHandle>& requested_fields,
                                                 float threshold, BalanceRecord& record) {
    if (!partition_.has_value() || partitions_.empty()) {
        record.reason = "no partition defined yet";
        return false;
    }
    const std::vector<FieldHandle> fields = unique_existing_proportional_fields(requested_fields);

    std::vector<DevicePartition> new_parts;
    if (!propose_division(new_parts, record, "MPI_Allreduce(balance times)")) return false;
    const std::vector<DevicePartition> old_parts = partitions_;

    const double distance = detail::l2_norm_diff(current_loads_, loads_from_partitions(new_parts));
    const bool accept = distance >= static_cast<double>(threshold);
    record.reason = "division change " + detail::format_number(distance, 3) + (accept ? " >= " : " < ") +
                    "threshold " + detail::format_number(threshold, 4);
    if (!accept) {
        reset_balance_window_events();
        return false;
    }

    apply_partitions(fields, old_parts, new_parts, record);
    return true;
}

bool Runtime::Impl::maybe_rebalance_profiled(const std::vector<FieldHandle>& requested_fields,
                                             const AutoBalancePolicy& policy, BalanceRecord& record) {
    if (!partition_.has_value() || partitions_.empty()) {
        record.reason = "no partition defined yet";
        return false;
    }
    if (policy.total_iterations <= 0) {
        reset_balance_window_events();
        record.reason = "AutoBalancePolicy::total_iterations not set";
        return false;
    }
    const std::vector<FieldHandle> fields = unique_existing_proportional_fields(requested_fields);

    if (!profiling_loaded_ || profiling_file_loaded_ != policy.profiling_file) {
        load_profiling_data(policy.profiling_file);
    }

    // 1. Measure and propose.
    std::vector<DevicePartition> new_parts;
    if (!propose_division(new_parts, record, "MPI_Allreduce(profiled balance times)")) return false;
    const std::vector<DevicePartition> old_parts = partitions_;

    // 2. Cost model: migration cost vs. time saved until the end of the run.
    const double migration_cost = get_migration_overhead(max_bytes_received_by_a_rank(fields, old_parts, new_parts));

    const std::size_t total_iterations = static_cast<std::size_t>(policy.total_iterations);
    const std::size_t remaining_iterations =
        (iteration_counter_ < total_iterations) ? (total_iterations - iteration_counter_) : 0;

    const double gain_per_iteration =
        std::max(0.0, record.iteration_seconds_before - record.iteration_seconds_after);
    const double estimated_gain = gain_per_iteration * static_cast<double>(remaining_iterations);

    // 3. Decision (must be identical on every rank).
    const int local_decision = (remaining_iterations != 0 && migration_cost < estimated_gain) ? 1 : 0;
    int min_decision = 0;
    int max_decision = 0;
    {
        PhaseTimer mpi_timer(*this, Phase::mpi_balance);
        detail::check_mpi(MPI_Allreduce(&local_decision, &min_decision, 1, MPI_INT, MPI_MIN, comm_),
                          "MPI_Allreduce(profiled rebalance decision min)");
        detail::check_mpi(MPI_Allreduce(&local_decision, &max_decision, 1, MPI_INT, MPI_MAX, comm_),
                          "MPI_Allreduce(profiled rebalance decision max)");
    }
    if (min_decision != max_decision) {
        throw Error("maybe_rebalance_profiled(): inconsistent rebalance decision across ranks");
    }

    if (remaining_iterations == 0) {
        record.reason = "no iterations left";
    } else {
        record.reason = "saves " + detail::format_seconds(estimated_gain) + " in the " +
                        detail::format_count(remaining_iterations) +
                        (remaining_iterations == 1 ? " iteration left" : " iterations left") +
                        (local_decision ? ", more than" : ", not more than") + " moving the data costs (" +
                        detail::format_seconds(migration_cost) + ")";
    }

    if (local_decision == 0) {
        reset_balance_window_events();
        return false;
    }

    apply_partitions(fields, old_parts, new_parts, record);
    return true;
}

// Largest number of bytes a single rank would receive from other ranks.
// Old and new partitions are both sorted by offset, so a linear sweep finds
// every overlap.
std::size_t Runtime::Impl::max_bytes_received_by_a_rank(const std::vector<FieldHandle>& fields,
                                                        const std::vector<DevicePartition>& old_parts,
                                                        const std::vector<DevicePartition>& new_parts) const {
    std::vector<std::size_t> rank_recv_bytes(static_cast<std::size_t>(size_), 0);

    for (FieldHandle fh : fields) {
        auto fit = fields_.find(fh.value);
        if (fit == fields_.end()) continue;
        const std::size_t elem_bytes = fit->second.element_bytes();

        std::size_t src_idx = 0;
        std::size_t dst_idx = 0;
        while (src_idx < old_parts.size() && dst_idx < new_parts.size()) {
            const DevicePartition& src = old_parts[src_idx];
            const DevicePartition& dst = new_parts[dst_idx];
            const std::size_t src_end = src.global_offset + src.element_count;
            const std::size_t dst_end = dst.global_offset + dst.element_count;

            const std::size_t start = std::max(src.global_offset, dst.global_offset);
            const std::size_t end = std::min(src_end, dst_end);
            if (start < end && src.owning_rank != dst.owning_rank) {
                rank_recv_bytes[static_cast<std::size_t>(dst.owning_rank)] += (end - start) * elem_bytes;
            }

            if (src_end <= dst_end) ++src_idx;
            if (dst_end <= src_end) ++dst_idx;
        }
    }

    return rank_recv_bytes.empty() ? 0 : *std::max_element(rank_recv_bytes.begin(), rank_recv_bytes.end());
}

// ---------------------------------------------------------------------------
// Migration-cost profile
// ---------------------------------------------------------------------------

// File format (tools/migration_profiler output), one header line then:
//   <max_volume_bytes> <seconds_per_byte> <latency_seconds>
void Runtime::Impl::load_profiling_data(const std::string& profiling_file) {
    int n_points = 0;

    if (rank_ == 0) {
        profiling_data_.clear();

        std::ifstream file(profiling_file);
        if (file.is_open()) {
            std::string header;
            std::getline(file, header);

            std::size_t v = 0;
            double m = 0.0;
            double b = 0.0;
            while (file >> v >> m >> b) {
                profiling_data_.push_back(detail::ProfileSegment{v, m, b});
            }
        } else {
            std::cerr << "[HARMONY Warning] Rank 0 could not open profiling file '" << profiling_file
                      << "'. Using a conservative default cost model." << std::endl;
            profiling_data_.push_back(detail::ProfileSegment{16777216ull, 1.0e-9, 1.0e-6});
        }

        std::sort(profiling_data_.begin(), profiling_data_.end(),
                  [](const detail::ProfileSegment& a, const detail::ProfileSegment& b) {
                      return a.max_volume < b.max_volume;
                  });
        n_points = static_cast<int>(profiling_data_.size());
    }

    PhaseTimer timer(*this, Phase::mpi_balance);
    detail::check_mpi(MPI_Bcast(&n_points, 1, MPI_INT, 0, comm_), "MPI_Bcast(n_points)");
    if (rank_ != 0) {
        profiling_data_.resize(static_cast<std::size_t>(n_points));
    }
    if (n_points > 0) {
        detail::check_mpi(
            MPI_Bcast(profiling_data_.data(), static_cast<int>(n_points * sizeof(detail::ProfileSegment)),
                      MPI_BYTE, 0, comm_),
            "MPI_Bcast(profiling_data)");
    }

    profiling_loaded_ = true;
    profiling_file_loaded_ = profiling_file;
}

double Runtime::Impl::get_migration_overhead(std::size_t volume) const {
    if (volume == 0 || profiling_data_.empty()) return 0.0;

    auto it = std::lower_bound(profiling_data_.begin(), profiling_data_.end(), volume,
                               [](const detail::ProfileSegment& p, std::size_t v) { return p.max_volume < v; });
    if (it == profiling_data_.end()) it = std::prev(profiling_data_.end());

    const double overhead = it->m * static_cast<double>(volume) + it->b;
    return (overhead > 0.0) ? overhead : 0.0;
}

} // namespace harmony
