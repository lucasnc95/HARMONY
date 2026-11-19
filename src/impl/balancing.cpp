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
#include "harmony/algorithms.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>

namespace harmony {

// Kernel seconds of every partition in the current window, known by all ranks.
std::vector<double> Runtime::Impl::collect_global_balance_times_from_window(const char* context) {
    if (!simulated_times_.empty()) {
        if (!devices_.empty() && simulated_times_.size() == partitions_.size()) {
            std::vector<double> local_elapsed(devices_.size(), 0.0);
            for (std::size_t i = 0; i < partitions_.size(); ++i) {
                const DevicePartition& dp = partitions_[i];
                if (dp.owning_rank == rank_ && dp.local_index >= 0 &&
                    static_cast<std::size_t>(dp.local_index) < local_elapsed.size()) {
                    local_elapsed[static_cast<std::size_t>(dp.local_index)] = simulated_times_[i];
                }
            }
            update_device_timings(local_elapsed);
        }
        return simulated_times_;
    }

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
    for (std::size_t i = 0; i < old_parts.size() && i < times.size(); ++i) {
        now = std::max(now, times[i]);
    }
    std::vector<double> capacity;
    (void)detail::compute_partition_capacities(times, old_parts, nullptr, false, capacity);

    bool all_measured = true;
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

unsigned int Runtime::Impl::synchronize_hwtopolb_seed(unsigned int policy_seed) {
    unsigned int attempt_seed = policy_seed;
    if (rank_ == 0) {
        if (attempt_seed == 0) {
            std::random_device rd;
            attempt_seed = rd();
            if (attempt_seed == 0) attempt_seed = 1u;
        } else if (balance_attempts_ > 0) {
            attempt_seed = policy_seed + static_cast<unsigned int>(balance_attempts_);
            if (attempt_seed == 0) attempt_seed = 1u;
        }
    }
    PhaseTimer mpi_timer(*this, Phase::mpi_balance);
    detail::check_mpi(MPI_Bcast(&attempt_seed, 1, MPI_UNSIGNED, 0, comm_), "MPI_Bcast(hwtopolb seed)");
    return attempt_seed;
}

void Runtime::Impl::apply_collective_power_cap(std::vector<float>& proposed,
                                               const AutoBalancePolicy& policy,
                                               BalanceRecord& record) {
    int power_cap_error = 0;
    std::string power_cap_msg;
    double throttle = 1.0;
    int has_power_metrics = 0;
    if (rank_ == 0 && !proposed.empty()) {
        try {
            if (!topo_metrics_.has_value()) {
                throw Error("Power cap requires topology power measurements");
            }
            has_power_metrics = (!topo_metrics_->current_power_watts.empty() &&
                                 !topo_metrics_->thermal_tdp_watts.empty()) ? 1 : 0;
            proposed = apply_power_cap(
                proposed, *topo_metrics_, policy.power_budget_watts, &throttle, &current_loads_);
        } catch (const std::exception& e) {
            power_cap_error = 1;
            power_cap_msg = e.what();
        } catch (...) {
            power_cap_error = 1;
            power_cap_msg = "Power cap infeasible or measurements unavailable";
        }
    }
    {
        PhaseTimer mpi_timer(*this, Phase::mpi_balance);
        detail::check_mpi(MPI_Bcast(&power_cap_error, 1, MPI_INT, 0, comm_), "MPI_Bcast(power cap status)");
    }
    if (power_cap_error != 0) {
        int msg_len = (rank_ == 0) ? static_cast<int>(power_cap_msg.size()) : 0;
        {
            PhaseTimer mpi_timer(*this, Phase::mpi_balance);
            detail::check_mpi(MPI_Bcast(&msg_len, 1, MPI_INT, 0, comm_), "MPI_Bcast(power cap error length)");
            if (rank_ != 0) power_cap_msg.resize(static_cast<std::size_t>(msg_len));
            if (msg_len > 0) {
                detail::check_mpi(MPI_Bcast(power_cap_msg.data(), msg_len, MPI_CHAR, 0, comm_),
                                  "MPI_Bcast(power cap error message)");
            }
        }
        throw Error(power_cap_msg.empty() ? "Power cap infeasible or measurements unavailable" : power_cap_msg);
    }
    {
        PhaseTimer mpi_timer(*this, Phase::mpi_balance);
        detail::check_mpi(MPI_Bcast(&has_power_metrics, 1, MPI_INT, 0, comm_), "MPI_Bcast(power cap active)");
        detail::check_mpi(MPI_Bcast(&throttle, 1, MPI_DOUBLE, 0, comm_), "MPI_Bcast(power cap throttle)");
    }
    record.power_capped = (has_power_metrics != 0);
    record.throttle_factor = throttle;
}

// Measures the window, proposes a division proportional to each device's
// throughput and fills the measured part of the record. Returns false (and
// the reason) when there is nothing to decide.
bool Runtime::Impl::propose_division(std::vector<DevicePartition>& new_parts,
                                     const AutoBalancePolicy& policy, BalanceRecord& record,
                                     const char* context) {
    record.window_iterations = iteration_counter_ - balance_window_start_iteration_;
    const std::vector<double> times = collect_global_balance_times_from_window(context);
    record.window_seconds = times;

    const bool apply_contention = policy.use_contention_adjustment && topo_metrics_.has_value() &&
                                  !topo_metrics_->memory_contention_factor.empty();
    record.contention_adjusted = apply_contention;
    record.algorithm = (policy.algorithm == BalanceAlgorithm::hwtopolb) ? "hwtopolb" : "erad";
    record.perturbation_factor =
        (policy.algorithm == BalanceAlgorithm::hwtopolb) ? policy.perturbation_factor : 0.0f;
    record.seed = policy.seed;
    record.power_capped = false;
    record.power_budget_watts = policy.use_power_cap ? policy.power_budget_watts : 0.0;
    record.throttle_factor = 1.0;

    std::vector<float> proposed;
    if (policy.algorithm == BalanceAlgorithm::hwtopolb && policy.perturbation_factor > 0.0f) {
        const unsigned int attempt_seed = synchronize_hwtopolb_seed(policy.seed);
        record.seed = attempt_seed;
        proposed = apply_contention
            ? hwtopolb_loads(times, partitions_, policy.perturbation_factor, attempt_seed,
                             *topo_metrics_, true)
            : hwtopolb_loads(times, partitions_, policy.perturbation_factor, attempt_seed);
    } else {
        proposed = apply_contention
            ? erad_loads(times, partitions_, *topo_metrics_, true)
            : erad_loads(times, partitions_);
    }

    if (policy.use_power_cap) {
        apply_collective_power_cap(proposed, policy, record);
    }

    int proposed_count = (rank_ == 0) ? static_cast<int>(proposed.size()) : 0;
    {
        PhaseTimer mpi_timer(*this, Phase::mpi_balance);
        detail::check_mpi(MPI_Bcast(&proposed_count, 1, MPI_INT, 0, comm_),
                          "MPI_Bcast(balance proposal size)");
        if (rank_ != 0) {
            proposed.resize(static_cast<std::size_t>(proposed_count));
        }
        if (proposed_count > 0) {
            detail::check_mpi(MPI_Bcast(proposed.data(), proposed_count, MPI_FLOAT, 0, comm_),
                              "MPI_Bcast(balance proposal)");
        }
    }

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

void Runtime::Impl::validate_collective_balance_inputs(const std::vector<FieldHandle>& fields,
                                                         const char* context) {
    unsigned long long field_signature = 1469598103934665603ULL;
    for (FieldHandle fh : fields) {
        field_signature ^= static_cast<unsigned long long>(fh.value);
        field_signature *= 1099511628211ULL;
    }

    const unsigned long long local_state[6] = {
        partition_.has_value() ? 1ULL : 0ULL,
        static_cast<unsigned long long>(partitions_.size()),
        static_cast<unsigned long long>(simulated_times_.size()),
        static_cast<unsigned long long>(fields.size()),
        static_cast<unsigned long long>(current_loads_.size()),
        field_signature
    };
    unsigned long long min_state[6] = {};
    unsigned long long max_state[6] = {};
    {
        PhaseTimer mpi_timer(*this, Phase::mpi_balance);
        detail::check_mpi(
            MPI_Allreduce(local_state, min_state, 6, MPI_UNSIGNED_LONG_LONG, MPI_MIN, comm_),
            "MPI_Allreduce(balance input minima)");
        detail::check_mpi(
            MPI_Allreduce(local_state, max_state, 6, MPI_UNSIGNED_LONG_LONG, MPI_MAX, comm_),
            "MPI_Allreduce(balance input maxima)");
    }
    for (int i = 0; i < 6; ++i) {
        if (min_state[i] != max_state[i]) {
            throw Error(std::string(context) + ": inconsistent balance inputs across ranks");
        }
    }
}

double Runtime::Impl::estimate_total_migration_cost_seconds(
    const std::vector<FieldHandle>& fields,
    const std::vector<DevicePartition>& old_parts,
    const std::vector<DevicePartition>& new_parts) const {
    if (!topo_metrics_.has_value()) return 0.0;

    double total_cost_s = 0.0;
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

            std::size_t start = 0;
            std::size_t len = 0;
            if (detail::intersect_1d(src.global_offset, src.element_count,
                                     dst.global_offset, dst.element_count, start, len)) {
                const int src_dev = (src.device_global_index >= 0)
                    ? src.device_global_index : static_cast<int>(src_idx);
                const int dst_dev = (dst.device_global_index >= 0)
                    ? dst.device_global_index : static_cast<int>(dst_idx);
                if (src_dev != dst_dev) {
                    total_cost_s += estimate_migration_cost_bytes(
                        *topo_metrics_, src_dev, dst_dev,
                        src.owning_rank, dst.owning_rank, len * elem_bytes);
                }
            }

            if (src_end <= dst_end) ++src_idx;
            if (dst_end <= src_end) ++dst_idx;
        }
    }
    return total_cost_s;
}

bool Runtime::Impl::passes_numa_cost_gate(const std::vector<FieldHandle>& fields,
                                          const std::vector<DevicePartition>& old_parts,
                                          const std::vector<DevicePartition>& new_parts,
                                          const AutoBalancePolicy& policy,
                                          BalanceRecord& record) {
    int local_skip = 0;
    double total_migration_cost_s = 0.0;
    double expected_gain_s = 0.0;

    if (topo_metrics_.has_value() && !policy.use_power_cap) {
        const double numa_ratio_limit = (policy.numa_cost_gain_ratio_threshold >= 0.0)
            ? policy.numa_cost_gain_ratio_threshold
            : numa_cost_gain_ratio_threshold_;
        total_migration_cost_s = estimate_total_migration_cost_seconds(fields, old_parts, new_parts);

        const std::vector<double>& times = record.window_seconds;
        double t_max_current = 0.0;
        for (std::size_t i = 0; i < old_parts.size() && i < times.size(); ++i) {
            t_max_current = std::max(t_max_current, times[i]);
        }
        std::vector<double> capacity;
        const double total_cap = detail::compute_partition_capacities(
            times, old_parts, &(*topo_metrics_), policy.use_contention_adjustment, capacity);

        double t_max_projected = t_max_current;
        if (total_cap > 0.0) {
            t_max_projected = 0.0;
            for (std::size_t i = 0; i < new_parts.size(); ++i) {
                if (new_parts[i].element_count == 0) continue;
                if (i >= capacity.size() || capacity[i] <= 0.0) {
                    t_max_projected = t_max_current;
                    break;
                }
                t_max_projected = std::max(
                    t_max_projected, static_cast<double>(new_parts[i].element_count) / capacity[i]);
            }
        }
        expected_gain_s = std::max(0.0, t_max_current - t_max_projected);
        local_skip = (total_migration_cost_s > numa_ratio_limit * expected_gain_s) ? 1 : 0;
    }

    int global_skip = 0;
    {
        PhaseTimer mpi_timer(*this, Phase::mpi_balance);
        detail::check_mpi(
            MPI_Allreduce(&local_skip, &global_skip, 1, MPI_INT, MPI_LOR, comm_),
            "MPI_Allreduce(NUMA decision to skip rebalance)");
    }

    if (global_skip != 0) {
        if (verbose_ && rank_ == 0) {
            std::cout << "[NUMA] migration cost exceeds gain threshold, skipping rebalance" << std::endl;
        }
        record.reason = "[NUMA] migration cost exceeds gain threshold (cost=" +
                        detail::format_seconds(total_migration_cost_s) +
                        ", gain=" + detail::format_seconds(expected_gain_s) + ")";
        reset_balance_window_events();
        return false;
    }
    return true;
}

bool Runtime::Impl::maybe_rebalance_from_timings(const std::vector<FieldHandle>& requested_fields,
                                                 const AutoBalancePolicy& policy, BalanceRecord& record) {
    if (!partition_.has_value() || partitions_.empty()) {
        record.reason = "no partition defined yet";
        return false;
    }
    const std::vector<FieldHandle> fields = unique_existing_proportional_fields(requested_fields);

    std::vector<DevicePartition> new_parts;
    if (!propose_division(new_parts, policy, record, "MPI_Allreduce(balance times)")) return false;
    const std::vector<DevicePartition> old_parts = partitions_;

    const double distance = detail::l2_norm_diff(current_loads_, loads_from_partitions(new_parts));
    const bool accept = distance >= static_cast<double>(policy.threshold);
    record.reason = "division change " + detail::format_number(distance, 3) + (accept ? " >= " : " < ") +
                    "threshold " + detail::format_number(policy.threshold, 4);
    if (!accept) {
        reset_balance_window_events();
        return false;
    }

    if (!passes_numa_cost_gate(fields, old_parts, new_parts, policy, record)) {
        return false;
    }

    apply_partitions(fields, old_parts, new_parts, record);
    return true;
}

namespace {

constexpr double kHierarchicalHysteresisMinGain = 0.05;

void compute_rank_cuts(const std::vector<float>& inter_cum_loads,
                       const std::vector<int>& rank_device_counts,
                       std::size_t global_elements,
                       std::size_t gran,
                       std::vector<std::size_t>& rank_cuts) {
    const std::size_t n_ranks = inter_cum_loads.size();
    const std::size_t total_units = global_elements / gran;
    rank_cuts.assign(n_ranks + 1, 0);

    std::size_t last_active = n_ranks;
    for (std::size_t r = n_ranks; r > 0; --r) {
        if (rank_device_counts[r - 1] > 0) {
            last_active = r - 1;
            break;
        }
    }

    for (std::size_t r = 0; r + 1 < n_ranks; ++r) {
        if (rank_device_counts[r] == 0) {
            rank_cuts[r + 1] = rank_cuts[r];
            continue;
        }
        if (r >= last_active) {
            rank_cuts[r + 1] = global_elements;
            continue;
        }
        const double raw = static_cast<double>(inter_cum_loads[r]) * static_cast<double>(total_units);
        std::size_t cut = static_cast<std::size_t>(std::llround(raw)) * gran;
        rank_cuts[r + 1] = std::clamp(cut, rank_cuts[r], global_elements);
    }
    rank_cuts.back() = global_elements;
}

void compute_local_device_cuts(std::size_t rank_start,
                               std::size_t rank_end,
                               std::size_t gran,
                               const std::vector<double>& p_local,
                               std::vector<unsigned long long>& local_offsets,
                               std::vector<unsigned long long>& local_counts) {
    const std::size_t n_local = p_local.size();
    const std::size_t rank_elements = rank_end - rank_start;
    const std::size_t rank_units = rank_elements / gran;

    std::vector<std::size_t> dev_cuts(n_local + 1, rank_start);
    double cum_p = 0.0;
    for (std::size_t li = 0; li + 1 < n_local; ++li) {
        cum_p += p_local[li];
        const double raw = cum_p * static_cast<double>(rank_units);
        const std::size_t cut = rank_start + static_cast<std::size_t>(std::llround(raw)) * gran;
        dev_cuts[li + 1] = std::clamp(cut, dev_cuts[li], rank_end);
    }
    dev_cuts.back() = rank_end;

    local_offsets.resize(n_local);
    local_counts.resize(n_local);
    for (std::size_t li = 0; li < n_local; ++li) {
        local_offsets[li] = static_cast<unsigned long long>(dev_cuts[li]);
        local_counts[li] = static_cast<unsigned long long>(dev_cuts[li + 1] - dev_cuts[li]);
    }
}

struct RankPhase1Result {
    std::vector<std::size_t> part_indices;
    std::vector<double> p_local;
    double local_capacity{0.0};
    double local_max_time{0.0};
    std::size_t rank_elements{0};
};

void compute_inter_node_loads(int num_ranks,
                              std::size_t global_elements,
                              const std::vector<double>& rank_data,
                              const std::vector<int>& rank_device_counts,
                              BalanceAlgorithm algorithm,
                              float perturbation_factor,
                              unsigned int attempt_seed,
                              bool verbose,
                              std::vector<float>& inter_cum_loads,
                              int& abort_migration_flag,
                              double& global_max_time,
                              double& projected_time) {
    double total_capacity = 0.0;
    global_max_time = 0.0;
    double inter_baseline_time = 0.0;
    projected_time = 0.0;
    std::vector<double> capacities(static_cast<std::size_t>(num_ranks), 0.0);
    for (int r = 0; r < num_ranks; ++r) {
        const double r_elements = rank_data[static_cast<std::size_t>(r * 3 + 0)];
        const double r_max_time = rank_data[static_cast<std::size_t>(r * 3 + 1)];
        const double r_capacity = rank_data[static_cast<std::size_t>(r * 3 + 2)];
        total_capacity += r_capacity;
        global_max_time = std::max(global_max_time, r_max_time);
        const double r_intra_time = (r_capacity > 0.0) ? (r_elements / r_capacity) : r_max_time;
        inter_baseline_time = std::max(inter_baseline_time, r_intra_time);
        capacities[static_cast<std::size_t>(r)] = r_capacity;
    }

    if (total_capacity > 0.0) {
        projected_time = static_cast<double>(global_elements) / total_capacity;
    }

    abort_migration_flag = 0;
    if (inter_baseline_time > 0.0 && projected_time > 0.0 &&
        (inter_baseline_time - projected_time) < kHierarchicalHysteresisMinGain * inter_baseline_time) {
        abort_migration_flag = 1;
        if (verbose) {
            std::cout << "[HIER] Inter-node migration aborted due to hysteresis (gain < 5%). projected_time="
                      << projected_time << " global_max_time=" << global_max_time << std::endl;
        }
    }

    inter_cum_loads.assign(static_cast<std::size_t>(num_ranks), 1.0f);
    if (abort_migration_flag != 0) {
        double elements_accum = 0.0;
        for (int r = 0; r < num_ranks; ++r) {
            elements_accum += rank_data[static_cast<std::size_t>(r * 3)];
            inter_cum_loads[static_cast<std::size_t>(r)] = static_cast<float>(
                elements_accum / static_cast<double>(global_elements));
        }
        inter_cum_loads.back() = 1.0f;
    } else {
        std::vector<double> shares(static_cast<std::size_t>(num_ranks), 0.0);
        if (total_capacity <= 0.0) {
            int active_ranks = 0;
            for (int r = 0; r < num_ranks; ++r) {
                if (static_cast<std::size_t>(r) < rank_device_counts.size() &&
                    rank_device_counts[static_cast<std::size_t>(r)] > 0) {
                    ++active_ranks;
                }
            }
            if (active_ranks == 0) {
                std::fill(shares.begin(), shares.end(), 1.0 / static_cast<double>(num_ranks));
            } else {
                for (int r = 0; r < num_ranks; ++r) {
                    if (static_cast<std::size_t>(r) < rank_device_counts.size() &&
                        rank_device_counts[static_cast<std::size_t>(r)] > 0) {
                        shares[static_cast<std::size_t>(r)] = 1.0 / static_cast<double>(active_ranks);
                    }
                }
            }
        } else {
            for (int r = 0; r < num_ranks; ++r) {
                shares[static_cast<std::size_t>(r)] = capacities[static_cast<std::size_t>(r)] / total_capacity;
            }
        }

        if (algorithm == BalanceAlgorithm::hwtopolb && perturbation_factor > 0.0f) {
            std::vector<std::size_t> active_indices;
            std::vector<double> active_shares;
            for (int r = 0; r < num_ranks; ++r) {
                const bool active = static_cast<std::size_t>(r) >= rank_device_counts.size() ||
                                    rank_device_counts[static_cast<std::size_t>(r)] > 0;
                if (active) {
                    active_indices.push_back(static_cast<std::size_t>(r));
                    active_shares.push_back(shares[static_cast<std::size_t>(r)]);
                }
            }
            detail::perturb_shares(active_shares, perturbation_factor, attempt_seed);
            for (std::size_t k = 0; k < active_indices.size(); ++k) {
                shares[active_indices[k]] = active_shares[k];
            }
        }

        double cum_cap = 0.0;
        for (int r = 0; r < num_ranks; ++r) {
            cum_cap += shares[static_cast<std::size_t>(r)];
            inter_cum_loads[static_cast<std::size_t>(r)] = static_cast<float>(cum_cap);
        }
        inter_cum_loads.back() = 1.0f;
    }

    if (verbose) {
        const std::vector<float> inter_rank_loads = detail::cumulative_to_individual(inter_cum_loads);
        std::cout << "[HIER] inter-node loads: [";
        for (std::size_t r = 0; r < inter_rank_loads.size(); ++r) {
            if (r > 0) std::cout << ", ";
            std::cout << inter_rank_loads[r];
        }
        std::cout << "]" << std::endl;
    }
}

} // namespace

bool Runtime::Impl::maybe_rebalance_hierarchical(const std::vector<FieldHandle>& requested_fields,
                                                 const AutoBalancePolicy& policy,
                                                 BalanceRecord& record) {
    if (!partition_.has_value() || partitions_.empty()) {
        record.reason = "no partition defined yet";
        return false;
    }

    record.window_iterations = iteration_counter_ - balance_window_start_iteration_;
    const std::vector<FieldHandle> fields = unique_existing_proportional_fields(requested_fields);
    const std::vector<DevicePartition> old_parts = partitions_;
    const bool apply_contention = policy.use_contention_adjustment && topo_metrics_.has_value() &&
                                  !topo_metrics_->memory_contention_factor.empty();
    record.contention_adjusted = apply_contention;
    record.algorithm = (policy.algorithm == BalanceAlgorithm::hwtopolb) ? "hwtopolb" : "erad";
    record.perturbation_factor =
        (policy.algorithm == BalanceAlgorithm::hwtopolb) ? policy.perturbation_factor : 0.0f;
    record.seed = policy.seed;
    record.power_capped = false;
    record.power_budget_watts = policy.use_power_cap ? policy.power_budget_watts : 0.0;
    record.throttle_factor = 1.0;

    const bool use_hwtopolb =
        (policy.algorithm == BalanceAlgorithm::hwtopolb && policy.perturbation_factor > 0.0f);
    unsigned int attempt_seed = policy.seed;
    if (use_hwtopolb) {
        attempt_seed = synchronize_hwtopolb_seed(policy.seed);
        record.seed = attempt_seed;
    }

    const int num_ranks = std::max(1, size());
    const bool in_process_sim_ranks = (mpi_size_ == 1 && simulated_ranks_ > 1);
    const std::vector<double> window_elapsed = simulated_times_.empty()
        ? compute_balance_window_elapsed_local()
        : std::vector<double>{};

    auto evaluate_phase1 = [&](int target_rank, bool update_timings) {
        RankPhase1Result res;
        for (std::size_t i = 0; i < old_parts.size(); ++i) {
            if (old_parts[i].owning_rank == target_rank) res.part_indices.push_back(i);
        }
        const std::size_t n_local = res.part_indices.size();
        std::vector<double> local_times(n_local, 0.0);
        for (std::size_t li = 0; li < n_local; ++li) {
            const std::size_t p_idx = res.part_indices[li];
            const int g = old_parts[p_idx].device_global_index;
            if (!simulated_times_.empty()) {
                if (g >= 0 && static_cast<std::size_t>(g) < simulated_times_.size()) {
                    local_times[li] = simulated_times_[static_cast<std::size_t>(g)];
                } else if (li < simulated_times_.size()) {
                    local_times[li] = simulated_times_[li];
                }
            } else if (li < window_elapsed.size() && window_elapsed[li] > 1.0e-12) {
                local_times[li] = window_elapsed[li];
            } else if (li < device_timings_.size()) {
                if (device_timings_[li].kernel_seconds_last > 1.0e-12) {
                    local_times[li] = device_timings_[li].kernel_seconds_last;
                } else if (device_timings_[li].kernel_seconds_avg > 1.0e-12) {
                    local_times[li] = device_timings_[li].kernel_seconds_avg;
                }
            }
            if (local_times[li] <= 1.0e-12) local_times[li] = 1.0;
            local_times[li] = std::max(local_times[li], 1.0e-9);
        }
        if (update_timings && !simulated_times_.empty()) {
            update_device_timings(local_times);
        }

        // Unit-element floor on active devices prevents permanent starvation of 0-element ranks.
        res.p_local.assign(n_local, 0.0);
        for (std::size_t li = 0; li < n_local; ++li) {
            const std::size_t p_idx = res.part_indices[li];
            const double elements = std::max(1.0, static_cast<double>(old_parts[p_idx].element_count));
            double cap = elements / local_times[li];
            if (apply_contention) {
                const int dev_idx = (old_parts[p_idx].device_global_index >= 0)
                    ? old_parts[p_idx].device_global_index
                    : static_cast<int>(p_idx);
                cap = adjusted_capacity(*topo_metrics_, dev_idx, cap);
            }
            res.p_local[li] = cap;
            res.local_capacity += cap;
            res.rank_elements += old_parts[p_idx].element_count;
        }
        if (n_local > 0) {
            if (res.local_capacity > 0.0) {
                for (double& share : res.p_local) share /= res.local_capacity;
            } else {
                std::fill(res.p_local.begin(), res.p_local.end(), 1.0 / static_cast<double>(n_local));
            }
            if (use_hwtopolb && n_local > 1) {
                unsigned int local_seed =
                    attempt_seed ^ (static_cast<unsigned int>(target_rank) * 0x9e3779b9u);
                if (local_seed == 0) local_seed = 1u;
                detail::perturb_shares(res.p_local, policy.perturbation_factor, local_seed);
            }
            res.local_max_time = std::max(*std::max_element(local_times.begin(), local_times.end()), 1.0e-9);
        }
        return res;
    };

    const RankPhase1Result my_phase1 = evaluate_phase1(rank_, true);

    if (verbose_ && rank_ == 0) {
        std::cout << "[HIER] intra-node loads: [";
        for (std::size_t li = 0; li < my_phase1.p_local.size(); ++li) {
            if (li > 0) std::cout << ", ";
            std::cout << my_phase1.p_local[li];
        }
        std::cout << "]" << std::endl;
    }

    {
        PhaseTimer wait_timer(*this, Phase::device_wait);
        synchronize_all_local_devices(false);
    }

    std::vector<int> rank_device_counts = all_device_counts_;
    if (rank_device_counts.size() != static_cast<std::size_t>(num_ranks)) {
        rank_device_counts.assign(static_cast<std::size_t>(num_ranks), 0);
        for (const DevicePartition& dp : old_parts) {
            if (dp.owning_rank >= 0 && dp.owning_rank < num_ranks) {
                ++rank_device_counts[static_cast<std::size_t>(dp.owning_rank)];
            }
        }
    }

    const std::size_t gran = std::max<std::size_t>(1, partition_->granularity);
    std::vector<DevicePartition> new_parts = old_parts;
    int abort_migration_flag = 0;
    double global_max_time = 0.0;
    double projected_time = 0.0;

    if (in_process_sim_ranks) {
        std::vector<RankPhase1Result> all_phase1(static_cast<std::size_t>(num_ranks));
        std::vector<double> rank_data(static_cast<std::size_t>(num_ranks) * 3, 0.0);
        for (int r = 0; r < num_ranks; ++r) {
            all_phase1[static_cast<std::size_t>(r)] = (r == rank_) ? my_phase1 : evaluate_phase1(r, false);
            rank_data[static_cast<std::size_t>(r * 3 + 0)] = static_cast<double>(all_phase1[r].rank_elements);
            rank_data[static_cast<std::size_t>(r * 3 + 1)] = all_phase1[r].local_max_time;
            rank_data[static_cast<std::size_t>(r * 3 + 2)] = all_phase1[r].local_capacity;
        }
        std::vector<float> inter_cum_loads;
        compute_inter_node_loads(num_ranks, partition_->global_elements, rank_data, rank_device_counts,
                                 policy.algorithm, policy.perturbation_factor, attempt_seed, verbose_,
                                 inter_cum_loads, abort_migration_flag, global_max_time, projected_time);
        std::vector<std::size_t> rank_cuts;
        compute_rank_cuts(inter_cum_loads, rank_device_counts, partition_->global_elements, gran, rank_cuts);

        for (int r = 0; r < num_ranks; ++r) {
            const RankPhase1Result& rp = all_phase1[static_cast<std::size_t>(r)];
            std::size_t r_start = 0;
            std::size_t r_end = 0;
            if (abort_migration_flag != 0 && !rp.part_indices.empty()) {
                r_start = old_parts[rp.part_indices.front()].global_offset;
                r_end = old_parts[rp.part_indices.back()].global_offset +
                        old_parts[rp.part_indices.back()].element_count;
            } else {
                r_start = rank_cuts[static_cast<std::size_t>(r)];
                r_end = rank_cuts[static_cast<std::size_t>(r) + 1];
            }
            std::vector<unsigned long long> offs;
            std::vector<unsigned long long> cnts;
            compute_local_device_cuts(r_start, r_end, gran, rp.p_local, offs, cnts);
            for (std::size_t li = 0; li < rp.part_indices.size(); ++li) {
                const std::size_t p_idx = rp.part_indices[li];
                new_parts[p_idx].global_offset = static_cast<std::size_t>(offs[li]);
                new_parts[p_idx].element_count = static_cast<std::size_t>(cnts[li]);
            }
        }
    } else {
        double local_data[3] = {
            static_cast<double>(my_phase1.rank_elements), my_phase1.local_max_time, my_phase1.local_capacity
        };
        std::vector<double> rank_data(static_cast<std::size_t>(mpi_size_) * 3, 0.0);
        {
            PhaseTimer mpi_timer(*this, Phase::mpi_balance);
            detail::check_mpi(
                MPI_Gather(local_data, 3, MPI_DOUBLE, rank_data.data(), 3, MPI_DOUBLE, 0, comm_),
                "MPI_Gather(hierarchical rank capacities)");
        }

        std::vector<float> inter_cum_loads(static_cast<std::size_t>(mpi_size_), 0.0f);
        if (rank_ == 0) {
            compute_inter_node_loads(mpi_size_, partition_->global_elements, rank_data, rank_device_counts,
                                     policy.algorithm, policy.perturbation_factor, attempt_seed, verbose_,
                                     inter_cum_loads, abort_migration_flag, global_max_time, projected_time);
        }

        {
            PhaseTimer mpi_timer(*this, Phase::mpi_balance);
            detail::check_mpi(
                MPI_Bcast(inter_cum_loads.data(), mpi_size_, MPI_FLOAT, 0, comm_),
                "MPI_Bcast(hierarchical inter_cum_loads)");
            detail::check_mpi(
                MPI_Bcast(&abort_migration_flag, 1, MPI_INT, 0, comm_),
                "MPI_Bcast(hierarchical abort_migration_flag)");
        }

        std::vector<std::size_t> rank_cuts;
        compute_rank_cuts(inter_cum_loads, rank_device_counts, partition_->global_elements, gran, rank_cuts);

        const std::size_t n_local = my_phase1.part_indices.size();
        std::size_t rank_start = 0;
        std::size_t rank_end = 0;
        if (abort_migration_flag != 0 && n_local > 0) {
            rank_start = old_parts[my_phase1.part_indices.front()].global_offset;
            rank_end = old_parts[my_phase1.part_indices.back()].global_offset +
                       old_parts[my_phase1.part_indices.back()].element_count;
        } else {
            rank_start = rank_cuts[static_cast<std::size_t>(rank_)];
            rank_end = rank_cuts[static_cast<std::size_t>(rank_) + 1];
        }

        std::vector<unsigned long long> local_offsets;
        std::vector<unsigned long long> local_counts;
        compute_local_device_cuts(rank_start, rank_end, gran, my_phase1.p_local, local_offsets, local_counts);

        const std::size_t total_devices = old_parts.size();
        if (abort_migration_flag == 0) {
            std::vector<int> displs(static_cast<std::size_t>(mpi_size_), 0);
            for (int r = 1; r < mpi_size_; ++r) {
                displs[static_cast<std::size_t>(r)] =
                    displs[static_cast<std::size_t>(r - 1)] + rank_device_counts[static_cast<std::size_t>(r - 1)];
            }
            std::vector<unsigned long long> all_offsets(total_devices, 0ULL);
            std::vector<unsigned long long> all_counts(total_devices, 0ULL);
            {
                PhaseTimer mpi_timer(*this, Phase::mpi_balance);
                detail::check_mpi(
                    MPI_Allgatherv(local_offsets.data(), static_cast<int>(n_local), MPI_UNSIGNED_LONG_LONG,
                                   all_offsets.data(), rank_device_counts.data(), displs.data(),
                                   MPI_UNSIGNED_LONG_LONG, comm_),
                    "MPI_Allgatherv(hierarchical all_offsets)");
                detail::check_mpi(
                    MPI_Allgatherv(local_counts.data(), static_cast<int>(n_local), MPI_UNSIGNED_LONG_LONG,
                                   all_counts.data(), rank_device_counts.data(), displs.data(),
                                   MPI_UNSIGNED_LONG_LONG, comm_),
                    "MPI_Allgatherv(hierarchical all_counts)");
            }
            for (std::size_t g = 0; g < total_devices; ++g) {
                new_parts[g].global_offset = static_cast<std::size_t>(all_offsets[g]);
                new_parts[g].element_count = static_cast<std::size_t>(all_counts[g]);
            }

            // C1 collective partition conservation check.
            std::size_t covered = 0;
            for (const DevicePartition& dp : new_parts) covered += dp.element_count;
            unsigned long long covered_ull = static_cast<unsigned long long>(covered);
            unsigned long long min_covered = 0;
            unsigned long long max_covered = 0;
            {
                PhaseTimer mpi_timer(*this, Phase::mpi_balance);
                detail::check_mpi(
                    MPI_Allreduce(&covered_ull, &min_covered, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, comm_),
                    "MPI_Allreduce(C1 partition conservation min)");
                detail::check_mpi(
                    MPI_Allreduce(&covered_ull, &max_covered, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, comm_),
                    "MPI_Allreduce(C1 partition conservation max)");
            }
            if (min_covered != static_cast<unsigned long long>(partition_->global_elements) ||
                max_covered != static_cast<unsigned long long>(partition_->global_elements)) {
                reset_balance_window_events();
                record.reason = "hierarchical partition conservation check failed";
                return false;
            }
        } else {
            const int max_devs_per_rank = rank_device_counts.empty()
                ? 0
                : *std::max_element(rank_device_counts.begin(), rank_device_counts.end());
            if (max_devs_per_rank > 1) {
                // Multiple devices on at least one rank: publish intra-node cuts cleanly via one Allreduce.
                std::vector<unsigned long long> local_metadata(total_devices * 2, 0ULL);
                std::vector<unsigned long long> global_metadata(total_devices * 2, 0ULL);
                for (std::size_t li = 0; li < n_local; ++li) {
                    const std::size_t p_idx = my_phase1.part_indices[li];
                    local_metadata[p_idx * 2] = local_offsets[li];
                    local_metadata[p_idx * 2 + 1] = local_counts[li];
                }
                {
                    PhaseTimer mpi_timer(*this, Phase::mpi_balance);
                    detail::check_mpi(
                        MPI_Allreduce(local_metadata.data(), global_metadata.data(),
                                      static_cast<int>(global_metadata.size()), MPI_UNSIGNED_LONG_LONG,
                                      MPI_SUM, comm_),
                        "MPI_Allreduce(hierarchical local partition metadata)");
                }
                for (std::size_t g = 0; g < total_devices; ++g) {
                    new_parts[g].global_offset = static_cast<std::size_t>(global_metadata[g * 2]);
                    new_parts[g].element_count = static_cast<std::size_t>(global_metadata[g * 2 + 1]);
                }
            }
        }
    }

    if (policy.use_power_cap) {
        std::vector<float> proposed = loads_from_partitions(new_parts);
        apply_collective_power_cap(proposed, policy, record);
        int proposed_count = (rank_ == 0) ? static_cast<int>(proposed.size()) : 0;
        {
            PhaseTimer mpi_timer(*this, Phase::mpi_balance);
            detail::check_mpi(MPI_Bcast(&proposed_count, 1, MPI_INT, 0, comm_),
                              "MPI_Bcast(hierarchical power cap size)");
            if (rank_ != 0) {
                proposed.resize(static_cast<std::size_t>(proposed_count));
            }
            if (proposed_count > 0) {
                detail::check_mpi(MPI_Bcast(proposed.data(), proposed_count, MPI_FLOAT, 0, comm_),
                                  "MPI_Bcast(hierarchical power cap loads)");
            }
        }
        std::vector<DevicePartition> capped_parts = partitions_from_loads(proposed, /*allow_empty=*/true);
        if (!capped_parts.empty() && capped_parts.size() == new_parts.size()) {
            new_parts = std::move(capped_parts);
        }
    }

    if (!in_process_sim_ranks && mpi_size_ > 1) {
        double timing_pair[2] = {global_max_time, projected_time};
        PhaseTimer mpi_timer(*this, Phase::mpi_balance);
        detail::check_mpi(MPI_Bcast(timing_pair, 2, MPI_DOUBLE, 0, comm_),
                          "MPI_Bcast(hierarchical timing estimates)");
        global_max_time = timing_pair[0];
        projected_time = timing_pair[1];
    }
    {
        const double window_n = static_cast<double>(std::max<std::size_t>(record.window_iterations, 1));
        record.iteration_seconds_before = global_max_time / window_n;
        record.iteration_seconds_after = (abort_migration_flag != 0 && !record.power_capped)
            ? record.iteration_seconds_before
            : (projected_time / window_n);
    }

    int local_invalid_partition = 0;
    std::size_t covered_elements = 0;
    for (const DevicePartition& part : new_parts) {
        if (part.global_offset != covered_elements ||
            part.element_count > partition_->global_elements - covered_elements) {
            local_invalid_partition = 1;
            break;
        }
        covered_elements += part.element_count;
    }
    if (covered_elements != partition_->global_elements) local_invalid_partition = 1;
    if ((abort_migration_flag == 0 || policy.use_power_cap) && !in_process_sim_ranks) {
        int any_invalid_partition = 0;
        {
            PhaseTimer mpi_timer(*this, Phase::mpi_balance);
            detail::check_mpi(
                MPI_Allreduce(&local_invalid_partition, &any_invalid_partition, 1, MPI_INT, MPI_MAX, comm_),
                "MPI_Allreduce(hierarchical partition coverage)");
        }
        if (any_invalid_partition != 0) {
            throw Error("Hierarchical partition has a gap, overlap, or missing elements");
        }
    } else if (local_invalid_partition != 0) {
        throw Error("Hierarchical partition has a gap, overlap, or missing elements");
    }

    const bool unchanged = detail::same_partition_layout(old_parts, new_parts);
    if (unchanged) {
        current_loads_ = loads_from_partitions(partitions_);
        {
            PhaseTimer wait_timer(*this, Phase::device_wait);
            synchronize_all_local_devices(true);
        }
        reset_balance_window_events();
        record.reason = (abort_migration_flag != 0 && !record.power_capped)
            ? "inter-node migration aborted due to hysteresis (gain < 5%)"
            : "the division is already balanced";
    } else {
        apply_partitions(fields, old_parts, new_parts, record,
                         /*synchronize_ranks=*/(!in_process_sim_ranks &&
                                                (abort_migration_flag == 0 || policy.use_power_cap)));
        record.reason = (abort_migration_flag != 0 && !record.power_capped)
            ? "intra-node rebalance applied (inter-node migration aborted due to hysteresis)"
            : "hierarchical two-phase rebalance applied";
    }

    if (verbose_ && rank_ == 0) {
        std::cout << "[HIER] final partition: [";
        for (std::size_t i = 0; i < partitions_.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << "dev " << partitions_[i].device_global_index
                      << ": offset=" << partitions_[i].global_offset
                      << " count=" << partitions_[i].element_count;
        }
        std::cout << "]" << std::endl;
    }

    return !unchanged;
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
    if (!propose_division(new_parts, policy, record, "MPI_Allreduce(profiled balance times)")) return false;
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
    std::vector<std::size_t> rank_recv_bytes(static_cast<std::size_t>(std::max(1, size())), 0);

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
