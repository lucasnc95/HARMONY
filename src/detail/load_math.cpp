#include "detail/load_math.hpp"
#include "harmony/algorithms.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>

namespace harmony {

namespace {

std::vector<float> cumulative_from_shares(const std::vector<double>& shares) {
    std::vector<float> loads(shares.size(), 0.0f);
    if (shares.empty()) return loads;

    double cumulative = 0.0;
    float prev = 0.0f;
    for (std::size_t i = 0; i < shares.size(); ++i) {
        cumulative += shares[i];
        const float val = std::clamp(static_cast<float>(cumulative), prev, 1.0f);
        loads[i] = val;
        prev = val;
    }
    loads.back() = 1.0f;
    return loads;
}

std::vector<float> erad_loads_impl(
    const std::vector<double>& times,
    const std::vector<DevicePartition>& partitions,
    const TopoMetrics* metrics,
    bool use_contention_adjustment) {
    if (times.empty() || times.size() != partitions.size()) {
        return {};
    }

    std::vector<double> capacity;
    const double total_capacity = detail::compute_partition_capacities(
        times, partitions, metrics, use_contention_adjustment, capacity);
    if (total_capacity <= 0.0) {
        return detail::equal_cumulative_loads(times.size());
    }

    std::vector<double> shares(capacity.size(), 0.0);
    for (std::size_t i = 0; i < capacity.size(); ++i) {
        shares[i] = capacity[i] / total_capacity;
    }
    return cumulative_from_shares(shares);
}

std::vector<float> hwtopolb_loads_impl(
    const std::vector<double>& times,
    const std::vector<DevicePartition>& partitions,
    float perturbation_factor,
    unsigned int seed,
    const TopoMetrics* metrics,
    bool use_contention_adjustment) {
    if (times.empty() || times.size() != partitions.size()) {
        return {};
    }
    if (!std::isfinite(perturbation_factor)) {
        throw Error("hwtopolb_loads(): perturbation_factor must be finite");
    }
    if (perturbation_factor <= 0.0f) {
        return erad_loads_impl(times, partitions, metrics, use_contention_adjustment);
    }

    const std::size_t n = times.size();
    std::vector<double> capacity;
    const double total_capacity = detail::compute_partition_capacities(
        times, partitions, metrics, use_contention_adjustment, capacity);

    std::vector<double> share(n, 0.0);
    if (total_capacity <= 0.0) {
        const double equal_share = 1.0 / static_cast<double>(n);
        std::fill(share.begin(), share.end(), equal_share);
    } else {
        for (std::size_t i = 0; i < n; ++i) {
            share[i] = capacity[i] / total_capacity;
        }
    }

    detail::perturb_shares(share, perturbation_factor, seed);
    return cumulative_from_shares(share);
}

} // namespace

double adjusted_capacity(
    const TopoMetrics& metrics,
    int global_device,
    double raw_throughput) {
    if (global_device < 0 ||
        static_cast<std::size_t>(global_device) >= metrics.memory_contention_factor.size()) {
        throw Error("Device index out of range for memory_contention_factor");
    }
    const double factor = metrics.memory_contention_factor[static_cast<std::size_t>(global_device)];
    if (!std::isfinite(factor) || factor < 1.0) {
        throw Error("memory_contention_factor must be finite and >= 1.0");
    }
    if (!std::isfinite(raw_throughput) || raw_throughput < 0.0) {
        throw Error("raw_throughput must be finite and >= 0.0");
    }
    return raw_throughput / factor;
}

std::vector<float> erad_loads(
    const std::vector<double>& times,
    const std::vector<DevicePartition>& partitions) {
    return erad_loads_impl(times, partitions, nullptr, false);
}

std::vector<float> erad_loads(
    const std::vector<double>& times,
    const std::vector<DevicePartition>& partitions,
    const TopoMetrics& metrics,
    bool use_contention_adjustment) {
    return erad_loads_impl(times, partitions, &metrics, use_contention_adjustment);
}

std::vector<float> hwtopolb_loads(
    const std::vector<double>& times,
    const std::vector<DevicePartition>& partitions,
    float perturbation_factor,
    unsigned int seed) {
    return hwtopolb_loads_impl(times, partitions, perturbation_factor, seed, nullptr, false);
}

std::vector<float> hwtopolb_loads(
    const std::vector<double>& times,
    const std::vector<DevicePartition>& partitions,
    float perturbation_factor,
    unsigned int seed,
    const TopoMetrics& metrics,
    bool use_contention_adjustment) {
    return hwtopolb_loads_impl(
        times, partitions, perturbation_factor, seed, &metrics, use_contention_adjustment);
}

std::vector<float> apply_power_cap(
    const std::vector<float>& loads,
    const TopoMetrics& metrics,
    double power_budget_watts,
    double* out_throttle_factor,
    const std::vector<float>* current_loads) {
    if (out_throttle_factor != nullptr) {
        *out_throttle_factor = 1.0;
    }
    if (!std::isfinite(power_budget_watts) || power_budget_watts < 0.0) {
        throw Error("Invalid power_budget_watts");
    }
    if (loads.empty()) return {};
    if (metrics.current_power_watts.empty() || metrics.thermal_tdp_watts.empty()) {
        return loads;
    }

    const std::size_t n = loads.size();
    const auto& measured_loads = (current_loads != nullptr) ? *current_loads : loads;
    if (measured_loads.size() != n || metrics.current_power_watts.size() != n ||
        metrics.thermal_tdp_watts.size() != n) {
        throw Error("Invalid power cap input dimensions");
    }

    std::vector<double> desired(n, 0.0);
    std::vector<double> capacity(n, 0.0);
    std::vector<double> slope(n, 0.0);
    std::vector<double> shares(n, 0.0);
    double previous_desired = 0.0;
    double previous_current = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double proposed = loads[i];
        const double measured = measured_loads[i];
        const double watts = metrics.current_power_watts[i];
        const double tdp = metrics.thermal_tdp_watts[i];
        if (!std::isfinite(proposed) || !std::isfinite(measured) ||
            !std::isfinite(watts) || !std::isfinite(tdp) ||
            proposed < previous_desired || measured < previous_current ||
            watts < 0.0 || tdp <= 0.0) {
            throw Error("Invalid power cap measurement or load vector");
        }
        desired[i] = proposed - previous_desired;
        const double current_share = measured - previous_current;
        previous_desired = proposed;
        previous_current = measured;
        if (current_share <= 0.0 || watts <= 0.0) {
            if (desired[i] > 0.0) {
                throw Error("Power cap requires measured positive load and power on every proposed device");
            }
            capacity[i] = 0.0;
            slope[i] = 0.0;
        } else {
            slope[i] = watts / current_share;
            capacity[i] = std::min(1.0, tdp / slope[i]);
        }
        shares[i] = std::min(desired[i], capacity[i]);
    }
    if (std::fabs(previous_desired - 1.0) > 1.0e-5 ||
        std::fabs(previous_current - 1.0) > 1.0e-5) {
        throw Error("Power cap loads must cover all elements");
    }

    const double total_capacity = std::accumulate(capacity.begin(), capacity.end(), 0.0);
    if (total_capacity < 1.0 - 1.0e-9) {
        if (out_throttle_factor != nullptr) {
            *out_throttle_factor = (total_capacity > 0.0)
                ? (1.0 / total_capacity)
                : std::numeric_limits<double>::infinity();
        }
        throw Error("Power cap is infeasible without actual device throttling");
    }

    double missing = 1.0 - std::accumulate(shares.begin(), shares.end(), 0.0);
    while (missing > 1.0e-10) {
        double headroom = 0.0;
        for (std::size_t i = 0; i < n; ++i) headroom += capacity[i] - shares[i];
        if (headroom < missing - 1.0e-9) {
            throw Error("Power cap has insufficient remaining capacity");
        }
        const double requested = missing;
        for (std::size_t i = 0; i < n; ++i) {
            const double room = capacity[i] - shares[i];
            shares[i] += std::min(room, requested * room / headroom);
        }
        missing = 1.0 - std::accumulate(shares.begin(), shares.end(), 0.0);
    }

    const auto predicted_power = [&]() {
        double total = 0.0;
        for (std::size_t i = 0; i < n; ++i) total += slope[i] * shares[i];
        return total;
    };
    if (power_budget_watts > 0.0 && predicted_power() > power_budget_watts + 1.0e-8) {
        std::vector<std::size_t> order(n);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(),
            [&](std::size_t a, std::size_t b) { return slope[a] < slope[b]; });
        std::fill(shares.begin(), shares.end(), 0.0);
        double remaining = 1.0;
        for (std::size_t i : order) {
            shares[i] = std::min(remaining, capacity[i]);
            remaining -= shares[i];
        }
        const double min_power = predicted_power();
        if (remaining > 1.0e-9 || min_power > power_budget_watts + 1.0e-8) {
            if (out_throttle_factor != nullptr) {
                *out_throttle_factor = (remaining > 1.0e-9)
                    ? ((total_capacity > 0.0) ? (1.0 / total_capacity)
                                              : std::numeric_limits<double>::infinity())
                    : (min_power / power_budget_watts);
            }
            throw Error("Power budget is infeasible without actual device throttling");
        }
    }

    return cumulative_from_shares(shares);
}

int numa_distance_between(const TopoMetrics& metrics, int node_a, int node_b) {
    if (metrics.numa_distance.empty()) {
        throw Error("NUMA distance matrix is empty");
    }
    const std::size_t n_dist = metrics.numa_distance.size();
    const std::size_t num_nodes =
        static_cast<std::size_t>(std::llround(std::sqrt(static_cast<double>(n_dist))));
    if (num_nodes == 0 || num_nodes * num_nodes != n_dist) {
        throw Error("Invalid NUMA distance matrix dimension");
    }
    if (node_a < 0 || static_cast<std::size_t>(node_a) >= num_nodes ||
        node_b < 0 || static_cast<std::size_t>(node_b) >= num_nodes) {
        throw Error("Invalid NUMA node index");
    }
    const std::size_t a = static_cast<std::size_t>(node_a);
    const std::size_t b = static_cast<std::size_t>(node_b);
    int dist = std::max(metrics.numa_distance[a * num_nodes + b],
                        metrics.numa_distance[b * num_nodes + a]);
    const int diag_min = std::min(metrics.numa_distance[a * num_nodes + a],
                                  metrics.numa_distance[b * num_nodes + b]);
    return std::max(dist, diag_min);
}

double estimate_migration_cost_bytes(
    const TopoMetrics& metrics,
    int src_global_device,
    int dst_global_device,
    int src_rank,
    int dst_rank,
    std::size_t bytes,
    double numa_penalty_ns_per_hop) {
    if (src_global_device < 0 || dst_global_device < 0) {
        throw Error("Device index cannot be negative");
    }
    if (!std::isfinite(numa_penalty_ns_per_hop) || numa_penalty_ns_per_hop < 0.0) {
        throw Error("numa_penalty_ns_per_hop must be finite and non-negative");
    }

    auto device_pcie = [&](std::size_t dev_idx) {
        const double lat = (dev_idx < metrics.pcie_latency_ns.size())
            ? metrics.pcie_latency_ns[dev_idx] : 5000.0;
        const double bw = (dev_idx < metrics.pcie_bandwidth_gbps.size() &&
                           metrics.pcie_bandwidth_gbps[dev_idx] > 0.0)
            ? metrics.pcie_bandwidth_gbps[dev_idx] : 10.0;
        return std::pair<double, double>{lat, bw};
    };

    const std::size_t s_idx = static_cast<std::size_t>(src_global_device);
    const std::size_t d_idx = static_cast<std::size_t>(dst_global_device);
    const auto [pcie_lat_src, pcie_bw_src] = device_pcie(s_idx);
    const auto [pcie_lat_dst, pcie_bw_dst] = device_pcie(d_idx);

    double latency_s = 0.0;
    double transfer_s = 0.0;

    if (src_rank != dst_rank && src_rank >= 0 && dst_rank >= 0) {
        // Inter-rank migration: PCIe(src) + MPI(src, dst) + PCIe(dst).
        latency_s = (pcie_lat_src + pcie_lat_dst) * 1e-9;
        transfer_s = (static_cast<double>(bytes) / (pcie_bw_src * 1e9)) +
                     (static_cast<double>(bytes) / (pcie_bw_dst * 1e9));

        const std::size_t num_ranks =
            static_cast<std::size_t>(std::llround(std::sqrt(static_cast<double>(metrics.mpi_latency_ns.size()))));
        if (num_ranks == 0 || num_ranks * num_ranks != metrics.mpi_latency_ns.size() ||
            metrics.mpi_bandwidth_gbps.size() != metrics.mpi_latency_ns.size() ||
            static_cast<std::size_t>(src_rank) >= num_ranks ||
            static_cast<std::size_t>(dst_rank) >= num_ranks) {
            throw Error("MPI migration metrics do not cover the requested ranks");
        }
        const std::size_t mpi_idx =
            static_cast<std::size_t>(src_rank) * num_ranks + static_cast<std::size_t>(dst_rank);
        const double mpi_bw = metrics.mpi_bandwidth_gbps[mpi_idx];
        const double mpi_lat = metrics.mpi_latency_ns[mpi_idx];
        if (!std::isfinite(mpi_bw) || mpi_bw <= 0.0 || !std::isfinite(mpi_lat) || mpi_lat < 0.0) {
            throw Error("Invalid MPI migration metrics");
        }
        latency_s += mpi_lat * 1e-9;
        transfer_s += static_cast<double>(bytes) / (mpi_bw * 1e9);
    } else {
        // Intra-rank migration: PCIe(src) + symmetric NUMA hop penalty.
        const int src_node = (s_idx < metrics.device_numa_node.size()) ? metrics.device_numa_node[s_idx] : 0;
        const int dst_node = (d_idx < metrics.device_numa_node.size()) ? metrics.device_numa_node[d_idx] : 0;
        const double extra_latency_ns = (src_node == dst_node)
            ? 0.0
            : static_cast<double>(numa_distance_between(metrics, src_node, dst_node)) * numa_penalty_ns_per_hop;
        latency_s = (pcie_lat_src + extra_latency_ns) * 1e-9;
        transfer_s = static_cast<double>(bytes) / (pcie_bw_src * 1e9);
    }

    return latency_s + transfer_s;
}

double estimate_migration_cost_bytes(
    const TopoMetrics& metrics,
    int src_global_device,
    int dst_global_device,
    std::size_t bytes,
    double numa_penalty_ns_per_hop) {
    return estimate_migration_cost_bytes(
        metrics, src_global_device, dst_global_device, -1, -1, bytes, numa_penalty_ns_per_hop);
}

} // namespace harmony

namespace harmony::detail {

std::vector<float> cumulative_to_individual(const std::vector<float>& cumulative) {
    std::vector<float> individual(cumulative.size(), 0.0f);
    float prev = 0.0f;
    for (std::size_t i = 0; i < cumulative.size(); ++i) {
        individual[i] = cumulative[i] - prev;
        prev = cumulative[i];
    }
    return individual;
}

std::vector<float> equal_cumulative_loads(std::size_t parts) {
    std::vector<float> loads(parts, 0.0f);
    if (parts == 0) return loads;

    const float step = 1.0f / static_cast<float>(parts);
    float acc = 0.0f;
    for (std::size_t i = 0; i < parts; ++i) {
        acc += step;
        loads[i] = acc;
    }
    loads.back() = 1.0f;
    return loads;
}

std::vector<float> compute_loads_from_partition_throughput(
    const std::vector<double>& times,
    const std::vector<DevicePartition>& partitions) {
    return harmony::erad_loads(times, partitions);
}

double l2_norm_diff(const std::vector<float>& current,
                                 const std::vector<float>& proposed) {
    if (current.empty() || current.size() != proposed.size()) {
        return std::numeric_limits<double>::infinity();
    }
    double acc = 0.0;
    for (std::size_t i = 0; i < proposed.size(); ++i) {
        const double d = static_cast<double>(current[i]) - static_cast<double>(proposed[i]);
        acc += d * d;
    }
    return std::sqrt(acc);
}

bool same_partition_layout(const std::vector<DevicePartition>& a,
                           const std::vector<DevicePartition>& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i].global_offset != b[i].global_offset ||
            a[i].element_count != b[i].element_count ||
            a[i].owning_rank   != b[i].owning_rank ||
            a[i].local_index   != b[i].local_index) {
            return false;
        }
    }
    return true;
}

bool intersect_1d(std::size_t off1, std::size_t len1,
                  std::size_t off2, std::size_t len2,
                  std::size_t& out_off, std::size_t& out_len) {
    const std::size_t lo = std::max(off1, off2);
    const std::size_t hi = std::min(off1 + len1, off2 + len2);
    if (hi <= lo) {
        out_off = 0;
        out_len = 0;
        return false;
    }
    out_off = lo;
    out_len = hi - lo;
    return true;
}

double compute_partition_capacities(
    const std::vector<double>& times,
    const std::vector<DevicePartition>& partitions,
    const TopoMetrics* metrics,
    bool use_contention_adjustment,
    std::vector<double>& out_capacity) {
    const std::size_t n = std::min(times.size(), partitions.size());
    out_capacity.assign(n, 0.0);
    const bool apply_contention =
        use_contention_adjustment && metrics != nullptr && !metrics->memory_contention_factor.empty();
    double total_capacity = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        if (partitions[i].element_count == 0 || times[i] <= 1.0e-12) continue;
        double cap = static_cast<double>(partitions[i].element_count) / times[i];
        if (apply_contention) {
            const int dev_idx = (partitions[i].device_global_index >= 0)
                ? partitions[i].device_global_index
                : static_cast<int>(i);
            cap = adjusted_capacity(*metrics, dev_idx, cap);
        }
        out_capacity[i] = cap;
        total_capacity += cap;
    }
    return total_capacity;
}

void perturb_shares(std::vector<double>& shares, float perturbation_factor, unsigned int seed) {
    if (shares.empty() || perturbation_factor <= 0.0f) return;
    if (!std::isfinite(perturbation_factor)) {
        throw Error("perturb_shares(): perturbation_factor must be finite");
    }

    std::mt19937 gen;
    if (seed != 0) {
        gen.seed(seed);
    } else {
        std::random_device rd;
        gen.seed(rd());
    }

    std::uniform_real_distribution<float> dist(-perturbation_factor, perturbation_factor);
    double sum_perturbed = 0.0;
    for (double& s : shares) {
        const double delta_i = static_cast<double>(dist(gen));
        s = std::max(1.0e-6, s * (1.0 + delta_i));
        sum_perturbed += s;
    }
    if (sum_perturbed > 0.0) {
        for (double& s : shares) {
            s /= sum_perturbed;
        }
    }
}

} // namespace harmony::detail
