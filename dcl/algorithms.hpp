#ifndef DCL_ALGORITHMS_HPP
#define DCL_ALGORITHMS_HPP

#include "types.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <optional>
#include <random>
#include <vector>

namespace dcl {

/**
 * @brief ERAD deterministic load balancing algorithm (R4.1).
 *
 * Deterministic strategy: computes loads purely from partition throughput
 * (elements per second), without stochastic perturbation.
 *
 * @param times Execution times per partition.
 * @param partitions Partition metadata for all active partitions.
 * @return Cumulative load vector (monotone non-decreasing, ending at 1.0f).
 */
inline std::vector<float> erad_loads(
    const std::vector<double>& times,
    const std::vector<DevicePartition>& partitions
) {
    if (times.empty() || times.size() != partitions.size()) {
        return {};
    }

    std::vector<double> capacity(times.size(), 0.0);
    double total_capacity = 0.0;

    for (std::size_t i = 0; i < times.size(); ++i) {
        if (partitions[i].element_count == 0 || times[i] <= 1.0e-12) {
            continue;
        }

        capacity[i] =
            static_cast<double>(partitions[i].element_count) / times[i];
        total_capacity += capacity[i];
    }

    std::vector<float> loads(times.size(), 0.0f);
    if (total_capacity <= 0.0) {
        const float step = 1.0f / static_cast<float>(times.size());
        float acc = 0.0f;
        for (std::size_t i = 0; i < loads.size(); ++i) {
            acc += step;
            loads[i] = acc;
        }
        loads.back() = 1.0f;
        return loads;
    }

    double cumulative = 0.0;
    for (std::size_t i = 0; i < capacity.size(); ++i) {
        cumulative += capacity[i] / total_capacity;
        loads[i] = static_cast<float>(cumulative);
    }

    loads.back() = 1.0f;
    return loads;
}

/**
 * @brief HWTOPOLB stochastic load balancing algorithm (R4.2).
 *
 * Computes base unperturbed shares from partition throughput, then if
 * perturbation_factor > 0.0f applies a bounded uniform random perturbation
 * delta_i in [-perturbation_factor, +perturbation_factor], clamps shares,
 * normalizes them, and returns cumulative loads.
 * If perturbation_factor == 0.0f, shares remain identical to ERAD.
 *
 * @param times Execution times per partition.
 * @param partitions Partition metadata for all active partitions.
 * @param perturbation_factor Maximum relative perturbation magnitude.
 * @return Cumulative load vector (monotone non-decreasing, ending at 1.0f).
 */
inline std::vector<float> hwtopolb_loads(
    const std::vector<double>& times,
    const std::vector<DevicePartition>& partitions,
    float perturbation_factor,
    unsigned int seed = 0
) {
    if (times.empty() || times.size() != partitions.size()) {
        return {};
    }

    if (perturbation_factor <= 0.0f) {
        return erad_loads(times, partitions);
    }

    std::vector<double> capacity(times.size(), 0.0);
    double total_capacity = 0.0;

    for (std::size_t i = 0; i < times.size(); ++i) {
        if (partitions[i].element_count == 0 || times[i] <= 1.0e-12) {
            continue;
        }

        capacity[i] =
            static_cast<double>(partitions[i].element_count) / times[i];
        total_capacity += capacity[i];
    }

    const std::size_t n = times.size();
    std::vector<float> share(n, 0.0f);
    if (total_capacity <= 0.0) {
        const float equal_share = 1.0f / static_cast<float>(n);
        for (std::size_t i = 0; i < n; ++i) {
            share[i] = equal_share;
        }
    } else {
        for (std::size_t i = 0; i < n; ++i) {
            share[i] = static_cast<float>(capacity[i] / total_capacity);
        }
    }

    static std::mt19937 gen;
    static unsigned int last_seed = 0;
    
    if (seed != 0 && seed != last_seed) {
        gen.seed(seed);
        last_seed = seed;
    } else if (seed == 0 && last_seed == 0) {
        std::random_device rd;
        gen.seed(rd());
        last_seed = 1; // Mark as initialized
    }
    
    std::uniform_real_distribution<float> dist(-perturbation_factor, perturbation_factor);

    std::vector<float> perturbed_share(n, 0.0f);
    float sum_perturbed = 0.0f;
    for (std::size_t i = 0; i < n; ++i) {
        const float delta_i = dist(gen);
        perturbed_share[i] = std::max(1e-6f, share[i] * (1.0f + delta_i));
        sum_perturbed += perturbed_share[i];
    }

    if (sum_perturbed <= 0.0f) {
        return erad_loads(times, partitions);
    }

    std::vector<float> loads(n, 0.0f);
    double cumulative = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const float final_share = perturbed_share[i] / sum_perturbed;
        cumulative += static_cast<double>(final_share);
        loads[i] = static_cast<float>(cumulative);
    }

    loads.back() = 1.0f;
    return loads;
}

/**
 * @brief Power-aware load cap algorithm (R5.3).
 *
 * Converts cumulative partition loads to individual device shares, identifies
 * devices exceeding thermal TDP or system power budget limits, reduces their load
 * shares proportionally, and redistributes the excess load to underloaded devices
 * according to spare thermal headroom or existing shares.
 *
 * @param loads Cumulative load vector (monotone non-decreasing, ending at 1.0f).
 * @param metrics Topology metrics containing thermal_tdp_watts and current_power_watts.
 * @param power_budget_watts Overall cluster/node power budget cap (0.0 if unconstrained).
 * @return Valid cumulative load vector (monotone non-decreasing, ending at 1.0f).
 */
inline std::vector<float> apply_power_cap(
    const std::vector<float>& loads,
    const TopoMetrics& metrics,
    double power_budget_watts = 0.0,
    double* out_throttle_factor = nullptr,
    const std::vector<float>* current_loads = nullptr
) {
    if (loads.empty()) return {};
    if (out_throttle_factor) *out_throttle_factor = 1.0;
    if (metrics.current_power_watts.empty() || metrics.thermal_tdp_watts.empty()) {
        return loads;
    }

    const std::size_t n = loads.size();
    const auto& measured_loads = current_loads ? *current_loads : loads;
    if (measured_loads.size() != n || metrics.current_power_watts.size() != n ||
        metrics.thermal_tdp_watts.size() != n ||
        !std::isfinite(power_budget_watts) || power_budget_watts < 0.0) {
        throw Error("Invalid power cap input dimensions or budget");
    }

    std::vector<double> desired(n), capacity(n), slope(n), shares(n);
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
    if (std::fabs(previous_desired - 1.0) > 1e-5 ||
        std::fabs(previous_current - 1.0) > 1e-5) {
        throw Error("Power cap loads must cover all elements");
    }

    const double total_capacity = std::accumulate(capacity.begin(), capacity.end(), 0.0);
    if (total_capacity < 1.0 - 1e-9) {
        throw Error("Power cap is infeasible without actual device throttling");
    }

    double missing = 1.0 - std::accumulate(shares.begin(), shares.end(), 0.0);
    while (missing > 1e-10) {
        double headroom = 0.0;
        for (std::size_t i = 0; i < n; ++i) headroom += capacity[i] - shares[i];
        if (headroom < missing - 1e-9) {
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
    if (power_budget_watts > 0.0 && predicted_power() > power_budget_watts + 1e-8) {
        // Find the minimum-power feasible allocation before rejecting the budget.
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
        if (remaining > 1e-9 || predicted_power() > power_budget_watts + 1e-8) {
            throw Error("Power budget is infeasible without actual device throttling");
        }
    }

    std::vector<float> result(n, 0.0f);
    double cumulative = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        cumulative += shares[i];
        result[i] = static_cast<float>(cumulative);
    }
    result.back() = 1.0f;
    return result;
}

} // namespace dcl

using dcl::apply_power_cap;
using dcl::erad_loads;
using dcl::hwtopolb_loads;

#endif // DCL_ALGORITHMS_HPP
