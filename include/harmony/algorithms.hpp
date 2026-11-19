// Public load-balancing algorithms and power/contention helpers (R4, R5).

#ifndef HARMONY_ALGORITHMS_HPP
#define HARMONY_ALGORITHMS_HPP

#include "harmony/types.hpp"

#include <cstddef>
#include <vector>

namespace harmony {

// ERAD deterministic load balancing (R4.1): computes cumulative loads purely
// from partition throughput (elements per second), without stochastic
// perturbation. Returns a monotone non-decreasing vector ending at 1.0f, or
// empty if inputs are empty or mismatched.
std::vector<float> erad_loads(
    const std::vector<double>& times,
    const std::vector<DevicePartition>& partitions);

// ERAD with optional contention adjustment (R5.2).
std::vector<float> erad_loads(
    const std::vector<double>& times,
    const std::vector<DevicePartition>& partitions,
    const TopoMetrics& metrics,
    bool use_contention_adjustment = true);

// HWTOPOLB stochastic load balancing (R4.2): computes base shares from
// partition throughput, then if perturbation_factor > 0.0f applies a bounded
// uniform perturbation delta_i in [-perturbation_factor, +perturbation_factor],
// clamps shares, normalizes them, and returns cumulative loads.
// When perturbation_factor <= 0.0f, matches erad_loads bit-for-bit.
// When seed != 0, output is deterministic for (times, partitions, factor, seed).
std::vector<float> hwtopolb_loads(
    const std::vector<double>& times,
    const std::vector<DevicePartition>& partitions,
    float perturbation_factor,
    unsigned int seed = 0);

// HWTOPOLB with optional contention adjustment (R5.2).
std::vector<float> hwtopolb_loads(
    const std::vector<double>& times,
    const std::vector<DevicePartition>& partitions,
    float perturbation_factor,
    unsigned int seed,
    const TopoMetrics& metrics,
    bool use_contention_adjustment = true);

// Adjusts raw throughput by TopoMetrics::memory_contention_factor[global_device] (R5.2).
// Throws harmony::Error on invalid index, non-finite/negative throughput, or factor < 1.0.
double adjusted_capacity(
    const TopoMetrics& metrics,
    int global_device,
    double raw_throughput);

// Power-aware load cap algorithm (R5.3): enforces per-device thermal TDP ceilings
// and optional system-wide power_budget_watts using linear power slopes derived
// from current_loads (or loads if current_loads is null). Sets *out_throttle_factor
// to 1.0 when feasible, or to the required global throttle factor (> 1.0) before
// throwing harmony::Error when total desired/minimum power exceeds thermal capacity.
std::vector<float> apply_power_cap(
    const std::vector<float>& loads,
    const TopoMetrics& metrics,
    double power_budget_watts = 0.0,
    double* out_throttle_factor = nullptr,
    const std::vector<float>* current_loads = nullptr);

// Symmetric NUMA hop distance between two NUMA nodes (R2).
// Validates that metrics.numa_distance is a non-empty N x N matrix and clamps
// against the diagonal minimum. Throws harmony::Error on invalid inputs.
int numa_distance_between(const TopoMetrics& metrics, int node_a, int node_b);

// Estimates migration time in seconds for transferring `bytes` from
// (src_global_device, src_rank) to (dst_global_device, dst_rank) (R2).
// Inter-rank transfers model PCIe(src) + MPI(src,dst) + PCIe(dst); intra-rank
// transfers model PCIe + symmetric NUMA hop penalty. Throws harmony::Error on
// negative device indices or invalid MPI metrics when ranks differ.
double estimate_migration_cost_bytes(
    const TopoMetrics& metrics,
    int src_global_device,
    int dst_global_device,
    int src_rank,
    int dst_rank,
    std::size_t bytes,
    double numa_penalty_ns_per_hop = 50.0);

// Intra-rank convenience overload for estimate_migration_cost_bytes.
double estimate_migration_cost_bytes(
    const TopoMetrics& metrics,
    int src_global_device,
    int dst_global_device,
    std::size_t bytes,
    double numa_penalty_ns_per_hop = 50.0);

} // namespace harmony

#endif // HARMONY_ALGORITHMS_HPP
