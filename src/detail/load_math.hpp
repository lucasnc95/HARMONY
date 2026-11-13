// Pure helpers for load vectors and partition layouts (no OpenCL, no MPI).
//
// Loads are always *cumulative*: for 4 devices with equal shares the vector
// is {0.25, 0.50, 0.75, 1.00}.

#ifndef HARMONY_DETAIL_LOAD_MATH_HPP
#define HARMONY_DETAIL_LOAD_MATH_HPP

#include "harmony/types.hpp"

#include <cstddef>
#include <vector>

namespace harmony::detail {

// {0.25, 0.5, 1.0} -> {0.25, 0.25, 0.5}
std::vector<float> cumulative_to_individual(const std::vector<float>& cumulative);

// {1/n, 2/n, ..., 1.0}
std::vector<float> equal_cumulative_loads(std::size_t parts);

// New cumulative loads proportional to each partition's throughput
// (elements / measured seconds). Returns an empty vector if sizes differ.
std::vector<float> compute_loads_from_partition_throughput(
    const std::vector<double>& times,
    const std::vector<DevicePartition>& partitions);

// L2 distance between two cumulative load vectors; +inf if `current` is
// empty or the sizes differ.
double l2_norm_diff(const std::vector<float>& current,
                                 const std::vector<float>& proposed);

// True if both layouts assign the same ranges to the same devices.
bool same_partition_layout(const std::vector<DevicePartition>& a,
                           const std::vector<DevicePartition>& b);

// Intersection of [off1, off1+len1) and [off2, off2+len2).
bool intersect_1d(std::size_t off1, std::size_t len1,
                  std::size_t off2, std::size_t len2,
                  std::size_t& out_off, std::size_t& out_len);

} // namespace harmony::detail

#endif // HARMONY_DETAIL_LOAD_MATH_HPP
