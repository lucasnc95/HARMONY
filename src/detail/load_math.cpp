#include "detail/load_math.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

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
    if (times.empty() || times.size() != partitions.size()) {
        return {};
    }

    std::vector<double> capacity(times.size(), 0.0);
    double total_capacity = 0.0;
    for (std::size_t i = 0; i < times.size(); ++i) {
        if (partitions[i].element_count == 0 || times[i] <= 1.0e-12) continue;
        capacity[i] = static_cast<double>(partitions[i].element_count) / times[i];
        total_capacity += capacity[i];
    }

    if (total_capacity <= 0.0) {
        return equal_cumulative_loads(times.size());
    }

    std::vector<float> loads(times.size(), 0.0f);
    double cumulative = 0.0;
    for (std::size_t i = 0; i < capacity.size(); ++i) {
        cumulative += capacity[i] / total_capacity;
        loads[i] = static_cast<float>(cumulative);
    }
    loads.back() = 1.0f;
    return loads;
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

} // namespace harmony::detail
