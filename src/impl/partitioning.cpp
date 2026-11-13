// Domain decomposition: mapping between cumulative loads and partitions.

#include "runtime_impl.hpp"
#include "detail/format.hpp"
#include "detail/load_math.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>

namespace harmony {

void Runtime::Impl::set_partition(const PartitionSpec& spec) {
    if (spec.global_elements == 0) throw Error("PartitionSpec.global_elements must be > 0");
    if (spec.units_per_element == 0) throw Error("PartitionSpec.units_per_element must be > 0");
    if (spec.bytes_per_unit == 0) throw Error("PartitionSpec.bytes_per_unit must be > 0");
    if (spec.granularity == 0) throw Error("PartitionSpec.granularity must be > 0");

    partition_ = spec;
    rebuild_partitions_equal_like_wrapper();
    current_loads_ = detail::equal_cumulative_loads(partitions_.size());
    reset_balance_window_events();
}

// Global devices are numbered rank by rank, so the owner of global device g
// is found by walking the per-rank device counts.
int Runtime::Impl::owner_rank_of_global_device(int g) const {
    int acc = 0;
    for (int r = 0; r < size_; ++r) {
        if (g < acc + all_device_counts_[r]) return r;
        acc += all_device_counts_[r];
    }
    return size_ - 1;
}

// Local index of global device g on this rank, or -1 if another rank owns it.
int Runtime::Impl::local_index_of_global_device(int g) const {
    int acc = 0;
    for (int r = 0; r < size_; ++r) {
        if (g < acc + all_device_counts_[r]) {
            return (r == rank_) ? (g - acc) : -1;
        }
        acc += all_device_counts_[r];
    }
    return -1;
}

// Splits the domain evenly, rounding the cuts down to the granularity.
void Runtime::Impl::rebuild_partitions_equal_like_wrapper() {
    partitions_.clear();
    if (!partition_.has_value()) return;

    int total_devices = 0;
    for (int c : all_device_counts_) total_devices += c;
    if (total_devices <= 0) return;

    const std::size_t n = partition_->global_elements;
    const std::size_t gran = std::max<std::size_t>(1, partition_->granularity);
    const std::size_t devices = static_cast<std::size_t>(total_devices);

    std::vector<std::size_t> cuts(devices + 1, 0);
    const std::size_t base = n / devices;
    const std::size_t rem = n % devices;
    std::size_t acc = 0;
    for (std::size_t g = 0; g < devices; ++g) {
        acc += base + (g < rem ? 1u : 0u);
        cuts[g + 1] = (g + 1 == devices) ? n : (acc / gran) * gran;
    }

    for (int g = 0; g < total_devices; ++g) {
        const std::size_t gi = static_cast<std::size_t>(g);
        DevicePartition dp;
        dp.device_global_index = g;
        dp.owning_rank = owner_rank_of_global_device(g);
        dp.local_index = local_index_of_global_device(g);
        dp.global_offset = cuts[gi];
        dp.element_count = cuts[gi + 1] - cuts[gi];
        partitions_.push_back(dp);
    }
}

std::vector<DevicePartition> Runtime::Impl::partitions_from_loads(const std::vector<float>& loads) const {
    std::vector<DevicePartition> out;
    if (!partition_.has_value() || loads.empty()) return out;

    const std::size_t n = partition_->global_elements;
    const std::size_t gran = std::max<std::size_t>(1, partition_->granularity);

    // Sanitize: finite, in [0, 1], non-decreasing, last == 1.
    std::vector<float> cumulative = loads;
    float prev = 0.0f;
    for (float& c : cumulative) {
        if (!std::isfinite(c)) return out;
        c = std::clamp(std::max(c, prev), 0.0f, 1.0f);
        prev = c;
    }
    cumulative.back() = 1.0f;

    const std::size_t parts = cumulative.size();
    std::vector<std::size_t> cuts(parts + 1, 0);

    if ((n % gran) == 0 && (n / gran) >= parts) {
        // Work in granularity units and keep at least one unit per partition.
        const std::size_t total_units = n / gran;
        std::vector<std::size_t> unit_cuts(parts + 1, 0);

        for (std::size_t i = 0; i + 1 < parts; ++i) {
            const double raw = static_cast<double>(cumulative[i]) * static_cast<double>(total_units);
            std::size_t cut = static_cast<std::size_t>(std::llround(raw));

            const std::size_t min_cut = unit_cuts[i] + 1;
            const std::size_t max_cut = total_units - (parts - (i + 1));
            cut = std::clamp(cut, min_cut, max_cut);
            unit_cuts[i + 1] = cut;
        }
        unit_cuts.back() = total_units;

        for (std::size_t i = 0; i < unit_cuts.size(); ++i) {
            cuts[i] = unit_cuts[i] * gran;
        }
    } else {
        // Fallback: element cuts rounded down to the granularity (may be empty).
        for (std::size_t i = 0; i < parts; ++i) {
            if (i + 1 == parts) {
                cuts[i + 1] = n;
            } else {
                const double raw = static_cast<double>(cumulative[i]) * static_cast<double>(n);
                std::size_t cut = static_cast<std::size_t>(std::llround(raw));
                cut = (cut / gran) * gran;
                cuts[i + 1] = std::clamp(cut, cuts[i], n);
            }
        }
        cuts.back() = n;
    }

    for (std::size_t g = 0; g < parts; ++g) {
        DevicePartition dp;
        dp.device_global_index = static_cast<int>(g);
        dp.owning_rank = owner_rank_of_global_device(static_cast<int>(g));
        dp.local_index = (dp.owning_rank == rank_) ? local_index_of_global_device(static_cast<int>(g)) : -1;
        dp.global_offset = cuts[g];
        dp.element_count = cuts[g + 1] - cuts[g];
        out.push_back(dp);
    }
    return out;
}

std::vector<float> Runtime::Impl::loads_from_partitions(const std::vector<DevicePartition>& parts) const {
    std::vector<float> loads(parts.size(), 1.0f);
    if (!partition_.has_value() || partition_->global_elements == 0 || parts.empty()) {
        return loads;
    }

    const double total = static_cast<double>(partition_->global_elements);
    for (std::size_t i = 0; i < parts.size(); ++i) {
        const std::size_t end = parts[i].global_offset + parts[i].element_count;
        loads[i] = static_cast<float>(static_cast<double>(end) / total);
        if (i > 0 && loads[i] < loads[i - 1]) loads[i] = loads[i - 1];
        if (loads[i] > 1.0f) loads[i] = 1.0f;
    }
    loads.back() = 1.0f;
    return loads;
}

void Runtime::Impl::print_partitions(const std::string& title, std::ostream& os) const {
    if (rank_ != 0) return;
    const double total = partition_.has_value() ? static_cast<double>(partition_->global_elements) : 0.0;

    detail::Table table({{"Dev", false}, {"Rank", false}, {"Name", false}, {"First element", true},
                         {"Elements", true}, {"Share", true}});
    for (const DevicePartition& p : partitions_) {
        std::string name = "?";
        for (const DeviceInfo& d : global_devices_) {
            if (d.global_index == p.device_global_index) name = d.name;
        }
        table.add_row({std::to_string(p.device_global_index), std::to_string(p.owning_rank),
                       detail::truncate(name, 32), detail::format_count(p.global_offset),
                       detail::format_count(p.element_count),
                       detail::format_percent(total > 0.0 ? p.element_count / total : 0.0)});
    }
    std::ostringstream out;
    out << title << "\n";
    table.print(out, "  ");
    os << out.str() << std::flush;
}

} // namespace harmony
