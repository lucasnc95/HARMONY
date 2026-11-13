// Data migration after a repartition, and the manual rebalance_to() API.

#include "runtime_impl.hpp"
#include "detail/load_math.hpp"

#include <algorithm>
#include <cmath>

namespace harmony {

namespace {

constexpr std::size_t kRedistributeStagingBytes = 64ull * 1024ull * 1024ull;
constexpr int kRedistributeTagBase = 3000;
constexpr int kRedistributeFieldTagStride = 4096;

} // namespace

void Runtime::Impl::rebalance_to(const std::vector<float>& loads) {
    if (!partition_.has_value() || partitions_.empty()) return;
    PhaseTimer timer(*this, Phase::balance);
    const double t0 = MPI_Wtime();
    if (loads.size() != partitions_.size()) {
        throw Error("rebalance_to(): loads size must match number of partitions");
    }

    std::vector<float> cumulative = loads;
    float prev = 0.0f;
    for (float& c : cumulative) {
        if (!std::isfinite(c)) throw Error("rebalance_to(): non-finite cumulative load");
        if (c < 0.0f) throw Error("rebalance_to(): negative cumulative load not allowed");
        c = std::min(std::max(c, prev), 1.0f);
        prev = c;
    }
    cumulative.back() = 1.0f;

    const std::vector<DevicePartition> old_parts = partitions_;
    const std::vector<DevicePartition> new_parts = partitions_from_loads(cumulative);
    if (new_parts.size() != old_parts.size()) {
        throw Error("rebalance_to(): rounded partition count mismatch");
    }

    BalanceRecord record;
    record.iteration = iteration_counter_;
    record.policy = "manual";
    for (const DevicePartition& p : old_parts) record.elements_before.push_back(p.element_count);

    if (detail::same_partition_layout(old_parts, new_parts)) {
        current_loads_ = loads_from_partitions(partitions_);
        reset_balance_window_events();
        record.reason = "the requested loads give the current division";
    } else {
        apply_partitions(all_field_handles(), old_parts, new_parts, record);
        record.applied = true;
        record.reason = "requested by the application (rebalance_to)";
    }

    for (const DevicePartition& p : partitions_) record.elements_after.push_back(p.element_count);
    record.decision_seconds = MPI_Wtime() - t0 - record.migration_seconds;
    print_balance_interval_metrics(std::move(record));
}

std::vector<FieldHandle> Runtime::Impl::all_field_handles() const {
    std::vector<FieldHandle> out;
    out.reserve(fields_.size());
    for (const auto& kv : fields_) out.push_back(FieldHandle{kv.first});
    return out;
}

// Migrates the given fields and switches to the new layout.
void Runtime::Impl::apply_partitions(const std::vector<FieldHandle>& fields,
                                     const std::vector<DevicePartition>& old_parts,
                                     const std::vector<DevicePartition>& new_parts,
                                     BalanceRecord& record) {
    PhaseTimer timer(*this, Phase::migration);
    const double t0 = MPI_Wtime();

    synchronize_untimed(true);
    redistribute_selected_registered_fields(fields, old_parts, new_parts);
    partitions_ = new_parts;
    current_loads_ = loads_from_partitions(partitions_);
    synchronize_untimed(true);
    reset_balance_window_events();

    // Partition borders moved: every ghost region now holds old data.
    for (auto& kv : fields_) kv.second.valid_halo = 0;

    record.bytes_moved = bytes_moved(fields, old_parts, new_parts);
    record.migration_seconds = MPI_Wtime() - t0;
}

// Bytes of the proportional fields whose owner device changes. Every rank
// computes the same value from the partition tables.
std::size_t Runtime::Impl::bytes_moved(const std::vector<FieldHandle>& fields,
                                       const std::vector<DevicePartition>& old_parts,
                                       const std::vector<DevicePartition>& new_parts) const {
    std::size_t total = 0;
    for (FieldHandle fh : unique_existing_proportional_fields(fields)) {
        const std::size_t elem_bytes = fields_.at(fh.value).element_bytes();
        for (const DevicePartition& src : old_parts) {
            for (const DevicePartition& dst : new_parts) {
                if (src.device_global_index == dst.device_global_index) continue;
                std::size_t off = 0;
                std::size_t len = 0;
                if (detail::intersect_1d(src.global_offset, src.element_count, dst.global_offset,
                                         dst.element_count, off, len)) {
                    total += len * elem_bytes;
                }
            }
        }
    }
    return total;
}

// Only proportional fields are moved (none = replicated, total = reserved).
void Runtime::Impl::redistribute_selected_registered_fields(const std::vector<FieldHandle>& fields,
                                        const std::vector<DevicePartition>& old_parts,
                                        const std::vector<DevicePartition>& new_parts) {
    for (FieldHandle fh : unique_existing_proportional_fields(fields)) {
        redistribute_field_intersection(fh, old_parts, new_parts);
        PhaseTimer timer(*this, Phase::mpi_migration);
        detail::check_mpi(MPI_Barrier(comm_), "MPI_Barrier(redistribute field)");
    }
}

// Copies every intersection between an old and a new partition from the old
// owner to the new owner (device -> host -> [MPI] -> host -> device).
void Runtime::Impl::redistribute_field_intersection(FieldHandle fh,
                                       const std::vector<DevicePartition>& old_parts,
                                       const std::vector<DevicePartition>& new_parts) {
    detail::RegisteredField& rf = field_or_throw(fh, "redistribute_field_intersection()");
    const std::size_t elem_bytes = rf.element_bytes();
    const std::size_t total_bytes = rf.total_bytes();

    auto replica_or_throw = [&](int local_index, const char* what) -> std::size_t {
        const std::size_t d = static_cast<std::size_t>(local_index);
        if (d >= rf.replicas.size() || rf.replicas[d] == nullptr) {
            throw Error(std::string("redistribute_field_intersection(): invalid ") + what + " replica");
        }
        return d;
    };

    for (const DevicePartition& src : old_parts) {
        for (const DevicePartition& dst : new_parts) {
            std::size_t inter_off = 0;
            std::size_t inter_len = 0;
            if (!detail::intersect_1d(src.global_offset, src.element_count,
                                      dst.global_offset, dst.element_count, inter_off, inter_len)) {
                continue;
            }

            const std::size_t byte_off = inter_off * elem_bytes;
            const std::size_t byte_len = inter_len * elem_bytes;
            if (byte_off + byte_len > total_bytes) {
                throw Error("redistribute_field_intersection(): byte range out of bounds");
            }

            const bool src_here = src.owning_rank == rank_;
            const bool dst_here = dst.owning_rank == rank_;

            // Same local device: the data is already in place.
            if (src_here && dst_here && src.local_index >= 0 && dst.local_index >= 0 &&
                src.local_index == dst.local_index) {
                continue;
            }
            if (!src_here && !dst_here) continue;

            std::vector<unsigned char> chunk(std::min(byte_len, kRedistributeStagingBytes));

            for (std::size_t done = 0; done < byte_len; done += chunk.size()) {
                const std::size_t chunk_len = std::min(chunk.size(), byte_len - done);
                const std::size_t chunk_off = byte_off + done;

                if (src_here && src.local_index >= 0) {
                    const std::size_t d = replica_or_throw(src.local_index, "source");
                    cl_event read_ev = nullptr;
                    detail::check_cl(
                        clEnqueueReadBuffer(local_devices_[d].transfer_queue, rf.replicas[d], CL_FALSE,
                                            chunk_off, chunk_len, chunk.data(), 0, nullptr, &read_ev),
                        "clEnqueueReadBuffer(redistribute read)");
                    detail::check_cl(clWaitForEvents(1, &read_ev), "clWaitForEvents(redistribute read)");
                    track_transfer_event(d, read_ev);
                    clReleaseEvent(read_ev);
                }

                if (!(src_here && dst_here)) {
                    const int tag_base = kRedistributeTagBase + fh.value * kRedistributeFieldTagStride;
                    mpi_transfer_bytes_chunked(src_here ? chunk.data() : nullptr,
                                               dst_here ? chunk.data() : nullptr,
                                               chunk_len,
                                               src_here ? dst.owning_rank : src.owning_rank,
                                               tag_base + static_cast<int>(done / chunk.size()),
                                               src_here, dst_here);
                }

                if (dst_here && dst.local_index >= 0) {
                    const std::size_t d = replica_or_throw(dst.local_index, "destination");
                    cl_event write_ev = nullptr;
                    detail::check_cl(
                        clEnqueueWriteBuffer(local_devices_[d].transfer_queue, rf.replicas[d], CL_TRUE,
                                             chunk_off, chunk_len, chunk.data(), 0, nullptr, &write_ev),
                        src_here ? "clEnqueueWriteBuffer(redistribute local)"
                                 : "clEnqueueWriteBuffer(redistribute remote)");
                    track_transfer_event(d, write_ev);
                    clReleaseEvent(write_ev);
                }
            }
        }
    }

    synchronize_all_local_devices(false);
}

} // namespace harmony
