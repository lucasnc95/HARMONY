// Halo exchange between neighbouring partitions.
//
// For every pair of consecutive partitions (left, right) and every halo field
// two messages exist:
//   left -> right: the last `halo` elements of left go right before right's
//                  slice (its left ghost region);
//   right -> left: the first `halo` elements of right go right after left's
//                  slice (its right ghost region).
//
// All messages of all fields and pairs are processed together, in stages,
// so nothing waits for an unrelated transfer:
//   1. post every MPI receive;
//   2. enqueue every device read (and device-to-device copies when both
//      devices share an OpenCL context: no host round trip);
//   3. as each read finishes, start its MPI send (or, for devices of the
//      same rank in different contexts, its device write);
//   4. as each receive arrives, enqueue its device write;
//   5. wait for the sends and the device writes.
// Copies inside one context are not waited for: the border kernels depend on
// them through the field's ghost-region events.
//
// Dependencies use the field's access trackers: a read of a partition's
// border waits for the kernels writing that partition; a write into the
// ghost region waits for the previous writers and readers of the ghost
// region (border kernels of an earlier step, possibly on another queue).

#include "runtime_impl.hpp"

#include <algorithm>
#include <climits>
#include <map>

namespace harmony {

namespace {

enum class Route {
    device_copy,   // same rank, same OpenCL context: clEnqueueCopyBuffer
    host_copy,     // same rank, different contexts: read -> host -> write
    send,          // this rank owns the source only
    receive        // this rank owns the destination only
};

struct HaloMessage {
    FieldHandle field;
    detail::RegisteredField* rf{nullptr};
    Route route{Route::send};
    std::size_t src_device{0};      // local index (when the source is here)
    std::size_t dst_device{0};      // local index (when the destination is here)
    int peer_rank{-1};
    std::size_t src_offset{0};      // bytes
    std::size_t dst_offset{0};      // bytes
    std::size_t bytes{0};
    int tag{0};
    unsigned char* host{nullptr};   // staging buffer
    cl_event read_ev{nullptr};
    cl_event write_ev{nullptr};
};

void wait_and_release(cl_event& ev, const char* what) {
    if (ev == nullptr) return;
    detail::check_cl(clWaitForEvents(1, &ev), what);
    clReleaseEvent(ev);
    ev = nullptr;
}

} // namespace

void Runtime::Impl::exchange_halo_set(const std::vector<std::pair<FieldHandle, std::size_t>>& fields) {
    if (fields.empty() || partitions_.size() < 2) return;
    PhaseTimer timer(*this, Phase::halo);

    // Tags are unique per (field slot, pair, direction) on every rank.
    const std::size_t max_pairs = partitions_.size();

    // ---- Plan ---------------------------------------------------------------
    std::vector<HaloMessage> messages;
    for (std::size_t slot = 0; slot < fields.size(); ++slot) {
        const FieldHandle fh = fields[slot].first;
        const std::size_t halo = fields[slot].second;
        detail::RegisteredField& rf = field_or_throw(fh, "halo exchange");
        rf.valid_halo = halo;  // up to date once this exchange completes
        if (halo == 0) continue;

        // Partitions smaller than the halo do not take part in the exchange.
        std::vector<std::size_t> active;
        for (std::size_t p = 0; p < partitions_.size(); ++p) {
            if (partitions_[p].element_count >= halo) active.push_back(p);
        }
        if (active.size() < 2) continue;
        const std::size_t pairs = active.size() - 1;

        const std::size_t elem_bytes = rf.element_bytes();
        const std::size_t halo_bytes = halo * elem_bytes;
        if (halo_bytes > static_cast<std::size_t>(INT_MAX)) {
            throw Error("halo exchange: halo larger than 2 GiB per message is not supported");
        }

        for (std::size_t pair = 0; pair < pairs; ++pair) {
            const DevicePartition& left = partitions_[active[pair]];
            const DevicePartition& right = partitions_[active[pair + 1]];

            for (int dir = 0; dir < 2; ++dir) {
                const DevicePartition& src = (dir == 0) ? left : right;
                const DevicePartition& dst = (dir == 0) ? right : left;
                const bool src_here = src.owning_rank == rank_;
                const bool dst_here = dst.owning_rank == rank_;
                if (!src_here && !dst_here) continue;

                HaloMessage m;
                m.field = fh;
                m.rf = &rf;
                m.bytes = halo_bytes;
                m.tag = static_cast<int>((slot * max_pairs + pair) * 2 + static_cast<std::size_t>(dir));
                m.src_offset = (dir == 0 ? left.global_offset + left.element_count - halo : right.global_offset) *
                               elem_bytes;
                m.dst_offset = (dir == 0 ? right.global_offset - halo : left.global_offset + left.element_count) *
                               elem_bytes;
                if (src_here) m.src_device = static_cast<std::size_t>(src.local_index);
                if (dst_here) m.dst_device = static_cast<std::size_t>(dst.local_index);

                if (src_here && dst_here) {
                    m.route = (local_devices_[m.src_device].context == local_devices_[m.dst_device].context)
                                  ? Route::device_copy
                                  : Route::host_copy;
                } else {
                    m.route = src_here ? Route::send : Route::receive;
                    m.peer_rank = src_here ? dst.owning_rank : src.owning_rank;
                }
                messages.push_back(m);
            }
        }
    }
    if (messages.empty()) return;

    if (static_cast<long long>(fields.size() * max_pairs * 2) > mpi_tag_upper_bound()) {
        throw Error("halo exchange: too many halo messages for MPI_TAG_UB");
    }

    // Host staging buffers, reused across iterations.
    if (halo_buffers_.size() < messages.size()) halo_buffers_.resize(messages.size());
    for (std::size_t i = 0; i < messages.size(); ++i) {
        if (messages[i].route == Route::device_copy) continue;
        if (halo_buffers_[i].size() < messages[i].bytes) halo_buffers_[i].resize(messages[i].bytes);
        messages[i].host = halo_buffers_[i].data();
    }

    // ---- 1. Post every receive ------------------------------------------------
    std::vector<MPI_Request> recv_requests;
    std::vector<std::size_t> recv_message;
    for (std::size_t i = 0; i < messages.size(); ++i) {
        HaloMessage& m = messages[i];
        if (m.route != Route::receive) continue;
        recv_requests.emplace_back();
        recv_message.push_back(i);
        detail::check_mpi(MPI_Irecv(m.host, static_cast<int>(m.bytes), MPI_BYTE, m.peer_rank, m.tag, halo_comm_,
                                    &recv_requests.back()),
                          "MPI_Irecv(halo)");
    }

    // ---- 2. Enqueue every device read and device-to-device copy ---------------
    for (HaloMessage& m : messages) {
        if (m.route == Route::receive) continue;
        cl_mem src_mem = m.rf->replicas[m.src_device];
        std::vector<cl_event> deps = m.rf->core[m.src_device].writers;

        if (m.route == Route::device_copy) {
            const detail::AccessTracker& ghost = m.rf->ghost[m.dst_device];
            deps.insert(deps.end(), ghost.writers.begin(), ghost.writers.end());
            deps.insert(deps.end(), ghost.readers.begin(), ghost.readers.end());
            detail::check_cl(
                clEnqueueCopyBuffer(local_devices_[m.dst_device].transfer_queue, src_mem,
                                    m.rf->replicas[m.dst_device], m.src_offset, m.dst_offset, m.bytes,
                                    static_cast<cl_uint>(deps.size()), deps.empty() ? nullptr : deps.data(),
                                    &m.write_ev),
                "clEnqueueCopyBuffer(halo)");
        } else {
            detail::check_cl(
                clEnqueueReadBuffer(local_devices_[m.src_device].transfer_queue, src_mem, CL_FALSE, m.src_offset,
                                    m.bytes, m.host, static_cast<cl_uint>(deps.size()),
                                    deps.empty() ? nullptr : deps.data(), &m.read_ev),
                "clEnqueueReadBuffer(halo)");
        }
        set_field_read_event(m.rf->core[m.src_device], m.route == Route::device_copy ? m.write_ev : m.read_ev);
        if (m.route == Route::device_copy) track_transfer_event(m.dst_device, m.write_ev);
        else track_transfer_event(m.src_device, m.read_ev);
    }
    for (detail::LocalDevice& ld : local_devices_) clFlush(ld.transfer_queue);

    auto enqueue_write = [&](HaloMessage& m) {
        const detail::AccessTracker& ghost = m.rf->ghost[m.dst_device];
        std::vector<cl_event> deps = ghost.writers;
        deps.insert(deps.end(), ghost.readers.begin(), ghost.readers.end());
        detail::check_cl(
            clEnqueueWriteBuffer(local_devices_[m.dst_device].transfer_queue, m.rf->replicas[m.dst_device],
                                 CL_FALSE, m.dst_offset, m.bytes, m.host, static_cast<cl_uint>(deps.size()),
                                 deps.empty() ? nullptr : deps.data(), &m.write_ev),
            "clEnqueueWriteBuffer(halo)");
        track_transfer_event(m.dst_device, m.write_ev);
        clFlush(local_devices_[m.dst_device].transfer_queue);
    };

    // ---- 3. Send (or write locally) as soon as each read is done -------------
    // The reads wait for the kernels that produced the halo field, so this
    // wait usually measures the previous step's computation, not the copy.
    std::vector<MPI_Request> send_requests;
    for (HaloMessage& m : messages) {
        if (m.route != Route::send && m.route != Route::host_copy) continue;
        {
            PhaseTimer wait_timer(*this, Phase::device_wait);
            wait_and_release(m.read_ev, "clWaitForEvents(halo read)");
        }
        if (m.route == Route::send) {
            send_requests.emplace_back();
            detail::check_mpi(MPI_Isend(m.host, static_cast<int>(m.bytes), MPI_BYTE, m.peer_rank, m.tag,
                                        halo_comm_, &send_requests.back()),
                              "MPI_Isend(halo)");
        } else {
            enqueue_write(m);
        }
    }

    // ---- 4. Write each received halo as soon as it arrives --------------------
    for (std::size_t done = 0; done < recv_requests.size(); ++done) {
        int index = MPI_UNDEFINED;
        {
            PhaseTimer wait_timer(*this, Phase::mpi_halo);
            detail::check_mpi(MPI_Waitany(static_cast<int>(recv_requests.size()), recv_requests.data(), &index,
                                          MPI_STATUS_IGNORE),
                              "MPI_Waitany(halo)");
        }
        if (index == MPI_UNDEFINED) break;
        enqueue_write(messages[recv_message[static_cast<std::size_t>(index)]]);
    }

    // ---- 5. Finish ------------------------------------------------------------
    if (!send_requests.empty()) {
        PhaseTimer wait_timer(*this, Phase::mpi_halo);
        detail::check_mpi(MPI_Waitall(static_cast<int>(send_requests.size()), send_requests.data(),
                                      MPI_STATUSES_IGNORE),
                          "MPI_Waitall(halo sends)");
    }

    // The new ghost writes replace the previous writers and readers of the
    // ghost region (they all waited for them).
    std::map<std::pair<int, std::size_t>, std::vector<cl_event>> new_writers;
    for (HaloMessage& m : messages) {
        if (m.write_ev != nullptr) new_writers[{m.field.value, m.dst_device}].push_back(m.write_ev);
    }
    for (auto& [key, events] : new_writers) {
        detail::AccessTracker& ghost = fields_.at(key.first).ghost[key.second];
        detail::release_events(ghost.writers);
        detail::release_events(ghost.readers);
        ghost.writers = events;  // takes the reference of each event
    }

    // Host buffers are reused next time, so writes from them must finish.
    for (HaloMessage& m : messages) {
        if (m.write_ev == nullptr || m.route == Route::device_copy) continue;
        PhaseTimer wait_timer(*this, Phase::device_wait);
        detail::check_cl(clWaitForEvents(1, &m.write_ev), "clWaitForEvents(halo write)");
    }
}

} // namespace harmony
