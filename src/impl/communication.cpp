// MPI transfers split in INT_MAX-sized chunks, and gather() to rank 0.

#include "runtime_impl.hpp"

#include <algorithm>
#include <climits>
#include <cstring>
#include <sstream>

namespace harmony {

namespace {

// Staging buffer size used by gather() on non-root ranks.
constexpr std::size_t kGatherStagingBytes = 64ull * 1024ull * 1024ull;
constexpr int kGatherTagBase = 1000;

int mpi_chunk_size(std::size_t remaining) {
    return static_cast<int>(std::min<std::size_t>(remaining, static_cast<std::size_t>(INT_MAX)));
}

} // namespace

int Runtime::Impl::mpi_tag_upper_bound() const {
    int tag_flag = 0;
    void* tag_attr = nullptr;
    detail::check_mpi(MPI_Comm_get_attr(comm_, MPI_TAG_UB, &tag_attr, &tag_flag),
                      "MPI_Comm_get_attr(MPI_TAG_UB)");
    return (tag_flag != 0 && tag_attr != nullptr) ? *static_cast<int*>(tag_attr) : 32767;
}

// Every chunk of a transfer uses its own tag (tag_base + chunk index).
void Runtime::Impl::check_mpi_tag_base(int tag_base, std::size_t total_bytes, const char* context) const {
    if (tag_base < 0) {
        throw Error(std::string(context) + ": negative MPI tag");
    }
    const std::size_t chunks =
        (total_bytes + static_cast<std::size_t>(INT_MAX) - 1) / static_cast<std::size_t>(INT_MAX);
    const int last_tag = tag_base + static_cast<int>(chunks == 0 ? 0 : chunks - 1);
    if (last_tag > mpi_tag_upper_bound()) {
        throw Error(std::string(context) + ": MPI tag exceeds MPI_TAG_UB");
    }
}

void Runtime::Impl::mpi_transfer_bytes_chunked(const unsigned char* sendbuf, unsigned char* recvbuf,
                                               std::size_t total_bytes, int peer_rank, int tag_base,
                                               bool do_send, bool do_recv) {
    if (do_send && sendbuf == nullptr) throw Error("mpi_transfer_bytes_chunked(): send buffer is null");
    if (do_recv && recvbuf == nullptr) throw Error("mpi_transfer_bytes_chunked(): recv buffer is null");
    check_mpi_tag_base(tag_base, total_bytes, "mpi_transfer_bytes_chunked()");
    PhaseTimer timer(*this, Phase::mpi_migration);  // only used to migrate data

    std::size_t offset = 0;
    int chunk_id = 0;
    while (offset < total_bytes) {
        const int chunk = mpi_chunk_size(total_bytes - offset);
        MPI_Request reqs[2];
        int req_count = 0;
        if (do_recv) {
            detail::check_mpi(MPI_Irecv(recvbuf + offset, chunk, MPI_BYTE, peer_rank, tag_base + chunk_id,
                                        comm_, &reqs[req_count++]),
                              "MPI_Irecv(chunked transfer)");
        }
        if (do_send) {
            detail::check_mpi(MPI_Isend(sendbuf + offset, chunk, MPI_BYTE, peer_rank, tag_base + chunk_id,
                                        comm_, &reqs[req_count++]),
                              "MPI_Isend(chunked transfer)");
        }
        if (req_count > 0) {
            detail::check_mpi(MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE), "MPI_Waitall(chunked transfer)");
        }
        offset += static_cast<std::size_t>(chunk);
        ++chunk_id;
    }
}

// Rank 0 reads its partitions straight into host_dst and receives the
// others; the remaining ranks read their partitions through a staging
// buffer and send them to rank 0.
void Runtime::Impl::gather(FieldHandle field, void* host_dst, std::size_t bytes) {
    // Only rank 0 receives the data: the other ranks may pass nullptr / 0.
    if (rank_ == 0 && host_dst == nullptr) throw Error("gather() host_dst is null on rank 0");
    PhaseTimer timer(*this, Phase::gather);
    detail::RegisteredField& rf = field_or_throw(field, "gather()");

    const std::size_t total_bytes = rf.total_bytes();
    if (rank_ == 0 && bytes < total_bytes) throw Error("gather() destination buffer too small");

    // Every pending computation/communication must finish before reading.
    synchronize_untimed(true);

    unsigned char* dst = static_cast<unsigned char*>(host_dst);
    if (rank_ == 0) std::memset(dst, 0, total_bytes);

    std::vector<unsigned char> staging;
    if (rank_ != 0) staging.resize(kGatherStagingBytes);

    const int tag_ub = mpi_tag_upper_bound();
    if (tag_ub < kGatherTagBase ||
        partitions_.size() > static_cast<std::size_t>(tag_ub - kGatherTagBase + 1)) {
        throw Error("gather() has too many partitions for MPI_TAG_UB");
    }

    // The partitions must cover the field exactly once.
    std::vector<std::pair<std::size_t, std::size_t>> ranges;
    ranges.reserve(partitions_.size());
    for (const DevicePartition& dp : partitions_) {
        if (dp.element_count == 0) continue;
        if (dp.global_offset > rf.spec.global_elements ||
            dp.element_count > rf.spec.global_elements - dp.global_offset) {
            throw Error("gather() partition range outside field bounds");
        }
        ranges.emplace_back(dp.global_offset, dp.global_offset + dp.element_count);
    }
    std::sort(ranges.begin(), ranges.end());

    std::size_t cursor = 0;
    for (const auto& range : ranges) {
        if (range.first != cursor) {
            std::ostringstream oss;
            oss << "gather() partition coverage gap/overlap at element " << cursor
                << " next_range_begin=" << range.first;
            throw Error(oss.str());
        }
        cursor = range.second;
    }
    if (cursor != rf.spec.global_elements) {
        std::ostringstream oss;
        oss << "gather() partition coverage ends at element " << cursor
            << " expected=" << rf.spec.global_elements;
        throw Error(oss.str());
    }

    const std::size_t element_bytes = rf.element_bytes();

    for (std::size_t p = 0; p < partitions_.size(); ++p) {
        const DevicePartition& dp = partitions_[p];
        const int gather_tag = kGatherTagBase + static_cast<int>(p);
        const std::size_t part_off_bytes = dp.global_offset * element_bytes;
        const std::size_t part_bytes = dp.element_count * element_bytes;
        if (part_bytes == 0) continue;

        if (dp.owning_rank == rank_) {
            // This rank owns the partition: read it from the device.
            if (dp.local_index < 0 || static_cast<std::size_t>(dp.local_index) >= local_devices_.size()) {
                throw Error("Invalid local_index in gather()");
            }
            const std::size_t d = static_cast<std::size_t>(dp.local_index);
            if (rf.replicas[d] == nullptr) throw Error("Null field replica in gather()");

            for (std::size_t done = 0; done < part_bytes;) {
                const std::size_t chunk = std::min(part_bytes - done, kGatherStagingBytes);

                if (rank_ == 0) {
                    detail::check_cl(
                        clEnqueueReadBuffer(local_devices_[d].transfer_queue, rf.replicas[d], CL_TRUE,
                                            part_off_bytes + done, chunk, dst + part_off_bytes + done,
                                            0, nullptr, nullptr),
                        "clEnqueueReadBuffer(gather root direct)");
                } else {
                    detail::check_cl(
                        clEnqueueReadBuffer(local_devices_[d].transfer_queue, rf.replicas[d], CL_TRUE,
                                            part_off_bytes + done, chunk, staging.data(),
                                            0, nullptr, nullptr),
                        "clEnqueueReadBuffer(gather staging)");

                    for (std::size_t sent = 0; sent < chunk;) {
                        const int mpi_chunk = mpi_chunk_size(chunk - sent);
                        detail::check_mpi(MPI_Send(staging.data() + sent, mpi_chunk, MPI_BYTE, 0, gather_tag, comm_),
                                          "MPI_Send(gather)");
                        sent += static_cast<std::size_t>(mpi_chunk);
                    }
                }
                done += chunk;
            }
        } else if (rank_ == 0) {
            // Another rank owns the partition: rank 0 receives it.
            for (std::size_t recvd = 0; recvd < part_bytes;) {
                const std::size_t chunk = std::min(part_bytes - recvd, kGatherStagingBytes);
                for (std::size_t chunk_recvd = 0; chunk_recvd < chunk;) {
                    const int mpi_chunk = mpi_chunk_size(chunk - chunk_recvd);
                    detail::check_mpi(MPI_Recv(dst + part_off_bytes + recvd + chunk_recvd, mpi_chunk, MPI_BYTE,
                                               dp.owning_rank, gather_tag, comm_, MPI_STATUS_IGNORE),
                                      "MPI_Recv(gather)");
                    chunk_recvd += static_cast<std::size_t>(mpi_chunk);
                }
                recvd += chunk;
            }
        }
    }
}

} // namespace harmony
