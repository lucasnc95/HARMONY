// Time measurement and the timing report (the data; logging.cpp prints it).
//
// CPU time of each process is split into exclusive phases (PhaseTimer): when
// a phase starts inside another one, the outer phase is paused. The phases
// therefore add up to the time spent inside the runtime, and whatever is
// left of the wall time is "not measured" (application code, ...).
//
// Device times come from OpenCL profiling events: compute is the union of
// the kernel intervals, transfers the extra time with host <-> device copies,
// and idle the rest.

#include "runtime_impl.hpp"
#include "detail/format.hpp"

#include <algorithm>
#include <sys/resource.h>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace harmony {

namespace {

int index_of(Phase p) { return static_cast<int>(p); }

} // namespace

// ---------------------------------------------------------------------------
// Exclusive phase timers
// ---------------------------------------------------------------------------

Runtime::Impl::PhaseTimer::PhaseTimer(Impl& impl, Phase phase) : impl_(impl) { impl_.enter_phase(phase); }

Runtime::Impl::PhaseTimer::~PhaseTimer() { impl_.leave_phase(); }

void Runtime::Impl::enter_phase(Phase phase) {
    const double now = MPI_Wtime();
    if (!phase_stack_.empty()) phase_seconds_[index_of(phase_stack_.back())] += now - phase_mark_;
    phase_stack_.push_back(phase);
    phase_mark_ = now;
}

double Runtime::Impl::phase_seconds_now(Phase phase) const {
    double s = phase_seconds_[index_of(phase)];
    if (!phase_stack_.empty() && phase_stack_.back() == phase) s += MPI_Wtime() - phase_mark_;
    return s;
}

void Runtime::Impl::leave_phase() {
    const double now = MPI_Wtime();
    if (!phase_stack_.empty()) {
        phase_seconds_[index_of(phase_stack_.back())] += now - phase_mark_;
        phase_stack_.pop_back();
    }
    phase_mark_ = now;
}

// ---------------------------------------------------------------------------
// Measured interval
// ---------------------------------------------------------------------------

void Runtime::Impl::start_timing() {
    synchronize_untimed(true);
    update_balance_window_events(true);

    phase_seconds_.fill(0.0);
    std::fill(kernel_busy_total_.begin(), kernel_busy_total_.end(), 0.0);
    std::fill(active_busy_total_.begin(), active_busy_total_.end(), 0.0);
    balance_log_.clear();
    balance_attempts_ = 0;
    timing_wall_.reset();
    timing_snapshot_.reset();
    timing_start_iteration_ = iteration_counter_;
    timing_start_ = MPI_Wtime();
    phase_mark_ = timing_start_;
}

double Runtime::Impl::stop_timing() {
    synchronize_untimed(true);  // waiting for the last kernels is part of the run
    update_balance_window_events(true);

    const double local = MPI_Wtime() - timing_start_;
    double wall = local;
    detail::check_mpi(MPI_Allreduce(&local, &wall, 1, MPI_DOUBLE, MPI_MAX, comm_), "MPI_Allreduce(wall time)");
    timing_wall_ = wall;

    TimingSnapshot snap;
    snap.phase_seconds = phase_seconds_;
    snap.kernel_busy = kernel_busy_total_;
    snap.active_busy = active_busy_total_;
    snap.iterations = iteration_counter_ - timing_start_iteration_;
    snap.balance_records = balance_log_.size();
    timing_snapshot_ = std::move(snap);
    return wall;
}

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------

TimingReport Runtime::Impl::timing_report() {
    update_balance_window_events(true);

    // After stop_timing() the frozen counters are reported.
    const auto& seconds = timing_snapshot_ ? timing_snapshot_->phase_seconds : phase_seconds_;
    const auto& kernel_busy = timing_snapshot_ ? timing_snapshot_->kernel_busy : kernel_busy_total_;
    const auto& active_busy = timing_snapshot_ ? timing_snapshot_->active_busy : active_busy_total_;

    TimingReport report;
    report.iterations = timing_snapshot_ ? timing_snapshot_->iterations : iteration_counter_ - timing_start_iteration_;
    if (timing_wall_.has_value()) {
        report.wall_seconds = *timing_wall_;
    } else {
        const double local = MPI_Wtime() - timing_start_;
        detail::check_mpi(MPI_Allreduce(&local, &report.wall_seconds, 1, MPI_DOUBLE, MPI_MAX, comm_),
                          "MPI_Allreduce(wall time)");
    }
    const double wall = report.wall_seconds;

    // ---- Processes: phase times and host name of every rank -----------------
    constexpr int kPhases = static_cast<int>(Phase::count);
    std::vector<double> all_phases(static_cast<std::size_t>(kPhases) * size_);
    detail::check_mpi(MPI_Allgather(seconds.data(), kPhases, MPI_DOUBLE, all_phases.data(), kPhases, MPI_DOUBLE,
                                    comm_),
                      "MPI_Allgather(phase times)");

    char host[MPI_MAX_PROCESSOR_NAME] = {};
    int host_len = 0;
    MPI_Get_processor_name(host, &host_len);
    std::vector<char> hosts(static_cast<std::size_t>(MPI_MAX_PROCESSOR_NAME) * size_);
    detail::check_mpi(MPI_Allgather(host, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, hosts.data(), MPI_MAX_PROCESSOR_NAME,
                                    MPI_CHAR, comm_),
                      "MPI_Allgather(host names)");

    // Peak resident memory of each process (ru_maxrss is in KiB on Linux).
    struct rusage usage {};
    getrusage(RUSAGE_SELF, &usage);
    const double local_peak = static_cast<double>(usage.ru_maxrss) * 1024.0;
    std::vector<double> all_peaks(static_cast<std::size_t>(size_));
    detail::check_mpi(MPI_Allgather(&local_peak, 1, MPI_DOUBLE, all_peaks.data(), 1, MPI_DOUBLE, comm_),
                      "MPI_Allgather(peak memory)");

    for (int r = 0; r < size_; ++r) {
        const double* p = &all_phases[static_cast<std::size_t>(r) * kPhases];
        RankReport rr;
        rr.rank = r;
        rr.host = std::string(&hosts[static_cast<std::size_t>(r) * MPI_MAX_PROCESSOR_NAME]);
        rr.launch_seconds = p[index_of(Phase::launch)];
        rr.device_wait_seconds = p[index_of(Phase::device_wait)];
        rr.halo_seconds = p[index_of(Phase::halo)];
        rr.mpi_halo_seconds = p[index_of(Phase::mpi_halo)];
        rr.mpi_migration_seconds = p[index_of(Phase::mpi_migration)];
        rr.mpi_balance_seconds = p[index_of(Phase::mpi_balance)];
        rr.mpi_sync_seconds = p[index_of(Phase::mpi_sync)];
        rr.migration_seconds = p[index_of(Phase::migration)];
        rr.balance_seconds = p[index_of(Phase::balance)];
        rr.gather_seconds = p[index_of(Phase::gather)];
        double measured = 0.0;
        for (int k = 0; k < kPhases; ++k) measured += p[k];
        rr.unmeasured_seconds = std::max(0.0, wall - measured);
        rr.peak_memory_bytes = static_cast<std::size_t>(all_peaks[static_cast<std::size_t>(r)]);
        report.ranks.push_back(rr);
    }

    // ---- Devices: {global index, compute, active} from every rank -----------
    std::vector<double> local_dev;
    for (std::size_t d = 0; d < local_devices_.size(); ++d) {
        local_dev.push_back(static_cast<double>(local_devices_[d].global_index));
        local_dev.push_back(kernel_busy[d]);
        local_dev.push_back(active_busy[d]);
    }
    std::vector<int> counts(size_), displs(size_);
    int total = 0;
    for (int r = 0; r < size_; ++r) {
        counts[r] = all_device_counts_.empty() ? 0 : all_device_counts_[r] * 3;
        displs[r] = total;
        total += counts[r];
    }
    std::vector<double> all_dev(static_cast<std::size_t>(total));
    detail::check_mpi(MPI_Allgatherv(local_dev.data(), static_cast<int>(local_dev.size()), MPI_DOUBLE,
                                     all_dev.data(), counts.data(), displs.data(), MPI_DOUBLE, comm_),
                      "MPI_Allgatherv(device times)");

    for (std::size_t i = 0; i + 2 < all_dev.size(); i += 3) {
        DeviceReport dr;
        dr.global_index = static_cast<int>(all_dev[i]);
        dr.compute_seconds = all_dev[i + 1];
        dr.transfer_seconds = std::max(0.0, all_dev[i + 2] - all_dev[i + 1]);
        dr.idle_seconds = std::max(0.0, wall - all_dev[i + 2]);
        for (const DeviceInfo& info : global_devices_) {
            if (info.global_index != dr.global_index) continue;
            dr.rank = info.rank;
            dr.name = info.name;
            dr.kind = info.kind;
        }
        for (const DevicePartition& p : partitions_) {
            if (p.device_global_index == dr.global_index) dr.elements = p.element_count;
        }
        report.devices.push_back(dr);
    }

    // ---- Critical path: the process of the slowest device ---------------------
    TimelineReport& tl = report.timeline;
    double slowest = -1.0;
    for (const DeviceReport& d : report.devices) {
        if (d.compute_seconds > slowest) {
            slowest = d.compute_seconds;
            tl.device = d.global_index;
            tl.rank = d.rank;
        }
    }
    if (tl.rank < 0 && !report.ranks.empty()) tl.rank = 0;
    for (const RankReport& p : report.ranks) {
        if (p.rank != tl.rank) continue;
        const double opencl = p.launch_seconds + p.device_wait_seconds;
        tl.opencl_compute_seconds = std::min(opencl, std::max(0.0, slowest));
        tl.opencl_other_seconds = opencl - tl.opencl_compute_seconds;
        tl.halo_seconds = p.halo_seconds;
        tl.mpi_halo_seconds = p.mpi_halo_seconds;
        tl.mpi_migration_seconds = p.mpi_migration_seconds;
        tl.mpi_balance_seconds = p.mpi_balance_seconds;
        tl.mpi_sync_seconds = p.mpi_sync_seconds;
        tl.migration_seconds = p.migration_seconds;
        tl.balance_seconds = p.balance_seconds;
        tl.gather_seconds = p.gather_seconds;
        tl.unmeasured_seconds = p.unmeasured_seconds;
    }

    const std::size_t records = timing_snapshot_ ? timing_snapshot_->balance_records : balance_log_.size();
    report.balance.assign(balance_log_.begin(), balance_log_.begin() + static_cast<std::ptrdiff_t>(records));
    report.results = results_;
    return report;
}

} // namespace harmony
