// Time measurement and the logs printed by rank 0:
//   - one entry per load-balancing attempt, while the program runs;
//   - the timing summary (print_timing_report), at the end.
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
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace harmony {

namespace {

int index_of(Phase p) { return static_cast<int>(p); }

// Attempts listed in the summary; beyond this only the applied ones.
constexpr std::size_t kMaxSummaryRows = 30;
// Changes shown in the table of shares over time.
constexpr std::size_t kMaxDivisionColumns = 12;

// "1.234 s (12.3 %)"
std::string seconds_and_share(double seconds, double wall) {
    std::string out = detail::format_seconds(seconds);
    if (wall > 0.0) out += " (" + detail::format_percent(seconds / wall) + ")";
    return out;
}

const DeviceInfo* find_device(const std::vector<DeviceInfo>& devices, std::size_t index) {
    for (const DeviceInfo& d : devices) {
        if (d.global_index == static_cast<int>(index)) return &d;
    }
    return nullptr;
}

std::string device_name(const std::vector<DeviceInfo>& devices, std::size_t index, std::size_t width = 36) {
    const DeviceInfo* d = find_device(devices, index);
    return d ? detail::truncate(d->name, width) : "?";
}

std::string device_type(const std::vector<DeviceInfo>& devices, std::size_t index) {
    const DeviceInfo* d = find_device(devices, index);
    return d ? detail::device_kind_name(d->kind) : "?";
}

// "CPU dev 3"
std::string device_short(const std::vector<DeviceInfo>& devices, std::size_t index) {
    return device_type(devices, index) + " dev " + std::to_string(index);
}

std::size_t slowest_device(const std::vector<double>& seconds) {
    std::size_t slowest = 0;
    for (std::size_t i = 1; i < seconds.size(); ++i) {
        if (seconds[i] > seconds[slowest]) slowest = i;
    }
    return slowest;
}

std::string iterations_text(long long n) {
    return detail::format_count(static_cast<std::size_t>(n)) + (n == 1 ? " iteration" : " iterations");
}

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
// Balancing log
// ---------------------------------------------------------------------------

// Printed once, the first time a step with balancing runs.
void Runtime::Impl::print_balance_policy(const AutoBalancePolicy& policy) {
    if (!verbose_ || rank_ != 0) return;
    const bool is_dynamic = policy.mode == BalanceMode::dynamic_threshold ||
                            policy.mode == BalanceMode::dynamic_profiled;
    const bool is_threshold = policy.mode == BalanceMode::dynamic_threshold ||
                              policy.mode == BalanceMode::static_threshold;
    const int interval = std::max(1, policy.interval);
    const int first = policy.warmup > 0 ? policy.warmup : interval;

    std::ostringstream out;
    out << "[HARMONY] Load balancing: " << (is_dynamic ? "dynamic " : "static ")
        << (is_threshold ? "threshold" : "profiled") << "\n"
        << "          Warm-up: " << iterations_text(first) << " with the initial division, then ";
    if (is_dynamic) out << "an attempt every " << iterations_text(interval);
    else out << "a single attempt";
    out << ".\n          Each attempt measures the last " << iterations_text(std::min(first, interval))
        << " and proposes shares proportional to the speed of each device;\n          ";
    if (is_threshold) {
        out << "the new division is applied when it changes by at least " << detail::format_number(policy.threshold, 4)
            << " (L2 distance of the cumulative loads).\n";
    } else {
        out << "the new division is applied when the time it saves until iteration "
            << detail::format_count(static_cast<std::size_t>(std::max(0, policy.total_iterations)))
            << " is larger than the cost of moving the data\n          (cost profile: "
            << (policy.profiling_file.empty() ? "-" : policy.profiling_file) << ").\n";
    }
    std::cout << out.str() << std::flush;
}

// Rank 0 prints every attempt as it happens:
//   [HARMONY] Balance #N | iteration I | policy | CHANGED / kept the division
//             Why: ...
//             Time per iteration: ... (+ a table per device when it changed)
void Runtime::Impl::print_balance_interval_metrics(BalanceRecord record) {
    record.number = ++balance_attempts_;

    if (verbose_ && rank_ == 0) {
        const std::string pad = "          ";
        std::ostringstream out;
        out << "[HARMONY] Balance #" << record.number << " | iteration " << detail::format_count(record.iteration)
            << " | " << record.policy << " | " << (record.applied ? "CHANGED the division" : "kept the division")
            << "\n" << pad << "Why: " << record.reason << "\n";

        const std::size_t devices = record.elements_after.size();
        const bool measured = record.window_iterations > 0 && record.window_seconds.size() == devices;
        if (measured) {
            const std::size_t slow = slowest_device(record.window_seconds);
            out << pad << "Time per iteration: " << detail::format_seconds(record.iteration_seconds_before)
                << " now (slowest: " << device_short(global_devices_, slow) << ") -> "
                << detail::format_seconds(record.iteration_seconds_after)
                << (record.applied ? " expected" : " with the best division") << " | measured over "
                << iterations_text(static_cast<long long>(record.window_iterations)) << "\n";
        }

        if (record.applied) {
            double total = 0.0;
            for (std::size_t e : record.elements_after) total += static_cast<double>(e);
            const std::size_t slow = measured ? slowest_device(record.window_seconds) : devices;

            std::vector<detail::Table::Column> columns = {{"Dev", false}, {"Rank", false}, {"Type", false},
                                                          {"Name", false}};
            if (measured) columns.push_back({"Kernel time/iteration", true});
            columns.insert(columns.end(), {{"Elements before", true}, {"Elements after", true}, {"Change", true},
                                           {"Share before", true}, {"Share after", true}});
            if (measured) columns.push_back({"Note", false});
            detail::Table table(columns);

            const double n = static_cast<double>(std::max<std::size_t>(record.window_iterations, 1));
            for (std::size_t i = 0; i < devices; ++i) {
                const std::size_t before = i < record.elements_before.size() ? record.elements_before[i] : 0;
                const std::size_t after = record.elements_after[i];
                const DeviceInfo* info = find_device(global_devices_, i);
                std::vector<std::string> row = {std::to_string(i), info ? std::to_string(info->rank) : "?",
                                                device_type(global_devices_, i), device_name(global_devices_, i)};
                if (measured) row.push_back(detail::format_seconds(record.window_seconds[i] / n));
                row.insert(row.end(), {detail::format_count(before), detail::format_count(after),
                                       detail::format_signed_count(static_cast<long long>(after) -
                                                                   static_cast<long long>(before)),
                                       detail::format_percent(total > 0 ? before / total : 0.0),
                                       detail::format_percent(total > 0 ? after / total : 0.0)});
                if (measured) row.push_back(i == slow ? "<- slowest" : "");
                table.add_row(row);
            }
            table.print(out, pad);
            out << pad << "Data moved: " << detail::format_bytes(record.bytes_moved) << " in "
                << detail::format_seconds(record.migration_seconds) << " | decision took "
                << detail::format_seconds(record.decision_seconds) << "\n";
        }
        std::cout << out.str() << std::flush;
    }
    balance_log_.push_back(std::move(record));
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
    return report;
}

void Runtime::Impl::print_timing_report(std::ostream& os) {
    const TimingReport r = timing_report();
    if (rank_ != 0) return;

    const double wall = r.wall_seconds;
    const double iters = static_cast<double>(std::max<std::size_t>(r.iterations, 1));
    const std::string rule(100, '=');
    std::ostringstream out;

    out << rule << "\n"
        << "HARMONY - TIMING SUMMARY\n"
        << rule << "\n"
        << "Wall time: " << detail::format_seconds(wall) << "    Iterations: " << detail::format_count(r.iterations);
    if (r.iterations > 0) out << " (" << detail::format_seconds(wall / iters) << " per iteration)";
    out << "\nMPI processes: " << r.ranks.size() << "    Devices: " << r.devices.size() << "\n";
    if (balance_policy_) {
        const AutoBalancePolicy& p = *balance_policy_;
        const bool is_dynamic = p.mode == BalanceMode::dynamic_threshold || p.mode == BalanceMode::dynamic_profiled;
        const bool is_threshold = p.mode == BalanceMode::dynamic_threshold || p.mode == BalanceMode::static_threshold;
        const int interval = std::max(1, p.interval);
        out << "Load balancing: " << (is_dynamic ? "dynamic " : "static ") << (is_threshold ? "threshold" : "profiled")
            << ", warm-up " << detail::format_count(static_cast<std::size_t>(p.warmup > 0 ? p.warmup : interval));
        if (is_dynamic) out << ", interval " << detail::format_count(static_cast<std::size_t>(interval));
        if (is_threshold) out << ", threshold " << detail::format_number(p.threshold, 4);
        out << "\n";
    } else {
        out << "Load balancing: off\n";
    }
    out << "\n";

    // ---- Critical path in one line --------------------------------------------
    {
        const TimelineReport& tl = r.timeline;
        const bool gather = tl.gather_seconds > 0.0;
        out << "TIMELINE OF THE CRITICAL PATH (the parts add up to the wall time)\n"
            << "  Process " << tl.rank << ", which drives the slowest device";
        if (tl.device >= 0) {
            out << " (" << device_type(global_devices_, static_cast<std::size_t>(tl.device)) << " dev " << tl.device
                << ", " << device_name(global_devices_, static_cast<std::size_t>(tl.device)) << ")";
        }
        out << ".\n  OpenCL compute = kernel time of that device; OpenCL overhead = the rest of the time the process\n"
            << "  was blocked in OpenCL (launching, copies, driver latency). MPI = waits for messages and collectives.\n\n";
        std::vector<detail::Table::Column> cols = {
            {"Wall", true},          {"OpenCL compute", true}, {"OpenCL overhead", true},
            {"Halo prep", true},     {"MPI halo", true},       {"MPI migration", true},
            {"MPI balancing", true}, {"MPI barriers", true},   {"Data migration", true},
            {"Balancing", true}};
        if (gather) cols.push_back({"Gather", true});
        cols.push_back({"Not measured", true});
        detail::Table line(cols);
        std::vector<std::string> row = {
            detail::format_seconds(wall),
            seconds_and_share(tl.opencl_compute_seconds, wall),
            seconds_and_share(tl.opencl_other_seconds, wall),
            seconds_and_share(tl.halo_seconds, wall),
            seconds_and_share(tl.mpi_halo_seconds, wall),
            seconds_and_share(tl.mpi_migration_seconds, wall),
            seconds_and_share(tl.mpi_balance_seconds, wall),
            seconds_and_share(tl.mpi_sync_seconds, wall),
            seconds_and_share(tl.migration_seconds, wall),
            seconds_and_share(tl.balance_seconds, wall)};
        if (gather) row.push_back(seconds_and_share(tl.gather_seconds, wall));
        row.push_back(seconds_and_share(tl.unmeasured_seconds, wall));
        line.add_row(row);
        line.print(out, "  ");
        out << "\n";
    }

    // ---- Devices ------------------------------------------------------------
    double total_elements = 0.0;
    for (const DeviceReport& d : r.devices) total_elements += static_cast<double>(d.elements);

    out << "TIME OF EACH DEVICE (Compute + Transfers + Idle = wall time)\n"
        << "  Compute:    OpenCL kernels running\n"
        << "  Transfers:  copies between host and device memory (halos and data migration)\n"
        << "  Idle:       no work (waiting for other devices, MPI or the CPU)\n\n";
    detail::Table devices({{"Dev", false}, {"Rank", false}, {"Type", false}, {"Name", false},
                           {"Elements", true}, {"Share", true}, {"Compute", true}, {"Transfers", true},
                           {"Idle", true}});
    for (const DeviceReport& d : r.devices) {
        devices.add_row({std::to_string(d.global_index), std::to_string(d.rank), detail::device_kind_name(d.kind),
                         detail::truncate(d.name, 36), detail::format_count(d.elements),
                         detail::format_percent(total_elements > 0 ? d.elements / total_elements : 0.0),
                         seconds_and_share(d.compute_seconds, wall), seconds_and_share(d.transfer_seconds, wall),
                         seconds_and_share(d.idle_seconds, wall)});
    }
    devices.print(out, "  ");
    if (r.devices.size() > 1) {
        std::size_t busiest = 0, least = 0;
        for (std::size_t i = 1; i < r.devices.size(); ++i) {
            if (r.devices[i].compute_seconds > r.devices[busiest].compute_seconds) busiest = i;
            if (r.devices[i].compute_seconds < r.devices[least].compute_seconds) least = i;
        }
        const DeviceReport& b = r.devices[busiest];
        const DeviceReport& l = r.devices[least];
        out << "  Busiest device: " << detail::device_kind_name(b.kind) << " dev " << b.global_index << ", "
            << detail::format_seconds(b.compute_seconds) << " of kernels = "
            << detail::format_seconds(b.compute_seconds / iters) << " per iteration.";
        if (l.compute_seconds > 0.0) {
            out << " Least busy: " << detail::device_kind_name(l.kind) << " dev " << l.global_index << ", "
                << detail::format_number(b.compute_seconds / l.compute_seconds, 3) << "x less.";
        }
        out << "\n  A large Idle on the fast devices means they wait for the busiest one.\n";
    }

    // ---- Processes ----------------------------------------------------------
    bool any_gather = false;
    for (const RankReport& p : r.ranks) any_gather = any_gather || p.gather_seconds > 0.0;

    out << "\nTIME OF EACH MPI PROCESS (the CPU that drives the devices; the columns add up to the wall time)\n"
        << "  OpenCL launch:   preparing and enqueueing kernels\n"
        << "  OpenCL wait:     CPU waiting for the devices to finish kernels and copies\n"
        << "  Halo exchange:   preparing and copying borders between devices (without the waits)\n"
        << "  MPI:             waiting for MPI messages and collective operations\n"
        << "  Data migration:  moving data between devices after a rebalance (without MPI)\n"
        << "  Balancing:       measuring times and deciding the new division (without MPI)\n";
    if (any_gather) out << "  Gather:          collecting a field on rank 0 (Runtime::gather)\n";
    out << "  Not measured:    outside HARMONY (application code between calls)\n"
        << "  High OpenCL wait or MPI usually means waiting for a slower device or process.\n\n";

    std::vector<detail::Table::Column> pcols = {{"Rank", false}, {"Host", false}, {"OpenCL launch", true},
                                                {"OpenCL wait", true}, {"Halo exchange", true},
                                                {"MPI", true}, {"Data migration", true}, {"Balancing", true}};
    if (any_gather) pcols.push_back({"Gather", true});
    pcols.push_back({"Not measured", true});
    detail::Table procs(pcols);
    for (const RankReport& p : r.ranks) {
        std::vector<std::string> row = {std::to_string(p.rank), p.host, seconds_and_share(p.launch_seconds, wall),
                                        seconds_and_share(p.device_wait_seconds, wall),
                                        seconds_and_share(p.halo_seconds, wall),
                                        seconds_and_share(p.mpi_seconds(), wall),
                                        seconds_and_share(p.migration_seconds, wall),
                                        seconds_and_share(p.balance_seconds, wall)};
        if (any_gather) row.push_back(seconds_and_share(p.gather_seconds, wall));
        row.push_back(seconds_and_share(p.unmeasured_seconds, wall));
        procs.add_row(row);
    }
    procs.print(out, "  ");

    out << "\n  MPI time by reason\n";
    detail::Table mpi({{"Rank", false}, {"Halo exchange", true}, {"Data migration", true},
                       {"Balancing", true}, {"Barriers", true}, {"Total", true}});
    for (const RankReport& p : r.ranks) {
        mpi.add_row({std::to_string(p.rank), detail::format_seconds(p.mpi_halo_seconds),
                     detail::format_seconds(p.mpi_migration_seconds), detail::format_seconds(p.mpi_balance_seconds),
                     detail::format_seconds(p.mpi_sync_seconds), detail::format_seconds(p.mpi_seconds())});
    }
    mpi.print(out, "  ");

    // ---- Balancing ----------------------------------------------------------
    out << "\nLOAD BALANCING\n";
    if (r.balance.empty()) {
        out << "  No attempts.\n";
    } else {
        std::size_t applied = 0, bytes = 0;
        double migration = 0.0, decision = 0.0;
        bool mixed_policies = false;
        for (const BalanceRecord& b : r.balance) {
            decision += b.decision_seconds;
            mixed_policies = mixed_policies || b.policy != r.balance.front().policy;
            if (!b.applied) continue;
            ++applied;
            bytes += b.bytes_moved;
            migration += b.migration_seconds;
        }
        out << "  " << r.balance.size() << (r.balance.size() == 1 ? " attempt, " : " attempts, ") << applied
            << " changed the division. Data moved: " << detail::format_bytes(bytes) << " in "
            << detail::format_seconds(migration) << ". Decisions: " << detail::format_seconds(decision) << ".\n"
            << "  Time/iteration = kernel time of the slowest device per iteration, measured before the attempt\n"
            << "  and expected after it (for a kept division: expected with the best division).\n\n";

        const bool show_all = r.balance.size() <= kMaxSummaryRows;
        std::vector<detail::Table::Column> cols = {{"#", true}, {"Iteration", true}};
        if (mixed_policies) cols.push_back({"Policy", false});
        cols.insert(cols.end(), {{"Result", false}, {"Time/iteration", true}, {"Data moved", true},
                                 {"Moved in", true}, {"Why", false}});
        detail::Table bal(cols);
        std::size_t hidden = 0;
        for (const BalanceRecord& b : r.balance) {
            if (!show_all && !b.applied) {
                ++hidden;
                continue;
            }
            std::vector<std::string> row = {std::to_string(b.number), detail::format_count(b.iteration)};
            if (mixed_policies) row.push_back(b.policy);
            row.push_back(b.applied ? "changed" : "kept");
            row.push_back(b.window_iterations > 0 ? detail::format_seconds(b.iteration_seconds_before) + " -> " +
                                                        detail::format_seconds(b.iteration_seconds_after)
                                                  : "-");
            row.push_back(b.applied ? detail::format_bytes(b.bytes_moved) : "-");
            row.push_back(b.applied ? detail::format_seconds(b.migration_seconds) : "-");
            row.push_back(b.reason);
            bal.add_row(row);
        }
        bal.print(out, "  ");
        if (hidden > 0) out << "  (" << hidden << " attempts that kept the division are not listed)\n";

        // Share of every device at the start and after each change.
        std::vector<const BalanceRecord*> changes;
        for (const BalanceRecord& b : r.balance) {
            if (b.applied) changes.push_back(&b);
        }
        if (!changes.empty()) {
            const std::size_t first = changes.size() > kMaxDivisionColumns ? changes.size() - kMaxDivisionColumns : 0;
            out << "\n  Share of the domain per device: at the start and after each change (#attempt @iteration)\n";
            std::vector<detail::Table::Column> dcols = {{"Dev", false}, {"Type", false}, {"Start", true}};
            for (std::size_t c = first; c < changes.size(); ++c) {
                dcols.push_back({"#" + std::to_string(changes[c]->number) + " @" +
                                     detail::format_count(changes[c]->iteration),
                                 true});
            }
            detail::Table div(dcols);
            auto share = [](const std::vector<std::size_t>& v, std::size_t i) {
                double total = 0.0;
                for (std::size_t e : v) total += static_cast<double>(e);
                return detail::format_percent(total > 0.0 && i < v.size() ? v[i] / total : 0.0);
            };
            const std::size_t devices = changes.front()->elements_before.size();
            for (std::size_t i = 0; i < devices; ++i) {
                std::vector<std::string> row = {std::to_string(i), device_type(global_devices_, i),
                                                share(changes[first]->elements_before, i)};
                for (std::size_t c = first; c < changes.size(); ++c) row.push_back(share(changes[c]->elements_after, i));
                div.add_row(row);
            }
            div.print(out, "  ");
            if (first > 0) out << "  (only the last " << kMaxDivisionColumns << " changes are shown)\n";
        }
    }
    out << rule << "\n";
    os << out.str() << std::flush;
}

// ---------------------------------------------------------------------------
// CSV report
// ---------------------------------------------------------------------------

void Runtime::Impl::write_timing_report(const std::string& path) {
    const TimingReport r = timing_report();
    if (rank_ != 0) return;

    std::ofstream out(path);
    if (!out) {
        std::cerr << "[HARMONY Warning] could not write the timing report to '" << path << "'" << std::endl;
        return;
    }
    // Commas would break the columns.
    auto clean = [](std::string text) {
        for (char& c : text) {
            if (c == ',' || c == '\n' || c == '\r') c = ';';
        }
        return text;
    };

    std::size_t applied = 0, bytes = 0;
    for (const BalanceRecord& b : r.balance) {
        if (!b.applied) continue;
        ++applied;
        bytes += b.bytes_moved;
    }

    out << std::setprecision(9);
    out << "#run,wall_s,iterations,mpi_processes,devices,balance_attempts,balance_applied,bytes_moved\n"
        << "run," << r.wall_seconds << "," << r.iterations << "," << r.ranks.size() << "," << r.devices.size()
        << "," << r.balance.size() << "," << applied << "," << bytes << "\n";

    const TimelineReport& tl = r.timeline;
    out << "#timeline,rank,device,wall_s,opencl_compute_s,opencl_other_s,halo_s,mpi_halo_s,mpi_migration_s,"
           "mpi_balance_s,mpi_sync_s,migration_s,balance_s,gather_s,unmeasured_s\n"
        << "timeline," << tl.rank << "," << tl.device << "," << r.wall_seconds << "," << tl.opencl_compute_seconds
        << "," << tl.opencl_other_seconds << "," << tl.halo_seconds << "," << tl.mpi_halo_seconds << ","
        << tl.mpi_migration_seconds << "," << tl.mpi_balance_seconds << "," << tl.mpi_sync_seconds << ","
        << tl.migration_seconds << "," << tl.balance_seconds << "," << tl.gather_seconds << ","
        << tl.unmeasured_seconds << "\n";

    out << "#device,index,rank,type,name,elements,compute_s,transfer_s,idle_s\n";
    for (const DeviceReport& d : r.devices) {
        out << "device," << d.global_index << "," << d.rank << "," << detail::device_kind_name(d.kind) << ","
            << clean(d.name) << "," << d.elements << "," << d.compute_seconds << "," << d.transfer_seconds << ","
            << d.idle_seconds << "\n";
    }

    out << "#process,rank,host,launch_s,device_wait_s,halo_s,mpi_halo_s,mpi_migration_s,mpi_balance_s,"
           "mpi_sync_s,migration_s,balance_s,gather_s,unmeasured_s\n";
    for (const RankReport& p : r.ranks) {
        out << "process," << p.rank << "," << clean(p.host) << "," << p.launch_seconds << ","
            << p.device_wait_seconds << "," << p.halo_seconds << "," << p.mpi_halo_seconds << ","
            << p.mpi_migration_seconds << "," << p.mpi_balance_seconds << "," << p.mpi_sync_seconds << ","
            << p.migration_seconds << "," << p.balance_seconds << "," << p.gather_seconds << ","
            << p.unmeasured_seconds << "\n";
    }

    out << "#balance,number,iteration,policy,applied,iteration_s_before,iteration_s_after,bytes_moved,"
           "migration_s,decision_s,reason\n";
    for (const BalanceRecord& b : r.balance) {
        out << "balance," << b.number << "," << b.iteration << "," << clean(b.policy) << "," << (b.applied ? 1 : 0)
            << "," << b.iteration_seconds_before << "," << b.iteration_seconds_after << "," << b.bytes_moved << ","
            << b.migration_seconds << "," << b.decision_seconds << "," << clean(b.reason) << "\n";
    }
}

} // namespace harmony
