// Everything HARMONY prints. Rank 0 writes to standard output:
//
//   run header    before the first step: application, execution settings, devices
//   balancing     one entry per load-balancing decision (Runtime::set_verbose)
//   summary       Runtime::print_timing_report()
//
// and Runtime::write_timing_report() writes the same data as CSV. Warnings
// go to standard error with the prefix "[HARMONY] warning:". The format is
// described in README.md ("Log reference").

#include "runtime_impl.hpp"
#include "detail/format.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace harmony {

namespace {

constexpr std::size_t kRuleWidth = 100;
constexpr std::size_t kKeyWidth = 24;
constexpr std::size_t kNameWidth = 36;
// Balancing table of the summary: beyond this many decisions only the
// applied ones are listed.
constexpr std::size_t kMaxSummaryRows = 30;
// Columns of the table of shares over time.
constexpr std::size_t kMaxDivisionColumns = 12;
const char* const kEventIndent = "          ";

std::string rule() { return std::string(kRuleWidth, '='); }

void title(std::ostream& out, const std::string& text) {
    out << rule() << "\n" << text << "\n" << rule() << "\n";
}

// "  Key                     value"
void key_value(std::ostream& out, const std::string& key, const std::string& value) {
    out << "  " << std::left << std::setw(static_cast<int>(kKeyWidth)) << key << std::right << value << "\n";
}

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

std::string device_name(const std::vector<DeviceInfo>& devices, std::size_t index) {
    const DeviceInfo* d = find_device(devices, index);
    return d ? detail::truncate(d->name, kNameWidth) : "?";
}

std::string device_type(const std::vector<DeviceInfo>& devices, std::size_t index) {
    const DeviceInfo* d = find_device(devices, index);
    return d ? detail::device_kind_name(d->kind) : "?";
}

std::size_t slowest_device(const std::vector<double>& seconds) {
    std::size_t slowest = 0;
    for (std::size_t i = 1; i < seconds.size(); ++i) {
        if (seconds[i] > seconds[slowest]) slowest = i;
    }
    return slowest;
}

bool is_dynamic(BalanceMode m) { return m == BalanceMode::dynamic_threshold || m == BalanceMode::dynamic_profiled; }
bool is_threshold(BalanceMode m) {
    return m == BalanceMode::dynamic_threshold || m == BalanceMode::static_threshold;
}

std::string policy_name(BalanceMode m) {
    if (m == BalanceMode::off) return "off";
    return std::string(is_dynamic(m) ? "dynamic" : "static") + " " + (is_threshold(m) ? "threshold" : "profiled");
}

std::string halo_sync_name(HaloSync s) {
    switch (s) {
        case HaloSync::overlap: return "overlap";
        case HaloSync::local: return "local";
        case HaloSync::global: return "global";
    }
    return "?";
}

std::string share(const std::vector<std::size_t>& elements, std::size_t i) {
    double total = 0.0;
    for (std::size_t e : elements) total += static_cast<double>(e);
    return detail::format_percent(total > 0.0 && i < elements.size() ? elements[i] / total : 0.0);
}

} // namespace

// ---------------------------------------------------------------------------
// Devices
// ---------------------------------------------------------------------------

detail::Table Runtime::Impl::device_table(bool with_partition) const {
    std::vector<detail::Table::Column> cols = {{"Dev", false},   {"Rank", false},          {"Host", false},
                                               {"Type", false},  {"Compute units", true}, {"Memory", true},
                                               {"Max buffer", true}};
    if (with_partition) cols.insert(cols.end(), {{"Elements", true}, {"Share", true}});
    cols.push_back({"Name", false});
    detail::Table table(cols);

    const double total = partition_.has_value() ? static_cast<double>(partition_->global_elements) : 0.0;
    for (const DeviceInfo& d : global_devices_) {
        std::vector<std::string> row = {std::to_string(d.global_index), std::to_string(d.rank), d.host,
                                        detail::device_kind_name(d.kind), std::to_string(d.compute_units),
                                        detail::format_bytes(d.global_memory), detail::format_bytes(d.max_buffer)};
        if (with_partition) {
            std::size_t elements = 0;
            for (const DevicePartition& p : partitions_) {
                if (p.device_global_index == d.global_index) elements = p.element_count;
            }
            row.push_back(detail::format_count(elements));
            row.push_back(detail::format_percent(total > 0.0 ? elements / total : 0.0));
        }
        row.push_back(detail::truncate(d.name, kNameWidth));
        table.add_row(row);
    }
    return table;
}

// ---------------------------------------------------------------------------
// Run header
// ---------------------------------------------------------------------------

void Runtime::Impl::print_run_header(const ExecutionStep& step) {
    run_header_printed_ = true;
    if (rank_ != 0) return;

    std::ostringstream out;
    title(out, "HARMONY" + (run_info_.application.empty() ? std::string() : "  " + run_info_.application));

    if (!run_info_.parameters.empty()) {
        out << "Application\n";
        for (const auto& [key, value] : run_info_.parameters) key_value(out, key, value);
    }

    out << "Execution\n";
    key_value(out, "MPI processes", std::to_string(size_));
    key_value(out, "Devices", std::to_string(global_devices_.size()));
    if (partition_.has_value()) {
        key_value(out, "Domain", detail::format_count(partition_->global_elements) + " elements, granularity " +
                                     detail::format_count(partition_->granularity));
    }
    std::size_t field_bytes = 0;
    for (const auto& kv : fields_) field_bytes += kv.second.total_bytes();
    key_value(out, "Data per device",
              detail::format_bytes(field_bytes) + " (" + std::to_string(fields_.size()) +
                  (fields_.size() == 1 ? " field)" : " fields)"));

    std::string kernels;
    for (const KernelInvocation& ki : step.invocations) {
        auto it = kernels_.find(ki.binding.kernel.value);
        if (!kernels.empty()) kernels += ", ";
        kernels += it != kernels_.end() ? it->second.spec.entry_point : "?";
    }
    key_value(out, "Step", step.name + ": " + kernels);

    if (step.halos.empty()) {
        key_value(out, "Halo exchange", "none");
    } else {
        for (const HaloSpec& hs : step.halos) {
            std::string names;
            for (FieldHandle h : hs.fields) {
                auto it = fields_.find(h.value);
                if (!names.empty()) names += ", ";
                names += it != fields_.end() ? it->second.spec.name : "?";
            }
            key_value(out, "Halo exchange", detail::format_count(hs.width_elements) + " elements: " + names);
        }
        key_value(out, "Halo synchronization", halo_sync_name(step.halo_sync));
    }

    const AutoBalancePolicy& p = step.balance;
    key_value(out, "Load balancing", policy_name(p.mode));
    if (p.mode != BalanceMode::off) {
        const std::string interval = detail::format_count(static_cast<std::size_t>(std::max(1, p.interval)));
        key_value(out, "Warm-up probe",
                  p.warmup > 0 ? "iteration " + detail::format_count(static_cast<std::size_t>(p.warmup)) : "none");
        key_value(out, "Balancing schedule",
                  is_dynamic(p.mode) ? "every " + interval + " iterations" : "iteration " + interval);
        if (is_threshold(p.mode)) {
            key_value(out, "Threshold", detail::format_number(p.threshold, 4));
        } else {
            key_value(out, "Migration profile", p.profiling_file.empty() ? "default" : p.profiling_file);
        }
    }

    out << "Devices\n";
    device_table(true).print(out, "  ");
    out << rule() << "\n";
    std::cout << out.str() << std::flush;
}

// ---------------------------------------------------------------------------
// Balancing log
// ---------------------------------------------------------------------------

// One entry per decision:
//   [HARMONY] Iteration I | <Warm-up probe | Balancing N> (policy) | applied/kept | reason
//             Time per iteration: ...
//             (table and migration line when applied)
void Runtime::Impl::print_balance_interval_metrics(BalanceRecord record) {
    record.number = record.warmup_probe ? 0 : ++balance_attempts_;

    if (verbose_ && rank_ == 0) {
        std::ostringstream out;
        out << "[HARMONY] Iteration " << detail::format_count(record.iteration) << " | "
            << (record.warmup_probe ? std::string("Warm-up probe") : "Balancing " + std::to_string(record.number))
            << " (" << record.policy << ") | " << (record.applied ? "applied" : "kept") << " | " << record.reason
            << "\n";

        const std::size_t devices = record.elements_after.size();
        const bool measured = record.window_iterations > 0 && record.window_seconds.size() == devices;
        if (measured) {
            out << kEventIndent << "Time per iteration: " << detail::format_seconds(record.iteration_seconds_before)
                << " measured over " << detail::format_count(record.window_iterations) << " iterations, "
                << detail::format_seconds(record.iteration_seconds_after)
                << (record.applied ? " predicted" : " with the proposed division") << "\n";
        }

        if (record.applied) {
            const std::size_t slow = measured ? slowest_device(record.window_seconds) : devices;
            std::vector<detail::Table::Column> columns = {{"Dev", false}, {"Type", false}, {"Name", false}};
            if (measured) columns.push_back({"Kernel time/iteration", true});
            columns.insert(columns.end(), {{"Elements before", true}, {"Elements after", true},
                                           {"Share before", true}, {"Share after", true}});
            if (measured) columns.push_back({"", false});
            detail::Table table(columns);

            const double n = static_cast<double>(std::max<std::size_t>(record.window_iterations, 1));
            for (std::size_t i = 0; i < devices; ++i) {
                const std::size_t before = i < record.elements_before.size() ? record.elements_before[i] : 0;
                std::vector<std::string> row = {std::to_string(i), device_type(global_devices_, i),
                                                device_name(global_devices_, i)};
                if (measured) row.push_back(detail::format_seconds(record.window_seconds[i] / n));
                row.insert(row.end(), {detail::format_count(before), detail::format_count(record.elements_after[i]),
                                       share(record.elements_before, i), share(record.elements_after, i)});
                if (measured) row.push_back(i == slow ? "slowest" : "");
                table.add_row(row);
            }
            table.print(out, kEventIndent);
            out << kEventIndent << "Migration: " << detail::format_bytes(record.bytes_moved) << " in "
                << detail::format_seconds(record.migration_seconds) << " | Decision: "
                << detail::format_seconds(record.decision_seconds) << "\n";
        }
        std::cout << out.str() << std::flush;
    }
    balance_log_.push_back(std::move(record));
}

// ---------------------------------------------------------------------------
// Summary
// ---------------------------------------------------------------------------

void Runtime::Impl::print_timing_report(std::ostream& os) {
    const TimingReport r = timing_report();
    if (rank_ != 0) return;

    const double wall = r.wall_seconds;
    const double iters = static_cast<double>(std::max<std::size_t>(r.iterations, 1));
    std::ostringstream out;

    std::size_t probes = 0, applied = 0, bytes = 0;
    double migration = 0.0;
    for (const BalanceRecord& b : r.balance) {
        probes += b.warmup_probe ? 1 : 0;
        if (!b.applied) continue;
        ++applied;
        bytes += b.bytes_moved;
        migration += b.migration_seconds;
    }
    const std::size_t attempts = r.balance.size() - probes;

    title(out, "HARMONY  summary" + (run_info_.application.empty() ? std::string() : "  " + run_info_.application));
    out << "Run\n";
    key_value(out, "Wall time", detail::format_seconds(wall));
    key_value(out, "Iterations", detail::format_count(r.iterations) +
                                     (r.iterations > 0 ? " (" + detail::format_seconds(wall / iters) + " per iteration)"
                                                       : std::string()));
    if (balance_policy_ && balance_policy_->mode != BalanceMode::off) {
        std::string text = policy_name(balance_policy_->mode) + ": ";
        if (probes > 0) text += "warm-up probe, ";
        text += std::to_string(attempts) + (attempts == 1 ? " attempt, " : " attempts, ") + std::to_string(applied) +
                " applied, " + detail::format_bytes(bytes) + " moved in " + detail::format_seconds(migration);
        key_value(out, "Load balancing", text);
    } else {
        key_value(out, "Load balancing", "off");
    }

    // ---- Critical path ------------------------------------------------------
    const TimelineReport& tl = r.timeline;
    out << "Critical path: rank " << tl.rank;
    if (tl.device >= 0) {
        const std::size_t dev = static_cast<std::size_t>(tl.device);
        out << ", device " << tl.device << " (" << device_type(global_devices_, dev) << ", "
            << device_name(global_devices_, dev) << ")";
    }
    out << "\n";
    {
        const bool gather = tl.gather_seconds > 0.0;
        std::vector<detail::Table::Column> cols = {
            {"Wall", true},          {"Kernels", true},       {"OpenCL other", true}, {"Halo", true},
            {"MPI halo", true},      {"MPI migration", true}, {"MPI balancing", true}, {"MPI barriers", true},
            {"Migration", true},     {"Balancing", true}};
        if (gather) cols.push_back({"Gather", true});
        cols.push_back({"Outside", true});
        detail::Table line(cols);
        std::vector<std::string> row = {detail::format_seconds(wall),
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
    }

    // ---- Devices ------------------------------------------------------------
    double total_elements = 0.0;
    for (const DeviceReport& d : r.devices) total_elements += static_cast<double>(d.elements);
    out << "Devices\n";
    detail::Table devices({{"Dev", false}, {"Rank", false}, {"Type", false}, {"Name", false},
                           {"Elements", true}, {"Share", true}, {"Compute", true}, {"Transfers", true},
                           {"Idle", true}, {"Kernel time/iteration", true}});
    for (const DeviceReport& d : r.devices) {
        devices.add_row({std::to_string(d.global_index), std::to_string(d.rank), detail::device_kind_name(d.kind),
                         detail::truncate(d.name, kNameWidth), detail::format_count(d.elements),
                         detail::format_percent(total_elements > 0 ? d.elements / total_elements : 0.0),
                         seconds_and_share(d.compute_seconds, wall), seconds_and_share(d.transfer_seconds, wall),
                         seconds_and_share(d.idle_seconds, wall), detail::format_seconds(d.compute_seconds / iters)});
    }
    devices.print(out, "  ");

    // ---- Processes ----------------------------------------------------------
    bool any_gather = false;
    for (const RankReport& p : r.ranks) any_gather = any_gather || p.gather_seconds > 0.0;
    out << "Processes\n";
    std::vector<detail::Table::Column> pcols = {{"Rank", false},        {"Host", false},     {"Launch", true},
                                                {"Device wait", true},  {"Halo", true},      {"MPI", true},
                                                {"Migration", true},    {"Balancing", true}};
    if (any_gather) pcols.push_back({"Gather", true});
    pcols.push_back({"Outside", true});
    pcols.push_back({"Peak RAM", true});
    detail::Table procs(pcols);
    for (const RankReport& p : r.ranks) {
        std::vector<std::string> row = {std::to_string(p.rank),
                                        p.host,
                                        seconds_and_share(p.launch_seconds, wall),
                                        seconds_and_share(p.device_wait_seconds, wall),
                                        seconds_and_share(p.halo_seconds, wall),
                                        seconds_and_share(p.mpi_seconds(), wall),
                                        seconds_and_share(p.migration_seconds, wall),
                                        seconds_and_share(p.balance_seconds, wall)};
        if (any_gather) row.push_back(seconds_and_share(p.gather_seconds, wall));
        row.push_back(seconds_and_share(p.unmeasured_seconds, wall));
        row.push_back(detail::format_bytes(p.peak_memory_bytes));
        procs.add_row(row);
    }
    procs.print(out, "  ");

    out << "MPI\n";
    detail::Table mpi({{"Rank", false}, {"Halo", true}, {"Migration", true}, {"Balancing", true},
                       {"Barriers", true}, {"Total", true}});
    for (const RankReport& p : r.ranks) {
        mpi.add_row({std::to_string(p.rank), detail::format_seconds(p.mpi_halo_seconds),
                     detail::format_seconds(p.mpi_migration_seconds), detail::format_seconds(p.mpi_balance_seconds),
                     detail::format_seconds(p.mpi_sync_seconds), detail::format_seconds(p.mpi_seconds())});
    }
    mpi.print(out, "  ");

    // ---- Load balancing -----------------------------------------------------
    if (!r.balance.empty()) {
        bool mixed_policies = false;
        for (const BalanceRecord& b : r.balance) mixed_policies = mixed_policies || b.policy != r.balance.front().policy;

        out << "Load balancing\n";
        const bool show_all = r.balance.size() <= kMaxSummaryRows;
        std::vector<detail::Table::Column> cols = {{"Decision", false}, {"Iteration", true}};
        if (mixed_policies) cols.push_back({"Policy", false});
        cols.insert(cols.end(), {{"Result", false}, {"Time/iteration", true}, {"Migration", true},
                                 {"Migration time", true}, {"Decision time", true}, {"Reason", false}});
        detail::Table bal(cols);
        std::size_t hidden = 0;
        for (const BalanceRecord& b : r.balance) {
            if (!show_all && !b.applied) {
                ++hidden;
                continue;
            }
            std::vector<std::string> row = {b.warmup_probe ? "warm-up" : std::to_string(b.number),
                                            detail::format_count(b.iteration)};
            if (mixed_policies) row.push_back(b.policy);
            row.push_back(b.applied ? "applied" : "kept");
            row.push_back(b.window_iterations > 0 ? detail::format_seconds(b.iteration_seconds_before) + " -> " +
                                                        detail::format_seconds(b.iteration_seconds_after)
                                                  : "-");
            row.push_back(b.applied ? detail::format_bytes(b.bytes_moved) : "-");
            row.push_back(b.applied ? detail::format_seconds(b.migration_seconds) : "-");
            row.push_back(detail::format_seconds(b.decision_seconds));
            row.push_back(b.reason);
            bal.add_row(row);
        }
        bal.print(out, "  ");
        if (hidden > 0) out << "  " << hidden << " decisions that kept the division are not listed\n";

        std::vector<const BalanceRecord*> changes;
        for (const BalanceRecord& b : r.balance) {
            if (b.applied) changes.push_back(&b);
        }
        if (!changes.empty()) {
            const std::size_t first = changes.size() > kMaxDivisionColumns ? changes.size() - kMaxDivisionColumns : 0;
            out << "Division (share of the domain per device)\n";
            std::vector<detail::Table::Column> dcols = {{"Dev", false}, {"Type", false},
                                                        {first > 0 ? "Before" : "Start", true}};
            for (std::size_t c = first; c < changes.size(); ++c) {
                dcols.push_back({(changes[c]->warmup_probe ? std::string("warm-up") : "#" + std::to_string(changes[c]->number)) +
                                     " @" + detail::format_count(changes[c]->iteration),
                                 true});
            }
            detail::Table div(dcols);
            const std::size_t count = changes.front()->elements_before.size();
            for (std::size_t i = 0; i < count; ++i) {
                std::vector<std::string> row = {std::to_string(i), device_type(global_devices_, i),
                                                share(changes[first]->elements_before, i)};
                for (std::size_t c = first; c < changes.size(); ++c) row.push_back(share(changes[c]->elements_after, i));
                div.add_row(row);
            }
            div.print(out, "  ");
            if (first > 0) out << "  last " << kMaxDivisionColumns << " changes shown\n";
        }
    }

    // ---- Application results ------------------------------------------------
    if (!r.results.empty()) {
        out << "Results\n";
        for (const ApplicationResult& res : r.results) key_value(out, res.name, res.value);
    }
    out << rule() << "\n";
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
        std::cerr << "[HARMONY] warning: cannot write the timing report to '" << path << "'" << std::endl;
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
           "mpi_sync_s,migration_s,balance_s,gather_s,unmeasured_s,peak_ram_bytes\n";
    for (const RankReport& p : r.ranks) {
        out << "process," << p.rank << "," << clean(p.host) << "," << p.launch_seconds << ","
            << p.device_wait_seconds << "," << p.halo_seconds << "," << p.mpi_halo_seconds << ","
            << p.mpi_migration_seconds << "," << p.mpi_balance_seconds << "," << p.mpi_sync_seconds << ","
            << p.migration_seconds << "," << p.balance_seconds << "," << p.gather_seconds << ","
            << p.unmeasured_seconds << "," << p.peak_memory_bytes << "\n";
    }

    out << "#balance,number,iteration,policy,applied,iteration_s_before,iteration_s_after,bytes_moved,"
           "migration_s,decision_s,warmup_probe,reason\n";
    for (const BalanceRecord& b : r.balance) {
        out << "balance," << b.number << "," << b.iteration << "," << clean(b.policy) << "," << (b.applied ? 1 : 0)
            << "," << b.iteration_seconds_before << "," << b.iteration_seconds_after << "," << b.bytes_moved << ","
            << b.migration_seconds << "," << b.decision_seconds << "," << (b.warmup_probe ? 1 : 0) << ","
            << clean(b.reason) << "\n";
    }

    out << "#result,name,value\n";
    for (const ApplicationResult& res : r.results) {
        out << "result," << clean(res.name) << "," << clean(res.value) << "\n";
    }
}

} // namespace harmony
