// Dependency tracking, kernel statistics and synchronization.

#include "runtime_impl.hpp"

#include <algorithm>

namespace harmony {

namespace {

// Kernel events kept per device before finished ones are collected.
constexpr std::size_t kHarvestThreshold = 256;
// Readers kept per tracker before finished ones are dropped.
constexpr std::size_t kReaderPruneThreshold = 32;

bool is_complete(cl_event ev) {
    cl_int status = CL_QUEUED;
    if (clGetEventInfo(ev, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(status), &status, nullptr) != CL_SUCCESS) {
        return false;
    }
    return status <= CL_COMPLETE;  // complete, or terminated with an error
}

} // namespace

// ---------------------------------------------------------------------------
// Access trackers
// ---------------------------------------------------------------------------

void Runtime::Impl::set_field_read_event(detail::AccessTracker& tracker, cl_event ev) {
    if (tracker.readers.size() >= kReaderPruneThreshold) {
        // Fields that are only read (e.g. coefficients) would grow forever.
        auto done = std::stable_partition(tracker.readers.begin(), tracker.readers.end(),
                                          [](cl_event e) { return !is_complete(e); });
        for (auto it = done; it != tracker.readers.end(); ++it) clReleaseEvent(*it);
        tracker.readers.erase(done, tracker.readers.end());
    }
    detail::check_cl(clRetainEvent(ev), "clRetainEvent(read)");
    tracker.readers.push_back(ev);
}

// The new writer waited for every previous writer and reader, so it replaces them.
void Runtime::Impl::set_field_write_event(detail::AccessTracker& tracker, cl_event ev) {
    detail::release_events(tracker.writers);
    detail::release_events(tracker.readers);
    detail::check_cl(clRetainEvent(ev), "clRetainEvent(write)");
    tracker.writers.push_back(ev);
}

void Runtime::Impl::release_tracked_field_events() {
    for (auto& kv : fields_) {
        for (detail::AccessTracker& t : kv.second.core) {
            detail::release_events(t.writers);
            detail::release_events(t.readers);
        }
        for (detail::AccessTracker& t : kv.second.ghost) {
            detail::release_events(t.writers);
            detail::release_events(t.readers);
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel statistics and balance window
// ---------------------------------------------------------------------------

void Runtime::Impl::append_balance_window_kernel_events(std::size_t d, cl_event ev) {
    if (ev == nullptr || d >= pending_kernel_events_.size()) return;
    detail::check_cl(clRetainEvent(ev), "clRetainEvent(kernel statistics)");
    pending_kernel_events_[d].push_back(ev);
    if (pending_kernel_events_[d].size() >= kHarvestThreshold) update_balance_window_events(false);
}

void Runtime::Impl::track_transfer_event(std::size_t d, cl_event ev) {
    if (ev == nullptr || d >= pending_transfer_events_.size()) return;
    detail::check_cl(clRetainEvent(ev), "clRetainEvent(transfer statistics)");
    pending_transfer_events_[d].push_back(ev);
    if (pending_transfer_events_[d].size() >= kHarvestThreshold) update_balance_window_events(false);
}

// Moves finished commands into the counters. With wait=true every pending
// command is waited for; otherwise each queue stops at its first command
// still running. Busy times are unions of the profiled intervals, so
// commands running at the same time are not counted twice.
void Runtime::Impl::update_balance_window_events(bool wait) {
    struct Interval {
        double start;
        double end;
        bool kernel;
    };
    std::vector<Interval> done;

    auto collect = [&](std::deque<cl_event>& pending, bool kernel) {
        while (!pending.empty()) {
            cl_event ev = pending.front();
            if (wait) {
                detail::check_cl(clWaitForEvents(1, &ev), "clWaitForEvents(statistics)");
            } else if (!is_complete(ev)) {
                break;
            }
            double start = 0.0, end = 0.0;
            if (detail::event_interval(ev, start, end)) done.push_back({start, end, kernel});
            clReleaseEvent(ev);
            pending.pop_front();
        }
    };

    for (std::size_t d = 0; d < pending_kernel_events_.size(); ++d) {
        done.clear();
        collect(pending_kernel_events_[d], true);
        collect(pending_transfer_events_[d], false);
        std::sort(done.begin(), done.end(), [](const Interval& a, const Interval& b) { return a.start < b.start; });

        for (const Interval& it : done) {
            if (it.kernel) {
                const double from = std::max(it.start, kernel_busy_until_[d]);
                if (it.end > from) {
                    kernel_busy_total_[d] += it.end - from;
                    kernel_busy_window_[d] += it.end - from;
                }
                kernel_busy_until_[d] = std::max(kernel_busy_until_[d], it.end);
            }
            const double from = std::max(it.start, active_busy_until_[d]);
            if (it.end > from) active_busy_total_[d] += it.end - from;
            active_busy_until_[d] = std::max(active_busy_until_[d], it.end);
        }
    }
}

void Runtime::Impl::reset_balance_window_events() {
    PhaseTimer timer(*this, Phase::device_wait);
    update_balance_window_events(true);
    std::fill(kernel_busy_window_.begin(), kernel_busy_window_.end(), 0.0);
    balance_window_start_iteration_ = iteration_counter_;
}

// Busy seconds per local device since the window started.
std::vector<double> Runtime::Impl::compute_balance_window_elapsed_local() {
    {
        PhaseTimer timer(*this, Phase::device_wait);
        update_balance_window_events(true);
    }
    const std::vector<double> elapsed = kernel_busy_window_;
    update_device_timings(elapsed);
    return elapsed;
}

void Runtime::Impl::update_device_timings(const std::vector<double>& elapsed_local) {
    if (device_timings_.size() != elapsed_local.size()) {
        device_timings_.assign(elapsed_local.size(), DeviceTiming{});
    }
    for (std::size_t d = 0; d < elapsed_local.size(); ++d) {
        DeviceTiming& dt = device_timings_[d];
        dt.kernel_seconds_last = elapsed_local[d];
        dt.kernel_seconds_avg =
            (dt.kernel_seconds_avg * static_cast<double>(dt.samples) + elapsed_local[d]) /
            static_cast<double>(dt.samples + 1);
        ++dt.samples;
    }
}

// ---------------------------------------------------------------------------
// Synchronization
// ---------------------------------------------------------------------------

void Runtime::Impl::synchronize(bool force_finish) {
    synchronize_untimed(force_finish);
}

// Waiting for the devices and the barrier are charged to their own phases,
// inside whatever phase called this.
void Runtime::Impl::synchronize_untimed(bool force_finish) {
    {
        PhaseTimer timer(*this, Phase::device_wait);
        synchronize_all_local_devices(force_finish);
    }
    PhaseTimer timer(*this, Phase::mpi_sync);
    detail::check_mpi(MPI_Barrier(comm_), "MPI_Barrier(synchronize)");
}

// Waits for every command already submitted to the local queues.
// force_finish uses clFinish; otherwise a marker is enqueued and awaited.
void Runtime::Impl::synchronize_all_local_devices(bool force_finish) {
    std::vector<cl_command_queue> queues;
    for (detail::LocalDevice& ld : local_devices_) {
        for (cl_command_queue q : ld.kernel_queues) queues.push_back(q);
        queues.push_back(ld.transfer_queue);
    }

    if (force_finish) {
        for (cl_command_queue q : queues) {
            if (q != nullptr) detail::check_cl(clFinish(q), "clFinish");
        }
    } else {
        std::vector<cl_event> markers;
        for (cl_command_queue q : queues) {
            if (q == nullptr) continue;
            cl_event ev = nullptr;
#if CL_TARGET_OPENCL_VERSION >= 120
            detail::check_cl(clEnqueueMarkerWithWaitList(q, 0, nullptr, &ev), "enqueue marker");
#else
            detail::check_cl(clEnqueueMarker(q, &ev), "enqueue marker");
#endif
            markers.push_back(ev);
        }
        for (cl_event ev : markers) {
            detail::check_cl(clWaitForEvents(1, &ev), "clWaitForEvents(queue marker)");
            clReleaseEvent(ev);
        }
    }

    release_tracked_field_events();
    update_balance_window_events(false);  // everything is finished: collects all
}

} // namespace harmony
