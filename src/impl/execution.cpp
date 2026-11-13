// execute(): runs the kernel invocations of a step.
//
// A step has sequential semantics: its invocations behave as if run one
// after another, in order. From the access of every field argument the
// runtime derives:
//   - dependencies (read-after-write, write-after-read, write-after-write),
//     enforced with OpenCL events, so independent kernels can run at the
//     same time on different kernel queues of a device;
//   - which halos are out of date before a kernel reads them: a field's halo
//     becomes stale when the field is written and is exchanged only when a
//     later kernel reads it (the field is in one of the step's HaloSpecs).
//
// A kernel that needs a halo exchange runs in three phases per partition:
//   interior (no neighbour data) -> halo exchange -> border.
// Otherwise it is launched once over each partition ("full").

#include "runtime_impl.hpp"

#include <algorithm>
#include <sstream>

namespace harmony {

namespace {

bool reads(Access a) { return a != Access::write; }
bool writes(Access a) { return a != Access::read; }

const char* access_name(Access a) {
    switch (a) {
        case Access::read: return "read";
        case Access::write: return "write";
        case Access::read_write: return "read_write";
    }
    return "?";
}

using ResolvedArgs = std::vector<detail::ResolvedFieldArg>;

// Two invocations conflict when they share a field and at least one writes it.
bool invocations_conflict(const ResolvedArgs& a, const ResolvedArgs& b) {
    for (const auto& fa : a) {
        for (const auto& fb : b) {
            if (fb.field.value != fa.field.value) continue;
            if (writes(fa.access) || writes(fb.access)) return true;
        }
    }
    return false;
}

void append_unique(std::vector<cl_event>& out, const std::vector<cl_event>& events) {
    for (cl_event ev : events) {
        if (ev != nullptr && std::find(out.begin(), out.end(), ev) == out.end()) out.push_back(ev);
    }
}

} // namespace

void Runtime::Impl::execute(const ExecutionStep& step) {
    if (step.invocations.empty()) return;
    PhaseTimer timer(*this, Phase::launch);

    std::vector<ResolvedArgs> fields;
    fields.reserve(step.invocations.size());
    for (const KernelInvocation& ki : step.invocations) {
        fields.push_back(resolve_field_args(ki, step));
        validate_invocation(ki, fields.back());
    }
    const std::vector<std::size_t> queues = assign_queues(fields);

    for (std::size_t i = 0; i < step.invocations.size(); ++i) {
        run_invocation(step.invocations[i], fields[i], queues[i], step.halo_sync);
    }

    ++iteration_counter_;
    if (step.balance.mode != BalanceMode::off) {
        maybe_balance_after_step(step);
    }

    if (step.synchronize_at_end) {
        PhaseTimer sync_timer(*this, Phase::device_wait);
        synchronize_all_local_devices(false);
    }
}

// Access and halo of every field argument (see FieldArg for the defaults).
std::vector<detail::ResolvedFieldArg> Runtime::Impl::resolve_field_args(const KernelInvocation& ki,
                                                                                const ExecutionStep& step) const {
    std::vector<ResolvedFieldArg> out;
    for (const auto& item : ki.binding.args) {
        const FieldArg* fa = std::get_if<FieldArg>(&item.second);
        if (fa == nullptr) continue;

        auto it = fields_.find(fa->field.value);
        if (it == fields_.end()) {
            throw Error("execute(" + step.name + "): unknown field in kernel argument " + std::to_string(item.first));
        }
        const bool replicated = it->second.spec.redistribution != RedistributionDependency::proportional;

        std::size_t halo = 0;
        for (const HaloSpec& hs : step.halos) {
            for (FieldHandle h : hs.fields) {
                if (h.value == fa->field.value) halo = std::max(halo, hs.width_elements);
            }
        }

        ResolvedFieldArg r;
        r.field = fa->field;
        r.access = fa->access ? *fa->access : ((replicated || halo > 0) ? Access::read : Access::read_write);
        if (r.access == Access::read_write && halo > 0) {
            throw Error("execute(" + step.name + "): field '" + it->second.spec.name +
                        "' is in a halo exchange and is read_write in kernel argument " +
                        std::to_string(item.first) +
                        ": a kernel cannot read the neighbours of a field it overwrites; use two buffers");
        }
        r.halo = (r.access == Access::read) ? halo : 0;
        out.push_back(r);
    }
    return out;
}

void Runtime::Impl::validate_invocation(const KernelInvocation& ki, const ResolvedArgs& fields) const {
    auto kit = kernels_.find(ki.binding.kernel.value);
    if (kit == kernels_.end()) throw Error("execute(): unknown kernel");
    const std::string& kernel_name = kit->second.spec.entry_point;

    if (ki.geometry.global_size != 0 && partition_.has_value() &&
        ki.geometry.global_size != partition_->global_elements) {
        throw Error("execute(): kernel '" + kernel_name + "': LaunchGeometry::global_size must be 0 or " +
                    std::to_string(partition_->global_elements) +
                    " (the elements of the partition; each device runs its own part)");
    }

    for (const ResolvedFieldArg& fa : fields) {
        const FieldSpec& spec = fields_.at(fa.field.value).spec;
        if (spec.redistribution != RedistributionDependency::proportional && writes(fa.access)) {
            throw Error("execute(): kernel '" + kernel_name + "': field '" + spec.name +
                        "' is replicated on every device and cannot be " + access_name(fa.access));
        }
        // Reading neighbours of a field that the same kernel writes would
        // depend on the order the devices run; use two buffers instead.
        if (fa.halo == 0) continue;
        for (const ResolvedFieldArg& fb : fields) {
            if (fb.field.value == fa.field.value && writes(fb.access)) {
                throw Error("execute(): kernel '" + kernel_name + "': field '" + spec.name +
                            "' is read with a halo and written by the same kernel; use two buffers");
            }
        }
    }
}

// Kernels that depend on an earlier kernel of the step share its queue (the
// chain runs in order); independent kernels start on the next free queue.
// Dependencies across queues are still enforced by events.
std::vector<std::size_t> Runtime::Impl::assign_queues(const std::vector<ResolvedArgs>& fields) const {
    std::vector<std::size_t> queue(fields.size(), 0);
    std::size_t next_free = 0;
    for (std::size_t i = 0; i < fields.size(); ++i) {
        std::optional<std::size_t> parent;
        for (std::size_t j = 0; j < i; ++j) {
            if (invocations_conflict(fields[i], fields[j])) parent = j;
        }
        queue[i] = parent ? queue[*parent] : next_free++;
    }
    return queue;
}

void Runtime::Impl::run_invocation(const KernelInvocation& ki, const ResolvedArgs& fields, std::size_t queue_slot,
                                   HaloSync sync) {
    detail::RegisteredKernel& rk = kernels_.at(ki.binding.kernel.value);

    // Halos this kernel reads that are out of date (one entry per field).
    std::vector<std::pair<FieldHandle, std::size_t>> stale;
    if (partitions_.size() > 1) {
        for (const ResolvedFieldArg& fa : fields) {
            if (fa.halo == 0) continue;
            if (fields_.at(fa.field.value).valid_halo >= fa.halo) continue;

            auto it = std::find_if(stale.begin(), stale.end(),
                                   [&](const auto& s) { return s.first.value == fa.field.value; });
            if (it == stale.end()) stale.emplace_back(fa.field, fa.halo);
            else it->second = std::max(it->second, fa.halo);
        }
    }

    if (stale.empty()) {
        run_full_phase(rk, ki, fields, queue_slot);
    } else {
        std::size_t halo = 0;
        for (const auto& s : stale) halo = std::max(halo, s.second);

        run_interior_phase(rk, ki, fields, queue_slot, halo);
        wait_before_halo_exchange(sync);
        exchange_halo_set(stale);
        run_border_phase(rk, ki, fields, queue_slot, halo);
    }

    // Written fields: their neighbours' copies are now out of date.
    for (const ResolvedFieldArg& fa : fields) {
        if (writes(fa.access)) fields_.at(fa.field.value).valid_halo = 0;
    }
}

void Runtime::Impl::wait_before_halo_exchange(HaloSync sync) {
    if (sync == HaloSync::overlap) return;
    {
        PhaseTimer timer(*this, Phase::device_wait);
        synchronize_all_local_devices(false);
    }
    if (sync == HaloSync::global) {
        PhaseTimer timer(*this, Phase::mpi_sync);
        detail::check_mpi(MPI_Barrier(comm_), "MPI_Barrier(halo sync)");
    }
}

// ---------------------------------------------------------------------------
// Phases: full, interior and border launches over this rank's partitions
// ---------------------------------------------------------------------------

void Runtime::Impl::run_full_phase(detail::RegisteredKernel& rk, const KernelInvocation& ki,
                                   const ResolvedArgs& fields, std::size_t queue_slot) {
    for (const DevicePartition& dp : partitions_) {
        if (dp.owning_rank != rank_ || dp.local_index < 0 || dp.element_count == 0) continue;
        const std::size_t d = static_cast<std::size_t>(dp.local_index);
        cl_kernel kernel = rk.kernels_per_local_device[d];
        bind_kernel_args(d, kernel, ki.binding);
        enqueue_range(d, kernel, ki, fields, queue_slot, LaunchPart::full, dp.global_offset, dp.element_count);
    }
}

// Elements that do not need neighbour data: [offset + halo, end - halo).
void Runtime::Impl::run_interior_phase(detail::RegisteredKernel& rk, const KernelInvocation& ki,
                                       const ResolvedArgs& fields, std::size_t queue_slot, std::size_t halo) {
    for (const DevicePartition& dp : partitions_) {
        if (dp.owning_rank != rank_ || dp.local_index < 0 || dp.element_count <= 2 * halo) continue;
        const std::size_t d = static_cast<std::size_t>(dp.local_index);
        cl_kernel kernel = rk.kernels_per_local_device[d];
        bind_kernel_args(d, kernel, ki.binding);
        enqueue_range(d, kernel, ki, fields, queue_slot, LaunchPart::interior, dp.global_offset + halo,
                      dp.element_count - 2 * halo);
    }
}

// The first and last `halo` elements (small partitions are all border).
void Runtime::Impl::run_border_phase(detail::RegisteredKernel& rk, const KernelInvocation& ki,
                                     const ResolvedArgs& fields, std::size_t queue_slot, std::size_t halo) {
    for (const DevicePartition& dp : partitions_) {
        if (dp.owning_rank != rank_ || dp.local_index < 0 || dp.element_count == 0) continue;
        const std::size_t d = static_cast<std::size_t>(dp.local_index);
        cl_kernel kernel = rk.kernels_per_local_device[d];
        bind_kernel_args(d, kernel, ki.binding);

        const std::size_t left = std::min(halo, dp.element_count);
        const std::size_t right = (dp.element_count > halo) ? std::min(halo, dp.element_count - left) : 0;
        enqueue_range(d, kernel, ki, fields, queue_slot, LaunchPart::border, dp.global_offset, left);
        enqueue_range(d, kernel, ki, fields, queue_slot, LaunchPart::border,
                      dp.global_offset + dp.element_count - right, right);
    }
}

void Runtime::Impl::enqueue_range(std::size_t d, cl_kernel kernel, const KernelInvocation& ki,
                                  const ResolvedArgs& fields, std::size_t queue_slot, LaunchPart part,
                                  std::size_t offset, std::size_t count) {
    if (count == 0) return;

    const std::vector<cl_event> deps = field_dependencies_for_binding(fields, d, part);

    const std::size_t gwo = offset + ki.geometry.global_offset;
    const std::size_t gws = count;
    std::size_t lws_value = ki.geometry.local_size.value_or(0);
    const std::size_t* lws = ki.geometry.local_size ? &lws_value : nullptr;

    const std::vector<cl_command_queue>& queues = local_devices_[d].kernel_queues;
    cl_command_queue queue = queues[queue_slot % queues.size()];

    cl_event ev = nullptr;
    detail::check_cl(clEnqueueNDRangeKernel(queue, kernel, 1, &gwo, &gws, lws, static_cast<cl_uint>(deps.size()),
                                            deps.empty() ? nullptr : deps.data(), &ev),
                     part == LaunchPart::full       ? "clEnqueueNDRangeKernel(full)"
                     : part == LaunchPart::interior ? "clEnqueueNDRangeKernel(interior)"
                                                    : "clEnqueueNDRangeKernel(border)");

    record_field_events(fields, d, part, ev);
    append_balance_window_kernel_events(d, ev);
    clReleaseEvent(ev);
}

// Events a launch must wait for. The interior never touches the ghost
// region, so it does not wait for a halo exchange in progress.
std::vector<cl_event> Runtime::Impl::field_dependencies_for_binding(const ResolvedArgs& fields, std::size_t d,
                                                                    LaunchPart part) const {
    std::vector<cl_event> deps;
    for (const ResolvedFieldArg& fa : fields) {
        const detail::RegisteredField& rf = fields_.at(fa.field.value);
        append_unique(deps, rf.core[d].writers);
        if (writes(fa.access)) append_unique(deps, rf.core[d].readers);
        if (fa.halo > 0 && part != LaunchPart::interior) append_unique(deps, rf.ghost[d].writers);
    }
    return deps;
}

// Reads are recorded before writes, so a field passed twice ends as written.
void Runtime::Impl::record_field_events(const ResolvedArgs& fields, std::size_t d, LaunchPart part, cl_event ev) {
    for (const ResolvedFieldArg& fa : fields) {
        if (!reads(fa.access)) continue;
        detail::RegisteredField& rf = fields_.at(fa.field.value);
        set_field_read_event(rf.core[d], ev);
        if (fa.halo > 0 && part != LaunchPart::interior) set_field_read_event(rf.ghost[d], ev);
    }
    for (const ResolvedFieldArg& fa : fields) {
        if (writes(fa.access)) set_field_write_event(fields_.at(fa.field.value).core[d], ev);
    }
}

void Runtime::Impl::bind_kernel_args(std::size_t d, cl_kernel kernel, const KernelBinding& binding) {
    for (const auto& [index, arg] : binding.args) {
        if (const ScalarArg* sa = std::get_if<ScalarArg>(&arg)) {
            detail::check_cl(clSetKernelArg(kernel, index, sa->bytes.size(), sa->bytes.data()),
                             "clSetKernelArg(scalar)");
        } else {
            cl_mem mem = fields_.at(std::get<FieldArg>(arg).field.value).replicas[d];
            detail::check_cl(clSetKernelArg(kernel, index, sizeof(cl_mem), &mem), "clSetKernelArg(field)");
        }
    }
}

// ---------------------------------------------------------------------------
// Load balancing trigger
// ---------------------------------------------------------------------------

std::vector<FieldHandle> Runtime::Impl::fields_with_role(const ExecutionStep& step, StepFieldRole role) const {
    std::vector<FieldHandle> out;
    for (const StepFieldTag& tag : step.field_tags) {
        if (tag.role == role) out.push_back(tag.field);
    }
    return out;
}

std::vector<FieldHandle> Runtime::Impl::fields_used_by(const ExecutionStep& step) const {
    std::vector<FieldHandle> out;
    for (const KernelInvocation& ki : step.invocations) {
        for (const auto& item : ki.binding.args) {
            if (const FieldArg* fa = std::get_if<FieldArg>(&item.second)) out.push_back(fa->field);
        }
    }
    return unique_existing_proportional_fields(out);
}

// Schedule (iterations counted from the creation of the runtime):
//   first attempt at `warmup` (or `interval` when warmup is 0);
//   dynamic modes: then every `interval` iterations; static: only that one.
// Every decision measures at most the last `interval` iterations, so the
// window is restarted `interval` iterations before the first attempt.
void Runtime::Impl::maybe_balance_after_step(const ExecutionStep& step) {
    const AutoBalancePolicy& policy = step.balance;
    const std::size_t interval = static_cast<std::size_t>(std::max(1, policy.interval));
    const std::size_t first = policy.warmup > 0 ? static_cast<std::size_t>(policy.warmup) : interval;
    const std::size_t it = iteration_counter_;

    if (!balance_policy_announced_) {
        balance_policy_announced_ = true;
        balance_policy_ = policy;
        print_balance_policy(policy);
    }

    const bool is_threshold = policy.mode == BalanceMode::dynamic_threshold ||
                              policy.mode == BalanceMode::static_threshold;
    const bool is_dynamic = policy.mode == BalanceMode::dynamic_threshold ||
                            policy.mode == BalanceMode::dynamic_profiled;

    // Drop the start of the warm-up from the first measurement.
    if (first > interval && it == first - interval && balance_window_start_iteration_ < it) {
        reset_balance_window_events();
    }

    bool should_try = false;
    if (it >= first && (it - first) % interval == 0) {
        if (is_dynamic) {
            should_try = true;
        } else if (!static_balance_attempted_) {
            should_try = true;
            static_balance_attempted_ = true;
        }
    }
    if (!should_try) return;

    std::vector<FieldHandle> requested = fields_with_role(step, StepFieldRole::rebalance_source);
    if (requested.empty()) requested = fields_used_by(step);
    const std::vector<FieldHandle> fields = unique_existing_proportional_fields(requested);

    PhaseTimer timer(*this, Phase::balance);
    const double t0 = MPI_Wtime();

    BalanceRecord record;
    record.iteration = it;
    record.policy = std::string(is_dynamic ? "dynamic " : "static ") + (is_threshold ? "threshold" : "profiled");
    for (const DevicePartition& p : partitions_) record.elements_before.push_back(p.element_count);

    if (fields.empty()) {
        reset_balance_window_events();
        record.reason = "no proportional field to move";
    } else {
        record.applied = is_threshold ? maybe_rebalance_from_timings(fields, policy.threshold, record)
                                      : maybe_rebalance_profiled(fields, policy, record);
    }

    for (const DevicePartition& p : partitions_) record.elements_after.push_back(p.element_count);
    record.decision_seconds = MPI_Wtime() - t0 - record.migration_seconds;
    print_balance_interval_metrics(std::move(record));
}

// Keeps only registered, proportional fields; sorted and without duplicates.
std::vector<FieldHandle> Runtime::Impl::unique_existing_proportional_fields(
    const std::vector<FieldHandle>& input) const {
    std::vector<int> ids;
    for (FieldHandle fh : input) {
        auto it = fields_.find(fh.value);
        if (it != fields_.end() && it->second.spec.redistribution == RedistributionDependency::proportional) {
            ids.push_back(fh.value);
        }
    }
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());

    std::vector<FieldHandle> out;
    for (int id : ids) out.push_back(FieldHandle{id});
    return out;
}

} // namespace harmony
