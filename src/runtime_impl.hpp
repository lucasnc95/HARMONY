// Private implementation of harmony::Runtime.
//
// The class is declared here and implemented across src/impl/*.cpp, one file
// per responsibility:
//
//   devices.cpp         MPI start-up/shutdown, OpenCL discovery, teardown
//   resources.cpp       fields (buffers) and kernels
//   partitioning.cpp    domain decomposition and load <-> partition mapping
//   execution.cpp       execute(): kernel calls, queues, halo needs, balancing trigger
//   events.cpp          dependency tracking, kernel statistics, synchronization
//   communication.cpp   chunked MPI transfers and gather()
//   halo.cpp            halo exchange between neighbouring partitions
//   redistribution.cpp  data migration after a repartition, rebalance_to()
//   balancing.cpp       automatic load-balancing policies
//   timing.cpp          phase timers and the timing report

#ifndef HARMONY_RUNTIME_IMPL_HPP
#define HARMONY_RUNTIME_IMPL_HPP

#include "harmony/runtime.hpp"
#include "detail/checks.hpp"
#include "detail/opencl_resources.hpp"

#include <CL/cl.h>
#include <mpi.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace harmony {

// What the CPU of this process is doing. Times are exclusive: a phase that
// starts inside another pauses it (see PhaseTimer), so the phases add up to
// the time spent inside the runtime; the rest is "unmeasured".
enum class Phase : int {
    launch,         // execute(): preparing and enqueueing kernels
    device_wait,    // waiting for kernels/transfers to finish
    halo,           // halo exchange (planning, enqueueing copies)
    mpi_halo,       // MPI waits of the halo exchange
    mpi_migration,  // MPI transfers of a data migration
    mpi_balance,    // MPI collectives of a balancing decision
    mpi_sync,       // MPI barriers
    migration,      // moving data after a rebalance
    balance,        // measuring and deciding a rebalance
    gather,         // Runtime::gather()
    count
};

namespace detail {

// A field argument with its access and halo resolved for one step.
struct ResolvedFieldArg {
    FieldHandle field;
    Access access{Access::read};
    std::size_t halo{0};  // only for reads of fields in a halo exchange
};

} // namespace detail

class Runtime::Impl {
public:
    Impl(int& argc, char**& argv);
    ~Impl();

    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;

    // --- devices.cpp --------------------------------------------------------
    void initialize_mpi_from_runtime();
    void discover_devices(const DeviceSelection& selection);
    void print_devices(std::ostream& os) const;

    const std::vector<DeviceInfo>& devices() const noexcept { return devices_; }
    const std::vector<DeviceInfo>& global_devices() const noexcept { return global_devices_; }
    const std::vector<DevicePartition>& partitions() const noexcept { return partitions_; }
    const std::vector<DeviceTiming>& device_timings() const noexcept { return device_timings_; }
    int rank() const noexcept { return rank_; }
    int size() const noexcept { return size_; }
    MPI_Comm communicator() const noexcept { return comm_; }

    // --- resources.cpp ------------------------------------------------------
    FieldHandle create_field(const FieldSpec& spec);
    KernelHandle create_kernel(const KernelSpec& spec);

    // --- partitioning.cpp ---------------------------------------------------
    void set_partition(const PartitionSpec& spec);
    void print_partitions(const std::string& title, std::ostream& os) const;

    // --- execution.cpp ------------------------------------------------------
    void execute(const ExecutionStep& step);

    // --- redistribution.cpp -------------------------------------------------
    void rebalance_to(const std::vector<float>& loads);

    // --- communication.cpp --------------------------------------------------
    void gather(FieldHandle field, void* host_dst, std::size_t bytes);

    // --- events.cpp ---------------------------------------------------------
    void synchronize(bool force_finish);

    // --- timing.cpp ---------------------------------------------------------
    void start_timing();
    double stop_timing();
    TimingReport timing_report();
    void print_timing_report(std::ostream& os);
    void write_timing_report(const std::string& path);
    void set_verbose(bool verbose) { verbose_ = verbose; }

private:
    // While alive, the CPU time is charged to `phase` (the enclosing phase is
    // paused and resumed at destruction).
    class PhaseTimer {
    public:
        PhaseTimer(Impl& impl, Phase phase);
        ~PhaseTimer();
        PhaseTimer(const PhaseTimer&) = delete;
        PhaseTimer& operator=(const PhaseTimer&) = delete;

    private:
        Impl& impl_;
    };
    void enter_phase(Phase phase);
    void leave_phase();

    // --- devices.cpp --------------------------------------------------------
    void release_opencl_objects();
    void clear_runtime_state();
    void gather_global_device_info();

    // --- resources.cpp ------------------------------------------------------
    void write_initial_field_data(FieldHandle h, const void* host_ptr);
    detail::RegisteredField& field_or_throw(FieldHandle h, const char* context);

    // --- partitioning.cpp ---------------------------------------------------
    int owner_rank_of_global_device(int g) const;
    int local_index_of_global_device(int g) const;
    void rebuild_partitions_equal_like_wrapper();
    std::vector<DevicePartition> partitions_from_loads(const std::vector<float>& loads) const;
    std::vector<float> loads_from_partitions(const std::vector<DevicePartition>& parts) const;

    // --- execution.cpp ------------------------------------------------------
    // Which part of each partition a launch covers.
    enum class LaunchPart { full, interior, border };

    using ResolvedFieldArg = detail::ResolvedFieldArg;

    std::vector<ResolvedFieldArg> resolve_field_args(const KernelInvocation& ki, const ExecutionStep& step) const;
    void validate_invocation(const KernelInvocation& ki, const std::vector<ResolvedFieldArg>& fields) const;
    std::vector<std::size_t> assign_queues(const std::vector<std::vector<ResolvedFieldArg>>& fields) const;
    void run_invocation(const KernelInvocation& ki, const std::vector<ResolvedFieldArg>& fields,
                        std::size_t queue_slot, HaloSync sync);
    void run_full_phase(detail::RegisteredKernel& rk, const KernelInvocation& ki,
                        const std::vector<ResolvedFieldArg>& fields, std::size_t queue_slot);
    void run_interior_phase(detail::RegisteredKernel& rk, const KernelInvocation& ki,
                            const std::vector<ResolvedFieldArg>& fields, std::size_t queue_slot, std::size_t halo);
    void run_border_phase(detail::RegisteredKernel& rk, const KernelInvocation& ki,
                          const std::vector<ResolvedFieldArg>& fields, std::size_t queue_slot, std::size_t halo);
    void enqueue_range(std::size_t local_device, cl_kernel kernel, const KernelInvocation& ki,
                       const std::vector<ResolvedFieldArg>& fields, std::size_t queue_slot, LaunchPart part,
                       std::size_t offset, std::size_t count);
    std::vector<cl_event> field_dependencies_for_binding(const std::vector<ResolvedFieldArg>& fields,
                                                         std::size_t local_device, LaunchPart part) const;
    void record_field_events(const std::vector<ResolvedFieldArg>& fields, std::size_t local_device,
                             LaunchPart part, cl_event ev);
    void bind_kernel_args(std::size_t local_device, cl_kernel kernel, const KernelBinding& binding);
    void wait_before_halo_exchange(HaloSync sync);
    std::vector<FieldHandle> fields_with_role(const ExecutionStep& step, StepFieldRole role) const;
    std::vector<FieldHandle> fields_used_by(const ExecutionStep& step) const;
    void maybe_balance_after_step(const ExecutionStep& step);
    std::vector<FieldHandle> unique_existing_proportional_fields(const std::vector<FieldHandle>& input) const;

    // --- events.cpp ---------------------------------------------------------
    void set_field_read_event(detail::AccessTracker& tracker, cl_event ev);
    void set_field_write_event(detail::AccessTracker& tracker, cl_event ev);
    void release_tracked_field_events();

    // Kernel statistics: durations (sum) and busy time (union of the kernel
    // intervals of a device) go to the totals and to the balance window.
    void append_balance_window_kernel_events(std::size_t local_device, cl_event ev);
    void track_transfer_event(std::size_t local_device, cl_event ev);
    void update_balance_window_events(bool wait);
    void reset_balance_window_events();
    std::vector<double> compute_balance_window_elapsed_local();
    void update_device_timings(const std::vector<double>& elapsed_local);

    void synchronize_all_local_devices(bool force_finish);
    void synchronize_untimed(bool force_finish);  // local devices + MPI barrier

    // --- communication.cpp --------------------------------------------------
    int mpi_tag_upper_bound() const;
    void check_mpi_tag_base(int tag_base, std::size_t total_bytes, const char* context) const;
    void mpi_transfer_bytes_chunked(const unsigned char* sendbuf, unsigned char* recvbuf,
                                    std::size_t total_bytes, int peer_rank, int tag_base,
                                    bool do_send, bool do_recv);

    // --- halo.cpp -----------------------------------------------------------
    // Exchanges the halo of every (field, width) pair in a single batch.
    void exchange_halo_set(const std::vector<std::pair<FieldHandle, std::size_t>>& fields);

    // --- redistribution.cpp -------------------------------------------------
    void redistribute_field_intersection(FieldHandle fh,
                            const std::vector<DevicePartition>& old_parts,
                            const std::vector<DevicePartition>& new_parts);
    void redistribute_selected_registered_fields(const std::vector<FieldHandle>& fields,
                             const std::vector<DevicePartition>& old_parts,
                             const std::vector<DevicePartition>& new_parts);
    void apply_partitions(const std::vector<FieldHandle>& fields,
                          const std::vector<DevicePartition>& old_parts,
                          const std::vector<DevicePartition>& new_parts,
                          BalanceRecord& record);
    std::size_t bytes_moved(const std::vector<FieldHandle>& fields,
                            const std::vector<DevicePartition>& old_parts,
                            const std::vector<DevicePartition>& new_parts) const;
    std::vector<FieldHandle> all_field_handles() const;

    // --- balancing.cpp ------------------------------------------------------
    bool maybe_rebalance_from_timings(const std::vector<FieldHandle>& requested_fields, float threshold,
                                   BalanceRecord& record);
    bool maybe_rebalance_profiled(const std::vector<FieldHandle>& requested_fields,
                                  const AutoBalancePolicy& policy, BalanceRecord& record);
    std::vector<double> collect_global_balance_times_from_window(const char* context);
    bool propose_division(std::vector<DevicePartition>& new_parts, BalanceRecord& record, const char* context);
    std::size_t max_bytes_received_by_a_rank(const std::vector<FieldHandle>& fields,
                                             const std::vector<DevicePartition>& old_parts,
                                             const std::vector<DevicePartition>& new_parts) const;
    void load_profiling_data(const std::string& profiling_file);
    double get_migration_overhead(std::size_t volume) const;

    // --- timing.cpp ---------------------------------------------------------
    void print_balance_policy(const AutoBalancePolicy& policy);
    void print_balance_interval_metrics(BalanceRecord record);

    // --- state --------------------------------------------------------------
    bool mpi_initialized_by_runtime_{false};
    int& argc_;
    char**& argv_;
    int rank_{0};
    int size_{1};
    MPI_Comm comm_{MPI_COMM_WORLD};
    MPI_Comm halo_comm_{MPI_COMM_NULL};  // private communicator for halo messages

    std::vector<detail::PlatformContext> platforms_;
    std::vector<detail::LocalDevice> local_devices_;
    std::vector<DeviceInfo> devices_;
    std::vector<DeviceInfo> global_devices_;
    std::vector<int> all_device_counts_;  // devices per rank
    std::vector<DevicePartition> partitions_;
    std::optional<PartitionSpec> partition_;

    std::unordered_map<int, detail::RegisteredField> fields_;
    std::unordered_map<int, detail::RegisteredKernel> kernels_;
    int next_field_id_{0};
    int next_kernel_id_{0};

    std::size_t iteration_counter_{0};

    // Host staging buffers reused by every halo exchange.
    std::vector<std::vector<unsigned char>> halo_buffers_;

    // Device statistics, per local device. "Busy" times are unions of the
    // profiled intervals (overlapping commands count once).
    std::vector<std::deque<cl_event>> pending_kernel_events_;
    std::vector<std::deque<cl_event>> pending_transfer_events_;
    std::vector<double> kernel_busy_total_;
    std::vector<double> kernel_busy_window_;   // since the last balancing attempt
    std::vector<double> active_busy_total_;    // kernels or transfers
    std::vector<double> kernel_busy_until_;    // end of the last counted interval
    std::vector<double> active_busy_until_;

    // Load balancing.
    std::vector<float> current_loads_;  // cumulative
    std::vector<DeviceTiming> device_timings_;
    bool static_balance_attempted_{false};  // static modes: one attempt per run
    bool balance_policy_announced_{false};
    std::optional<AutoBalancePolicy> balance_policy_;  // last policy seen (for the report)
    std::size_t balance_window_start_iteration_{0};    // iteration the balance window started
    std::vector<detail::ProfileSegment> profiling_data_;
    bool profiling_loaded_{false};
    std::string profiling_file_loaded_;

    // Timing.
    std::array<double, static_cast<int>(Phase::count)> phase_seconds_{};
    std::vector<Phase> phase_stack_;
    double phase_mark_{0.0};
    double timing_start_{0.0};
    std::optional<double> timing_wall_;  // set by stop_timing()

    // Counters frozen by stop_timing(), so later calls (e.g. gather) do not
    // leak into the report.
    struct TimingSnapshot {
        std::array<double, static_cast<int>(Phase::count)> phase_seconds{};
        std::vector<double> kernel_busy;
        std::vector<double> active_busy;
        std::size_t iterations{0};
        std::size_t balance_records{0};
    };
    std::optional<TimingSnapshot> timing_snapshot_;
    std::size_t timing_start_iteration_{0};
    std::vector<BalanceRecord> balance_log_;
    bool verbose_{true};
    int balance_attempts_{0};
    bool devices_discovered_{false};
};

} // namespace harmony

#endif // HARMONY_RUNTIME_IMPL_HPP
