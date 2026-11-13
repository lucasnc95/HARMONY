// Public data types of the HARMONY runtime.
//
// Everything a user needs to describe devices, fields (distributed buffers),
// kernels, launch geometry, halo exchange and load-balancing policies.

#ifndef HARMONY_TYPES_HPP
#define HARMONY_TYPES_HPP

#include <mpi.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace harmony {

// Every error raised by the runtime (OpenCL, MPI or misuse of the API).
class Error : public std::runtime_error {
public:
    explicit Error(const std::string& msg) : std::runtime_error(msg) {}
};

// ---------------------------------------------------------------------------
// Enumerations
// ---------------------------------------------------------------------------

enum class DeviceKind {
    all,
    cpu,
    gpu,
    accelerator
};

enum class BufferUsage {
    read_only,
    write_only,
    read_write
};

// How a field must be moved when the domain is repartitioned.
enum class RedistributionDependency {
    none,          // replicated / read-only data (e.g. parameters): never moved
    proportional,  // element i belongs to the device that owns element i
    total          // reserved: every device needs the whole field
};

enum class BalanceMode {
    off,
    static_threshold,   // a single rebalance attempt (after the warm-up)
    dynamic_threshold,  // a rebalance attempt every `interval` iterations
    static_profiled,    // like static_threshold, decided by the cost model
    dynamic_profiled    // like dynamic_threshold, decided by the cost model
};

// How a kernel uses a field argument. The runtime derives from it the
// dependencies between kernels, which halos to exchange and when.
enum class Access {
    read,
    write,       // the kernel overwrites every element of its partition
    read_write
};

// What happens right before the halos are exchanged.
enum class HaloSync {
    overlap,  // no wait: the exchange runs while the interior is computed
    local,    // wait until this rank's devices finish the interior
    global    // local + MPI barrier (every rank starts the exchange together)
};

// ---------------------------------------------------------------------------
// Handles and kernel arguments
// ---------------------------------------------------------------------------

struct FieldHandle {
    int value{-1};
};

struct KernelHandle {
    int value{-1};
};

// Raw bytes of a scalar kernel argument (int, float, struct, ...).
struct ScalarArg {
    std::vector<unsigned char> bytes;

    ScalarArg() = default;

    template <typename T, typename = std::enable_if_t<std::is_trivially_copyable_v<T>>>
    ScalarArg(const T& value) : bytes(sizeof(T)) {
        std::memcpy(bytes.data(), &value, sizeof(T));
    }
};

// A field passed to a kernel and how the kernel uses it. When the access is
// not given (KernelBindingBuilder::arg(index, field)) the runtime uses:
//   replicated fields (RedistributionDependency::none)  -> read
//   fields in one of the step's halo exchanges          -> read
//   any other field                                     -> read_write
struct FieldArg {
    FieldHandle field;
    std::optional<Access> access{};
};

using KernelArg = std::variant<FieldArg, ScalarArg>;

// ---------------------------------------------------------------------------
// Devices and partitions
// ---------------------------------------------------------------------------

struct DeviceSelection {
    DeviceKind kind{DeviceKind::all};
    int max_devices_per_rank{0};  // 0 = no limit
    int kernel_queues{4};         // independent kernels that may run at once on a device

    // Compute units of every CPU device left free for the MPI process (which
    // drives the other devices and MPI). The CPU device is then created as an
    // OpenCL sub-device with the remaining units. 0 = use the whole CPU.
    int cpu_host_cores{0};
};

struct DeviceInfo {
    int rank{0};
    int local_index{-1};
    int global_index{-1};
    std::string name;
    DeviceKind kind{DeviceKind::all};
    unsigned int compute_units{0};
    std::uint64_t global_memory{0};  // bytes (CL_DEVICE_GLOBAL_MEM_SIZE)
    std::uint64_t max_buffer{0};     // largest buffer (CL_DEVICE_MAX_MEM_ALLOC_SIZE); each field is one buffer
};

// Contiguous slice [global_offset, global_offset + element_count) of the
// domain assigned to one device. local_index is -1 on non-owning ranks.
struct DevicePartition {
    int device_global_index{-1};
    int owning_rank{-1};
    int local_index{-1};
    std::size_t global_offset{0};
    std::size_t element_count{0};
};

struct DeviceTiming {
    double kernel_seconds_last{0.0};
    double kernel_seconds_avg{0.0};
    std::uint64_t samples{0};
};

// ---------------------------------------------------------------------------
// Specifications
// ---------------------------------------------------------------------------

// A distributed buffer. Its size is
// global_elements * units_per_element * bytes_per_unit bytes.
struct FieldSpec {
    std::string name;
    std::size_t global_elements = 0;
    std::size_t units_per_element = 0;
    std::size_t bytes_per_unit = 0;
    BufferUsage usage = BufferUsage::read_write;
    const void* host_ptr = nullptr;  // optional initial data
    RedistributionDependency redistribution = RedistributionDependency::proportional;
};

struct KernelSpec {
    std::string source_file;
    std::string entry_point;
    std::string build_options;
};

// Domain decomposition. Partition boundaries are always multiples of
// `granularity` elements (e.g. one XY plane of a 3D mesh).
struct PartitionSpec {
    std::size_t global_elements{0};
    std::size_t units_per_element{0};
    std::size_t bytes_per_unit{0};
    std::size_t granularity{1};
};

// ---------------------------------------------------------------------------
// Execution steps
// ---------------------------------------------------------------------------

// Where a kernel runs. The domain is split among the devices, so each device
// launches the kernel over its own partition only.
struct LaunchGeometry {
    std::size_t global_offset{0};             // added to every work-item id
    std::size_t global_size{0};               // 0 or PartitionSpec::global_elements
    std::optional<std::size_t> local_size{};  // work-group size (default: driver)
};

// A kernel with its arguments (built with Runtime::bind()).
struct KernelBinding {
    KernelHandle kernel;
    std::vector<std::pair<unsigned, KernelArg>> args;
};

// One kernel launch inside a step.
struct KernelInvocation {
    KernelBinding binding;
    LaunchGeometry geometry;
};

// Fields whose borders (`width_elements` elements on each side of every
// partition) are copied between neighbouring devices before a kernel reads
// them. The exchange only happens when the field changed since the last one.
struct HaloSpec {
    std::size_t width_elements{0};
    std::vector<FieldHandle> fields;
};

struct AutoBalancePolicy {
    BalanceMode mode{BalanceMode::off};

    // dynamic modes: a rebalance attempt every `interval` iterations.
    // static modes:  a single attempt (see warmup).
    int interval{0};

    // Threshold modes: minimum L2 distance between the current and the
    // proposed cumulative loads to accept a rebalance (e.g. 0.05).
    float threshold{0.0f};

    // Total number of iterations of the run (used by the profiled modes).
    int total_iterations{0};

    // Linear segments of the migration-cost profile
    // (output of tools/migration_profiler).
    std::string profiling_file;

    // Iterations run with the initial division before the first attempt
    // (0 = `interval`). The first attempt happens at iteration `warmup`;
    // dynamic modes then try again every `interval` iterations. Every
    // attempt uses the kernel times of at most the last `interval`
    // iterations, so the start of the run is not measured when
    // warmup > interval.
    int warmup{0};
};

// Extra meaning of a field in a step (Runtime::step().tag_field()).
enum class StepFieldRole {
    // Moved to its new owner when a rebalance is applied. Without any tag,
    // every proportional field used by the step is moved; tagging only the
    // fields that hold state (e.g. the buffer just written in a ping-pong
    // scheme) avoids moving data that will be overwritten anyway.
    rebalance_source
};

struct StepFieldTag {
    FieldHandle field;
    StepFieldRole role{StepFieldRole::rebalance_source};
};

// One iteration of the application: kernels with sequential semantics.
// The result is always the same as running them one after another; the
// runtime overlaps the ones that do not depend on each other.
struct ExecutionStep {
    std::string name;
    std::vector<KernelInvocation> invocations;
    std::vector<HaloSpec> halos;  // one per with_halo_exchange()
    HaloSync halo_sync{HaloSync::overlap};
    AutoBalancePolicy balance{};
    bool synchronize_at_end{false};
    std::vector<StepFieldTag> field_tags;
};

// ---------------------------------------------------------------------------
// Timing report (Runtime::timing_report / print_timing_report)
//
// Every time covers the interval between start_timing() and stop_timing().
// ---------------------------------------------------------------------------

// What one device did. compute + transfer + idle = wall time (kernels and
// transfers that overlap are counted once, as compute).
struct DeviceReport {
    int global_index{-1};
    int rank{-1};
    std::string name;
    DeviceKind kind{DeviceKind::all};
    std::size_t elements{0};       // size of its partition at the end
    double compute_seconds{0.0};   // OpenCL kernels running
    double transfer_seconds{0.0};  // copies host <-> device (halos, migration)
    double idle_seconds{0.0};      // neither
};

// How the CPU of one MPI process spent the wall time. Every field is
// exclusive (no time is counted twice), so they add up to the wall time.
struct RankReport {
    int rank{-1};
    std::string host;
    double launch_seconds{0.0};         // execute(): preparing and enqueueing kernels
    double device_wait_seconds{0.0};    // waiting for kernels/transfers to finish
    double halo_seconds{0.0};           // halo exchange (without its waits)
    double mpi_halo_seconds{0.0};       // MPI waits: halo exchange
    double mpi_migration_seconds{0.0};  // MPI waits: data migration
    double mpi_balance_seconds{0.0};    // MPI waits: balancing decisions
    double mpi_sync_seconds{0.0};       // MPI waits: barriers
    double migration_seconds{0.0};      // moving data after a rebalance (without MPI)
    double balance_seconds{0.0};        // measuring and deciding (without MPI)
    double gather_seconds{0.0};         // Runtime::gather()
    double unmeasured_seconds{0.0};     // outside the runtime (application code, ...)

    double mpi_seconds() const {
        return mpi_halo_seconds + mpi_migration_seconds + mpi_balance_seconds + mpi_sync_seconds;
    }
};

// One load-balancing attempt (automatic or rebalance_to()). The vectors have
// one entry per device (global index order).
struct BalanceRecord {
    int number{0};                       // 1, 2, 3, ...
    std::size_t iteration{0};
    std::string policy;                  // "static threshold", "dynamic profiled", ..., "manual"
    bool applied{false};
    std::string reason;
    std::size_t window_iterations{0};    // iterations measured for the decision
    std::vector<double> window_seconds;  // kernel time of each device in those iterations
    double iteration_seconds_before{0.0};  // slowest device, per iteration (measured)
    double iteration_seconds_after{0.0};   // slowest device, per iteration (expected)
    std::vector<std::size_t> elements_before;
    std::vector<std::size_t> elements_after;
    std::size_t bytes_moved{0};
    double decision_seconds{0.0};
    double migration_seconds{0.0};
};

// The wall time of the critical path in one line: the MPI process that
// drives the slowest device (the one with most kernel time), split in
// exclusive parts that add up to the wall time. The time that process spent
// blocked in OpenCL (launch + waits) is split in the kernel time of the
// slowest device ("compute") and the rest ("launch and copies").
struct TimelineReport {
    int rank{-1};    // critical process
    int device{-1};  // slowest device (global index)
    double opencl_compute_seconds{0.0};
    double opencl_other_seconds{0.0};   // overhead: launching kernels, copies, driver latency
    double halo_seconds{0.0};           // preparing halo copies (without waits)
    double mpi_halo_seconds{0.0};
    double mpi_migration_seconds{0.0};
    double mpi_balance_seconds{0.0};
    double mpi_sync_seconds{0.0};
    double migration_seconds{0.0};
    double balance_seconds{0.0};
    double gather_seconds{0.0};
    double unmeasured_seconds{0.0};
};

struct TimingReport {
    double wall_seconds{0.0};   // between start_timing() and stop_timing()
    std::size_t iterations{0};  // execute() calls in the same interval
    std::vector<DeviceReport> devices;
    std::vector<RankReport> ranks;
    std::vector<BalanceRecord> balance;
    TimelineReport timeline;
};

} // namespace harmony

#endif // HARMONY_TYPES_HPP
