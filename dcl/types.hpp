
#ifndef DCL_TYPES_HPP
#define DCL_TYPES_HPP

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>
#include <type_traits>
#ifndef OMPI_SKIP_MPICXX
#define OMPI_SKIP_MPICXX 1
#endif
#ifndef MPICH_SKIP_MPICXX
#define MPICH_SKIP_MPICXX 1
#endif
#include <mpi.h>

namespace dcl {

class Error : public std::runtime_error {
public:
    explicit Error(const std::string& msg) : std::runtime_error(msg) {}
};

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


enum class RedistributionDependency {
    none,
    proportional,
    total
};

struct FieldHandle {
    int value{-1};
};

struct KernelHandle {
    int value{-1};
};

struct ScalarArg {
    std::vector<unsigned char> bytes;

    ScalarArg() = default;

    template <typename T, typename = std::enable_if_t<std::is_trivially_copyable_v<T>>>
    explicit ScalarArg(const T& value) : bytes(sizeof(T)) {
        std::memcpy(bytes.data(), &value, sizeof(T));
    }
};

using KernelArg = std::variant<FieldHandle, ScalarArg>;

struct DeviceSelection {
    DeviceKind kind{DeviceKind::all};
    int max_devices_per_rank{0};
};

struct DeviceInfo {
    int rank{0};
    int local_index{-1};
    int global_index{-1};
    std::string name;
    DeviceKind kind{DeviceKind::all};
    unsigned int compute_units{0};
    std::string platform_name;
    std::string platform_version;
    std::string device_version;
    std::string driver_version;
    std::string identity_kind{"unavailable"};
    std::string identity;
};

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

struct FieldSpec {
    std::string name;
    std::size_t global_elements = 0;
    std::size_t units_per_element = 0;
    std::size_t bytes_per_unit = 0;
    BufferUsage usage = BufferUsage::read_write;
    const void* host_ptr = nullptr;
    RedistributionDependency redistribution = RedistributionDependency::proportional;
};

struct KernelSpec {
    std::string source_file;
    std::string entry_point;
    std::string build_options;
};

struct PartitionSpec {
    std::size_t global_elements{0};
    std::size_t units_per_element{0};
    std::size_t bytes_per_unit{0};
    std::size_t granularity{1};
};

struct LaunchGeometry {
    std::size_t global_offset{0};
    std::size_t global_size{0};
    std::optional<std::size_t> local_size{};
};

struct KernelBinding {
    KernelHandle kernel;
    std::vector<std::pair<unsigned, KernelArg>> args;
};

struct KernelInvocation {
    KernelBinding binding;
    LaunchGeometry geometry;
};

struct HaloSpec {
    std::size_t width_elements{0};
    std::vector<FieldHandle> fields;
};

enum class BalanceMode {
    off,
    static_threshold,
    dynamic_threshold,
    static_profiled,
    dynamic_profiled,
    hierarchical
};

struct AutoBalancePolicy {
    BalanceMode mode{BalanceMode::off};

    // dynamic: rebalance a cada N iterações
    // static: warmup de N iterações, tenta rebalancear uma única vez
    int interval{0};

    // threshold em norma L2 de cargas (ex.: 0.05)
    float threshold{0.0f};

    // número total de iterações da execução, usado no modo profiled
    int total_iterations{0};

    // arquivo com os segmentos lineares do profiling de migração
    std::string profiling_file;

    // R2: Threshold ratio for NUMA migration cost vs expected gain (default 50% = 0.50)
    double numa_cost_gain_ratio_threshold{0.50};

    // R5 forward-compatible fields
    bool use_contention_adjustment{false};
    bool use_power_cap{false};
    double power_budget_watts{0.0};
};

// Topology Metrics Data Structure (R1.1)
struct TopoMetrics {
    // Per-device PCIe round-trip latency in nanoseconds (ns)
    std::vector<double> pcie_latency_ns;
    // Per-device PCIe peak bandwidth in gigabytes per second (GB/s)
    std::vector<double> pcie_bandwidth_gbps;
    // Rank-to-rank MPI latency in nanoseconds, indexed by [src_rank * rank_count + dst_rank].
    std::vector<double> mpi_latency_ns;
    // Rank-to-rank MPI bandwidth in gigabytes per second with the same indexing.
    std::vector<double> mpi_bandwidth_gbps;
    bool synthetic{false};

    // Forward-compatible fields for NUMA-aware load balancing (R2)
    // Flat N x N matrix of NUMA distances (where N is the number of NUMA nodes), unitless distance metrics
    std::vector<int> numa_distance;
    // Mapping from global device index to NUMA node index
    std::vector<int> device_numa_node;

    // Forward-compatible fields for Resource Contention and Power Consumption (R5)
    // Per-device memory contention factor in [1.0, inf), where 1.0 = no contention
    std::vector<double> memory_contention_factor;
    // Per-device Thermal Design Power in watts (W)
    std::vector<double> thermal_tdp_watts;
    // Per-device current measured power draw in watts (W)
    std::vector<double> current_power_watts;
};


enum class StepFieldRole {
    none,
    read_source,
    write_target,
    halo_source,
    rebalance_source
};

struct StepFieldTag {
    FieldHandle field;
    StepFieldRole role{StepFieldRole::none};
};

struct ExecutionStep {
    std::string name;
    std::vector<KernelInvocation> invocations;
    HaloSpec halo{};
    AutoBalancePolicy balance{};
    bool synchronize_at_end{true};
    std::vector<StepFieldTag> field_tags;
};

} // namespace dcl

#endif
