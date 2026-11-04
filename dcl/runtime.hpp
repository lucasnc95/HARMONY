
#ifndef DCL_RUNTIME_HPP
#define DCL_RUNTIME_HPP

#include "types.hpp"
#include <memory>
#include <string>
#include <vector>

namespace dcl {

class Runtime;

class KernelBindingBuilder {
public:
    KernelBindingBuilder(Runtime& runtime, KernelHandle kernel);

    KernelBindingBuilder& arg(unsigned index, FieldHandle handle);
    KernelBindingBuilder& arg(unsigned index, ScalarArg scalar);

    KernelBinding build() const;

private:
    Runtime* runtime_{nullptr};
    KernelBinding binding_;
};

class StepBuilder {
public:
    StepBuilder(Runtime& runtime, std::string name);

    StepBuilder& invoke(const KernelBinding& binding, const LaunchGeometry& geometry);
    StepBuilder& with_halo_exchange(const HaloSpec& halo);
    StepBuilder& with_balance(const AutoBalancePolicy& policy);
    StepBuilder& tag_field(FieldHandle field, StepFieldRole role);
    StepBuilder& synchronize_at_end(bool value);

    ExecutionStep build() const;

private:
    Runtime* runtime_{nullptr};
    ExecutionStep step_;
};

class Runtime {
public:
    static Runtime create(int& argc, char**& argv);

    Runtime(Runtime&&) noexcept;
    Runtime& operator=(Runtime&&) noexcept;
    ~Runtime();

    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;

    void discover_devices(const DeviceSelection& selection);

    const std::vector<DeviceInfo>& devices() const noexcept;
    const std::vector<DevicePartition>& partitions() const noexcept;
    const std::vector<DeviceTiming>& device_timings() const noexcept;

    int rank() const noexcept;
    int size() const noexcept;
    MPI_Comm communicator() const noexcept;

    FieldHandle create_field(const FieldSpec& spec);
    KernelHandle create_kernel(const KernelSpec& spec);

    KernelBindingBuilder bind(KernelHandle kernel);
    StepBuilder step(std::string name);

    void set_partition(const PartitionSpec& spec);
    void execute(const ExecutionStep& step);
    void rebalance_to(const std::vector<float>& loads);
    void gather(FieldHandle field, void* host_dst, std::size_t bytes);
    void synchronize(bool force_finish = false);

    // Controls detailed per-iteration metric output. Timing collection and MPI
    // reductions still run in both modes. Enabled by default for compatibility.
    void set_verbose_metrics(bool enabled) noexcept;

    void set_topo_metrics(const TopoMetrics& metrics);
    const std::optional<TopoMetrics>& topo_metrics() const noexcept;
    void set_simulated_devices_count(int count) noexcept;
    void set_simulated_ranks_for_testing(int ranks) noexcept;

    void set_numa_cost_gain_ratio_threshold(double threshold) noexcept;
    double numa_cost_gain_ratio_threshold() const noexcept;

    void set_simulated_times(const std::vector<double>& times) noexcept;
    void clear_simulated_times() noexcept;

    bool maybe_rebalance_from_timings(
        const std::vector<FieldHandle>& rebalance_fields,
        float threshold,
        double numa_cost_gain_ratio_threshold = 0.50,
        bool use_contention_adjustment = false,
        bool use_power_cap = false,
        double power_budget_watts = 0.0
    );

    bool maybe_rebalance_from_timings(
        const std::vector<FieldHandle>& rebalance_fields,
        const AutoBalancePolicy& policy
    );

    bool maybe_rebalance_hierarchical(
        const std::vector<FieldHandle>& rebalance_fields = {}
    );

private:
    class Impl;
    explicit Runtime(std::unique_ptr<Impl> impl);

    std::unique_ptr<Impl> impl_;

    friend class KernelBindingBuilder;
    friend class StepBuilder;
};

} // namespace dcl

#endif
