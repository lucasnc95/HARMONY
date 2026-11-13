// Public entry point of the HARMONY runtime.
//
// Typical flow:
//   auto runtime = harmony::Runtime::create(argc, argv);  // initializes MPI if needed
//   runtime.discover_devices({harmony::DeviceKind::all, 0});
//   runtime.set_partition({...});
//   auto u = runtime.create_field({...});
//   auto kernel = runtime.create_kernel({"kernel.cl", "entry", ""});
//
//   auto binding = runtime.bind(kernel)
//       .arg(0, u, harmony::Access::read)
//       .arg(1, v, harmony::Access::write)
//       .arg(2, n)
//       .build();
//
//   auto step = runtime.step("advance")
//       .invoke(binding)                                  // more kernels: more invoke()
//       .with_halo_exchange({1, {u}})
//       .with_balance({harmony::BalanceMode::dynamic_profiled, 100, 0.0f, iterations,
//                      "profiling.txt", /*warmup=*/500})
//       .build();
//
//   runtime.start_timing();
//   for (...) runtime.execute(step);
//   runtime.stop_timing();
//   runtime.print_timing_report();
//   runtime.gather(v, host_buffer, bytes);                // result on rank 0

#ifndef HARMONY_RUNTIME_HPP
#define HARMONY_RUNTIME_HPP

#include "types.hpp"

#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace harmony {

class Runtime;

// Builds a KernelBinding: a kernel and its arguments.
class KernelBindingBuilder {
public:
    explicit KernelBindingBuilder(KernelHandle kernel);

    // Field argument. The access tells the runtime which kernels depend on
    // each other and when a halo is out of date (see FieldArg for the
    // default when it is omitted).
    KernelBindingBuilder& arg(unsigned index, FieldHandle field);
    KernelBindingBuilder& arg(unsigned index, FieldHandle field, Access access);
    KernelBindingBuilder& arg(unsigned index, ScalarArg scalar);

    KernelBinding build() const;
    operator KernelBinding() const { return build(); }

private:
    KernelBinding binding_;
};

// Builds an ExecutionStep (one iteration of the application).
class StepBuilder {
public:
    explicit StepBuilder(std::string name);

    // Kernels run in the order they are invoked (independent ones overlap).
    StepBuilder& invoke(const KernelBinding& binding, const LaunchGeometry& geometry = {});
    // May be called more than once (fields with different halo widths).
    StepBuilder& with_halo_exchange(const HaloSpec& halo);
    // What happens right before the halos are exchanged (default: overlap).
    StepBuilder& with_halo_sync(HaloSync sync);
    StepBuilder& with_balance(const AutoBalancePolicy& policy);
    StepBuilder& tag_field(FieldHandle field, StepFieldRole role);
    StepBuilder& synchronize_at_end(bool value);

    ExecutionStep build() const;
    operator ExecutionStep() const { return build(); }

private:
    ExecutionStep step_;
};

class Runtime {
public:
    // Initializes MPI (if the application did not) and returns the runtime.
    // MPI is finalized when the runtime is destroyed, only if it was
    // initialized here and no exception is propagating (in that case the
    // application should call MPI_Abort).
    static Runtime create(int& argc, char**& argv);

    Runtime(Runtime&&) noexcept;
    Runtime& operator=(Runtime&&) noexcept;
    ~Runtime();

    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;

    // Devices ----------------------------------------------------------------
    void discover_devices(const DeviceSelection& selection);

    const std::vector<DeviceInfo>& devices() const noexcept;         // this rank
    const std::vector<DeviceInfo>& global_devices() const noexcept;  // every rank
    const std::vector<DevicePartition>& partitions() const noexcept;
    const std::vector<DeviceTiming>& device_timings() const noexcept;

    // Tables printed by rank 0 only (not collective).
    void print_devices(std::ostream& os = std::cout) const;
    void print_partitions(const std::string& title = "Partitions", std::ostream& os = std::cout) const;

    // MPI --------------------------------------------------------------------
    int rank() const noexcept;
    int size() const noexcept;
    MPI_Comm communicator() const noexcept;

    // Resources --------------------------------------------------------------
    FieldHandle create_field(const FieldSpec& spec);
    KernelHandle create_kernel(const KernelSpec& spec);

    KernelBindingBuilder bind(KernelHandle kernel) const;
    StepBuilder step(std::string name) const;

    // Execution --------------------------------------------------------------
    void set_partition(const PartitionSpec& spec);
    // Runs every kernel of the step on every device (asynchronous: returns
    // once the work is submitted, except for halo exchanges and balancing).
    void execute(const ExecutionStep& step);

    // Repartitions the domain using cumulative loads, e.g. {0.25, 0.6, 1.0}.
    void rebalance_to(const std::vector<float>& loads);

    // Collects the whole field on rank 0 (collective call). Only rank 0 needs
    // a destination buffer; the other ranks may pass nullptr and 0.
    void gather(FieldHandle field, void* host_dst, std::size_t bytes);

    void synchronize(bool force_finish = false);

    // Timing -----------------------------------------------------------------
    // start_timing() and stop_timing() are collective: they wait for every
    // device and rank, so the interval covers exactly the work in between.
    // start_timing() also clears all counters. stop_timing() returns the
    // elapsed wall time (the same value on every rank).
    void start_timing();
    double stop_timing();

    // Collective. Phase times, kernel time per device and balancing history
    // since start_timing() (or since the runtime was created).
    TimingReport timing_report();

    // Collective. Rank 0 prints timing_report() as tables.
    void print_timing_report(std::ostream& os = std::cout);

    // Collective. Rank 0 writes timing_report() as CSV (for scripts): one
    // "device", "process" and "balance" row per item and one "run" row.
    // Every row type has its header in a line starting with "#".
    void write_timing_report(const std::string& path);

    // Rank 0 prints every load-balancing attempt as it happens (default: on).
    void set_verbose(bool verbose);

private:
    class Impl;
    explicit Runtime(std::unique_ptr<Impl> impl);

    std::unique_ptr<Impl> impl_;
};

} // namespace harmony

#endif // HARMONY_RUNTIME_HPP
