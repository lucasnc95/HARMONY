
#include "runtime_impl.hpp"
#include <iostream>

namespace dcl {

Runtime Runtime::create(int& argc, char**& argv) {
    std::unique_ptr<Impl> impl(new Impl(argc, argv));
    impl->initialize_mpi_from_runtime();
    return Runtime(std::move(impl));
}

Runtime::Runtime(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Runtime::Runtime(Runtime&&) noexcept = default;
Runtime& Runtime::operator=(Runtime&&) noexcept = default;
Runtime::~Runtime() = default;

void Runtime::discover_devices(const DeviceSelection& selection) {
    impl_->discover_devices(selection);
}

const std::vector<DeviceInfo>& Runtime::devices() const noexcept {
    return impl_->devices();
}

const std::vector<DevicePartition>& Runtime::partitions() const noexcept {
    return impl_->partitions();
}

const std::vector<DeviceTiming>& Runtime::device_timings() const noexcept {
    return impl_->device_timings();
}

int Runtime::rank() const noexcept { return impl_->rank(); }
int Runtime::size() const noexcept { return impl_->size(); }
MPI_Comm Runtime::communicator() const noexcept { return impl_->communicator(); }

FieldHandle Runtime::create_field(const FieldSpec& spec) {
    return impl_->create_field(spec);
}

KernelHandle Runtime::create_kernel(const KernelSpec& spec) {
    return impl_->create_kernel(spec);
}

KernelBindingBuilder Runtime::bind(KernelHandle kernel) {
    return KernelBindingBuilder(*this, kernel);
}

StepBuilder Runtime::step(std::string name) {
    return StepBuilder(*this, std::move(name));
}

void Runtime::set_partition(const PartitionSpec& spec) {
    impl_->set_partition(spec);
}

void Runtime::execute(const ExecutionStep& step) {
    impl_->execute(step);
}

void Runtime::rebalance_to(const std::vector<float>& loads) {
    impl_->rebalance_to(loads);
}

void Runtime::gather(FieldHandle field, void* host_dst, std::size_t bytes) {
    impl_->gather(field, host_dst, bytes);
}

void Runtime::synchronize(bool force_finish) {
    impl_->synchronize(force_finish);
}

void Runtime::set_verbose_metrics(bool enabled) noexcept {
    impl_->set_verbose_metrics(enabled);
}

void Runtime::set_topo_metrics(const TopoMetrics& metrics) {
    impl_->set_topo_metrics(metrics);
}

const std::optional<TopoMetrics>& Runtime::topo_metrics() const noexcept {
    return impl_->topo_metrics();
}

void Runtime::set_simulated_devices_count(int count) noexcept {
    impl_->set_simulated_devices_count(count);
}

void Runtime::set_simulated_ranks_for_testing(int ranks) noexcept {
    impl_->set_simulated_ranks_for_testing(ranks);
}

void Runtime::set_numa_cost_gain_ratio_threshold(double threshold) noexcept {
    impl_->set_numa_cost_gain_ratio_threshold(threshold);
}

double Runtime::numa_cost_gain_ratio_threshold() const noexcept {
    return impl_->numa_cost_gain_ratio_threshold();
}

void Runtime::set_simulated_times(const std::vector<double>& times) noexcept {
    impl_->set_simulated_times(times);
}

void Runtime::clear_simulated_times() noexcept {
    impl_->clear_simulated_times();
}

bool Runtime::maybe_rebalance_from_timings(
    const std::vector<FieldHandle>& rebalance_fields,
    float threshold,
    double numa_cost_gain_ratio_threshold,
    bool use_contention_adjustment,
    bool use_power_cap,
    double power_budget_watts
) {
    return impl_->maybe_rebalance_from_timings(
        rebalance_fields,
        threshold,
        numa_cost_gain_ratio_threshold,
        use_contention_adjustment,
        use_power_cap,
        power_budget_watts
    );
}

bool Runtime::maybe_rebalance_from_timings(
    const std::vector<FieldHandle>& rebalance_fields,
    const AutoBalancePolicy& policy
) {
    return impl_->maybe_rebalance_from_timings(
        rebalance_fields,
        policy
    );
}

bool Runtime::maybe_rebalance_hierarchical(
    const std::vector<FieldHandle>& rebalance_fields
) {
    return impl_->maybe_rebalance_hierarchical(rebalance_fields);
}

KernelBindingBuilder::KernelBindingBuilder(Runtime& runtime, KernelHandle kernel)
    : runtime_(&runtime) {
    binding_.kernel = kernel;
}

KernelBindingBuilder& KernelBindingBuilder::arg(unsigned index, FieldHandle handle) {
    binding_.args.push_back(std::make_pair(index, KernelArg(handle)));
    return *this;
}

KernelBindingBuilder& KernelBindingBuilder::arg(unsigned index, ScalarArg scalar) {
    binding_.args.push_back(std::make_pair(index, KernelArg(std::move(scalar))));
    return *this;
}

KernelBinding KernelBindingBuilder::build() const {
    return binding_;
}

StepBuilder::StepBuilder(Runtime& runtime, std::string name)
    : runtime_(&runtime) {
    step_.name = std::move(name);
}

StepBuilder& StepBuilder::invoke(const KernelBinding& binding, const LaunchGeometry& geometry) {
    KernelInvocation ki;
    ki.binding = binding;
    ki.geometry = geometry;
    step_.invocations.push_back(ki);
    return *this;
}

StepBuilder& StepBuilder::with_halo_exchange(const HaloSpec& halo) {
    step_.halo = halo;
    return *this;
}

StepBuilder& StepBuilder::with_balance(const AutoBalancePolicy& policy) {
    step_.balance = policy;
    return *this;
}

StepBuilder& StepBuilder::tag_field(FieldHandle field, StepFieldRole role) {
    step_.field_tags.push_back(dcl::StepFieldTag{field, role});
    return *this;
}

StepBuilder& StepBuilder::synchronize_at_end(bool value) {
    step_.synchronize_at_end = value;
    return *this;
}

ExecutionStep StepBuilder::build() const {
    return step_;
}

} // namespace dcl

#ifdef __CPPCHECK__
int main(int argc, char** argv) {
    dcl::Runtime rt = dcl::Runtime::create(argc, argv);
    dcl::KernelHandle kh;
    auto kb = rt.bind(kh);
    dcl::FieldHandle fh;
    kb.arg(0, fh);
    (void)kb.build();
    auto sb = rt.step("test");
    dcl::LaunchGeometry geom;
    sb.invoke(dcl::KernelBinding{}, geom);
    sb.with_halo_exchange(dcl::HaloSpec{});
    sb.with_balance(dcl::AutoBalancePolicy{});
    sb.tag_field(fh, dcl::StepFieldRole::none);
    sb.synchronize_at_end(true);
    (void)sb.build();
    rt.set_numa_cost_gain_ratio_threshold(0.5);
    (void)rt.numa_cost_gain_ratio_threshold();
    rt.set_simulated_times({});
    rt.clear_simulated_times();
    (void)rt.maybe_rebalance_from_timings({fh}, 0.05f, 0.5);
    (void)rt.maybe_rebalance_hierarchical({fh});
    (void)&dcl::load_topo_metrics;
    (void)&dcl::save_topo_metrics;
    (void)&dcl::adjusted_capacity;
    (void)&dcl::apply_power_cap;
    dcl::Runtime::Impl::cppcheck_anchor_unused();
    return 0;
}
#endif
