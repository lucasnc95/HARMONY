// Public API facade: forwards every call to Runtime::Impl, plus the builders.

#include "runtime_impl.hpp"

#include <utility>

namespace harmony {

// ---------------------------------------------------------------------------
// Runtime
// ---------------------------------------------------------------------------

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

const std::vector<DeviceInfo>& Runtime::devices() const noexcept { return impl_->devices(); }
const std::vector<DeviceInfo>& Runtime::global_devices() const noexcept { return impl_->global_devices(); }
const std::vector<DevicePartition>& Runtime::partitions() const noexcept { return impl_->partitions(); }
const std::vector<DeviceTiming>& Runtime::device_timings() const noexcept { return impl_->device_timings(); }

int Runtime::rank() const noexcept { return impl_->rank(); }
int Runtime::size() const noexcept { return impl_->size(); }
MPI_Comm Runtime::communicator() const noexcept { return impl_->communicator(); }

FieldHandle Runtime::create_field(const FieldSpec& spec) { return impl_->create_field(spec); }
KernelHandle Runtime::create_kernel(const KernelSpec& spec) { return impl_->create_kernel(spec); }

KernelBindingBuilder Runtime::bind(KernelHandle kernel) const { return KernelBindingBuilder(kernel); }
StepBuilder Runtime::step(std::string name) const { return StepBuilder(std::move(name)); }

void Runtime::set_partition(const PartitionSpec& spec) { impl_->set_partition(spec); }
void Runtime::execute(const ExecutionStep& step) { impl_->execute(step); }
void Runtime::rebalance_to(const std::vector<float>& loads) { impl_->rebalance_to(loads); }

void Runtime::gather(FieldHandle field, void* host_dst, std::size_t bytes) {
    impl_->gather(field, host_dst, bytes);
}

void Runtime::synchronize(bool force_finish) { impl_->synchronize(force_finish); }

void Runtime::print_devices(std::ostream& os) const { impl_->print_devices(os); }
void Runtime::print_partitions(const std::string& title, std::ostream& os) const {
    impl_->print_partitions(title, os);
}

void Runtime::start_timing() { impl_->start_timing(); }
double Runtime::stop_timing() { return impl_->stop_timing(); }
TimingReport Runtime::timing_report() { return impl_->timing_report(); }
void Runtime::print_timing_report(std::ostream& os) { impl_->print_timing_report(os); }
void Runtime::write_timing_report(const std::string& path) { impl_->write_timing_report(path); }
void Runtime::set_verbose(bool verbose) { impl_->set_verbose(verbose); }

// ---------------------------------------------------------------------------
// KernelBindingBuilder
// ---------------------------------------------------------------------------

KernelBindingBuilder::KernelBindingBuilder(KernelHandle kernel) { binding_.kernel = kernel; }

KernelBindingBuilder& KernelBindingBuilder::arg(unsigned index, FieldHandle field) {
    binding_.args.emplace_back(index, KernelArg(FieldArg{field, std::nullopt}));
    return *this;
}

KernelBindingBuilder& KernelBindingBuilder::arg(unsigned index, FieldHandle field, Access access) {
    binding_.args.emplace_back(index, KernelArg(FieldArg{field, access}));
    return *this;
}

KernelBindingBuilder& KernelBindingBuilder::arg(unsigned index, ScalarArg scalar) {
    binding_.args.emplace_back(index, KernelArg(std::move(scalar)));
    return *this;
}

KernelBinding KernelBindingBuilder::build() const { return binding_; }

// ---------------------------------------------------------------------------
// StepBuilder
// ---------------------------------------------------------------------------

StepBuilder::StepBuilder(std::string name) { step_.name = std::move(name); }

StepBuilder& StepBuilder::invoke(const KernelBinding& binding, const LaunchGeometry& geometry) {
    step_.invocations.push_back(KernelInvocation{binding, geometry});
    return *this;
}

StepBuilder& StepBuilder::with_halo_exchange(const HaloSpec& halo) {
    step_.halos.push_back(halo);
    return *this;
}

StepBuilder& StepBuilder::with_halo_sync(HaloSync sync) {
    step_.halo_sync = sync;
    return *this;
}

StepBuilder& StepBuilder::with_balance(const AutoBalancePolicy& policy) {
    step_.balance = policy;
    return *this;
}

StepBuilder& StepBuilder::tag_field(FieldHandle field, StepFieldRole role) {
    step_.field_tags.push_back(StepFieldTag{field, role});
    return *this;
}

StepBuilder& StepBuilder::synchronize_at_end(bool value) {
    step_.synchronize_at_end = value;
    return *this;
}

ExecutionStep StepBuilder::build() const { return step_; }

} // namespace harmony
