// stencil2d with the HARMONY runtime: partitioning, halos, load balancing and
// gather are handled by the library (compare with run_mpiocl.cpp).

#include "stencil2d.hpp"

#include <harmony/runtime.hpp>

#include <optional>
#include <stdexcept>

namespace stencil2d {

namespace {

harmony::DeviceKind to_harmony(Device d) {
    switch (d) {
        case Device::cpu: return harmony::DeviceKind::cpu;
        case Device::gpu: return harmony::DeviceKind::gpu;
        case Device::all: return harmony::DeviceKind::all;
    }
    return harmony::DeviceKind::all;
}

harmony::BalanceMode balance_mode(const Options& o) {
    if (o.balance_mode == "off") return harmony::BalanceMode::off;
    const bool threshold = o.balance_strategy == "threshold";
    if (!threshold && o.balance_strategy != "profiled") {
        throw std::invalid_argument("--balance-strategy must be threshold or profiled");
    }
    if (o.balance_mode == "static") {
        return threshold ? harmony::BalanceMode::static_threshold : harmony::BalanceMode::static_profiled;
    }
    if (o.balance_mode == "dynamic") {
        return threshold ? harmony::BalanceMode::dynamic_threshold : harmony::BalanceMode::dynamic_profiled;
    }
    throw std::invalid_argument("--balance-mode must be off, static or dynamic");
}

harmony::HaloSync halo_sync(const Options& o) {
    if (o.halo_sync == "local") return harmony::HaloSync::local;
    if (o.halo_sync == "global") return harmony::HaloSync::global;
    return harmony::HaloSync::overlap;
}

template <typename T>
RunResult run_typed(const Options& o, harmony::Runtime& runtime) {
    const std::size_t total = o.elements();
    std::vector<T> init = initial_data<T>(o);
    std::vector<T> zeros(o.is_stencil() ? 0 : total, T(0));  // only the map needs it

    // Stencils are split in whole rows (granularity nx) and need one row of halo.
    const std::size_t row = o.is_stencil() ? static_cast<std::size_t>(o.nx) : 1;
    runtime.set_partition({total, 1, sizeof(T), row});

    auto kernel = runtime.create_kernel({o.kernel_file, kernel_entry_point(o.kernel), o.build_options()});
    auto f_a = runtime.create_field({"a", total, 1, sizeof(T), harmony::BufferUsage::read_write, init.data(),
                                     harmony::RedistributionDependency::proportional});
    auto f_b = runtime.create_field({"b", total, 1, sizeof(T), harmony::BufferUsage::read_write,
                                     o.is_stencil() ? init.data() : zeros.data(),
                                     harmony::RedistributionDependency::proportional});
    // The devices have their copies: free the host ones (GBs for big grids).
    std::vector<T>().swap(init);
    std::vector<T>().swap(zeros);

    // All kernels have the signature (in, out, ...scalars). The stencils read
    // the rows above and below: a halo exchange of one row on `in`; the map has none.
    auto make_binding = [&](harmony::FieldHandle in, harmony::FieldHandle out) {
        auto binding = runtime.bind(kernel)
            .arg(0, in, harmony::Access::read)
            .arg(1, out, harmony::Access::write);
        switch (o.kernel) {
            case Kernel::ocean: binding.arg(2, o.nx).arg(3, o.ny); break;
            case Kernel::lu:    binding.arg(2, o.nx).arg(3, o.ny).arg(4, o.flop_repeat); break;
            case Kernel::map:   binding.arg(2, static_cast<int>(total)).arg(3, o.flop_repeat); break;
        }
        return binding.build();
    };

    const harmony::AutoBalancePolicy balance{balance_mode(o), o.rebalance_interval, o.rebalance_threshold,
                                             o.iterations, o.profiling_file, o.balance_warmup};

    // Only the buffer just written holds state; the other one is overwritten next.
    auto make_step = [&](const char* name, harmony::FieldHandle in, harmony::FieldHandle out) {
        auto step = runtime.step(name)
            .invoke(make_binding(in, out))
            .with_halo_sync(halo_sync(o))
            .with_balance(balance);
        if (o.is_stencil()) {
            step.with_halo_exchange({row, {in}}).tag_field(out, harmony::StepFieldRole::rebalance_source);
        } else {
            step.tag_field(in, harmony::StepFieldRole::rebalance_source);  // the map always reads `a`
        }
        return step.build();
    };

    // Stencils ping-pong between a and b; the map always reads a and writes b.
    const harmony::ExecutionStep step_ab = make_step("a_to_b", f_a, f_b);
    const harmony::ExecutionStep step_ba = o.is_stencil() ? make_step("b_to_a", f_b, f_a) : step_ab;

    runtime.start_timing();
    for (int it = 0; it < o.iterations; ++it) {
        runtime.execute((it % 2 == 0) ? step_ab : step_ba);
    }
    const double seconds = runtime.stop_timing();
    if (o.report) {
        runtime.print_partitions("Final partitions");
        runtime.print_timing_report();
    }
    if (!o.report_csv.empty()) runtime.write_timing_report(o.report_csv);

    RunResult result;
    result.seconds = seconds;
    if (o.gather) {
        // Only rank 0 receives the field (the others pass no buffer).
        const bool result_in_a = o.is_stencil() && (o.iterations % 2 == 0);
        std::vector<T> output(runtime.rank() == 0 ? total : 0, T(0));
        runtime.gather(result_in_a ? f_a : f_b, output.empty() ? nullptr : output.data(), output.size() * sizeof(T));
        result.output.assign(output.begin(), output.end());
    }
    return result;
}

} // namespace

RunResult run_harmony(const Options& o, int argc, char** argv) {
    auto runtime = harmony::Runtime::create(argc, argv);
    runtime.discover_devices({to_harmony(o.device), 0, 4, o.cpu_host_cores});
    runtime.print_devices();
    return o.is_double() ? run_typed<double>(o, runtime) : run_typed<float>(o, runtime);
}

} // namespace stencil2d
