// Quickstart: a 1D stencil distributed over every OpenCL device of every
// MPI rank, with halo exchange and dynamic load balancing.
//
// The result is checked against a sequential CPU version, so this example
// doubles as an end-to-end test of the runtime:
//   mpirun -np 2 ./bin/quickstart_stencil1d --n 100000 --iterations 200

#include <harmony/runtime.hpp>

#include <mpi.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef KERNEL_FILE
#define KERNEL_FILE "stencil1d.cl"
#endif

namespace {

struct Options {
    int n = 100000;
    int iterations = 200;
    int interval = 50;
    int warmup = 0;  // iterations before the first balancing attempt (0 = interval)
    std::string kernel_file = KERNEL_FILE;
    harmony::HaloSync halo_sync = harmony::HaloSync::overlap;
    bool verbose = true;  // log every balancing attempt (--verbose 0 to silence)
};

Options parse_options(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (i + 1 >= argc) throw std::invalid_argument("missing value for " + key);
        const std::string value = argv[++i];
        if (key == "--n") o.n = std::stoi(value);
        else if (key == "--iterations") o.iterations = std::stoi(value);
        else if (key == "--interval") o.interval = std::stoi(value);
        else if (key == "--warmup") o.warmup = std::stoi(value);
        else if (key == "--kernel-file") o.kernel_file = value;
        else if (key == "--verbose") o.verbose = (value == "1");
        else if (key == "--halo-sync") {
            if (value == "overlap") o.halo_sync = harmony::HaloSync::overlap;
            else if (value == "local") o.halo_sync = harmony::HaloSync::local;
            else if (value == "global") o.halo_sync = harmony::HaloSync::global;
            else throw std::invalid_argument("--halo-sync must be overlap, local or global");
        } else {
            throw std::invalid_argument("unknown option " + key +
                                        " (use --n --iterations --interval --warmup --halo-sync --verbose --kernel-file)");
        }
    }
    if (o.n < 3 || o.iterations <= 0) throw std::invalid_argument("--n must be >= 3 and --iterations > 0");
    return o;
}

std::vector<float> reference(std::vector<float> a, int iterations) {
    std::vector<float> b(a.size());
    const std::size_t n = a.size();
    for (int it = 0; it < iterations; ++it) {
        b[0] = a[0];
        b[n - 1] = a[n - 1];
        for (std::size_t i = 1; i + 1 < n; ++i) b[i] = (a[i - 1] + a[i] + a[i + 1]) / 3.0f;
        a.swap(b);
    }
    return a;
}

} // namespace

int main(int argc, char** argv) {
    try {
        // 1. Start the runtime (initializes MPI) and find the devices.
        auto runtime = harmony::Runtime::create(argc, argv);
        const Options opt = parse_options(argc, argv);

        runtime.discover_devices({harmony::DeviceKind::all, 0});
        runtime.set_verbose(opt.verbose);
        runtime.print_devices();

        // 2. Split the domain (n elements of one float) among all devices.
        const std::size_t total = static_cast<std::size_t>(opt.n);
        runtime.set_partition({total, 1, sizeof(float), 1});

        // 3. Register the data and the kernel.
        std::vector<float> initial(total);
        for (std::size_t i = 0; i < total; ++i) initial[i] = static_cast<float>(i % 97);

        auto a = runtime.create_field({"a", total, 1, sizeof(float), harmony::BufferUsage::read_write,
                                       initial.data(), harmony::RedistributionDependency::proportional});
        auto b = runtime.create_field({"b", total, 1, sizeof(float), harmony::BufferUsage::read_write,
                                       initial.data(), harmony::RedistributionDependency::proportional});
        auto kernel = runtime.create_kernel({opt.kernel_file, "Stencil1D", ""});

        // 4. One iteration: Stencil1D(in, out, n). It reads one neighbour on
        //    each side of `in`, so `in` has a halo exchange of 1 element.
        auto bind_b_from_a = runtime.bind(kernel)
            .arg(0, a, harmony::Access::read)
            .arg(1, b, harmony::Access::write)
            .arg(2, opt.n)
            .build();
        auto bind_a_from_b = runtime.bind(kernel)
            .arg(0, b, harmony::Access::read)
            .arg(1, a, harmony::Access::write)
            .arg(2, opt.n)
            .build();

        const harmony::AutoBalancePolicy balance{harmony::BalanceMode::dynamic_threshold, opt.interval, 0.01f,
                                                 opt.iterations, "", opt.warmup};

        const harmony::ExecutionStep a_to_b = runtime.step("a_to_b")
            .invoke(bind_b_from_a)
            .with_halo_exchange({1, {a}})
            .with_halo_sync(opt.halo_sync)
            .with_balance(balance)
            .tag_field(b, harmony::StepFieldRole::rebalance_source)  // only the fresh result needs to move
            .build();

        const harmony::ExecutionStep b_to_a = runtime.step("b_to_a")
            .invoke(bind_a_from_b)
            .with_halo_exchange({1, {b}})
            .with_halo_sync(opt.halo_sync)
            .with_balance(balance)
            .tag_field(a, harmony::StepFieldRole::rebalance_source)
            .build();

        // 5. Run (timed), then collect the result on rank 0.
        runtime.start_timing();
        for (int it = 0; it < opt.iterations; ++it) {
            runtime.execute((it % 2 == 0) ? a_to_b : b_to_a);
        }
        runtime.stop_timing();
        runtime.print_timing_report();

        std::vector<float> result(total);
        runtime.gather((opt.iterations % 2 == 0) ? a : b, result.data(), result.size() * sizeof(float));

        int status = 0;
        if (runtime.rank() == 0) {
            const std::vector<float> expected = reference(initial, opt.iterations);
            double max_error = 0.0;
            for (std::size_t i = 0; i < total; ++i) {
                max_error = std::max(max_error, static_cast<double>(std::fabs(result[i] - expected[i])));
            }
            status = (max_error <= 1.0e-4) ? 0 : 1;
            std::cout << "max |device - cpu| = " << max_error << (status == 0 ? "  [OK]\n" : "  [FAILED]\n");
        }
        MPI_Bcast(&status, 1, MPI_INT, 0, runtime.communicator());
        return status;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << std::endl;
        int initialized = 0;
        MPI_Initialized(&initialized);
        if (initialized) MPI_Abort(MPI_COMM_WORLD, 2);
        return 2;
    }
}
