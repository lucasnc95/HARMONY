// Several kernels in one step.
//
// Each step runs three kernels, written in program order:
//
//   gradient : reads u (1 neighbour each side) -> writes g
//   update   : reads u, reads g (1 neighbour)  -> writes v
//   energy   : reads u                         -> writes e
//
// Only the accesses are declared; HARMONY works out the rest:
//   - update waits for gradient (it reads g), on the same kernel queue;
//   - energy shares no written field with the others, so it runs on another
//     queue at the same time as gradient/update;
//   - u's halo is exchanged before gradient (u was written by the previous
//     step) and g's halo before update (g was just written by gradient).
//     Nobody asks for these exchanges: they follow from "read with halo".
//
// The result is compared with a sequential CPU version:
//   mpirun -np 2 ./bin/multi_kernel --n 200000 --iterations 100

#include <harmony/runtime.hpp>

#include <mpi.h>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef KERNEL_FILE
#define KERNEL_FILE "multi_kernel.cl"
#endif

namespace {

struct Options {
    int n = 200000;
    int iterations = 100;
    int interval = 25;
    int warmup = 0;  // iterations before the first balancing attempt (0 = interval)
    float c = 0.2f;
    std::string kernel_file = KERNEL_FILE;
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
        else throw std::invalid_argument("unknown option " + key + " (use --n --iterations --interval --warmup --verbose)");
    }
    if (o.n < 3 || o.iterations <= 0) throw std::invalid_argument("--n must be >= 3 and --iterations > 0");
    return o;
}

// Same computation, sequentially.
void reference(std::vector<float>& u, std::vector<float>& e, int iterations, float c) {
    const std::size_t n = u.size();
    std::vector<float> g(n), v(n);
    for (int it = 0; it < iterations; ++it) {
        for (std::size_t i = 0; i < n; ++i) g[i] = (i == 0 || i == n - 1) ? 0.0f : 0.5f * (u[i + 1] - u[i - 1]);
        for (std::size_t i = 0; i < n; ++i) {
            v[i] = (i == 0 || i == n - 1) ? u[i] : u[i] + c * 0.5f * (g[i + 1] - g[i - 1]);
        }
        for (std::size_t i = 0; i < n; ++i) e[i] = u[i] * u[i];
        u.swap(v);
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        auto runtime = harmony::Runtime::create(argc, argv);
        const Options opt = parse_options(argc, argv);

        runtime.discover_devices({harmony::DeviceKind::all, 0, /*kernel_queues=*/4});
        runtime.set_verbose(opt.verbose);
        runtime.print_devices();

        const std::size_t n = static_cast<std::size_t>(opt.n);
        runtime.set_partition({n, 1, sizeof(float), 1});

        // Varies everywhere, so every partition border carries real data.
        std::vector<float> initial(n);
        for (std::size_t i = 0; i < n; ++i) initial[i] = static_cast<float>((i * 7919) % 1000) / 1000.0f;

        auto field = [&](const char* name, const std::vector<float>& data) {
            return runtime.create_field({name, n, 1, sizeof(float), harmony::BufferUsage::read_write, data.data(),
                                         harmony::RedistributionDependency::proportional});
        };
        auto u = field("u", initial);
        auto v = field("v", initial);
        auto g = field("g", std::vector<float>(n, 0.0f));
        auto e = field("e", std::vector<float>(n, 0.0f));

        auto gradient = runtime.create_kernel({opt.kernel_file, "gradient", ""});
        auto update = runtime.create_kernel({opt.kernel_file, "update", ""});
        auto energy = runtime.create_kernel({opt.kernel_file, "energy", ""});

        const harmony::AutoBalancePolicy balance{harmony::BalanceMode::dynamic_threshold, opt.interval, 0.01f,
                                                 opt.iterations, "", opt.warmup};

        // One step = the three kernels, in program order:
        //   gradient: g = d(in)/dx        (reads the neighbours of `in`)
        //   update:   out = in + c dg/dx  (reads the neighbours of `g`)
        //   energy:   e = in^2            (independent of update: runs at the same time)
        // Without tag_field every field the step uses is moved on a rebalance.
        auto make_step = [&](const char* name, harmony::FieldHandle in, harmony::FieldHandle out) {
            auto bind_gradient = runtime.bind(gradient)
                .arg(0, in, harmony::Access::read)
                .arg(1, g, harmony::Access::write)
                .arg(2, opt.n)
                .build();
            auto bind_update = runtime.bind(update)
                .arg(0, in, harmony::Access::read)
                .arg(1, g, harmony::Access::read)
                .arg(2, out, harmony::Access::write)
                .arg(3, opt.c)
                .arg(4, opt.n)
                .build();
            auto bind_energy = runtime.bind(energy)
                .arg(0, in, harmony::Access::read)
                .arg(1, e, harmony::Access::write)
                .arg(2, opt.n)
                .build();

            return runtime.step(name)
                .invoke(bind_gradient)
                .invoke(bind_update)
                .invoke(bind_energy)
                .with_halo_exchange({1, {in, g}})
                .with_balance(balance)
                .build();
        };
        const harmony::ExecutionStep u_to_v = make_step("u_to_v", u, v);
        const harmony::ExecutionStep v_to_u = make_step("v_to_u", v, u);

        runtime.start_timing();
        for (int it = 0; it < opt.iterations; ++it) {
            runtime.execute((it % 2 == 0) ? u_to_v : v_to_u);
        }
        runtime.stop_timing();
        runtime.print_timing_report();

        std::vector<float> result(n), result_e(n);
        runtime.gather((opt.iterations % 2 == 0) ? u : v, result.data(), n * sizeof(float));
        runtime.gather(e, result_e.data(), n * sizeof(float));

        int status = 0;
        if (runtime.rank() == 0) {
            std::vector<float> ref = initial;
            std::vector<float> ref_e(n);
            reference(ref, ref_e, opt.iterations, opt.c);

            double err = 0.0, err_e = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                err = std::max(err, static_cast<double>(std::fabs(result[i] - ref[i])));
                err_e = std::max(err_e, static_cast<double>(std::fabs(result_e[i] - ref_e[i])));
            }
            status = (err <= 1.0e-5 && err_e <= 1.0e-5) ? 0 : 1;
            std::cout << "max |device - cpu|: u = " << err << ", e = " << err_e
                      << (status == 0 ? "  [OK]\n" : "  [FAILED]\n");
        }
        MPI_Bcast(&status, 1, MPI_INT, 0, runtime.communicator());
        return status;
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << std::endl;
        int initialized = 0;
        MPI_Initialized(&initialized);
        if (initialized) MPI_Abort(MPI_COMM_WORLD, 2);
        return 2;
    }
}
