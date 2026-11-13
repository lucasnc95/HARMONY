// Cardiac tissue simulation (monodomain) with a ToR-ORd-like cell model.
//
// Each time step is ONE HARMONY step with TWO kernels:
//
//   diffusion : read(vm, halo = 1 plane)  -> write(vm_tmp)
//   reaction  : read(vm_tmp), read_write(state) -> write(vm)        (cell model)
//
// The reaction waits for the diffusion (it reads vm_tmp) and the halo of vm
// is exchanged automatically before every diffusion (vm was written by the
// previous reaction). Nothing else is declared.
//
// Why it is a good balancing test: the cell model takes many more substeps
// where the tissue is depolarizing than where it rests (see torord_like.h),
// so the cost per element changes with the phase of the action potential.
// The wave starts at z = 0 and travels along z, the axis HARMONY partitions,
// so the expensive region moves from one partition (device) to the next,
// then the whole tissue goes through plateau, repolarization and rest.
//
//   mpirun -np 2 ./bin/cardiac_tissue --nx 24 --ny 24 --nz 384 --time 100
//   mpirun -np 2 ./bin/cardiac_tissue --balance-mode off      # compare

#include "torord_like.h"  // host copy of the model (double), for --check

#include <harmony/runtime.hpp>

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef KERNEL_FILE
#define KERNEL_FILE "tor_ord.cl"
#endif
#ifndef KERNEL_DIR
#define KERNEL_DIR "."
#endif

namespace {

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

struct Options {
    int nx = 24, ny = 24, nz = 384;
    double dx = 0.01;          // cm (100 um)
    double dt = 0.01;          // ms
    double time = 100.0;       // ms of simulated time
    double diffusivity = 0.0012;  // cm^2/ms
    double bcl = 500.0;        // ms between stimuli
    double stim_duration = 1.0;   // ms
    double stim_amplitude = -80.0;  // uA/uF
    int stim_planes = 3;
    double dv_max = 0.005;     // mV per substep
    double ca_tol = 2.0e-5;    // relative Ca_i change per substep
    int min_substeps = 1;
    int max_substeps = 1000;
    bool double_precision = false;
    harmony::BalanceMode balance = harmony::BalanceMode::dynamic_threshold;
    int rebalance_interval = 200;  // steps
    int balance_warmup = 0;        // steps before the first attempt (0 = rebalance_interval)
    int cpu_host_cores = 0;        // cores of each CPU device left to the MPI process
    float rebalance_threshold = 0.01f;
    std::string profiling_file = "profiling_results.txt";
    harmony::HaloSync halo_sync = harmony::HaloSync::overlap;
    harmony::DeviceKind device = harmony::DeviceKind::all;
    double progress_seconds = 60.0;  // progress line at least this often (wall time)
    std::string kernel_file = KERNEL_FILE;
    std::string kernel_dir = KERNEL_DIR;
    std::string timing_file;
    std::string report_csv;  // timing report as CSV (write_timing_report)
    bool check = false;
    bool verbose = true;  // log every balancing attempt (--verbose 0 to silence)
};

const char* usage =
    "options (all take a value):\n"
    "  --nx --ny --nz             tissue size in cells (default 24 24 384; the wave runs along z)\n"
    "  --dx                       cell size in cm (0.01)\n"
    "  --dt                       time step in ms (0.01)\n"
    "  --time                     simulated time in ms (100)\n"
    "  --diffusivity              cm^2/ms (0.0012)\n"
    "  --bcl                      ms between stimuli (500)\n"
    "  --stim-planes              stimulated z planes (3)\n"
    "  --dv-max --ca-tol          substep tolerances (0.005, 2e-5): smaller = heavier\n"
    "  --min-substeps --max-substeps  (1, 1000)\n"
    "  --precision                float | double (float)\n"
    "  --balance-mode             off | static | dynamic (dynamic)\n"
    "  --balance-strategy         threshold | profiled (threshold)\n"
    "  --rebalance-interval       steps between attempts (200)\n"
    "  --balance-warmup           steps before the first attempt (0 = interval)\n"
    "  --cpu-host-cores           cores of each CPU device left to the MPI process (0)\n"
    "  --report-csv               writes the timing report as CSV to this file\n"
    "  --rebalance-threshold      (0.01)\n"
    "  --profiling-file           migration profile for 'profiled'\n"
    "  --halo-sync                overlap | local | global (overlap)\n"
    "  --device                   all | gpu | cpu: OpenCL devices to use (all)\n"
    "  --progress                 seconds between progress lines (60)\n"
    "  --timing-file              appends the wall time\n"
    "  --check                    1 = compare with a CPU run (small tissues only)\n"
    "  --verbose                  0 = do not log each balancing attempt (1)\n";

Options parse_options(int argc, char** argv) {
    Options o;
    std::string schedule = "dynamic", strategy = "threshold";
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--help" || key == "-h") throw std::invalid_argument(usage);
        if (i + 1 >= argc) throw std::invalid_argument("missing value for " + key);
        const std::string v = argv[++i];
        if (key == "--nx") o.nx = std::stoi(v);
        else if (key == "--ny") o.ny = std::stoi(v);
        else if (key == "--nz") o.nz = std::stoi(v);
        else if (key == "--dx") o.dx = std::stod(v);
        else if (key == "--dt") o.dt = std::stod(v);
        else if (key == "--time") o.time = std::stod(v);
        else if (key == "--diffusivity") o.diffusivity = std::stod(v);
        else if (key == "--bcl") o.bcl = std::stod(v);
        else if (key == "--stim-planes") o.stim_planes = std::stoi(v);
        else if (key == "--dv-max") o.dv_max = std::stod(v);
        else if (key == "--ca-tol") o.ca_tol = std::stod(v);
        else if (key == "--min-substeps") o.min_substeps = std::stoi(v);
        else if (key == "--max-substeps") o.max_substeps = std::stoi(v);
        else if (key == "--precision") {
            if (v != "float" && v != "double") throw std::invalid_argument("--precision must be float or double");
            o.double_precision = (v == "double");
        } else if (key == "--balance-mode") schedule = v;
        else if (key == "--balance-strategy") strategy = v;
        else if (key == "--rebalance-interval") o.rebalance_interval = std::stoi(v);
        else if (key == "--balance-warmup") o.balance_warmup = std::stoi(v);
        else if (key == "--rebalance-threshold") o.rebalance_threshold = std::stof(v);
        else if (key == "--profiling-file") o.profiling_file = v;
        else if (key == "--cpu-host-cores") o.cpu_host_cores = std::stoi(v);
        else if (key == "--device") {
            if (v == "all") o.device = harmony::DeviceKind::all;
            else if (v == "gpu") o.device = harmony::DeviceKind::gpu;
            else if (v == "cpu") o.device = harmony::DeviceKind::cpu;
            else throw std::invalid_argument("--device must be all, gpu or cpu");
        } else if (key == "--progress") o.progress_seconds = std::stod(v);
        else if (key == "--halo-sync") {
            if (v == "overlap") o.halo_sync = harmony::HaloSync::overlap;
            else if (v == "local") o.halo_sync = harmony::HaloSync::local;
            else if (v == "global") o.halo_sync = harmony::HaloSync::global;
            else throw std::invalid_argument("--halo-sync must be overlap, local or global");
        } else if (key == "--kernel-file") o.kernel_file = v;
        else if (key == "--kernel-dir") o.kernel_dir = v;
        else if (key == "--timing-file") o.timing_file = v;
        else if (key == "--report-csv") o.report_csv = v;
        else if (key == "--check") o.check = (v == "1");
        else if (key == "--verbose") o.verbose = (v == "1");
        else throw std::invalid_argument("unknown option " + key + "\n" + usage);
    }

    if (schedule == "off") o.balance = harmony::BalanceMode::off;
    else if (schedule == "static" && strategy == "threshold") o.balance = harmony::BalanceMode::static_threshold;
    else if (schedule == "dynamic" && strategy == "threshold") o.balance = harmony::BalanceMode::dynamic_threshold;
    else if (schedule == "static" && strategy == "profiled") o.balance = harmony::BalanceMode::static_profiled;
    else if (schedule == "dynamic" && strategy == "profiled") o.balance = harmony::BalanceMode::dynamic_profiled;
    else throw std::invalid_argument("--balance-mode off|static|dynamic, --balance-strategy threshold|profiled");

    if (o.nx < 1 || o.ny < 1 || o.nz < 3) throw std::invalid_argument("need nx, ny >= 1 and nz >= 3");
    const double coef = o.diffusivity * o.dt / (o.dx * o.dx);
    if (coef > 1.0 / 6.0) {
        std::ostringstream msg;
        msg << "unstable diffusion: diffusivity*dt/dx^2 = " << coef << " > 1/6 (reduce --dt or increase --dx)";
        throw std::invalid_argument(msg.str());
    }
    return o;
}

const char* balance_name(harmony::BalanceMode m) {
    switch (m) {
        case harmony::BalanceMode::off: return "off";
        case harmony::BalanceMode::static_threshold: return "static / threshold";
        case harmony::BalanceMode::dynamic_threshold: return "dynamic / threshold";
        case harmony::BalanceMode::static_profiled: return "static / profiled";
        case harmony::BalanceMode::dynamic_profiled: return "dynamic / profiled";
    }
    return "?";
}

// ---------------------------------------------------------------------------
// Host data in the device precision
// ---------------------------------------------------------------------------

// Values are kept in double on the host and converted to the precision of
// the kernels when uploaded / downloaded.
std::vector<unsigned char> to_device(const std::vector<double>& values, bool dp) {
    std::vector<unsigned char> bytes(values.size() * (dp ? sizeof(double) : sizeof(float)));
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (dp) std::memcpy(bytes.data() + i * sizeof(double), &values[i], sizeof(double));
        else {
            const float f = static_cast<float>(values[i]);
            std::memcpy(bytes.data() + i * sizeof(float), &f, sizeof(float));
        }
    }
    return bytes;
}

std::vector<double> from_device(const std::vector<unsigned char>& bytes, bool dp) {
    const std::size_t n = bytes.size() / (dp ? sizeof(double) : sizeof(float));
    std::vector<double> values(n);
    for (std::size_t i = 0; i < n; ++i) {
        if (dp) std::memcpy(&values[i], bytes.data() + i * sizeof(double), sizeof(double));
        else {
            float f;
            std::memcpy(&f, bytes.data() + i * sizeof(float), sizeof(float));
            values[i] = f;
        }
    }
    return values;
}

// ---------------------------------------------------------------------------
// CPU reference: the same two kernels, sequentially, in double
// ---------------------------------------------------------------------------

struct Reference {
    const Options& o;
    std::size_t plane, cells;
    std::vector<double> vm, tmp, state, activation;

    explicit Reference(const Options& opt)
        : o(opt), plane(static_cast<std::size_t>(opt.nx) * opt.ny), cells(plane * opt.nz),
          vm(cells, TORORD_V0), tmp(cells), state(cells * TORORD_NSTATES), activation(cells, -1.0) {
        for (std::size_t i = 0; i < cells; ++i) torord_initial_state(&state[i * TORORD_NSTATES]);
    }

    void step(double t) {
        const double coef = o.diffusivity * o.dt / (o.dx * o.dx);
        for (std::size_t g = 0; g < cells; ++g) {
            const int x = static_cast<int>(g % o.nx), y = static_cast<int>((g / o.nx) % o.ny);
            const int z = static_cast<int>(g / plane);
            const double c = vm[g];
            const double s = (x > 0 ? vm[g - 1] : c) + (x < o.nx - 1 ? vm[g + 1] : c) +
                             (y > 0 ? vm[g - o.nx] : c) + (y < o.ny - 1 ? vm[g + o.nx] : c) +
                             (z > 0 ? vm[g - plane] : c) + (z < o.nz - 1 ? vm[g + plane] : c) - 6 * c;
            tmp[g] = c + coef * s;
        }
        for (std::size_t g = 0; g < cells; ++g) {
            const int x = static_cast<int>(g % o.nx);
            const int z = static_cast<int>(g / plane);
            const int celltype = (x < (3 * o.nx) / 10) ? CELL_ENDO : ((x < (6 * o.nx) / 10) ? CELL_MID : CELL_EPI);
            const double phase = t - o.bcl * std::floor(t / o.bcl);
            const double istim = (z < o.stim_planes && phase < o.stim_duration) ? o.stim_amplitude : 0.0;
            double v = tmp[g];
            torord_advance(&v, &state[g * TORORD_NSTATES], celltype, istim, o.dt, o.dv_max, o.ca_tol,
                           o.min_substeps, o.max_substeps);
            vm[g] = v;
            if (activation[g] < 0 && v >= -20) activation[g] = t + o.dt;
        }
    }
};

// ---------------------------------------------------------------------------
// Summaries printed by rank 0
// ---------------------------------------------------------------------------

void print_wave_summary(const Options& o, const std::vector<double>& activation) {
    const std::size_t plane = static_cast<std::size_t>(o.nx) * o.ny;
    const std::size_t center = static_cast<std::size_t>(o.nx / 2) + static_cast<std::size_t>(o.nx) * (o.ny / 2);
    const int z1 = o.nz / 4, z2 = (3 * o.nz) / 4;
    const double t1 = activation[center + plane * z1], t2 = activation[center + plane * z2];

    std::size_t activated = 0;
    for (double a : activation) activated += (a >= 0) ? 1 : 0;

    std::printf("\nWave\n  activated cells: %.1f %%\n", 100.0 * activated / activation.size());
    if (t1 >= 0 && t2 > t1) {
        const double cv = (z2 - z1) * o.dx / (t2 - t1) * 10.0;  // cm/ms -> m/s
        std::printf("  conduction velocity along z: %.2f m/s (planes %d -> %d activated at %.2f and %.2f ms)\n",
                    cv, z1, z2, t1, t2);
    } else {
        std::printf("  the wave did not cross the tissue yet (increase --time)\n");
    }
}

// Average substeps per cell and per time step, in bands of z planes: shows
// where the work was (the stimulated end and the path of the wave).
void print_work_profile(const Options& o, const std::vector<double>& work, std::size_t steps) {
    const std::size_t plane = static_cast<std::size_t>(o.nx) * o.ny;
    const int bands = std::min(8, o.nz);
    std::printf("\nCell-model work along z (substeps per cell per time step)\n");
    std::printf("  %-14s %10s\n", "z planes", "substeps");
    std::vector<double> avg(bands);
    double lo = 1e300, hi = 0;
    for (int b = 0; b < bands; ++b) {
        const int z0 = b * o.nz / bands, z1 = (b + 1) * o.nz / bands;
        double sum = 0;
        for (std::size_t g = plane * z0; g < plane * z1; ++g) sum += work[g];
        avg[b] = sum / (plane * (z1 - z0)) / static_cast<double>(std::max<std::size_t>(steps, 1));
        lo = std::min(lo, avg[b]);
        hi = std::max(hi, avg[b]);
    }
    for (int b = 0; b < bands; ++b) {
        const int z0 = b * o.nz / bands, z1 = (b + 1) * o.nz / bands;
        const std::size_t bar = static_cast<std::size_t>(std::lround(40.0 * avg[b] / std::max(hi, 1e-12)));
        std::printf("  %4d - %-7d %10.2f  %s\n", z0, z1 - 1, avg[b], std::string(bar, '#').c_str());
    }
    std::printf("  heaviest / lightest band: %.2fx\n", hi / std::max(lo, 1e-12));
}

} // namespace

int main(int argc, char** argv) {
    try {
        auto runtime = harmony::Runtime::create(argc, argv);
        const Options o = parse_options(argc, argv);
        const bool dp = o.double_precision;
        const std::size_t real_bytes = dp ? sizeof(double) : sizeof(float);

        runtime.discover_devices({o.device, 0, 4, o.cpu_host_cores});
        runtime.set_verbose(o.verbose);

        const std::size_t plane = static_cast<std::size_t>(o.nx) * o.ny;
        const std::size_t cells = plane * o.nz;
        const std::size_t steps = static_cast<std::size_t>(std::llround(o.time / o.dt));

        if (runtime.rank() == 0) {
            std::printf("HARMONY - cardiac tissue, ToR-ORd-like cell model, 2 kernels per step\n");
            std::printf("  Tissue      : %d x %d x %d cells (%.2f x %.2f x %.2f cm), %zu cells\n", o.nx, o.ny, o.nz,
                        o.nx * o.dx, o.ny * o.dx, o.nz * o.dx, cells);
            // Every device holds a full-size replica of each field.
            const double mib = cells * (TORORD_NSTATES + 4) * real_bytes / (1024.0 * 1024.0);
            std::printf("  State       : %d variables + V per cell, %s precision (%.1f %s per device)\n",
                        TORORD_NSTATES, dp ? "double" : "single", mib < 1.0 ? mib * 1024.0 : mib,
                        mib < 1.0 ? "KiB" : "MiB");
            std::printf("  Time        : %.1f ms in %zu steps of %.3f ms, stimulus every %.0f ms\n", o.time, steps,
                        o.dt, o.bcl);
            std::printf("  Cell model  : dv_max %.3g mV, ca_tol %.3g, substeps %d..%d\n", o.dv_max, o.ca_tol,
                        o.min_substeps, o.max_substeps);
            std::printf("  Balancing   : %s", balance_name(o.balance));
            if (o.balance != harmony::BalanceMode::off) {
                std::printf(", warm-up %d steps, every %d steps",
                            o.balance_warmup > 0 ? o.balance_warmup : o.rebalance_interval, o.rebalance_interval);
            }
            std::printf("\n\n");
        }
        runtime.print_devices();

        // One z plane per partition unit; the diffusion halo is one plane.
        runtime.set_partition({cells, 1, real_bytes, plane});

        // ---- Fields ----------------------------------------------------------
        // The initial data is built one field at a time, in the device
        // precision, and freed right after the upload: for millions of cells
        // the host copies would otherwise take gigabytes for the whole run.
        auto field = [&](const char* name, std::size_t units, std::size_t bytes, const void* data) {
            return runtime.create_field({name, cells, units, bytes, harmony::BufferUsage::read_write, data,
                                         harmony::RedistributionDependency::proportional});
        };
        auto constant = [&](double value) { return to_device(std::vector<double>(cells, value), dp); };

        harmony::FieldHandle vm, vm_tmp, state, activation, work;
        {
            const auto data = constant(TORORD_V0);
            vm = field("vm", 1, real_bytes, data.data());
            vm_tmp = field("vm_tmp", 1, real_bytes, data.data());
        }
        {
            std::vector<double> one(TORORD_NSTATES);
            torord_initial_state(one.data());
            const auto cell = to_device(one, dp);  // bytes of one cell
            std::vector<unsigned char> data(cells * cell.size());
            for (std::size_t i = 0; i < cells; ++i) std::memcpy(data.data() + i * cell.size(), cell.data(), cell.size());
            state = field("state", TORORD_NSTATES, real_bytes, data.data());
        }
        activation = field("activation", 1, real_bytes, constant(-1.0).data());
        work = field("work", 1, sizeof(float), std::vector<float>(cells, 0.0f).data());

        // ---- Kernels ---------------------------------------------------------
        const std::string build_options = "-I " + o.kernel_dir +
            (dp ? " -DREAL=double -DUSE_DOUBLE" : " -DREAL=float -cl-single-precision-constant");
        auto diffusion = runtime.create_kernel({o.kernel_file, "diffusion", build_options});
        auto reaction = runtime.create_kernel({o.kernel_file, "reaction", build_options});

        // Scalars of type `real` must match the kernel precision.
        auto real_arg = [dp](double x) { return dp ? harmony::ScalarArg(x) : harmony::ScalarArg(static_cast<float>(x)); };

        const harmony::AutoBalancePolicy balance{o.balance, o.rebalance_interval, o.rebalance_threshold,
                                                 static_cast<int>(steps), o.profiling_file, o.balance_warmup};
        const double coef = o.diffusivity * o.dt / (o.dx * o.dx);

        // diffusion(vm -> vm_tmp) reads the neighbouring planes of vm (halo of
        // one plane); reaction(vm_tmp, state -> vm) integrates the cell model.
        auto bind_diffusion = runtime.bind(diffusion)
            .arg(0, vm, harmony::Access::read)
            .arg(1, vm_tmp, harmony::Access::write)
            .arg(2, o.nx).arg(3, o.ny).arg(4, o.nz).arg(5, real_arg(coef))
            .build();

        // The reaction depends on the time t (stimulus), so its binding and
        // the step are rebuilt every iteration; building them is cheap.
        auto make_step = [&](double t) {
            auto bind_reaction = runtime.bind(reaction)
                .arg(0, vm_tmp, harmony::Access::read)
                .arg(1, state, harmony::Access::read_write)
                .arg(2, vm, harmony::Access::write)
                .arg(3, activation, harmony::Access::read_write)
                .arg(4, work, harmony::Access::read_write)
                .arg(5, o.nx).arg(6, o.ny).arg(7, o.nz)
                .arg(8, real_arg(t)).arg(9, real_arg(o.dt))
                .arg(10, real_arg(o.dv_max)).arg(11, real_arg(o.ca_tol))
                .arg(12, o.min_substeps).arg(13, o.max_substeps)
                .arg(14, real_arg(o.bcl)).arg(15, real_arg(o.stim_duration))
                .arg(16, real_arg(o.stim_amplitude)).arg(17, o.stim_planes)
                .build();

            // vm_tmp is rewritten every step, so it is not moved on a rebalance.
            return runtime.step("tissue")
                .invoke(bind_diffusion)
                .invoke(bind_reaction)
                .with_halo_exchange({plane, {vm}})
                .with_halo_sync(o.halo_sync)
                .with_balance(balance)
                .tag_field(vm, harmony::StepFieldRole::rebalance_source)
                .tag_field(state, harmony::StepFieldRole::rebalance_source)
                .tag_field(activation, harmony::StepFieldRole::rebalance_source)
                .tag_field(work, harmony::StepFieldRole::rebalance_source)
                .build();
        };

        // ---- Time loop -------------------------------------------------------
        if (runtime.rank() == 0) std::printf("\nRunning %zu steps...\n", steps);
        runtime.start_timing();
        const double wall0 = MPI_Wtime();
        double last_print = wall0;
        for (std::size_t k = 0; k < steps; ++k) {
            runtime.execute(make_step(k * o.dt));

            // Every 10 % of the steps and at least every --progress seconds.
            if (runtime.rank() != 0) continue;
            const double now = MPI_Wtime();
            const bool tenth = steps >= 10 && (k + 1) % (steps / 10) == 0;
            if (tenth || now - last_print >= o.progress_seconds) {
                const double done = static_cast<double>(k + 1) / steps;
                std::printf("  t = %8.2f ms  (%5.1f %%)  wall %9.1f s  estimated total %9.1f s\n",
                            (k + 1) * o.dt, 100.0 * done, now - wall0, (now - wall0) / done);
                std::fflush(stdout);
                last_print = now;
            }
        }
        const double elapsed = runtime.stop_timing();

        if (runtime.rank() == 0) std::printf("\n");
        runtime.print_partitions("Final partitions (planes follow the wave)");
        if (runtime.rank() == 0) std::printf("\n");
        runtime.print_timing_report();
        if (!o.report_csv.empty()) runtime.write_timing_report(o.report_csv);

        // ---- Results ---------------------------------------------------------
        std::vector<unsigned char> buf(cells * real_bytes);
        runtime.gather(activation, buf.data(), buf.size());
        const std::vector<double> act = from_device(buf, dp);
        runtime.gather(vm, buf.data(), buf.size());
        const std::vector<double> v_final = from_device(buf, dp);
        std::vector<float> work_f(cells);
        runtime.gather(work, work_f.data(), cells * sizeof(float));

        int status = 0;
        if (runtime.rank() == 0) {
            print_wave_summary(o, act);
            print_work_profile(o, std::vector<double>(work_f.begin(), work_f.end()), steps);

            if (o.check) {
                std::printf("\nChecking against a sequential CPU run (double precision)...\n");
                Reference ref(o);
                for (std::size_t k = 0; k < steps; ++k) ref.step(k * o.dt);
                double dv = 0, dact = 0;
                for (std::size_t g = 0; g < cells; ++g) {
                    dv = std::max(dv, std::fabs(v_final[g] - ref.vm[g]));
                    if ((act[g] < 0) != (ref.activation[g] < 0)) dact = 1e9;
                    else if (act[g] >= 0) dact = std::max(dact, std::fabs(act[g] - ref.activation[g]));
                }
                // Single precision shifts the upstroke by a few time steps at most.
                const double act_tol = dp ? 1e-6 : 5 * o.dt;
                const double v_tol = dp ? 1e-3 : 5.0;
                status = (dact <= act_tol && dv <= v_tol) ? 0 : 1;
                std::printf("  max |activation time difference| = %.4g ms (tolerance %.3g)\n", dact, act_tol);
                std::printf("  max |final V difference|         = %.4g mV (tolerance %.3g)\n", dv, v_tol);
                std::printf("  %s\n", status == 0 ? "[OK]" : "[FAILED]");
            }

            if (!o.timing_file.empty()) {
                std::ofstream out(o.timing_file, std::ios::app);
                out << std::fixed << std::setprecision(9) << elapsed << "\n";
            }
            std::printf("\nExecution time: %.3f s\n", elapsed);
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
