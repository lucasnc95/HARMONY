// Cardiac tissue snapshots for illustration (no time measurement).
//
// Runs the same simulation as main.cpp (same kernels and HARMONY step) and,
// every --every milliseconds, writes the tissue as a legacy VTK file:
//
//   <output>/tissue_0000.vtk, tissue_0001.vtk, ...   (open the series in ParaView)
//
// Fields of each file (one value per cell):
//   Vm                 membrane potential (mV)
//   Cai                intracellular calcium (mM)
//   activation_time    when the cell depolarized (ms; -1 = not yet)
//   substeps_per_step  cell-model substeps per time step since the previous
//                      snapshot: the computational cost of each phase of the
//                      action potential (upstroke >> plateau > rest)
//   celltype           0 endo, 1 epi, 2 mid
//   device             global index of the device that owns the cell
//   rank               MPI process that owns the cell
//
// Run it once, with a small tissue, to make the figures:
//   mpirun -np 2 ./bin/cardiac_snapshots --nx 24 --ny 24 --nz 256 --time 300 --every 2 --output vtk

#include "torord_like.h"  // TORORD_NSTATES, S_CAI, initial state, cell types
#include "vtk_writer.hpp"

#include <harmony/runtime.hpp>

#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
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

struct Options {
    int nx = 24, ny = 24, nz = 256;
    double dx = 0.01, dt = 0.01;   // cm, ms
    double time = 300.0;           // ms
    double every = 2.0;            // ms between snapshots
    double diffusivity = 0.0012;   // cm^2/ms
    double bcl = 500.0, stim_duration = 1.0, stim_amplitude = -80.0;
    int stim_planes = 3;
    double dv_max = 0.005, ca_tol = 2.0e-5;
    int min_substeps = 1, max_substeps = 1000;
    bool double_precision = false;
    harmony::DeviceKind device = harmony::DeviceKind::all;
    harmony::BalanceMode balance = harmony::BalanceMode::dynamic_threshold;
    int rebalance_interval = 100, balance_warmup = 0;
    float rebalance_threshold = 0.01f;
    std::string profiling_file = "profiling_results.txt";
    std::string output = "vtk";
    std::string kernel_file = KERNEL_FILE;
    std::string kernel_dir = KERNEL_DIR;
};

const char* usage =
    "cardiac_snapshots: writes the tissue as VTK files for illustration (no timing)\n"
    "  --nx --ny --nz            tissue size in cells (24 24 256)\n"
    "  --time                    simulated ms (300)\n"
    "  --every                   ms between snapshots (2)\n"
    "  --output                  output directory (vtk)\n"
    "  --precision               float | double (float)\n"
    "  --device                  all | gpu | cpu (all)\n"
    "  --balance-mode            off | static | dynamic (dynamic)\n"
    "  --balance-strategy        threshold | profiled (threshold)\n"
    "  --rebalance-interval      steps (100)\n"
    "  --balance-warmup          steps before the first attempt (0 = interval)\n"
    "  --rebalance-threshold     (0.01)\n"
    "  --profiling-file          migration profile for 'profiled'\n"
    "  --dx --dt --diffusivity --bcl --stim-planes --dv-max --ca-tol --min-substeps --max-substeps\n"
    "  --kernel-file --kernel-dir\n";

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
        else if (key == "--time") o.time = std::stod(v);
        else if (key == "--every") o.every = std::stod(v);
        else if (key == "--output") o.output = v;
        else if (key == "--dx") o.dx = std::stod(v);
        else if (key == "--dt") o.dt = std::stod(v);
        else if (key == "--diffusivity") o.diffusivity = std::stod(v);
        else if (key == "--bcl") o.bcl = std::stod(v);
        else if (key == "--stim-planes") o.stim_planes = std::stoi(v);
        else if (key == "--dv-max") o.dv_max = std::stod(v);
        else if (key == "--ca-tol") o.ca_tol = std::stod(v);
        else if (key == "--min-substeps") o.min_substeps = std::stoi(v);
        else if (key == "--max-substeps") o.max_substeps = std::stoi(v);
        else if (key == "--rebalance-interval") o.rebalance_interval = std::stoi(v);
        else if (key == "--balance-warmup") o.balance_warmup = std::stoi(v);
        else if (key == "--rebalance-threshold") o.rebalance_threshold = std::stof(v);
        else if (key == "--profiling-file") o.profiling_file = v;
        else if (key == "--balance-mode") schedule = v;
        else if (key == "--balance-strategy") strategy = v;
        else if (key == "--kernel-file") o.kernel_file = v;
        else if (key == "--kernel-dir") o.kernel_dir = v;
        else if (key == "--precision") {
            if (v != "float" && v != "double") throw std::invalid_argument("--precision float|double");
            o.double_precision = (v == "double");
        } else if (key == "--device") {
            if (v == "all") o.device = harmony::DeviceKind::all;
            else if (v == "gpu") o.device = harmony::DeviceKind::gpu;
            else if (v == "cpu") o.device = harmony::DeviceKind::cpu;
            else throw std::invalid_argument("--device all|gpu|cpu");
        } else {
            throw std::invalid_argument("unknown option " + key + "\n" + usage);
        }
    }
    if (schedule == "off") o.balance = harmony::BalanceMode::off;
    else if (schedule == "static" && strategy == "threshold") o.balance = harmony::BalanceMode::static_threshold;
    else if (schedule == "dynamic" && strategy == "threshold") o.balance = harmony::BalanceMode::dynamic_threshold;
    else if (schedule == "static" && strategy == "profiled") o.balance = harmony::BalanceMode::static_profiled;
    else if (schedule == "dynamic" && strategy == "profiled") o.balance = harmony::BalanceMode::dynamic_profiled;
    else throw std::invalid_argument("--balance-mode off|static|dynamic, --balance-strategy threshold|profiled");

    if (o.nx < 1 || o.ny < 1 || o.nz < 3) throw std::invalid_argument("need nx, ny >= 1 and nz >= 3");
    if (o.every < o.dt) throw std::invalid_argument("--every must be at least one time step");
    if (o.diffusivity * o.dt / (o.dx * o.dx) > 1.0 / 6.0) {
        throw std::invalid_argument("unstable diffusion: reduce --dt or increase --dx");
    }
    return o;
}

// Host values (double) <-> device bytes (float or double).
std::vector<unsigned char> to_device(const std::vector<double>& values, bool dp) {
    std::vector<unsigned char> bytes(values.size() * (dp ? sizeof(double) : sizeof(float)));
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (dp) {
            std::memcpy(bytes.data() + i * sizeof(double), &values[i], sizeof(double));
        } else {
            const float f = static_cast<float>(values[i]);
            std::memcpy(bytes.data() + i * sizeof(float), &f, sizeof(float));
        }
    }
    return bytes;
}

double value_at(const std::vector<unsigned char>& bytes, std::size_t i, bool dp) {
    if (dp) {
        double d;
        std::memcpy(&d, bytes.data() + i * sizeof(double), sizeof(double));
        return d;
    }
    float f;
    std::memcpy(&f, bytes.data() + i * sizeof(float), sizeof(float));
    return f;
}

// Global device and rank that own each cell, from the current partitions.
void owners(const std::vector<harmony::DevicePartition>& parts, std::size_t cells, std::vector<int>& device,
            std::vector<int>& rank) {
    device.assign(cells, -1);
    rank.assign(cells, -1);
    for (const harmony::DevicePartition& p : parts) {
        for (std::size_t g = p.global_offset; g < p.global_offset + p.element_count && g < cells; ++g) {
            device[g] = p.device_global_index;
            rank[g] = p.owning_rank;
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        auto runtime = harmony::Runtime::create(argc, argv);
        const Options o = parse_options(argc, argv);
        const bool dp = o.double_precision;
        const std::size_t real_bytes = dp ? sizeof(double) : sizeof(float);

        runtime.discover_devices({o.device, 0});
        runtime.print_devices();

        const std::size_t plane = static_cast<std::size_t>(o.nx) * o.ny;
        const std::size_t cells = plane * o.nz;
        const std::size_t steps = static_cast<std::size_t>(std::llround(o.time / o.dt));
        const std::size_t steps_per_snapshot = std::max<std::size_t>(1, std::llround(o.every / o.dt));

        runtime.set_partition({cells, 1, real_bytes, plane});

        // ---- Fields (same as main.cpp) ---------------------------------------
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
            const auto cell = to_device(one, dp);
            std::vector<unsigned char> data(cells * cell.size());
            for (std::size_t i = 0; i < cells; ++i) std::memcpy(data.data() + i * cell.size(), cell.data(), cell.size());
            state = field("state", TORORD_NSTATES, real_bytes, data.data());
        }
        activation = field("activation", 1, real_bytes, constant(-1.0).data());
        work = field("work", 1, sizeof(float), std::vector<float>(cells, 0.0f).data());

        // ---- Kernels and step (same as main.cpp) -----------------------------
        const std::string build_options = "-I " + o.kernel_dir +
            (dp ? " -DREAL=double -DUSE_DOUBLE" : " -DREAL=float -cl-single-precision-constant");
        auto diffusion = runtime.create_kernel({o.kernel_file, "diffusion", build_options});
        auto reaction = runtime.create_kernel({o.kernel_file, "reaction", build_options});
        auto real_arg = [dp](double x) { return dp ? harmony::ScalarArg(x) : harmony::ScalarArg(static_cast<float>(x)); };

        const harmony::AutoBalancePolicy balance{o.balance, o.rebalance_interval, o.rebalance_threshold,
                                                 static_cast<int>(steps), o.profiling_file, o.balance_warmup};
        const double coef = o.diffusivity * o.dt / (o.dx * o.dx);

        auto bind_diffusion = runtime.bind(diffusion)
            .arg(0, vm, harmony::Access::read)
            .arg(1, vm_tmp, harmony::Access::write)
            .arg(2, o.nx).arg(3, o.ny).arg(4, o.nz).arg(5, real_arg(coef))
            .build();

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

            return runtime.step("tissue")
                .invoke(bind_diffusion)
                .invoke(bind_reaction)
                .with_halo_exchange({plane, {vm}})
                .with_balance(balance)
                .tag_field(vm, harmony::StepFieldRole::rebalance_source)
                .tag_field(state, harmony::StepFieldRole::rebalance_source)
                .tag_field(activation, harmony::StepFieldRole::rebalance_source)
                .tag_field(work, harmony::StepFieldRole::rebalance_source)
                .build();
        };

        // ---- Snapshots -------------------------------------------------------
        if (runtime.rank() == 0) std::filesystem::create_directories(o.output);

        std::vector<int> celltype(cells);
        for (std::size_t g = 0; g < cells; ++g) {
            const int x = static_cast<int>(g % o.nx);
            celltype[g] = (x < (3 * o.nx) / 10) ? CELL_ENDO : ((x < (6 * o.nx) / 10) ? CELL_MID : CELL_EPI);
        }

        std::vector<unsigned char> v_buf(cells * real_bytes), a_buf(cells * real_bytes);
        std::vector<unsigned char> s_buf(cells * TORORD_NSTATES * real_bytes);
        std::vector<float> work_now(cells), work_before(cells, 0.0f);
        std::size_t last_snapshot_step = 0;
        int index = 0;

        // gather() is collective: every rank calls it, rank 0 writes the file.
        auto snapshot = [&](std::size_t step) {
            runtime.gather(vm, v_buf.data(), v_buf.size());
            runtime.gather(activation, a_buf.data(), a_buf.size());
            runtime.gather(state, s_buf.data(), s_buf.size());
            runtime.gather(work, work_now.data(), work_now.size() * sizeof(float));
            if (runtime.rank() != 0) return;

            const double t = step * o.dt;
            const double span = static_cast<double>(std::max<std::size_t>(1, step - last_snapshot_step));
            std::vector<float> v(cells), cai(cells), act(cells), substeps(cells);
            for (std::size_t g = 0; g < cells; ++g) {
                v[g] = static_cast<float>(value_at(v_buf, g, dp));
                act[g] = static_cast<float>(value_at(a_buf, g, dp));
                cai[g] = static_cast<float>(value_at(s_buf, g * TORORD_NSTATES + S_CAI, dp));
                substeps[g] = step > last_snapshot_step
                                  ? static_cast<float>((work_now[g] - work_before[g]) / span)
                                  : 0.0f;
            }
            std::vector<int> device, rank;
            owners(runtime.partitions(), cells, device, rank);

            char name[64];
            std::snprintf(name, sizeof(name), "tissue_%04d.vtk", index);
            std::ostringstream title;
            title << "HARMONY cardiac tissue, t = " << t << " ms (step " << step << ")";
            vtk::StructuredPoints out((std::filesystem::path(o.output) / name).string(), title.str(), o.nx, o.ny,
                                      o.nz, o.dx * 10.0);  // spacing in mm
            out.add("Vm", v);
            out.add("Cai", cai);
            out.add("activation_time", act);
            out.add("substeps_per_step", substeps);
            out.add("celltype", celltype);
            out.add("device", device);
            out.add("rank", rank);
            out.close();
            std::printf("  wrote %s/%s  (t = %.2f ms)\n", o.output.c_str(), name, t);
            std::fflush(stdout);
        };
        auto finish_snapshot = [&](std::size_t step) {
            snapshot(step);
            work_before = work_now;
            last_snapshot_step = step;
            ++index;
        };

        if (runtime.rank() == 0) {
            std::printf("Writing %zu snapshots of a %d x %d x %d tissue to %s/ (one every %.3g ms)\n",
                        steps / steps_per_snapshot + 1, o.nx, o.ny, o.nz, o.output.c_str(), o.every);
        }
        finish_snapshot(0);
        for (std::size_t k = 0; k < steps; ++k) {
            runtime.execute(make_step(k * o.dt));
            if ((k + 1) % steps_per_snapshot == 0 || k + 1 == steps) finish_snapshot(k + 1);
        }
        runtime.synchronize(true);
        if (runtime.rank() == 0) {
            std::printf("Done. In ParaView, open %s/tissue_..vtk (the series) and color by Vm,\n"
                        "substeps_per_step (cost) or device (who computes each region).\n",
                        o.output.c_str());
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << std::endl;
        int initialized = 0;
        MPI_Initialized(&initialized);
        if (initialized) MPI_Abort(MPI_COMM_WORLD, 2);
        return 2;
    }
}
