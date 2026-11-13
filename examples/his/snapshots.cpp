// HIS snapshots for illustration (no time measurement).
//
// Runs the same simulation as main.cpp (same kernel and HARMONY steps) and,
// every --every iterations, writes the mesh as a legacy VTK file:
//
//   <output>/his_0000.vtk, his_0001.vtk, ...   (open the series in ParaView)
//
// Fields of each file (one value per mesh point):
//   A_antigen, MR_resting_macrophage, MA_activated_macrophage, N_neutrophil,
//   CH_proinflammatory_cytokine, ND_apoptotic_neutrophil, G_protein_granules,
//   CA_antiinflammatory_cytokine   the 8 populations of the model
//   device   global index of the device that owns the point
//   rank     MPI process that owns the point
//
// Run it once, with a small mesh, to make the figures:
//   mpirun -np 2 ./bin/his_snapshots --x 32 --y 32 --z 256 --iterations 20000 --every 1000 --output vtk

#include "his_mesh.hpp"
#include "vtk_writer.hpp"

#include <harmony/runtime.hpp>

#include <mpi.h>

#include <cstdio>
#include <filesystem>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#ifndef KERNEL_FILE
#define KERNEL_FILE "his.cl"
#endif

namespace {

struct Options {
    int x = 32, y = 32, z = 256;
    int iterations = 20000;
    int every = 1000;
    bool use_double = false;
    harmony::DeviceKind device = harmony::DeviceKind::all;
    harmony::BalanceMode balance = harmony::BalanceMode::dynamic_threshold;
    int rebalance_interval = 1000, balance_warmup = 0;
    float rebalance_threshold = 0.0003125f;
    std::string profiling_file = "profiling_results.txt";
    std::string output = "vtk";
    std::string kernel_file = KERNEL_FILE;
};

const char* usage =
    "his_snapshots: writes the HIS mesh as VTK files for illustration (no timing)\n"
    "  --x --y --z               mesh size (32 32 256)\n"
    "  --iterations              time steps (20000)\n"
    "  --every                   iterations between snapshots (1000)\n"
    "  --output                  output directory (vtk)\n"
    "  --precision               float | double (float)\n"
    "  --device                  all | gpu | cpu (all)\n"
    "  --balance-mode            off | static | dynamic (dynamic)\n"
    "  --balance-strategy        threshold | profiled (threshold)\n"
    "  --rebalance-interval      iterations (1000)\n"
    "  --balance-warmup          iterations before the first attempt (0 = interval)\n"
    "  --rebalance-threshold     (0.0003125)\n"
    "  --profiling-file --kernel-file\n";

Options parse_options(int argc, char** argv) {
    Options o;
    std::string schedule = "dynamic", strategy = "threshold";
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--help" || key == "-h") throw std::invalid_argument(usage);
        if (i + 1 >= argc) throw std::invalid_argument("missing value for " + key);
        const std::string v = argv[++i];
        if (key == "--x") o.x = std::stoi(v);
        else if (key == "--y") o.y = std::stoi(v);
        else if (key == "--z") o.z = std::stoi(v);
        else if (key == "--iterations") o.iterations = std::stoi(v);
        else if (key == "--every") o.every = std::stoi(v);
        else if (key == "--output") o.output = v;
        else if (key == "--rebalance-interval") o.rebalance_interval = std::stoi(v);
        else if (key == "--balance-warmup") o.balance_warmup = std::stoi(v);
        else if (key == "--rebalance-threshold") o.rebalance_threshold = std::stof(v);
        else if (key == "--profiling-file") o.profiling_file = v;
        else if (key == "--balance-mode") schedule = v;
        else if (key == "--balance-strategy") strategy = v;
        else if (key == "--kernel-file") o.kernel_file = v;
        else if (key == "--precision") {
            if (v != "float" && v != "double") throw std::invalid_argument("--precision float|double");
            o.use_double = (v == "double");
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

    if (o.x <= 0 || o.y <= 0 || o.z <= 0) throw std::invalid_argument("mesh dimensions must be positive");
    if (o.iterations <= 0 || o.every <= 0) throw std::invalid_argument("--iterations and --every must be positive");
    return o;
}

template <typename T>
int run(const Options& opt, harmony::Runtime& runtime) {
    const std::size_t plane = static_cast<std::size_t>(opt.x) * static_cast<std::size_t>(opt.y);
    const std::size_t points = plane * static_cast<std::size_t>(opt.z);
    if (points * his::MALHA_TOTAL_CELULAS > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error("mesh too large for the int indices of the kernel");
    }

    const std::vector<int> params = his::make_parameters(static_cast<int>(points), opt.x, opt.y, opt.z);
    std::vector<T> mesh(points * his::MALHA_TOTAL_CELULAS, T(0));
    his::initialize(mesh, params);

    // ---- Same setup as main.cpp ----------------------------------------------
    const std::size_t granularity = plane;
    runtime.set_partition({points, his::MALHA_TOTAL_CELULAS, sizeof(T), granularity});

    auto params_field = runtime.create_field({"parametros", params.size(), 1, sizeof(int),
                                              harmony::BufferUsage::read_only, params.data(),
                                              harmony::RedistributionDependency::none});
    auto state_a = runtime.create_field({"malha_a", points, his::MALHA_TOTAL_CELULAS, sizeof(T),
                                         harmony::BufferUsage::read_write, mesh.data(),
                                         harmony::RedistributionDependency::proportional});
    auto state_b = runtime.create_field({"malha_b", points, his::MALHA_TOTAL_CELULAS, sizeof(T),
                                         harmony::BufferUsage::read_write, mesh.data(),
                                         harmony::RedistributionDependency::proportional});
    auto kernel = runtime.create_kernel(
        {opt.kernel_file, "ProcessarPontos", std::is_same_v<T, double> ? "-DUSE_DOUBLE" : ""});

    auto bind_a_from_b = runtime.bind(kernel)
        .arg(0, state_a, harmony::Access::write)
        .arg(1, state_b, harmony::Access::read)
        .arg(2, params_field)
        .build();
    auto bind_b_from_a = runtime.bind(kernel)
        .arg(0, state_b, harmony::Access::write)
        .arg(1, state_a, harmony::Access::read)
        .arg(2, params_field)
        .build();

    const harmony::AutoBalancePolicy balance{opt.balance, opt.rebalance_interval, opt.rebalance_threshold,
                                             opt.iterations, opt.profiling_file, opt.balance_warmup};

    const harmony::ExecutionStep step_a_from_b = runtime.step("his-step-a-from-b")
        .invoke(bind_a_from_b, harmony::LaunchGeometry{0, points, std::nullopt})
        .with_halo_exchange(harmony::HaloSpec{granularity, {state_b}})
        .with_balance(balance)
        .tag_field(state_a, harmony::StepFieldRole::rebalance_source)
        .build();
    const harmony::ExecutionStep step_b_from_a = runtime.step("his-step-b-from-a")
        .invoke(bind_b_from_a, harmony::LaunchGeometry{0, points, std::nullopt})
        .with_halo_exchange(harmony::HaloSpec{granularity, {state_a}})
        .with_balance(balance)
        .tag_field(state_b, harmony::StepFieldRole::rebalance_source)
        .build();

    // ---- Snapshots -----------------------------------------------------------
    if (runtime.rank() == 0) {
        std::filesystem::create_directories(opt.output);
        std::printf("Writing %d snapshots of a %d x %d x %d mesh to %s/ (one every %d iterations)\n",
                    opt.iterations / opt.every + 1, opt.x, opt.y, opt.z, opt.output.c_str(), opt.every);
    }

    int index = 0;
    // gather() is collective: every rank calls it, rank 0 writes the file.
    // After an odd number of iterations the newest data is in A, else in B
    // (both start with the initial mesh).
    auto snapshot = [&](int iteration) {
        runtime.gather((iteration % 2 != 0) ? state_a : state_b, mesh.data(), mesh.size() * sizeof(T));
        if (runtime.rank() == 0) {
            std::vector<int> device(points, -1), rank(points, -1);
            for (const harmony::DevicePartition& p : runtime.partitions()) {
                for (std::size_t g = p.global_offset; g < p.global_offset + p.element_count; ++g) {
                    device[g] = p.device_global_index;
                    rank[g] = p.owning_rank;
                }
            }

            char name[64];
            std::snprintf(name, sizeof(name), "his_%04d.vtk", index);
            std::ostringstream title;
            title << "HARMONY HIS, iteration " << iteration;
            vtk::StructuredPoints out((std::filesystem::path(opt.output) / name).string(), title.str(), opt.x, opt.y,
                                      opt.z, 0.1);  // deltaX = 0.1 in his.cl
            // Point g = x + X * (y + Y * z) stores its 8 values contiguously.
            std::vector<float> values(points);
            for (int c = 0; c < his::MALHA_TOTAL_CELULAS; ++c) {
                for (std::size_t g = 0; g < points; ++g) {
                    values[g] = static_cast<float>(mesh[g * his::MALHA_TOTAL_CELULAS + static_cast<std::size_t>(c)]);
                }
                out.add(his::cell_name(c), values);
            }
            out.add("device", device);
            out.add("rank", rank);
            out.close();
            std::printf("  wrote %s/%s  (iteration %d)\n", opt.output.c_str(), name, iteration);
            std::fflush(stdout);
        }
        ++index;
    };

    snapshot(0);
    for (int it = 0; it < opt.iterations; ++it) {
        runtime.execute((it % 2 == 0) ? step_a_from_b : step_b_from_a);
        if ((it + 1) % opt.every == 0 || it + 1 == opt.iterations) snapshot(it + 1);
    }
    runtime.synchronize(true);
    if (runtime.rank() == 0) {
        std::printf("Done. In ParaView, open %s/his_..vtk (the series) and color by A_antigen or device.\n",
                    opt.output.c_str());
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        auto runtime = harmony::Runtime::create(argc, argv);
        const Options opt = parse_options(argc, argv);
        runtime.discover_devices({opt.device, 0});
        runtime.print_devices();
        return opt.use_double ? run<double>(opt, runtime) : run<float>(opt, runtime);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << std::endl;
        int initialized = 0;
        MPI_Initialized(&initialized);
        if (initialized) MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return 1;
}
