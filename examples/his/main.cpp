// Human Immune System (HIS) simulation on a 3D mesh, distributed with HARMONY.
//
// Two buffers (A and B) are swapped every iteration. The domain is split in
// XY planes, and one plane of halo is exchanged between neighbouring devices.
//
// Usage (see README for all options):
//   mpirun -np 2 ./his --x 50 --y 50 --z 3200 --iterations 10000
//          --balance-mode dynamic --balance-strategy threshold
//          --rebalance-interval 1000 --rebalance-threshold 0.0003125
//          --balance-warmup 2000   (iterations before the first attempt)
//          --precision double      (default: float)

#include "his_mesh.hpp"

#include <harmony/runtime.hpp>

#include <mpi.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

#ifndef KERNEL_FILE
#define KERNEL_FILE "his.cl"
#endif

namespace {

struct Options {
    int x = 50;
    int y = 50;
    int z = 3200;
    int iterations = 10000;
    int rebalance_interval = 1000;
    int balance_warmup = 0;  // iterations before the first attempt (0 = rebalance_interval)
    float rebalance_threshold = 0.0003125f;
    harmony::BalanceMode balance_mode = harmony::BalanceMode::dynamic_threshold;
    std::string profiling_file = "profiling_results.txt";
    std::string timing_file;
    std::string report_csv;  // timing report as CSV (write_timing_report)
    std::string kernel_file = KERNEL_FILE;
    harmony::HaloSync halo_sync = harmony::HaloSync::overlap;
    bool gather_final = true;
    bool print_mesh = false;
    bool verbose = true;  // log every balancing attempt (--verbose 0 to silence)
    harmony::DeviceKind device = harmony::DeviceKind::all;
    int cpu_host_cores = 0;  // cores of each CPU device left to the MPI process
    bool use_double = false;  // --precision float|double
};

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

harmony::BalanceMode parse_balance_mode(const std::string& schedule, const std::string& strategy) {
    if (schedule == "off") return harmony::BalanceMode::off;
    if (schedule == "static" && strategy == "threshold") return harmony::BalanceMode::static_threshold;
    if (schedule == "dynamic" && strategy == "threshold") return harmony::BalanceMode::dynamic_threshold;
    if (schedule == "static" && strategy == "profiled") return harmony::BalanceMode::static_profiled;
    if (schedule == "dynamic" && strategy == "profiled") return harmony::BalanceMode::dynamic_profiled;
    throw std::invalid_argument(
        "Invalid balance configuration. Use --balance-mode off|static|dynamic "
        "and --balance-strategy threshold|profiled");
}

harmony::HaloSync parse_halo_sync(const std::string& value) {
    if (value == "overlap") return harmony::HaloSync::overlap;
    if (value == "local") return harmony::HaloSync::local;
    if (value == "global") return harmony::HaloSync::global;
    throw std::invalid_argument("--halo-sync must be overlap, local or global");
}

const char* halo_sync_name(harmony::HaloSync sync) {
    switch (sync) {
        case harmony::HaloSync::overlap: return "overlap (exchange while the interior is computed)";
        case harmony::HaloSync::local:   return "local (devices finish the interior first)";
        case harmony::HaloSync::global:  return "global (devices finish + MPI barrier)";
    }
    return "?";
}

bool parse_bool(const std::string& v) { return v == "1" || v == "true" || v == "yes"; }

// Every option is "--name value".
Options parse_options(int argc, char** argv) {
    Options o;
    std::string schedule = "dynamic";
    std::string strategy = "threshold";
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (i + 1 >= argc) throw std::invalid_argument("missing value for " + key);
        const std::string v = argv[++i];
        if (key == "--x") o.x = std::stoi(v);
        else if (key == "--y") o.y = std::stoi(v);
        else if (key == "--z") o.z = std::stoi(v);
        else if (key == "--iterations") o.iterations = std::stoi(v);
        else if (key == "--rebalance-interval") o.rebalance_interval = std::stoi(v);
        else if (key == "--balance-warmup") o.balance_warmup = std::stoi(v);
        else if (key == "--rebalance-threshold") o.rebalance_threshold = std::stof(v);
        else if (key == "--balance-mode") schedule = v;
        else if (key == "--balance-strategy") strategy = v;
        else if (key == "--profiling-file") o.profiling_file = v;
        else if (key == "--timing-file") o.timing_file = v;
        else if (key == "--report-csv") o.report_csv = v;
        else if (key == "--kernel-file") o.kernel_file = v;
        else if (key == "--gather-final") o.gather_final = parse_bool(v);
        else if (key == "--print-mesh") o.print_mesh = parse_bool(v);
        else if (key == "--halo-sync") o.halo_sync = parse_halo_sync(v);
        else if (key == "--verbose") o.verbose = parse_bool(v);
        else if (key == "--device") {
            if (v == "all") o.device = harmony::DeviceKind::all;
            else if (v == "gpu") o.device = harmony::DeviceKind::gpu;
            else if (v == "cpu") o.device = harmony::DeviceKind::cpu;
            else throw std::invalid_argument("--device must be all, gpu or cpu");
        }
        else if (key == "--cpu-host-cores") o.cpu_host_cores = std::stoi(v);
        else if (key == "--precision") {
            if (v == "float") o.use_double = false;
            else if (v == "double") o.use_double = true;
            else throw std::invalid_argument("--precision must be float or double");
        }
        else throw std::invalid_argument("unknown option " + key + " (see README)");
    }
    o.balance_mode = parse_balance_mode(schedule, strategy);

    if (o.x <= 0 || o.y <= 0 || o.z <= 0) throw std::invalid_argument("Mesh dimensions must be positive");
    if (o.iterations <= 0) throw std::invalid_argument("--iterations must be positive");
    return o;
}

// The kernel indexes the mesh with int, so every stride must fit in an int.
void check_fits_in_int(std::size_t value, const char* what) {
    if (value > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error(std::string(what) + " exceeds the int range used by the kernel");
    }
}

// Order-dependent checksum, handy to compare runs (same value as before for float).
template <typename T>
std::uint64_t checksum(const std::vector<T>& data) {
    using Bits = std::conditional_t<sizeof(T) == 4, std::uint32_t, std::uint64_t>;
    std::uint64_t h = 1469598103934665603ull;
    for (T v : data) {
        Bits bits = 0;
        std::memcpy(&bits, &v, sizeof(bits));
        h = (h ^ bits) * 1099511628211ull;
    }
    return h;
}

} // namespace

// The whole run, for T = float or double.
template <typename T>
int run(const Options& opt, harmony::Runtime& runtime) {
    // --- Mesh -------------------------------------------------------------
    const std::size_t plane = static_cast<std::size_t>(opt.x) * static_cast<std::size_t>(opt.y);
    const std::size_t points = plane * static_cast<std::size_t>(opt.z);
    check_fits_in_int(points * his::MALHA_TOTAL_CELULAS, "mesh size");

    const std::vector<int> params = his::make_parameters(static_cast<int>(points), opt.x, opt.y, opt.z);
    std::vector<T> mesh(points * his::MALHA_TOTAL_CELULAS, T(0));
    his::initialize(mesh, params);

    // --- Partition: whole XY planes per device ----------------------------
    const std::size_t granularity = plane;
    runtime.set_partition({points, his::MALHA_TOTAL_CELULAS, sizeof(T), granularity});

    // --- Fields and kernel ------------------------------------------------
    auto params_field = runtime.create_field({"parametros", params.size(), 1, sizeof(int),
                                              harmony::BufferUsage::read_only, params.data(),
                                              harmony::RedistributionDependency::none});
    auto state_a = runtime.create_field({"malha_a", points, his::MALHA_TOTAL_CELULAS, sizeof(T),
                                         harmony::BufferUsage::read_write, mesh.data(),
                                         harmony::RedistributionDependency::proportional});
    auto state_b = runtime.create_field({"malha_b", points, his::MALHA_TOTAL_CELULAS, sizeof(T),
                                         harmony::BufferUsage::read_write, mesh.data(),
                                         harmony::RedistributionDependency::proportional});

    // The kernel uses REAL = float, or double with -DUSE_DOUBLE.
    auto kernel = runtime.create_kernel({opt.kernel_file, "ProcessarPontos", std::is_same_v<T, double> ? "-DUSE_DOUBLE" : ""});

    // The devices have their copies now: only rank 0 keeps the host mesh
    // (to receive the final gather). Saves a whole buffer of RAM per node.
    const std::size_t mesh_bytes = mesh.size() * sizeof(T);
    if (runtime.rank() != 0 || !opt.gather_final) std::vector<T>().swap(mesh);

    // ProcessarPontos(out, in, params): one step per buffer direction.
    // The kernel reads the neighbouring XY planes of `in`: a halo of one plane.
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

    const harmony::AutoBalancePolicy balance{
        opt.balance_mode,
        opt.rebalance_interval,
        opt.rebalance_threshold,
        opt.iterations,
        opt.profiling_file,
        opt.balance_warmup
    };

    // Only the buffer just written is moved on a rebalance: the other one
    // is overwritten by the next step.
    const harmony::ExecutionStep step_a_from_b = runtime.step("his-step-a-from-b")
        .invoke(bind_a_from_b, harmony::LaunchGeometry{0, points, std::nullopt})
        .with_halo_exchange(harmony::HaloSpec{granularity, {state_b}})
        .with_halo_sync(opt.halo_sync)
        .with_balance(balance)
        .tag_field(state_a, harmony::StepFieldRole::rebalance_source)
        .synchronize_at_end(false)
        .build();

    const harmony::ExecutionStep step_b_from_a = runtime.step("his-step-b-from-a")
        .invoke(bind_b_from_a, harmony::LaunchGeometry{0, points, std::nullopt})
        .with_halo_exchange(harmony::HaloSpec{granularity, {state_a}})
        .with_halo_sync(opt.halo_sync)
        .with_balance(balance)
        .tag_field(state_b, harmony::StepFieldRole::rebalance_source)
        .synchronize_at_end(false)
        .build();

    if (runtime.rank() == 0) {
        const double mib = static_cast<double>(mesh_bytes) / (1024.0 * 1024.0);
        std::cout << "HARMONY - Human Immune System simulation\n"
                  << "  Mesh        : " << opt.x << " x " << opt.y << " x " << opt.z << " points ("
                  << std::fixed << std::setprecision(1) << mib << " MiB per buffer)\n" << std::defaultfloat
                  << "  Precision   : " << (std::is_same_v<T, double> ? "double" : "float") << "\n"
                  << "  Iterations  : " << opt.iterations << "\n"
                  << "  Balancing   : " << balance_name(opt.balance_mode);
        if (opt.balance_mode != harmony::BalanceMode::off) {
            std::cout << ", warm-up " << (opt.balance_warmup > 0 ? opt.balance_warmup : opt.rebalance_interval)
                      << " iterations, interval " << opt.rebalance_interval;
            if (opt.balance_mode == harmony::BalanceMode::static_threshold ||
                opt.balance_mode == harmony::BalanceMode::dynamic_threshold) {
                std::cout << ", threshold " << opt.rebalance_threshold;
            } else {
                std::cout << ", profile '" << opt.profiling_file << "'";
            }
        }
        std::cout << "\n  Halo sync   : " << halo_sync_name(opt.halo_sync) << "\n\n";
    }
    runtime.print_devices();
    if (runtime.rank() == 0) std::cout << "\n";
    runtime.print_partitions("Initial partitions");

    // --- Time loop --------------------------------------------------------
    // start/stop_timing wait for every device and rank, so the interval
    // covers exactly the iterations (including any rebalancing).
    runtime.start_timing();
    for (int iter = 0; iter < opt.iterations; ++iter) {
        runtime.execute((iter % 2 == 0) ? step_a_from_b : step_b_from_a);
    }
    const double elapsed = runtime.stop_timing();

    // --- Results ----------------------------------------------------------
    const bool final_is_a = (opt.iterations % 2 != 0);
    if (opt.gather_final) {
        runtime.gather(final_is_a ? state_a : state_b, mesh.data(), mesh.size() * sizeof(T));
    }

    if (runtime.rank() == 0) std::cout << "\n";
    runtime.print_partitions("Final partitions");
    if (runtime.rank() == 0) std::cout << "\n";
    runtime.print_timing_report();
    if (!opt.report_csv.empty()) runtime.write_timing_report(opt.report_csv);

    if (runtime.rank() == 0) {
        if (opt.gather_final) {
            if (opt.print_mesh) {
                his::print_antigen(mesh, params, final_is_a ? "STATE_A_FINAL" : "STATE_B_FINAL");
            }
            std::cout << "Checksum: " << std::hex << checksum(mesh) << std::dec << "\n";
        }

        if (!opt.timing_file.empty()) {
            std::ofstream timing_out(opt.timing_file, std::ios::app);
            if (!timing_out) {
                std::cerr << "Could not open timing file: " << opt.timing_file << "\n";
                return 3;
            }
            // One value per line; the PBS scripts compute statistics from it.
            timing_out << std::fixed << std::setprecision(9) << elapsed << "\n";
        }
        std::cout << "Execution time: " << std::fixed << std::setprecision(3) << elapsed << " s\n";
    }
    return 0;
}

int main(int argc, char** argv) {
    try {
        auto runtime = harmony::Runtime::create(argc, argv);
        const Options opt = parse_options(argc, argv);

        runtime.discover_devices({opt.device, 0, 4, opt.cpu_host_cores});
        runtime.set_verbose(opt.verbose);

        return opt.use_double ? run<double>(opt, runtime) : run<float>(opt, runtime);
    } catch (const std::exception& e) {
        // Stop every rank, so none is left waiting in a collective call.
        std::cerr << "error: " << e.what() << std::endl;
        int initialized = 0;
        MPI_Initialized(&initialized);
        if (initialized) MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return 1;
}
