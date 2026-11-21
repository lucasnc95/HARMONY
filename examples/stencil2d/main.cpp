// 2D stencil / map benchmark: HARMONY runtime vs. a hand-written MPI + OpenCL
// version of the same computation.
//
//   mpirun -np 2 ./stencil2d --impl harmony --kernel ocean --nx 1024 --ny 1024 --iters 100
//   mpirun -np 2 ./stencil2d --impl mpiocl  --kernel lu    --flop-repeat 64
//   mpirun -np 2 ./stencil2d --impl both    --kernel map   --n 16777216
//   mpirun -np 2 ./stencil2d --impl harmony --kernel lu --precision double --chains 4
//          --balance-mode static --balance-warmup 20 --timing-file times.txt
//
// --impl both runs the two versions and checks that the results match.

#include "stencil2d.hpp"

#include <mpi.h>

#include <fstream>
#include <iomanip>
#include <initializer_list>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

namespace {

using namespace stencil2d;

// Minimal "--name value" parser.
class Args {
public:
    Args(int argc, char** argv) {
        for (int i = 1; i < argc; ++i) {
            const std::string key = argv[i];
            if (i + 1 >= argc) throw std::invalid_argument("missing value for " + key);
            values_[key] = argv[++i];
        }
    }
    void require_known(std::initializer_list<const char*> known) const {
        for (const auto& kv : values_) {
            bool ok = false;
            for (const char* k : known) ok = ok || kv.first == k;
            if (!ok) throw std::invalid_argument("unknown option " + kv.first);
        }
    }
    std::string get(const std::string& key, const std::string& fallback) const {
        auto it = values_.find(key);
        return it == values_.end() ? fallback : it->second;
    }
    int get_int(const std::string& key, int fallback) const {
        auto it = values_.find(key);
        return it == values_.end() ? fallback : std::stoi(it->second);
    }
    float get_float(const std::string& key, float fallback) const {
        auto it = values_.find(key);
        return it == values_.end() ? fallback : std::stof(it->second);
    }

private:
    std::map<std::string, std::string> values_;
};

Options parse_options(int argc, char** argv) {
    const Args args(argc, argv);
    args.require_known({"--impl", "--kernel", "--device", "--iters", "--nx", "--ny", "--n", "--flop-repeat",
                        "--chains", "--precision", "--kernel-file", "--halo-sync", "--timing-file",
                        "--balance-mode", "--balance-strategy", "--rebalance-interval", "--balance-warmup",
                        "--rebalance-threshold", "--profiling-file", "--cpu-host-cores", "--report-csv", "--gather",
                        "--verbose"});
    Options o;

    const std::string impl = args.get("--impl", "harmony");
    if (impl == "harmony") o.impl = Impl::harmony;
    else if (impl == "mpiocl") o.impl = Impl::mpiocl;
    else if (impl == "both") o.impl = Impl::both;
    else throw std::invalid_argument("--impl must be harmony, mpiocl or both");

    const std::string kernel = args.get("--kernel", "ocean");
    if (kernel == "ocean") o.kernel = Kernel::ocean;
    else if (kernel == "lu") o.kernel = Kernel::lu;
    else if (kernel == "map") o.kernel = Kernel::map;
    else throw std::invalid_argument("--kernel must be ocean, lu or map");

    const std::string device = args.get("--device", "all");
    if (device == "cpu") o.device = Device::cpu;
    else if (device == "gpu") o.device = Device::gpu;
    else if (device == "all") o.device = Device::all;
    else throw std::invalid_argument("--device must be cpu, gpu or all");

    o.iterations = args.get_int("--iters", o.iterations);
    o.nx = args.get_int("--nx", o.nx);
    o.ny = args.get_int("--ny", o.ny);
    o.n = args.get_int("--n", o.n);
    o.flop_repeat = args.get_int("--flop-repeat", o.flop_repeat);
    o.chains = args.get_int("--chains", o.chains);
    const std::string precision = args.get("--precision", "float");
    if (precision == "float") o.precision = Precision::single;
    else if (precision == "double") o.precision = Precision::dual;
    else throw std::invalid_argument("--precision must be float or double");
    o.timing_file = args.get("--timing-file", o.timing_file);
    o.balance_mode = args.get("--balance-mode", o.balance_mode);
    o.balance_strategy = args.get("--balance-strategy", o.balance_strategy);
    o.rebalance_interval = args.get_int("--rebalance-interval", o.rebalance_interval);
    o.balance_warmup = args.get_int("--balance-warmup", o.balance_warmup);
    o.rebalance_threshold = args.get_float("--rebalance-threshold", o.rebalance_threshold);
    o.profiling_file = args.get("--profiling-file", o.profiling_file);
    o.cpu_host_cores = args.get_int("--cpu-host-cores", o.cpu_host_cores);
    o.report_csv = args.get("--report-csv", o.report_csv);
    o.gather = args.get("--gather", "1") == "1";
    if (o.impl == Impl::both && !o.gather) throw std::invalid_argument("--impl both needs --gather 1 (to compare)");
    o.kernel_file = args.get("--kernel-file", o.kernel_file);
    o.halo_sync = args.get("--halo-sync", o.halo_sync);
    o.verbose = args.get("--verbose", "1") == "1";
    if (o.halo_sync != "overlap" && o.halo_sync != "local" && o.halo_sync != "global") {
        throw std::invalid_argument("--halo-sync must be overlap, local or global");
    }

    if (o.chains < 1 || o.chains > 16) throw std::invalid_argument("--chains must be between 1 and 16");
    if (o.iterations <= 0 || o.nx < 3 || o.ny < 3 || o.n <= 0) {
        throw std::invalid_argument("Sizes and --iters must be positive (nx, ny >= 3)");
    }
    return o;
}

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int status = 0;
    try {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        const Options o = parse_options(argc, argv);

        // The hand-written baseline runs first; with --impl both HARMONY then
        // reports its time and the comparison of the results.
        RunResult mpiocl_result;
        if (o.impl != Impl::harmony) mpiocl_result = run_mpiocl(o);

        RunResult harmony_result;
        if (o.impl != Impl::mpiocl) {
            harmony_result = run_harmony(o, argc, argv, o.impl == Impl::both ? &mpiocl_result : nullptr);
            status = harmony_result.status;
        } else if (rank == 0) {
            std::cout << "MPI+OpenCL baseline: " << std::fixed << std::setprecision(3) << mpiocl_result.seconds
                      << " s\n";
        }

        // One line per run (--impl both: HARMONY and MPI+OpenCL seconds).
        if (rank == 0 && !o.timing_file.empty()) {
            std::ofstream out(o.timing_file, std::ios::app);
            if (!out) throw std::runtime_error("cannot open the timing file " + o.timing_file);
            out << std::fixed << std::setprecision(9);
            if (o.impl == Impl::harmony) out << harmony_result.seconds << "\n";
            else if (o.impl == Impl::mpiocl) out << mpiocl_result.seconds << "\n";
            else out << harmony_result.seconds << " " << mpiocl_result.seconds << "\n";
        }
        MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return status;
}
