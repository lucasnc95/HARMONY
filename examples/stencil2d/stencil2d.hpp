// Shared definitions of the stencil2d example.

#ifndef HARMONY_EXAMPLES_STENCIL2D_HPP
#define HARMONY_EXAMPLES_STENCIL2D_HPP

#include <cstddef>
#include <string>
#include <vector>

#ifndef KERNEL_FILE
#define KERNEL_FILE "stencil2d.cl"
#endif

namespace stencil2d {

enum class Impl { harmony, mpiocl, both };
enum class Device { cpu, gpu, all };
enum class Kernel { ocean, lu, map };
enum class Precision { single, dual };

struct Options {
    Impl impl = Impl::harmony;
    Kernel kernel = Kernel::ocean;
    Device device = Device::all;
    int iterations = 100;
    int nx = 1024;
    int ny = 1024;
    int n = 1 << 24;
    int flop_repeat = 64;
    int chains = 1;                     // independent chains in the flop loop (lu, map)
    Precision precision = Precision::single;
    std::string kernel_file = KERNEL_FILE;
    std::string timing_file;            // appends the wall time of each version (one line each)

    // HARMONY only.
    std::string halo_sync = "overlap";  // overlap | local | global
    bool report = true;                 // print the timing report
    std::string report_csv;             // timing report as CSV (write_timing_report)
    bool gather = true;                 // collect the result on rank 0 (needed by --impl both)
    std::string balance_mode = "off";   // off | static | dynamic
    std::string balance_strategy = "profiled";  // threshold | profiled
    int rebalance_interval = 100;
    int balance_warmup = 0;             // 0 = rebalance_interval
    float rebalance_threshold = 0.01f;
    std::string profiling_file = "profiling_results.txt";
    int cpu_host_cores = 0;             // cores of each CPU device left to the MPI process

    bool is_stencil() const { return kernel != Kernel::map; }
    bool is_double() const { return precision == Precision::dual; }
    std::size_t real_bytes() const { return is_double() ? sizeof(double) : sizeof(float); }

    // OpenCL build options shared by both implementations.
    std::string build_options() const {
        std::string opts = is_double() ? "-DREAL=double -DUSE_DOUBLE" : "-DREAL=float";
        return opts + " -DCHAINS=" + std::to_string(chains);
    }
    std::size_t elements() const {
        return is_stencil() ? static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny)
                            : static_cast<std::size_t>(n);
    }
};

inline const char* kernel_entry_point(Kernel k) {
    switch (k) {
        case Kernel::ocean: return "ocean_jacobi_2d_flat";
        case Kernel::lu:    return "lu_update_2d_heavy_flat";
        case Kernel::map:   return "map_heavy_1d";
    }
    return "";
}

// Initial data, identical for both implementations (T = float or double).
template <typename T>
std::vector<T> initial_data(const Options& o) {
    std::vector<T> data(o.elements(), T(0));
    if (o.is_stencil()) {
        for (int y = 0; y < o.ny; ++y) {
            for (int x = 0; x < o.nx; ++x) {
                data[static_cast<std::size_t>(y) * o.nx + x] = T(0.001f) * static_cast<T>((x + 3 * y) % 1000);
            }
        }
    } else {
        for (std::size_t i = 0; i < data.size(); ++i) data[i] = T(0.001f) * static_cast<T>(i);
    }
    return data;
}

struct RunResult {
    double seconds = 0.0;        // wall time of the iterations (max over ranks)
    std::vector<double> output;  // full result, valid on rank 0 only
};

// Implementations (run_harmony.cpp / run_mpiocl.cpp). MPI must be initialized.
RunResult run_harmony(const Options& o, int argc, char** argv);
RunResult run_mpiocl(const Options& o);

} // namespace stencil2d

#endif // HARMONY_EXAMPLES_STENCIL2D_HPP
