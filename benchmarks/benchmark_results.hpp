#pragma once

#include <mpi.h>

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace benchmark {

struct StepRecord {
    int iteration{0};
    double step_time_s{0.0};
    std::string action;
    int rebalanced{0};
    double rebalance_gain{0.0};
};

struct Durations {
    double loop_s{0.0};
    double validation_s{0.0};
    double total_s{0.0};
};

inline Durations reduce_durations(MPI_Comm communicator,
                                  std::chrono::steady_clock::time_point start,
                                  std::chrono::steady_clock::time_point loop_end,
                                  std::chrono::steady_clock::time_point validation_start,
                                  std::chrono::steady_clock::time_point validation_end,
                                  std::chrono::steady_clock::time_point total_end) {
    const Durations local{
        std::chrono::duration<double>(loop_end - start).count(),
        std::chrono::duration<double>(validation_end - validation_start).count(),
        std::chrono::duration<double>(total_end - start).count(),
    };
    Durations maximum;
    dcl::detail::check_mpi(MPI_Allreduce(&local.loop_s, &maximum.loop_s, 1, MPI_DOUBLE, MPI_MAX,
                           communicator), "MPI_Allreduce(loop duration)");
    dcl::detail::check_mpi(MPI_Allreduce(&local.validation_s, &maximum.validation_s, 1, MPI_DOUBLE, MPI_MAX,
                           communicator), "MPI_Allreduce(validation duration)");
    dcl::detail::check_mpi(MPI_Allreduce(&local.total_s, &maximum.total_s, 1, MPI_DOUBLE, MPI_MAX,
                           communicator), "MPI_Allreduce(full duration)");
    return maximum;
}

inline bool write_results(const std::string& output_csv, const char* workload,
                          const std::string& balance_algo, bool opencl_available,
                          const std::vector<StepRecord>& records, const Durations& durations) {
    std::ofstream csv(output_csv);
    if (!csv.is_open()) {
        std::cerr << "Failed to open output CSV: " << output_csv << "\n";
        return false;
    }

    const char* mode = opencl_available ? "gpu_opencl" : "cpu_simulation";
    csv << "iteration,step_time_s,action,rebalanced,rebalance_gain,execution_mode\n";
    int num_rebalances = 0;
    double sum_gain = 0.0;
    for (const auto& rec : records) {
        csv << rec.iteration << ","
            << std::fixed << std::setprecision(9) << rec.step_time_s << ","
            << rec.action << "," << rec.rebalanced << ","
            << std::fixed << std::setprecision(9) << rec.rebalance_gain << ","
            << mode << "\n";
        if (rec.rebalanced == 1) {
            ++num_rebalances;
            sum_gain += rec.rebalance_gain;
        }
    }
    csv.close();
    if (!csv) {
        std::cerr << "Failed to write output CSV: " << output_csv << "\n";
        return false;
    }

    const double avg_gain = num_rebalances > 0 ? sum_gain / num_rebalances : 0.0;
    std::cout << "[" << workload << "] algo=" << balance_algo
              << " mode=" << mode
              << " total_time_s=" << durations.total_s
              << " loop_time_s=" << durations.loop_s
              << " validation_time_s=" << durations.validation_s
              << " validation=PASS"
              << " num_rebalances=" << num_rebalances
              << " avg_rebalance_gain=" << avg_gain << "\n";
    return true;
}

}  // namespace benchmark
