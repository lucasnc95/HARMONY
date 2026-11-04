#include "dcl/runtime.hpp"
#include <mpi.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <limits>
#include <optional>
#include <vector>
#include <chrono>

// Tipos de celulas.
#define CELULA_A            0
#define CELULA_MR           1
#define CELULA_MA           2
#define CELULA_N            3
#define CELULA_CH           4
#define CELULA_ND           5
#define CELULA_G            6
#define CELULA_CA           7
#define MALHA_TOTAL_CELULAS 8

// Informacoes de acesso à estrutura "parametrosMalha".
#define OFFSET_COMPUTACAO               0
#define LENGTH_COMPUTACAO               1
#define COMPRIMENTO_GLOBAL_X            2
#define COMPRIMENTO_GLOBAL_Y            3
#define COMPRIMENTO_GLOBAL_Z            4
#define MALHA_DIMENSAO_POSICAO_Z        5
#define MALHA_DIMENSAO_POSICAO_Y        6
#define MALHA_DIMENSAO_POSICAO_X        7
#define MALHA_DIMENSAO_CELULAS          8
#define NUMERO_PARAMETROS_MALHA         9

static inline std::size_t idx_his(
    int celula,
    int x,
    int y,
    int z,
    const int* parametrosMalha
) {
    return static_cast<std::size_t>(celula * parametrosMalha[MALHA_DIMENSAO_CELULAS]) +
           static_cast<std::size_t>(z) * parametrosMalha[MALHA_DIMENSAO_POSICAO_Z] +
           static_cast<std::size_t>(y) * parametrosMalha[MALHA_DIMENSAO_POSICAO_Y] +
           static_cast<std::size_t>(x) * parametrosMalha[MALHA_DIMENSAO_POSICAO_X];
}

static void InicializarParametrosMalhaHIS(
    int* parametrosMalha,
    int offsetComputacao,
    int lengthComputacao,
    int xMalhaLength,
    int yMalhaLength,
    int zMalhaLength
) {
    parametrosMalha[OFFSET_COMPUTACAO] = offsetComputacao;
    parametrosMalha[LENGTH_COMPUTACAO] = lengthComputacao;
    parametrosMalha[COMPRIMENTO_GLOBAL_X] = xMalhaLength;
    parametrosMalha[COMPRIMENTO_GLOBAL_Y] = yMalhaLength;
    parametrosMalha[COMPRIMENTO_GLOBAL_Z] = zMalhaLength;
    parametrosMalha[MALHA_DIMENSAO_POSICAO_Z] = yMalhaLength * xMalhaLength * MALHA_TOTAL_CELULAS;
    parametrosMalha[MALHA_DIMENSAO_POSICAO_Y] = xMalhaLength * MALHA_TOTAL_CELULAS;
    parametrosMalha[MALHA_DIMENSAO_POSICAO_X] = MALHA_TOTAL_CELULAS;
    parametrosMalha[MALHA_DIMENSAO_CELULAS] = 1;
}

static void InicializarPontosHIS(float* malha, int* parametrosMalha) {
    for (int x = 0; x < parametrosMalha[COMPRIMENTO_GLOBAL_X]; ++x) {
        for (int y = 0; y < parametrosMalha[COMPRIMENTO_GLOBAL_Y]; ++y) {
            for (int z = 0; z < parametrosMalha[COMPRIMENTO_GLOBAL_Z]; ++z) {
                if (z >= static_cast<int>(0.75f * parametrosMalha[COMPRIMENTO_GLOBAL_Z])) {
                    malha[idx_his(CELULA_A, x, y, z, parametrosMalha)] = 100.0f;
                } else {
                    malha[idx_his(CELULA_A, x, y, z, parametrosMalha)] = 0.0f;
                }

                malha[idx_his(CELULA_MR, x, y, z, parametrosMalha)] = 1.0f;
                malha[idx_his(CELULA_MA, x, y, z, parametrosMalha)] = 0.0f;
                malha[idx_his(CELULA_N,  x, y, z, parametrosMalha)] = 0.0f;
                malha[idx_his(CELULA_CH, x, y, z, parametrosMalha)] = 0.0f;
                malha[idx_his(CELULA_ND, x, y, z, parametrosMalha)] = 0.0f;
                malha[idx_his(CELULA_G,  x, y, z, parametrosMalha)] = 0.0f;
                malha[idx_his(CELULA_CA, x, y, z, parametrosMalha)] = 0.0f;
            }
        }
    }
}

static void PrintMalhaCompletaUnida(float* malha, int* parametrosMalha, const char* nome) {
    int X = parametrosMalha[COMPRIMENTO_GLOBAL_X];
    int Y = parametrosMalha[COMPRIMENTO_GLOBAL_Y];
    int Z = parametrosMalha[COMPRIMENTO_GLOBAL_Z];

    std::cout << "\n=== " << nome << " ===\n";

    for (int x = 0; x < X; x++) {
        for (int y = 0; y < Y; y++) {
            for (int z = 0; z < Z; z++) {
                float v = malha[
                    (CELULA_A * parametrosMalha[MALHA_DIMENSAO_CELULAS]) +
                    (z * parametrosMalha[MALHA_DIMENSAO_POSICAO_Z]) +
                    (y * parametrosMalha[MALHA_DIMENSAO_POSICAO_Y]) +
                    (x * parametrosMalha[MALHA_DIMENSAO_POSICAO_X])
                ];
                std::printf("%8.4f ", v);
            }
            std::printf("\n");
        }
        std::printf("\n");
    }
}

static void print_partitions(const std::vector<dcl::DevicePartition>& parts) {
    std::cout << "=== Particoes globais ===\n";
    for (std::size_t i = 0; i < parts.size(); ++i) {
        const dcl::DevicePartition& p = parts[i];
        std::cout
            << "part[" << i << "] "
            << "device_global=" << p.device_global_index
            << " rank=" << p.owning_rank
            << " local_index=" << p.local_index
            << " offset=" << p.global_offset
            << " count=" << p.element_count
            << "\n";
    }
    std::cout << "\n";
}


static int parse_int_arg(char** begin, char** end, const std::string& name, int default_value) {
    for (char** it = begin; it != end; ++it) {
        if (name == *it && (it + 1) != end) {
            return std::atoi(*(it + 1));
        }
    }
    return default_value;
}

static float parse_float_arg(char** begin, char** end, const std::string& name, float default_value) {
    for (char** it = begin; it != end; ++it) {
        if (name == *it && (it + 1) != end) {
            return std::strtof(*(it + 1), nullptr);
        }
    }
    return default_value;
}

static std::string parse_string_arg(char** begin, char** end, const std::string& name, const std::string& default_value) {
    for (char** it = begin; it != end; ++it) {
        if (name == *it && (it + 1) != end) {
            return std::string(*(it + 1));
        }
    }
    return default_value;
}

static dcl::BalanceMode parse_balance_mode(const std::string& schedule, const std::string& strategy) {
    if (schedule == "off") return dcl::BalanceMode::off;
    if (schedule == "static" && strategy == "threshold") return dcl::BalanceMode::static_threshold;
    if (schedule == "dynamic" && strategy == "threshold") return dcl::BalanceMode::dynamic_threshold;
    if (schedule == "static" && strategy == "profiled") return dcl::BalanceMode::static_profiled;
    if (schedule == "dynamic" && strategy == "profiled") return dcl::BalanceMode::dynamic_profiled;
    throw std::runtime_error("Invalid balance configuration. Use --balance-mode off|static|dynamic and --balance-strategy threshold|profiled");
}

static std::size_t checked_mul_size_t(std::size_t a, std::size_t b, const char* what) {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
        throw std::overflow_error(std::string("Overflow computing ") + what);
    }
    return a * b;
}

static int checked_int_from_size_t(std::size_t value, const char* what) {
    if (value > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error(std::string(what) + " exceeds int range used by kernel parameters");
    }
    return static_cast<int>(value);
}

int main(int argc, char** argv) {
    try {
        using clock_t = std::chrono::steady_clock;

        auto runtime = dcl::Runtime::create(argc, argv);

        runtime.discover_devices({
            dcl::DeviceKind::all,
            0
        });

        std::cout << "=== Dispositivos locais do rank " << runtime.rank() << " ===\n";
        const std::vector<dcl::DeviceInfo>& devs = runtime.devices();
        for (std::size_t i = 0; i < devs.size(); ++i) {
            const dcl::DeviceInfo& d = devs[i];
            std::cout
                << "global=" << d.global_index
                << " local=" << d.local_index
                << " nome=\"" << d.name << "\""
                << " compute_units=" << d.compute_units
                << "\n";
        }
        std::cout << "\n";

        const int x = parse_int_arg(argv + 1, argv + argc, "--x", 50);
        const int y = parse_int_arg(argv + 1, argv + argc, "--y", 50);
        const int z = parse_int_arg(argv + 1, argv + argc, "--z", 3200);

        const int iterations = parse_int_arg(argv + 1, argv + argc, "--iterations", 10000);

        const int rebalance_interval = parse_int_arg(argv + 1, argv + argc, "--rebalance-interval", 1000);
        const float rebalance_threshold = parse_float_arg(argv + 1, argv + argc, "--rebalance-threshold", 0.0003125f);
        const std::string balance_mode_str = parse_string_arg(argv + 1, argv + argc, "--balance-mode", "dynamic");
        const std::string balance_strategy_str = parse_string_arg(argv + 1, argv + argc, "--balance-strategy", "threshold");
        const std::string profiling_file = parse_string_arg(argv + 1, argv + argc, "--profiling-file", "profiling_results.txt");
        const std::string timing_file = parse_string_arg(argv + 1, argv + argc, "--timing-file", "");
        const bool gather_final = parse_int_arg(argv + 1, argv + argc, "--gather-final", 1) != 0;
        const dcl::BalanceMode balance_mode = parse_balance_mode(balance_mode_str, balance_strategy_str);
        gather_final = true; // gather is intentionally disabled for benchmark runs.

        if (x <= 0 || y <= 0 || z <= 0) {
            throw std::runtime_error("Mesh dimensions must be positive");
        }

        const std::size_t xy =
            checked_mul_size_t(static_cast<std::size_t>(x), static_cast<std::size_t>(y), "x*y");
        const std::size_t total_elements_size =
            checked_mul_size_t(xy, static_cast<std::size_t>(z), "x*y*z");
        const int total_elements =
            checked_int_from_size_t(total_elements_size, "x*y*z");
        const std::size_t tam =
            checked_mul_size_t(total_elements_size, MALHA_TOTAL_CELULAS, "mesh storage");

        checked_int_from_size_t(
            checked_mul_size_t(xy, MALHA_TOTAL_CELULAS, "z stride"),
            "z stride"
        );
        checked_int_from_size_t(
            checked_mul_size_t(static_cast<std::size_t>(x), MALHA_TOTAL_CELULAS, "y stride"),
            "y stride"
        );

        std::vector<int> parametros(NUMERO_PARAMETROS_MALHA, 0);
        std::vector<float> malha(tam, 0.0f);

        InicializarParametrosMalhaHIS(parametros.data(), 0, total_elements, x, y, z);
        InicializarPontosHIS(malha.data(), parametros.data());

        auto kernel = runtime.create_kernel({
            "kernels.cl",
            "ProcessarPontos",
            ""
        });

        // Granularidade = um plano XY
        const std::size_t granularity =
            static_cast<std::size_t>(x) * static_cast<std::size_t>(y);

        runtime.set_partition({
            static_cast<std::size_t>(total_elements),
            MALHA_TOTAL_CELULAS,
            sizeof(float),
            granularity
        });

        if (runtime.rank() == 0) {
            std::cout << "=== PARTICOES INICIAIS ===\n";
            print_partitions(runtime.partitions());
            
        }

        auto params_field = runtime.create_field({
            "parametros",
            static_cast<std::size_t>(NUMERO_PARAMETROS_MALHA),
            1,
            sizeof(int),
            dcl::BufferUsage::read_only,
            parametros.data(),
            dcl::RedistributionDependency::none
        });

        auto state_a = runtime.create_field({
            "malha_a",
            static_cast<std::size_t>(total_elements),
            MALHA_TOTAL_CELULAS,
            sizeof(float),
            dcl::BufferUsage::read_write,
            malha.data(),
            dcl::RedistributionDependency::proportional
        });

        auto state_b = runtime.create_field({
            "malha_b",
            static_cast<std::size_t>(total_elements),
            MALHA_TOTAL_CELULAS,
            sizeof(float),
            dcl::BufferUsage::read_write,
            malha.data(),
            dcl::RedistributionDependency::proportional
        });

        auto bind_a_from_b = runtime.bind(kernel)
            .arg(0, state_a)
            .arg(1, state_b)
            .arg(2, params_field)
            .build();

        auto bind_b_from_a = runtime.bind(kernel)
            .arg(0, state_b)
            .arg(1, state_a)
            .arg(2, params_field)
            .build();

        dcl::ExecutionStep step_a_from_b = runtime.step("his-step-a-from-b")
            .invoke(
                bind_a_from_b,
                dcl::LaunchGeometry{
                    0,
                    static_cast<std::size_t>(total_elements),
                    std::optional<std::size_t>()
                }
            )
            .with_halo_exchange(dcl::HaloSpec{
                granularity,
                std::vector<dcl::FieldHandle>{state_b}
            })
            .with_balance(dcl::AutoBalancePolicy{
                balance_mode,
                rebalance_interval,
                rebalance_threshold,
                iterations,
                profiling_file
            })
            .tag_field(state_b, dcl::StepFieldRole::read_source)
            .tag_field(state_a, dcl::StepFieldRole::write_target)
            .tag_field(state_b, dcl::StepFieldRole::halo_source)
            .tag_field(state_a, dcl::StepFieldRole::rebalance_source)
            .synchronize_at_end(false)
            .build();

        dcl::ExecutionStep step_b_from_a = runtime.step("his-step-b-from-a")
            .invoke(
                bind_b_from_a,
                dcl::LaunchGeometry{
                    0,
                    static_cast<std::size_t>(total_elements),
                    std::optional<std::size_t>()
                }
            )
            .with_halo_exchange(dcl::HaloSpec{
                granularity,
                std::vector<dcl::FieldHandle>{state_a}
            })
            .with_balance(dcl::AutoBalancePolicy{
                balance_mode,
                rebalance_interval,
                rebalance_threshold,
                iterations,
                profiling_file
            })
            .tag_field(state_a, dcl::StepFieldRole::read_source)
            .tag_field(state_b, dcl::StepFieldRole::write_target)
            .tag_field(state_a, dcl::StepFieldRole::halo_source)
            .tag_field(state_b, dcl::StepFieldRole::rebalance_source)
            .synchronize_at_end(false)
            .build();
        
        if (runtime.rank() == 0) {
        std::cout<<"X = "<<x<<" Y = "<<y<<" Z = "<<z<<" Iterations = "<<iterations<<std::endl;
        }

        auto start = clock_t::now();

        for (int iter = 0; iter < iterations; ++iter) {
            if ((iter % 2) == 0) {
                runtime.execute(step_a_from_b);
            } else {
                runtime.execute(step_b_from_a);
            }
        }
        runtime.synchronize(true);
        auto end = clock_t::now();
        
        const bool final_is_a = (iterations % 2 != 0);
        const dcl::FieldHandle final_field = final_is_a ? state_a : state_b;
    
       
            runtime.gather(
                final_field,
                malha.data(),
                malha.size() * sizeof(float)
            );
       
     
        

        if (runtime.rank() == 0) {
            
            std::cout << "\n=== PARTICOES FINAIS ===\n";
            print_partitions(runtime.partitions());

            std::cout << "\n============================\n";
            std::cout << "ITERACAO " << iterations << "\n";
            
                PrintMalhaCompletaUnida(
                    malha.data(),
                    parametros.data(),
                    final_is_a ? "STATE_A_FINAL" : "STATE_B_FINAL"
                );
            
            
            std::chrono::duration<double> elapsed_seconds = end - start;
            const double elapsed_value = elapsed_seconds.count();

            if (!timing_file.empty()) {
                std::ofstream timing_out(timing_file, std::ios::app);
                if (!timing_out.is_open()) {
                    std::cerr << "Erro ao abrir arquivo de tempos: " << timing_file << "\n";
                    return 3;
                }
                // One numeric value per line. The PBS script reads this file to compute statistics.
                timing_out << std::fixed << std::setprecision(9) << elapsed_value << "\n";
            }

            std::cout << "Tempo de execução: " << elapsed_value << "s\n";
        }
        
        return 0;
    } catch (const dcl::Error& e) {
        std::cerr << "dcl error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "std error: " << e.what() << std::endl;
        return 2;
    }
}
