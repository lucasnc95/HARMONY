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
#include <string>

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



#include <cmath>
#include <algorithm>
#include <cstdint>

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


static inline float clampf(float v, float lo, float hi) {
    return std::max(lo, std::min(v, hi));
}

static inline float gauss1(float x, float c, float s) {
    float d = (x - c) / s;
    return std::exp(-0.5f * d * d);
}

static inline float gauss3(
    float x, float cx, float sx,
    float y, float cy, float sy,
    float z, float cz, float sz)
{
    float dx = (x - cx) / sx;
    float dy = (y - cy) / sy;
    float dz = (z - cz) / sz;

    return std::exp(-0.5f * (dx * dx + dy * dy + dz * dz));
}

static inline float hash01(int x, int y, int z) {
    std::uint32_t h = 2166136261u;

    h = (h ^ static_cast<std::uint32_t>(x)) * 16777619u;
    h = (h ^ static_cast<std::uint32_t>(y)) * 16777619u;
    h = (h ^ static_cast<std::uint32_t>(z)) * 16777619u;

    h ^= h >> 16;
    h *= 2246822519u;
    h ^= h >> 13;
    h *= 3266489917u;
    h ^= h >> 16;

    return static_cast<float>(h & 0x00FFFFFFu) / static_cast<float>(0x01000000u);
}

static void InicializarPontosHIS_CargaIrregular3D(float* malha, int* parametrosMalha) {
    const int X = parametrosMalha[COMPRIMENTO_GLOBAL_X];
    const int Y = parametrosMalha[COMPRIMENTO_GLOBAL_Y];
    const int Z = parametrosMalha[COMPRIMENTO_GLOBAL_Z];

    constexpr float PI = 3.14159265358979323846f;

    for (int x = 0; x < X; ++x) {
        for (int y = 0; y < Y; ++y) {
            for (int z = 0; z < Z; ++z) {

                float xn = (X > 1) ? static_cast<float>(x) / static_cast<float>(X - 1) : 0.0f;
                float yn = (Y > 1) ? static_cast<float>(y) / static_cast<float>(Y - 1) : 0.0f;
                float zn = (Z > 1) ? static_cast<float>(z) / static_cast<float>(Z - 1) : 0.0f;

                float antigeno = 0.0f;

                // ============================================================
                // 1) Região diagonal 3D.
                //
                // O centro em z depende de x e y. Portanto não é uma fatia
                // z constante repetida em todo o domínio.
                // ============================================================
                float centroZDiagonal = 0.12f + 0.38f * xn + 0.22f * yn;

                float regiaoDiagonal = gauss3(
                    xn, 0.35f, 0.22f,
                    yn, 0.35f, 0.18f,
                    zn, centroZDiagonal, 0.055f
                );

                antigeno += 85.0f * regiaoDiagonal;

                // ============================================================
                // 2) Tubo helicoidal ao longo do eixo z.
                //
                // A posição do centro no plano x-y muda com z.
                // Isso cria uma região ativa comprida, mas não plana.
                // ============================================================
                float fase = 2.0f * PI * (5.0f * zn + 0.15f * std::sin(2.0f * PI * zn));

                float hx = 0.50f + 0.30f * std::sin(fase);
                float hy = 0.50f + 0.30f * std::cos(0.80f * fase + 0.6f);

                float dxh = (xn - hx) / 0.060f;
                float dyh = (yn - hy) / 0.075f;

                float tuboHelicoidal = std::exp(-0.5f * (dxh * dxh + dyh * dyh));

                // Envolve o tubo para ele não ocupar igualmente todo o z.
                float envelopeTubo =
                    0.65f * gauss1(zn, 0.35f, 0.18f) +
                    1.00f * gauss1(zn, 0.72f, 0.14f);

                antigeno += 70.0f * tuboHelicoidal * envelopeTubo;

                // ============================================================
                // 3) Manchas localizadas em regiões diferentes.
                // ============================================================
                float mancha1 = gauss3(
                    xn, 0.18f, 0.075f,
                    yn, 0.78f, 0.090f,
                    zn, 0.62f, 0.035f
                );

                float mancha2 = gauss3(
                    xn, 0.82f, 0.085f,
                    yn, 0.22f, 0.080f,
                    zn, 0.86f, 0.030f
                );

                float mancha3 = gauss3(
                    xn, 0.62f, 0.120f,
                    yn, 0.68f, 0.110f,
                    zn, 0.18f, 0.045f
                );

                antigeno += 100.0f * mancha1;
                antigeno += 95.0f  * mancha2;
                antigeno += 60.0f  * mancha3;

                // ============================================================
                // 4) Perturbação determinística.
                //
                // Serve para quebrar simetria e criar gradientes locais.
                // Não usa rand(), então o resultado é reproduzível.
                // ============================================================
                float ruido = hash01(x, y, z);

                float ondulacao =
                    0.5f +
                    0.5f * std::sin(
                        2.0f * PI *
                        (
                            7.0f  * xn +
                            11.0f * yn +
                            17.0f * zn
                        )
                    );

                float fatorIrregular =
                    0.82f +
                    0.22f * ondulacao +
                    0.10f * (ruido - 0.5f);

                antigeno *= fatorIrregular;

                // Mantém valores em uma faixa razoável para o modelo.
                antigeno = clampf(antigeno, 0.0f, 120.0f);

                // Zera valores muito pequenos para criar regiões realmente inativas.
                if (antigeno < 0.05f) {
                    antigeno = 0.0f;
                }

                malha[idx_his(CELULA_A,  x, y, z, parametrosMalha)] = antigeno;

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
/*    for (int x = 0; x < parametrosMalha[COMPRIMENTO_GLOBAL_X]; ++x) {
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
    }*/
    int X = parametrosMalha[COMPRIMENTO_GLOBAL_X];
    int Y = parametrosMalha[COMPRIMENTO_GLOBAL_Y];
    int Z = parametrosMalha[COMPRIMENTO_GLOBAL_Z];

    for (int x = 0; x < X; ++x) {
        for (int y = 0; y < Y; ++y) {
            for (int z = 0; z < Z; ++z) {

                // Valor propositalmente diferente por eixo.
                //
                // Exemplo:
                // x = unidade
                // y = dezena
                // z = centena
                //
                // A(3, 4, 2) = 243
                float antigeno = 100.0f * z + 10.0f * y + 1.0f * x;

                malha[idx_his(CELULA_A,  x, y, z, parametrosMalha)] = antigeno;

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
    if (schedule == "hierarchical" || strategy == "hierarchical") return dcl::BalanceMode::hierarchical;
    throw std::runtime_error("Invalid balance configuration. Use --balance-mode off|static|dynamic|hierarchical and --balance-strategy threshold|profiled|hierarchical");
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

        int iterations = parse_int_arg(argv + 1, argv + argc, "--iterations", 10000);

        const int rebalance_interval = parse_int_arg(argv + 1, argv + argc, "--rebalance-interval", 1000);
        const float rebalance_threshold = parse_float_arg(argv + 1, argv + argc, "--rebalance-threshold", 0.0003125f);
        const std::string balance_mode_str = parse_string_arg(argv + 1, argv + argc, "--balance-mode", "dynamic");
        const std::string balance_strategy_str = parse_string_arg(argv + 1, argv + argc, "--balance-strategy", "threshold");
        const std::string profiling_file = parse_string_arg(argv + 1, argv + argc, "--profiling-file", "profiling_results.txt");
        const std::string timing_file = parse_string_arg(argv + 1, argv + argc, "--timing-file", "");
        const std::string metrics_file = parse_string_arg(argv + 1, argv + argc, "--metrics-file", "");
        const std::string metrics_run_id = parse_string_arg(argv + 1, argv + argc, "--metrics-run-id", "0");
        const int first_rebalance_interval = parse_int_arg(argv + 1, argv + argc, "--first-rebalance-interval", 0);
        const bool gather_final = parse_int_arg(argv + 1, argv + argc, "--gather-final", 1) != 0;
        const dcl::BalanceMode balance_mode = parse_balance_mode(balance_mode_str, balance_strategy_str);

        if (!metrics_file.empty()) {
            setenv("DCL_METRICS_FILE", metrics_file.c_str(), 1);
            setenv("DCL_METRICS_RUN_ID", metrics_run_id.c_str(), 1);
        }

        if (first_rebalance_interval > 0) {
            const std::string first_interval_str = std::to_string(first_rebalance_interval);
            setenv("DCL_FIRST_REBALANCE_INTERVAL", first_interval_str.c_str(), 1);
        }

        //gather_final = true; // gather is intentionally disabled for benchmark runs.

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
        InicializarPontosHIS_CargaIrregular3D(malha.data(), parametros.data());

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

        if (gather_final) {
            runtime.gather(
                final_field,
                malha.data(),
                malha.size() * sizeof(float)
            );
        }

        if (runtime.rank() == 0) {
            
            std::cout << "\n=== PARTICOES FINAIS ===\n";
            print_partitions(runtime.partitions());

            std::cout << "\n============================\n";
            std::cout << "ITERACAO " << iterations << "\n";
            
  /*              PrintMalhaCompletaUnida(
                    malha.data(),
                    parametros.data(),
                    final_is_a ? "STATE_A_FINAL" : "STATE_B_FINAL"
                );
            
  */          
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
