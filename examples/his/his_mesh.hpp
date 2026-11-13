// Mesh layout of the Human Immune System (HIS) simulation.
//
// The mesh has X*Y*Z points and each point stores MALHA_TOTAL_CELULAS values
// (one per cell/substance type), float or double. Indices and parameter positions must match
// the definitions in his.cl.

#ifndef HARMONY_EXAMPLES_HIS_MESH_HPP
#define HARMONY_EXAMPLES_HIS_MESH_HPP

#include <cstddef>
#include <cstdio>
#include <iostream>
#include <vector>

namespace his {

// Cell types (one value per type at each mesh point).
enum Celula : int {
    CELULA_A = 0,
    CELULA_MR = 1,
    CELULA_MA = 2,
    CELULA_N = 3,
    CELULA_CH = 4,
    CELULA_ND = 5,
    CELULA_G = 6,
    CELULA_CA = 7,
    MALHA_TOTAL_CELULAS = 8
};

// Positions inside the "parametrosMalha" array passed to the kernel.
enum ParametroMalha : int {
    OFFSET_COMPUTACAO = 0,
    LENGTH_COMPUTACAO = 1,
    COMPRIMENTO_GLOBAL_X = 2,
    COMPRIMENTO_GLOBAL_Y = 3,
    COMPRIMENTO_GLOBAL_Z = 4,
    MALHA_DIMENSAO_POSICAO_Z = 5,
    MALHA_DIMENSAO_POSICAO_Y = 6,
    MALHA_DIMENSAO_POSICAO_X = 7,
    MALHA_DIMENSAO_CELULAS = 8,
    NUMERO_PARAMETROS_MALHA = 9
};

inline std::vector<int> make_parameters(int total_points, int x, int y, int z) {
    std::vector<int> p(NUMERO_PARAMETROS_MALHA, 0);
    p[OFFSET_COMPUTACAO] = 0;
    p[LENGTH_COMPUTACAO] = total_points;
    p[COMPRIMENTO_GLOBAL_X] = x;
    p[COMPRIMENTO_GLOBAL_Y] = y;
    p[COMPRIMENTO_GLOBAL_Z] = z;
    p[MALHA_DIMENSAO_POSICAO_Z] = y * x * MALHA_TOTAL_CELULAS;
    p[MALHA_DIMENSAO_POSICAO_Y] = x * MALHA_TOTAL_CELULAS;
    p[MALHA_DIMENSAO_POSICAO_X] = MALHA_TOTAL_CELULAS;
    p[MALHA_DIMENSAO_CELULAS] = 1;
    return p;
}

inline std::size_t index_of(int celula, int x, int y, int z, const std::vector<int>& p) {
    return static_cast<std::size_t>(celula) * p[MALHA_DIMENSAO_CELULAS] +
           static_cast<std::size_t>(z) * p[MALHA_DIMENSAO_POSICAO_Z] +
           static_cast<std::size_t>(y) * p[MALHA_DIMENSAO_POSICAO_Y] +
           static_cast<std::size_t>(x) * p[MALHA_DIMENSAO_POSICAO_X];
}

// Initial condition: antigen (A) in the top quarter of the mesh and resting
// macrophages (MR) everywhere.
template <typename T>
void initialize(std::vector<T>& mesh, const std::vector<int>& p) {
    const int X = p[COMPRIMENTO_GLOBAL_X];
    const int Y = p[COMPRIMENTO_GLOBAL_Y];
    const int Z = p[COMPRIMENTO_GLOBAL_Z];
    for (int x = 0; x < X; ++x) {
        for (int y = 0; y < Y; ++y) {
            for (int z = 0; z < Z; ++z) {
                for (int c = 0; c < MALHA_TOTAL_CELULAS; ++c) {
                    mesh[index_of(c, x, y, z, p)] = T(0);
                }
                if (z >= static_cast<int>(0.75f * Z)) {
                    mesh[index_of(CELULA_A, x, y, z, p)] = T(100);
                }
                mesh[index_of(CELULA_MR, x, y, z, p)] = T(1);
            }
        }
    }
}

// Prints the antigen concentration of every point.
template <typename T>
void print_antigen(const std::vector<T>& mesh, const std::vector<int>& p, const char* title) {
    std::cout << "\n=== " << title << " ===\n";
    for (int x = 0; x < p[COMPRIMENTO_GLOBAL_X]; ++x) {
        for (int y = 0; y < p[COMPRIMENTO_GLOBAL_Y]; ++y) {
            for (int z = 0; z < p[COMPRIMENTO_GLOBAL_Z]; ++z) {
                std::printf("%8.4f ", static_cast<double>(mesh[index_of(CELULA_A, x, y, z, p)]));
            }
            std::printf("\n");
        }
        std::printf("\n");
    }
    std::fflush(stdout);
}

// Names of the cell types (for the VTK output).
inline const char* cell_name(int celula) {
    // As described in the comments of his.cl.
    static const char* names[MALHA_TOTAL_CELULAS] = {
        "A_antigen", "MR_resting_macrophage", "MA_activated_macrophage", "N_neutrophil",
        "CH_proinflammatory_cytokine", "ND_apoptotic_neutrophil", "G_protein_granules",
        "CA_antiinflammatory_cytokine"};
    return (celula >= 0 && celula < MALHA_TOTAL_CELULAS) ? names[celula] : "?";
}

} // namespace his

#endif // HARMONY_EXAMPLES_HIS_MESH_HPP
