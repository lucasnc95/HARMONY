// Cardiac tissue (monodomain, operator splitting) with two kernels per step:
//
//   diffusion : vm_tmp = vm + coef * laplacian(vm)       (7-point, no-flux borders)
//   reaction  : integrates the cell model over dt from vm_tmp, writes vm
//
// The tissue is an nx * ny * nz box flattened to 1D (index = x + nx*(y + ny*z)),
// so HARMONY splits it in whole z planes and the diffusion reads one plane
// on each side of a partition (halo = nx * ny elements).
//
// Build options (set by main.cpp): -I<this directory> -DREAL=float|double.

#include "torord_like.h"

__kernel void diffusion(__global const real* vm, __global real* vm_tmp,
                        const int nx, const int ny, const int nz, const real coef)
{
    const int gid = (int)get_global_id(0);
    const int plane = nx * ny;
    if (gid >= plane * nz) return;

    const int x = gid % nx;
    const int y = (gid / nx) % ny;
    const int z = gid / plane;

    const real c = vm[gid];
    const real xm = (x > 0) ? vm[gid - 1] : c;
    const real xp = (x < nx - 1) ? vm[gid + 1] : c;
    const real ym = (y > 0) ? vm[gid - nx] : c;
    const real yp = (y < ny - 1) ? vm[gid + nx] : c;
    const real zm = (z > 0) ? vm[gid - plane] : c;
    const real zp = (z < nz - 1) ? vm[gid + plane] : c;

    vm_tmp[gid] = c + coef * (xm + xp + ym + yp + zm + zp - 6 * c);
}

__kernel void reaction(__global const real* vm_tmp, __global real* state, __global real* vm,
                       __global real* activation, __global float* work,
                       const int nx, const int ny, const int nz,
                       const real t, const real dt, const real dv_max, const real ca_tol,
                       const int min_substeps, const int max_substeps,
                       const real bcl, const real stim_duration, const real stim_amplitude,
                       const int stim_planes)
{
    const int gid = (int)get_global_id(0);
    const int plane = nx * ny;
    if (gid >= plane * nz) return;

    const int x = gid % nx;
    const int z = gid / plane;

    // Transmural heterogeneity along x: endo | mid | epi.
    const int celltype = (x < (3 * nx) / 10) ? CELL_ENDO : ((x < (6 * nx) / 10) ? CELL_MID : CELL_EPI);

    // Periodic stimulus on the first planes (the wave then travels along z).
    const real phase = t - bcl * floor(t / bcl);
    const real istim = (z < stim_planes && phase < stim_duration) ? stim_amplitude : (real)0;

    real s[TORORD_NSTATES];
    __global real* cell = state + (size_t)gid * TORORD_NSTATES;
    for (int k = 0; k < TORORD_NSTATES; ++k) s[k] = cell[k];

    real v = vm_tmp[gid];
    const int n = torord_advance(&v, s, celltype, istim, dt, dv_max, ca_tol, min_substeps, max_substeps);

    for (int k = 0; k < TORORD_NSTATES; ++k) cell[k] = s[k];
    vm[gid] = v;
    work[gid] += (float)n;

    // Time of the first upstroke through -20 mV.
    if (activation[gid] < 0 && v >= -20) activation[gid] = t + dt;
}
