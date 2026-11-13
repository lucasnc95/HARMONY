// 2D stencils on a flattened nx*ny grid plus a 1D map kernel.
//
//   ocean_jacobi_2d_flat     5-point Jacobi average (memory bound)
//   lu_update_2d_heavy_flat  9-point weighted average + extra flops
//   map_heavy_1d             element-wise map + extra flops (no halo)
//
// Grid borders are copied unchanged.
//
// Build options (set by the host):
//   -DREAL=double -DUSE_DOUBLE   double precision (default: float)
//   -DCHAINS=N                   independent chains in the flop loop (default 1)
//
// The flop loop does 4 * flop_repeat * CHAINS floating-point operations per
// element. Each chain depends on its previous value, so with CHAINS = 1 a
// work-item cannot overlap its operations; GPUs hide that with thousands of
// work-items, CPUs need independent chains (instruction-level parallelism)
// to keep their vector units busy.

#ifdef USE_DOUBLE
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

#ifndef REAL
#define REAL float
#endif

#ifndef CHAINS
#define CHAINS 1
#endif

// Literals in the precision of REAL (float literals keep the float path
// bit-identical to the original kernels).
#ifdef USE_DOUBLE
#define K(x) (x)
#else
#define K(x) (x##f)
#endif

inline REAL heavy_flops(REAL v, const int flop_repeat)
{
#if CHAINS == 1
    for (int k = 0; k < flop_repeat; ++k) {
        v = v * K(1.000001) + K(0.000001);
        v = v * K(0.999999) + K(0.000001);
    }
    return v;
#else
    REAL acc[CHAINS];
#pragma unroll
    for (int j = 0; j < CHAINS; ++j) acc[j] = v + (REAL)j * K(0.001);

    for (int k = 0; k < flop_repeat; ++k) {
#pragma unroll
        for (int j = 0; j < CHAINS; ++j) {
            acc[j] = acc[j] * K(1.000001) + K(0.000001);
            acc[j] = acc[j] * K(0.999999) + K(0.000001);
        }
    }

    REAL sum = K(0.0);
#pragma unroll
    for (int j = 0; j < CHAINS; ++j) sum += acc[j];
    return sum / (REAL)CHAINS;
#endif
}

__kernel void ocean_jacobi_2d_flat(
    __global const REAL* in,
    __global REAL* out,
    const int nx,
    const int ny)
{
    const int gid = get_global_id(0);
    const int total = nx * ny;
    if (gid >= total) return;

    const int x = gid % nx;
    const int y = gid / nx;

    if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1) {
        out[gid] = in[gid];
        return;
    }

    const REAL center = in[gid];
    const REAL left   = in[gid - 1];
    const REAL right  = in[gid + 1];
    const REAL up     = in[gid - nx];
    const REAL down   = in[gid + nx];
    out[gid] = K(0.2) * (center + left + right + up + down);
}

__kernel void lu_update_2d_heavy_flat(
    __global const REAL* in,
    __global REAL* out,
    const int nx,
    const int ny,
    const int flop_repeat)
{
    const int gid = get_global_id(0);
    const int total = nx * ny;
    if (gid >= total) return;

    const int x = gid % nx;
    const int y = gid / nx;

    if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1) {
        out[gid] = in[gid];
        return;
    }

    const REAL c  = in[gid];
    const REAL l  = in[gid - 1];
    const REAL r  = in[gid + 1];
    const REAL u  = in[gid - nx];
    const REAL d  = in[gid + nx];
    const REAL ul = in[gid - nx - 1];
    const REAL ur = in[gid - nx + 1];
    const REAL dl = in[gid + nx - 1];
    const REAL dr = in[gid + nx + 1];

    const REAL v = K(0.25) * c
                 + K(0.125) * (l + r + u + d)
                 + K(0.0625) * (ul + ur + dl + dr);

    out[gid] = heavy_flops(v, flop_repeat);
}

__kernel void map_heavy_1d(
    __global const REAL* in,
    __global REAL* out,
    const int n,
    const int flop_repeat)
{
    const int gid = get_global_id(0);
    if (gid >= n) return;

    const REAL v = in[gid] * K(1.234567) + K(0.765432);
    out[gid] = heavy_flops(v, flop_repeat);
}
