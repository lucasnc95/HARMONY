// Three kernels used in the same step (see main.cpp).

// g[i] = (u[i+1] - u[i-1]) / 2; zero at both ends of the domain.
__kernel void gradient(__global const float* u, __global float* g, const int n)
{
    const int i = (int)get_global_id(0);
    if (i >= n) return;
    g[i] = (i == 0 || i == n - 1) ? 0.0f : 0.5f * (u[i + 1] - u[i - 1]);
}

// v[i] = u[i] + c * (g[i+1] - g[i-1]) / 2: one diffusion step written as the
// divergence of the gradient. The ends of the domain are kept fixed.
__kernel void update(__global const float* u, __global const float* g, __global float* v,
                     const float c, const int n)
{
    const int i = (int)get_global_id(0);
    if (i >= n) return;
    v[i] = (i == 0 || i == n - 1) ? u[i] : u[i] + c * 0.5f * (g[i + 1] - g[i - 1]);
}

// e[i] = u[i]^2 (does not depend on the two kernels above).
__kernel void energy(__global const float* u, __global float* e, const int n)
{
    const int i = (int)get_global_id(0);
    if (i >= n) return;
    e[i] = u[i] * u[i];
}
