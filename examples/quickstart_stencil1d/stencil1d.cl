// 3-point average: out[i] = (in[i-1] + in[i] + in[i+1]) / 3.
// The first and last elements are copied unchanged.
__kernel void Stencil1D(
    __global const float* in,
    __global float* out,
    const int n)
{
    const int gid = (int)get_global_id(0);
    if (gid >= n) {
        return;
    }

    if (gid == 0 || gid == n - 1) {
        out[gid] = in[gid];
        return;
    }

    out[gid] = (in[gid - 1] + in[gid] + in[gid + 1]) / 3.0f;
}
