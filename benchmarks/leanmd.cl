__kernel void leanmd_pair_force(
    __global const float* pos_x,
    __global const float* pos_y,
    __global const float* pos_z,
    __global float* force_out,
    const int N,
    const int cutoff
) {
    int i = get_global_id(0);
    if (i >= N) return;
    float xi = pos_x[i];
    float yi = pos_y[i];
    float zi = pos_z[i];
    float f_total = 0.0f;
    int start_j = (i > cutoff) ? (i - cutoff) : 0;
    int end_j = (i + cutoff < N) ? (i + cutoff) : (N - 1);
    for (int j = start_j; j <= end_j; ++j) {
        if (i == j) continue;
        float dx = xi - pos_x[j];
        float dy = yi - pos_y[j];
        float dz = zi - pos_z[j];
        float r2 = dx*dx + dy*dy + dz*dz + 1.0e-6f;
        float r2inv = 1.0f / r2;
        float r6inv = r2inv * r2inv * r2inv;
        f_total += r2inv * r6inv * (2.0f * r6inv - 1.0f);
    }
    force_out[i] = f_total;
}
