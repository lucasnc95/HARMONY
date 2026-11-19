__kernel void kneighbor_stencil(
    __global const float* in_data,
    __global float* out_data,
    const int N,
    const int K
) {
    int gid = get_global_id(0);
    if (gid >= N) return;
    float sum = 0.0f;
    int count = 0;
    for (int o = -K; o <= K; ++o) {
        int idx = gid + o;
        if (idx >= 0 && idx < N) {
            sum += in_data[idx];
            count++;
        }
    }
    out_data[gid] = sum / (float)count;
}
