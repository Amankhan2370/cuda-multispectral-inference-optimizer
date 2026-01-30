/**
 * fused_preprocess.cu
 * Stub for a fused CUDA kernel: normalization + spectral weighting + layout transform.
 *
 * Kernel fusion benefits:
 * - Fewer global memory round-trips: read input once, write output once.
 * - Better arithmetic intensity: reuse in registers across normalization,
 *   weighting, and layout steps instead of writing then re-reading.
 * - Lower launch overhead: one kernel instead of three.
 * - Smaller working set: no intermediate device buffers for normalized
 *   or weighted NHWC; only final NCHW output.
 *
 * No actual optimization logic required yet; structure only.
 */

#include <cuda_runtime.h>
#include "../include/cuda_utils.h"

/**
 * Fused preprocess: NHWC input -> normalize per channel -> optional spectral
 * weighting -> NCHW output. Stub: copy NHWC -> NCHW only.
 */
__global__ void fused_preprocess_kernel(const float* __restrict__ in,
                                        float* __restrict__ out,
                                        const float* __restrict__ mean,
                                        const float* __restrict__ inv_std,
                                        const float* __restrict__ weight,
                                        int N, int H, int W, int C) {
  (void)mean;
  (void)inv_std;
  (void)weight;

  int n = blockIdx.z;
  int h = blockIdx.y * blockDim.y + threadIdx.y;
  int w = blockIdx.x * blockDim.x + threadIdx.x;

  if (n >= N || h >= H || w >= W) return;

  for (int c = 0; c < C; ++c) {
    int in_idx = n * (H * W * C) + h * (W * C) + w * C + c;
    int out_idx = n * (C * H * W) + c * (H * W) + h * W + w;
    out[out_idx] = in[in_idx];
  }
}

extern "C" void launch_fused_preprocess(const float* d_in, float* d_out,
                                        const float* d_mean,
                                        const float* d_inv_std,
                                        const float* d_weight,
                                        int N, int H, int W, int C) {
  dim3 block(16, 16);
  dim3 grid((W + 15) / 16, (H + 15) / 16, N);
  fused_preprocess_kernel<<<grid, block>>>(d_in, d_out, d_mean, d_inv_std,
                                           d_weight, N, H, W, C);
  CUDA_CHECK_LAST();
}
