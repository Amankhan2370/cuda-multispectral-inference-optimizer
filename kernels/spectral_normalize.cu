/**
 * spectral_normalize.cu
 * Per-channel normalization kernel for multispectral inference.
 *
 * Memory coalescing: threads in a warp access consecutive elements along
 * the channel dimension so that global loads are coalesced. Grid-stride
 * loop over batch and spatial dims keeps occupancy high.
 *
 * TODO: Tune block size for target GPU (e.g. 256 for Ampere).
 * TODO: Consider shared memory for repeated mean/variance if needed.
 */

#include <cuda_runtime.h>
#include "../include/cuda_utils.h"

/**
 * Per-channel normalization (e.g. (x - mean) / sqrt(var + eps)).
 * Assumes input layout [N, C, H, W]; operates over C with stats
 * computed from N,H,W. Output same shape as input.
 *
 * Coalescing intent: threads in block iterate over contiguous channel
 * indices so that adjacent threads access adjacent addresses.
 */
__global__ void spectral_normalize_kernel(const float* __restrict__ in,
                                          float* __restrict__ out,
                                          const float* __restrict__ mean,
                                          const float* __restrict__ inv_std,
                                          int N, int C, int H, int W) {
  int total = N * C * H * W;
  int stride = blockDim.x * gridDim.x;

  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += stride) {
    int c = (idx / (H * W)) % C;
    float x = in[idx];
    out[idx] = (x - mean[c]) * inv_std[c];
  }
}

extern "C" void launch_spectral_normalize(const float* d_in, float* d_out,
                                          const float* d_mean,
                                          const float* d_inv_std,
                                          int N, int C, int H, int W) {
  int total = N * C * H * W;
  int block = 256;
  int grid = (total + block - 1) / block;
  spectral_normalize_kernel<<<grid, block>>>(d_in, d_out, d_mean, d_inv_std,
                                             N, C, H, W);
  CUDA_CHECK_LAST();
}
