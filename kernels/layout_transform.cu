/**
 * layout_transform.cu
 * NHWC -> NCHW layout transform kernel for inference.
 *
 * Why this matters for inference: many frameworks and backends (e.g. TensorRT,
 * ONNX, cuDNN) expect NCHW. Camera/sensor pipelines often produce NHWC.
 * A single fused transpose kernel avoids extra round-trips and buffers.
 *
 * No hardcoded dimensions; all passed as kernel arguments.
 */

#include <cuda_runtime.h>
#include "../include/cuda_utils.h"

/**
 * Copy with layout change: in [N,H,W,C] -> out [N,C,H,W].
 * Each thread writes one output element; coalescing is best when
 * threads are assigned to contiguous N,C,H,W indices (e.g. linearized by W).
 */
__global__ void nhwc_to_nchw_kernel(const float* __restrict__ in,
                                    float* __restrict__ out,
                                    int N, int H, int W, int C) {
  int n = blockIdx.z;
  int h = blockIdx.y * blockDim.y + threadIdx.y;
  int w = blockIdx.x * blockDim.x + threadIdx.x;

  if (n >= N || h >= H || w >= W) return;

  int in_base = n * (H * W * C) + h * (W * C) + w * C;
  int out_base = n * (C * H * W) + h * W + w;

  for (int c = 0; c < C; ++c) {
    out[out_base + c * (H * W)] = in[in_base + c];
  }
}

extern "C" void launch_nhwc_to_nchw(const float* d_in, float* d_out,
                                    int N, int H, int W, int C) {
  dim3 block(16, 16);
  dim3 grid((W + 15) / 16, (H + 15) / 16, N);
  nhwc_to_nchw_kernel<<<grid, block>>>(d_in, d_out, N, H, W, C);
  CUDA_CHECK_LAST();
}
