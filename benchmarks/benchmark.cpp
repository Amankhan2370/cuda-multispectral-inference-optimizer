/**
 * C++ CUDA benchmark driver for multispectral inference kernels.
 * Allocates device buffers, launches kernels (stub calls), uses CUDA events
 * for timing, prints placeholder performance metrics.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "../include/cuda_utils.h"

extern "C" void launch_spectral_normalize(const float* d_in, float* d_out,
                                          const float* d_mean,
                                          const float* d_inv_std,
                                          int N, int C, int H, int W);
extern "C" void launch_nhwc_to_nchw(const float* d_in, float* d_out,
                                    int N, int H, int W, int C);
extern "C" void launch_fused_preprocess(const float* d_in, float* d_out,
                                        const float* d_mean,
                                        const float* d_inv_std,
                                        const float* d_weight,
                                        int N, int H, int W, int C);

int main(int argc, char** argv) {
  int N = 4, C = 8, H = 224, W = 224;
  if (argc >= 5) {
    N = atoi(argv[1]);
    C = atoi(argv[2]);
    H = atoi(argv[3]);
    W = atoi(argv[4]);
  }

  size_t nchw_bytes = (size_t)N * C * H * W * sizeof(float);
  size_t nhwc_bytes = (size_t)N * H * W * C * sizeof(float);
  size_t c_bytes   = (size_t)C * sizeof(float);

  float *d_in_nchw = nullptr, *d_in_nhwc = nullptr, *d_out = nullptr;
  float *d_mean = nullptr, *d_inv_std = nullptr, *d_weight = nullptr;

  CUDA_CHECK(cudaMalloc(&d_in_nchw, nchw_bytes));
  CUDA_CHECK(cudaMalloc(&d_in_nhwc, nhwc_bytes));
  CUDA_CHECK(cudaMalloc(&d_out, nchw_bytes));
  CUDA_CHECK(cudaMalloc(&d_mean, c_bytes));
  CUDA_CHECK(cudaMalloc(&d_inv_std, c_bytes));
  CUDA_CHECK(cudaMalloc(&d_weight, c_bytes));
  CUDA_CHECK(cudaMemset(d_in_nchw, 0, nchw_bytes));
  CUDA_CHECK(cudaMemset(d_in_nhwc, 0, nhwc_bytes));

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  const int warmup = 5, repeat = 100;

  /* Spectral normalize: NCHW in/out */
  for (int i = 0; i < warmup; i++)
    launch_spectral_normalize(d_in_nchw, d_out, d_mean, d_inv_std, N, C, H, W);
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < repeat; i++)
    launch_spectral_normalize(d_in_nchw, d_out, d_mean, d_inv_std, N, C, H, W);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms_spectral = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms_spectral, start, stop));
  printf("spectral_normalize: %.3f ms/iter (%d iters)\n", ms_spectral / repeat, repeat);

  /* NHWC -> NCHW */
  for (int i = 0; i < warmup; i++)
    launch_nhwc_to_nchw(d_in_nhwc, d_out, N, H, W, C);
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < repeat; i++)
    launch_nhwc_to_nchw(d_in_nhwc, d_out, N, H, W, C);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms_layout = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms_layout, start, stop));
  printf("layout_transform:   %.3f ms/iter (%d iters)\n", ms_layout / repeat, repeat);

  /* Fused preprocess: NHWC in, NCHW out */
  for (int i = 0; i < warmup; i++)
    launch_fused_preprocess(d_in_nhwc, d_out, d_mean, d_inv_std, d_weight, N, H, W, C);
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < repeat; i++)
    launch_fused_preprocess(d_in_nhwc, d_out, d_mean, d_inv_std, d_weight, N, H, W, C);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms_fused = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms_fused, start, stop));
  printf("fused_preprocess:    %.3f ms/iter (%d iters)\n", ms_fused / repeat, repeat);

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_in_nchw));
  CUDA_CHECK(cudaFree(d_in_nhwc));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFree(d_mean));
  CUDA_CHECK(cudaFree(d_inv_std));
  CUDA_CHECK(cudaFree(d_weight));

  return 0;
}
