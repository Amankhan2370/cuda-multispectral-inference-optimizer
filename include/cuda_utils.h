/**
 * cuda_utils.h
 * Common CUDA error-checking macros and utilities for inference pipelines.
 * No external dependencies beyond CUDA runtime.
 */

#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                       \
    }                                                                          \
  } while (0)

#define CUDA_CHECK_LAST() CUDA_CHECK(cudaGetLastError())

#endif /* CUDA_UTILS_H */
