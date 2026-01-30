cmake_minimum_required(VERSION 3.18)
project(cuda_multispectral_inference LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

add_library(spectral_normalize kernels/spectral_normalize.cu)
target_include_directories(spectral_normalize PRIVATE ${CMAKE_SOURCE_DIR}/include)
set_target_properties(spectral_normalize PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_library(layout_transform kernels/layout_transform.cu)
target_include_directories(layout_transform PRIVATE ${CMAKE_SOURCE_DIR}/include)
set_target_properties(layout_transform PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_library(fused_preprocess kernels/fused_preprocess.cu)
target_include_directories(fused_preprocess PRIVATE ${CMAKE_SOURCE_DIR}/include)
set_target_properties(fused_preprocess PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(benchmark benchmarks/benchmark.cpp)
target_link_libraries(benchmark PRIVATE spectral_normalize layout_transform fused_preprocess CUDA::cudart)
target_include_directories(benchmark PRIVATE ${CMAKE_SOURCE_DIR}/include)
