# CUDA Multispectral Inference Optimizer

This repository contains CUDA kernels and tooling for optimizing multispectral image preprocess and layout transforms in inference pipelines. The focus is on latency and memory efficiency at deployment time: per-channel normalization, NHWC–NCHW layout conversion, and a fused preprocess kernel that combines these steps. No training code is included. This is inference-only and reflects the kind of work you see in production AI infrastructure (preprocess, layout, and benchmarking on GPU).

---

## Overview

Multispectral and hyperspectral inputs (e.g. multiple bands per pixel) are common in remote sensing and specialized vision pipelines. Before running a model, data is usually normalized per channel and converted from sensor-friendly layouts (e.g. NHWC) to framework-friendly ones (NCHW). Doing this with many small kernel launches and temporary buffers can dominate latency. This project provides minimal, compilable scaffolding for custom CUDA kernels that perform normalization and layout transform, plus a stub for a single fused kernel, so you can measure and optimize preprocess cost on your own hardware.

---

## Why multispectral inference is hard

**Memory:** Multispectral tensors are large (batch × channels × height × width). Multiple passes (normalize, then layout) double or triple device memory traffic and require intermediate buffers. Bandwidth, not compute, often limits throughput.

**Latency:** Many small kernels mean more launch overhead and more round-trips to global memory. For real-time or low-latency serving, preprocess can be a significant fraction of total inference time. Fusing steps into fewer kernels reduces launches and improves arithmetic intensity.

---

## What this project demonstrates

- **Per-channel normalization** in a single CUDA kernel with coalescing-friendly indexing and grid-stride loops.
- **NHWC → NCHW layout transform** as a dedicated kernel, with no hardcoded dimensions, suitable for plugging into inference graphs.
- **Fused preprocess stub** that conceptually combines normalization, optional spectral weighting, and layout transform in one kernel (structure only; optimization left to you).
- **Baseline comparison** via a PyTorch inference-only script that runs the same operations with torch ops so you can compare latency.
- **C++ benchmark driver** that times the CUDA kernels with CUDA events and prints placeholder metrics.
- **Profiling workflow** documented in Markdown (Nsight Systems / Compute) with placeholders for your own screenshots and numbers.

---

## Architecture overview

**Baseline:** PyTorch on GPU (or CPU): generate dummy multispectral tensor, normalize per channel with `mean`/`var`, optionally permute for layout. Good for sanity-checking correctness and getting a latency reference.

**Optimized path:** Custom CUDA kernels: `spectral_normalize.cu` (normalize), `layout_transform.cu` (NHWC→NCHW), and `fused_preprocess.cu` (stub for a single fused kernel). The C++ benchmark links these and measures them with CUDA events. You can replace the stub with a real fused implementation and compare against the baseline and the separate kernels.

---

## CUDA techniques used

- **Grid-stride loops** in the normalization kernel so that total thread count is independent of problem size and occupancy stays reasonable.
- **Coalesced global memory access** by design: threads in a warp map to contiguous channel/spatial indices where applicable so that loads/stores are coalesced.
- **No hardcoded dimensions** in the layout or fused kernels; N, C, H, W are passed as kernel arguments so the same code works for different input sizes.
- **Kernel fusion (stub)** to illustrate a single kernel that would do normalize + optional weighting + layout in one pass, reducing global memory round-trips and launch overhead.
- **CUDA events** in the benchmark for host-synchronized timing of kernel execution.

---

## Repository structure

```
cuda-multispectral-inference-optimizer/
├── kernels/
│   ├── spectral_normalize.cu   # Per-channel normalization kernel
│   ├── layout_transform.cu     # NHWC → NCHW
│   └── fused_preprocess.cu     # Fused preprocess stub
├── baseline/
│   └── pytorch_pipeline.py      # PyTorch inference-only baseline
├── include/
│   └── cuda_utils.h             # CUDA error-checking macros
├── benchmarks/
│   └── benchmark.cpp           # C++ CUDA benchmark driver
├── profiling/
│   └── analysis.md              # Profiling notes and placeholders
├── CMakeLists.txt
└── README.md
```

---

## How to build

Requirements: CUDA toolkit, CMake 3.18+, and a C++17-capable compiler. No other dependencies.

```bash
cd cuda-multispectral-inference-optimizer
mkdir build && cd build
cmake ..
cmake --build .
```

The executable `benchmark` is built in the build directory and links the three kernel libraries.

---

## How to run benchmarks

From the build directory:

```bash
./benchmark
```

Optional arguments: `N C H W` (batch, channels, height, width). Example:

```bash
./benchmark 4 8 224 224
```

The program runs warmup iterations, then timed iterations for each of the three kernel paths and prints placeholder timing (ms/iter). Run on a machine with a CUDA-capable GPU; results depend on your driver and GPU.

PyTorch baseline (for comparison):

```bash
python baseline/pytorch_pipeline.py
```

Requires PyTorch and a CUDA-enabled build if you want GPU timing.

---

## Profiling workflow

Use **NVIDIA Nsight Systems** for timeline profiling (kernel launch count, overlap, host/device sync). Use **Nsight Compute** for per-kernel metrics (occupancy, memory throughput, warp execution).

1. Capture a trace that includes the benchmark runs (or your own harness).
2. Identify the dominant kernels and any unnecessary syncs or small launches.
3. Compare baseline (PyTorch or separate kernels) vs the fused kernel once implemented.
4. Fill in `profiling/analysis.md` with screenshots and numbers from your runs. Do not invent results; the template uses placeholders like "Insert Nsight screenshot here".

---

## Security and credentials

This project does not contain credentials, API keys, or secrets. Any placeholders that might later hold paths or keys are marked in code with comments such as:

- `# add your api key here`
- `# add your model or weights here if needed`

Do not commit real keys or paths. Keep configuration and secrets outside the repository.

---

## Who this project is for

Senior engineers and ML infrastructure engineers who work on inference pipelines, GPU performance, or multispectral/hyperspectral deployment. The code is minimal on purpose: it compiles, runs, and illustrates structure and intent. You are expected to tune block/grid sizes, add real fused logic, and plug in your own data paths and model interfaces. It is not a training codebase and does not implement learning; it is focused on inference, CUDA kernels, benchmarking, and profiling.
