<p align="center">
  <strong>CUDA Multispectral Inference Optimizer</strong>
</p>
<p align="center">
  Inference-optimized CUDA kernels for multispectral preprocess, layout transforms, and benchmarking.
</p>

<p align="center">
  <a href="https://developer.nvidia.com/cuda-toolkit"><img src="https://img.shields.io/badge/CUDA-12.x-76B900?style=flat-square&logo=nvidia&logoColor=white" alt="CUDA 12.x"/></a>
  <a href="https://isocpp.org"><img src="https://img.shields.io/badge/C%2B%2B-17-00599C?style=flat-square&logo=cplusplus&logoColor=white" alt="C++17"/></a>
  <a href="https://cmake.org"><img src="https://img.shields.io/badge/CMake-3.18%2B-064F8C?style=flat-square&logo=cmake&logoColor=white" alt="CMake 3.18+"/></a>
  <a href="https://www.python.org"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.10+"/></a>
  <a href="https://pytorch.org"><img src="https://img.shields.io/badge/PyTorch-baseline-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch"/></a>
  <a href="#license"><img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square" alt="License MIT"/></a>
</p>

---

## Table of contents

- [Overview](#overview)
- [Problem](#problem)
- [Components](#components)
- [Architecture](#architecture)
- [Tools and requirements](#tools-and-requirements)
- [Repository structure](#repository-structure)
- [Quick start](#quick-start)
- [Build](#build)
- [Benchmarks](#benchmarks)
- [Profiling](#profiling)
- [Security and credentials](#security-and-credentials)
- [License](#license)

---

## Overview

Multispectral and hyperspectral inputs (multiple bands per pixel) are common in remote sensing and specialized vision. Inference pipelines typically normalize per channel and convert sensor layouts (e.g. NHWC) to framework layouts (NCHW). Multiple kernel launches and temporary buffers can dominate latency. This repository provides minimal, compilable CUDA scaffolding: per-channel normalization, NHWC→NCHW layout transform, and a fused preprocess stub. Inference-only; no training code.

---

## Problem

| Factor | Impact |
|--------|--------|
| **Memory** | Large tensors (N×C×H×W). Multiple passes increase device traffic and intermediate buffers; bandwidth often limits throughput. |
| **Latency** | Many small kernels add launch overhead and extra global memory round-trips. Preprocess can be a large fraction of total inference time; fusion reduces launches and improves arithmetic intensity. |

---

## Components

| Component | Description |
|-----------|-------------|
| `spectral_normalize.cu` | Per-channel normalization; coalescing-friendly indexing, grid-stride loops. |
| `layout_transform.cu` | NHWC → NCHW; no hardcoded dimensions; suitable for inference graphs. |
| `fused_preprocess.cu` | Stub: normalize + optional spectral weighting + layout in one kernel (structure only). |
| `pytorch_pipeline.py` | PyTorch inference-only baseline (normalize + permute) for correctness and latency reference. |
| `benchmark.cpp` | C++ driver; CUDA events; warmup + repeat; reports ms/iter for each kernel path. |
| `profiling/analysis.md` | Template for Nsight Systems / Compute results and screenshots. |

---

## Architecture

**Pipeline (conceptual)**

```text
  Input (NHWC)  →  Normalize (per channel)  →  [Optional: spectral weight]  →  Layout (NCHW)  →  Output
```

**Baseline (PyTorch)**  
Dummy multispectral tensor → per-channel `mean`/`var` normalize → permute for layout. Used for correctness and latency baseline.

**Optimized path (CUDA)**  
Three kernel entry points: `spectral_normalize`, `layout_transform`, `fused_preprocess`. The C++ benchmark links all three and times each path. Replace the fused stub with a real implementation and compare against baseline and separate kernels.

**CUDA techniques used**

| Technique | Purpose |
|-----------|---------|
| Grid-stride loops | Normalization kernel scales with problem size; keeps occupancy reasonable. |
| Coalesced global access | Warp threads map to contiguous channel/spatial indices where applicable. |
| Dimension-agnostic kernels | N, C, H, W passed as arguments; one implementation for arbitrary sizes. |
| Kernel fusion (stub) | Single kernel for normalize + weighting + layout to reduce round-trips and launch overhead. |
| CUDA events | Host-synchronized timing in the benchmark for kernel execution. |

---

## Tools and requirements

| Tool | Version | Purpose |
|------|---------|---------|
| CUDA Toolkit | 12.x | Kernels, runtime, CUDA events |
| CMake | 3.18+ | Configure and build C++/CUDA |
| C++ compiler | C++17 | Benchmark driver and host code |
| Python | 3.10+ | Optional; PyTorch baseline |
| PyTorch | (CUDA build optional) | Baseline pipeline |
| Nsight Systems | — | Timeline profiling |
| Nsight Compute | — | Per-kernel analysis |

No other dependencies for the C++/CUDA build.

---

## Repository structure

```text
cuda-multispectral-inference-optimizer/
├── kernels/
│   ├── spectral_normalize.cu
│   ├── layout_transform.cu
│   └── fused_preprocess.cu
├── baseline/
│   └── pytorch_pipeline.py
├── include/
│   └── cuda_utils.h
├── benchmarks/
│   └── benchmark.cpp
├── profiling/
│   └── analysis.md
├── CMakeLists.txt
└── README.md
```

---

## Quick start

```bash
git clone https://github.com/Amankhan2370/cuda-multispectral-inference-optimizer.git
cd cuda-multispectral-inference-optimizer
mkdir build && cd build
cmake .. && cmake --build .
./benchmark 4 8 224 224
```

---

## Build

From the repository root:

```bash
mkdir build && cd build
cmake ..
cmake --build .
```

Artifact: `build/benchmark` (links the three kernel libraries).

---

## Benchmarks

| Command | Description |
|---------|-------------|
| `./benchmark` | Default shape (N=4, C=8, H=224, W=224). |
| `./benchmark N C H W` | Custom batch, channels, height, width. |

Warmup then timed iterations for each kernel path; output is ms/iter. Requires a CUDA-capable GPU.

**PyTorch baseline (optional)**

```bash
python baseline/pytorch_pipeline.py
```

Requires PyTorch; use a CUDA build for GPU timing.

---

## Profiling

| Tool | Use |
|------|-----|
| **Nsight Systems** | Timeline: kernel count, overlap, host/device sync. |
| **Nsight Compute** | Per-kernel: occupancy, memory throughput, warp utilization. |

1. Capture a trace of the benchmark (or your harness).
2. Identify dominant kernels and unnecessary syncs or small launches.
3. Compare baseline vs separate kernels vs fused kernel once implemented.
4. Document results in `profiling/analysis.md` (placeholders provided; do not invent numbers).

---

## Security and credentials

No credentials, API keys, or secrets are included. Placeholders for user-supplied values are marked in code, e.g.:

- `# add your api key here`
- `# add your model or weights here if needed`

Do not commit real keys or paths; keep them outside the repository.

---

## License

MIT.
