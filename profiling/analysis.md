# Profiling analysis

Placeholder document for baseline vs optimized kernel profiling. Fill in with real runs and screenshots.

---

## Baseline profiling

- **Setup**: PyTorch baseline (`baseline/pytorch_pipeline.py`) or separate CUDA kernels (normalize + layout) without fusion.
- **Tool**: Nsight Systems for timeline; Nsight Compute for kernel occupancy and memory throughput.

Insert Nsight screenshot here (timeline of baseline preprocess).

- **Metrics**: kernel launch count, total time in normalization vs layout, memory bandwidth.

---

## Optimized profiling

- **Setup**: Fused preprocess kernel + tuned grid/block and any shared-memory or vectorized loads.
- **Tool**: Same as above; compare kernel duration and memory traffic.

Insert Nsight screenshot here (timeline of fused kernel).

- **Metrics**: single-kernel time, achieved bandwidth, occupancy.

---

## Metrics analyzed

| Metric | Baseline | Optimized | Notes |
|--------|----------|-----------|--------|
| Kernel count | — | — | Fewer launches with fusion |
| Total preprocess time (ms) | — | — | Fill after runs |
| Memory throughput (GB/s) | — | — | From Nsight Compute |
| Occupancy | — | — | From Nsight Compute |

Update this table with actual numbers from your environment. Do not invent results.
