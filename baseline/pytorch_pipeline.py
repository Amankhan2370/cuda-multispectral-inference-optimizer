"""
PyTorch inference-only baseline for multispectral preprocess.
Generates dummy multispectral tensor, applies normalization and layout
transform with torch ops, measures latency. No training.
# add your model or weights here if needed
"""

import time
import torch

def run_baseline(batch_size: int = 4, channels: int = 8, height: int = 224, width: int = 224):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Dummy multispectral input [N, C, H, W]
    x = torch.randn(batch_size, channels, height, width, device=device, dtype=torch.float32)

    # Per-channel normalization (inference path: use fixed stats or precomputed)
    mean = x.mean(dim=(0, 2, 3), keepdim=True)
    var = x.var(dim=(0, 2, 3), keepdim=True) + 1e-5
    x_norm = (x - mean) * (var ** -0.5)

    # Layout: NCHW -> NHWC (for comparison) then back to NCHW
    x_nhwc = x_norm.permute(0, 2, 3, 1).contiguous()
    x_back = x_nhwc.permute(0, 3, 1, 2).contiguous()

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(100):
        m = x.mean(dim=(0, 2, 3), keepdim=True)
        v = x.var(dim=(0, 2, 3), keepdim=True) + 1e-5
        out = (x - m) * (v ** -0.5)
        out = out.permute(0, 2, 3, 1).contiguous().permute(0, 3, 1, 2).contiguous()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    print(f"Baseline (PyTorch) 100 iters: {elapsed*1000:.2f} ms total, {elapsed*10:.2f} ms/iter")
    return x_back

if __name__ == "__main__":
    run_baseline()
