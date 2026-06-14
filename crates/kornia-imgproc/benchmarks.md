# kornia-imgproc GPU benchmarks

## How to run

```sh
cargo run --example bench_gpu_color --features gpu-cubecl --release
```

## Run history

Results are appended newest-first. Each row pins the commit hash so
regressions can be bisected without re-reading the full report.

| Date | Commit | Branch | 512×512 GPU ms | 1024×1024 GPU ms | 1920×1080 GPU ms | 3840×2160 GPU ms | GPU vs CPU (1080p) |
|------|--------|--------|---------------:|-----------------:|-----------------:|-----------------:|-------------------:|
| — | — | — | — | — | — | — | — |

*No run recorded yet. See "Add a result row" below.*

---

## Add a result row

Run the benchmark on your machine and append a row in this format:

```
| 2026-XX-XX | `<git rev-parse --short HEAD>` | `<branch>` | X.XXX | X.XXX | X.XXX | X.XXX | X.X× |
```

Include the following fields in the run-info block below so the hardware
context is captured alongside the numbers.

---

## Run info (to be filled)

| Field | Value |
|-------|-------|
| Date | — |
| Commit | — |
| Branch | — |
| GPU | NVIDIA GeForce GTX 1650 (Turing, 4 GB GDDR6, 192 GB/s peak BW) |
| Driver | — (`nvidia-smi` → Driver Version row) |
| CUDA toolkit | 12.4 (`nvcc --version`) |
| OS | Ubuntu 22.04 (x86\_64) |
| Rust | — (`rustc --version`) |
| Warmup iters | 20 |
| Timed iters | 200 |
| Metric | pure kernel time (no H↔D transfers); sync via one readback after all iters |

### Note on the CUDA environment

The development machine (GTX 1650, nvcc 12.4) has a kernel driver / user-space
library version mismatch (`nvidia-smi` reports "Driver/library version mismatch:
NVML 580.159") that prevents any CUDA kernel from initialising.  Results will be
added once the driver is reseated or a separate CUDA-capable machine is used.
