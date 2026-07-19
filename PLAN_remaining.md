# Remaining Work — kornia-rs GSoC GPU acceleration

**Context:** Week 3 plan items (bicubic block_dim, Lanczos, warp-perspective,
remap, smem tiling) are merged or in flight. INTER_AREA was dropped (no
performance improvement). This plan covers what is still missing from the
original GSoC proposal, in the order to implement.

---

## 1. VPI benchmarks [x]

**Branch:** `feat/vpi-benchmarks`
**Effort:** 2–3 days

NVIDIA VPI (Vision Programming Interface) is the key comparison target for
Jetson / embedded deployments. The proposal required VPI comparison alongside
OpenCV. Currently we only have the OpenCV correctness scripts — no VPI numbers
at all.

**What to build:**
- `crates/kornia-imgproc/examples/check_correctness_vpi.py` — same structure
  as `check_correctness_cuda.py`: dump GPU output as JSON, compare pixel-by-pixel
  against VPI reference
- Cover: resize (nearest, bilinear, bicubic), warp_affine (bilinear, nearest),
  warp_perspective (bilinear)
- `crates/kornia-imgproc/examples/bench_vpi.py` — timing comparison:
  kornia-rs GPU vs VPI CUDA backend at 1920×1080 and 3840×2160
  Metrics: latency (ms), throughput (images/s), speedup vs VPI

**Install VPI (x86 with CUDA):**
```
sudo apt install libnvvpi3 vpi3-dev python3-vpi3
```

**Note:** VPI may not be available in CI — gate the script behind a runtime
check (`try: import vpi3 except ImportError: skip`) and note in the PR that
results are from local hardware.

---

## 2. Managed memory / Unified memory [x]

**Branch:** `feat/managed-memory`
**Effort:** 2–3 days

`MemoryDomain::Unified` exists in the type system but the actual allocation
path is not implemented. On Jetson (integrated GPU), CPU and GPU share
physical memory — `cudaMallocManaged` allocates memory accessible from both
sides with no PCIe copy. On discrete GPUs (GTX/RTX), managed memory still
works but falls back through the PCIe interconnect.

**What is missing:**
1. `CudaUnifiedAllocator` — wraps `cuMemAllocManaged` / `cuMemFree`, sets
   `MemoryDomain::Unified` on the resulting `TensorStorage`
2. `stream.clone_unified()` or `alloc_unified()` helper on the stream
3. Launcher path: detect `MemoryDomain::Unified` and skip the H2D copy step
   (the data is already accessible to the GPU)
4. Test: allocate unified tensor, run resize/warp kernel directly without
   explicit H2D, verify output matches the copy path
5. Benchmark: compare unified-memory path vs explicit H2D copy path at
   1920×1080 on GTX 1650 — report whether it helps or hurts on discrete GPU
   (on x86 it typically hurts due to demand paging; on Jetson it eliminates
   the copy entirely)

**Files:** `crates/kornia-tensor/src/cuda/`, `crates/kornia-imgproc/src/cuda/`

---

## 3. Benchmark suite with transfer breakdown [ ]

**Branch:** `feat/benchmark-suite`
**Effort:** 2–3 days

The proposal's final deliverable #4 is a reproducible CPU vs GPU benchmark
with H2D / kernel / D2H breakdown. Currently there are ad-hoc bench files
but no unified, reproducible suite with the format the proposal specified.

**Three layers (from proposal §3.7.2):**

**Layer A — kernel microbenchmarks** (sizes: 1M, 10M, 100M elements)
- Currently not implemented for tensor ops (elementwise, reduction)
- Metrics: device kernel time, total time, effective throughput

**Layer B — tensor op benchmarks** (to be done after item 4 below)
- Will benchmark the allocator-dispatched tensor ops
- Skip until dispatch traits are implemented

**Layer C — image operation benchmarks** ← implement now
- resize (bilinear, nearest, bicubic, lanczos) at 1920×1080 and 3840×2160
- warp_affine (bilinear) at 1920×1080 and 3840×2160
- warp_perspective (bilinear) at 1920×1080 and 3840×2160
- remap (bilinear) at 1920×1080 and 3840×2160
- Metrics per operation: CPU baseline (ms), H2D (ms), kernel (ms), D2H (ms),
  total GPU round-trip (ms), speedup over CPU (kernel-only and round-trip)

**Output format:** Markdown table, appended to a `benchmarks.md` file at repo
root with hardware info header (GPU name, driver, CUDA version, date).

**How to split H2D / kernel / D2H:**
Use `CudaEvent` record before H2D, after H2D (= before kernel), after kernel
(= before D2H), after D2H. Report each segment separately.

---

## 4. Tensor-level GPU dispatch traits [ ]

**Branch:** `feat/tensor-gpu-dispatch`
**Effort:** 3–4 days

The proposal's core architecture (§3.2) was allocator-dispatched tensor ops
so the same `tensor.unary(Abs)` call routes to CPU or GPU based on the
tensor's allocator/domain. This is the proposal's midterm deliverable and is
not yet implemented.

**What to implement:**

Traits (in `kornia-tensor`):
```rust
pub trait TensorUnaryKernel<T, const N: usize>: TensorAllocator {
    fn unary(input: &Tensor<T, N, Self>, output: &mut Tensor<T, N, Self>, op: UnaryOp)
        -> Result<(), TensorError>;
}

pub trait TensorBinaryKernel<T, const N: usize>: TensorAllocator {
    fn binary(a: &Tensor<T, N, Self>, b: &Tensor<T, N, Self>,
              out: &mut Tensor<T, N, Self>, op: BinaryOp) -> Result<(), TensorError>;
}

pub trait TensorReduceKernel<T, const N: usize>: TensorAllocator {
    fn reduce(input: &Tensor<T, N, Self>, dim: usize, op: ReduceOp)
        -> Result<Tensor<T, N, Self>, TensorError>;
}
```

Operations to cover:
- `UnaryOp`: `Abs`, `Relu`, `Clamp { min, max }`, `Neg`
- `BinaryOp`: `Add`, `Sub`, `Mul`, `Div`, `Min`, `Max`
- `ReduceOp`: `Sum`, `Mean`

GPU kernel implementation: one CUDA C kernel per op group (unary, binary,
reduce), JIT-compiled via NVRTC, one thread per element.

CPU implementation: delegates to existing slice-based loops (existing code,
just wrapped in the trait).

**Parity tests:** GPU vs CPU for every op at sizes 1K, 1M. Max error = 0 for
integer inputs, ≤ 1 ULP for f32.

---

## 5. Imgproc backend dispatch traits [ ]

**Branch:** `feat/imgproc-backend-dispatch`
**Effort:** 3–4 days
**Depends on:** item 4 (dispatch trait pattern established)

The proposal's §3.3 goal: the public `resize()`, `warp_affine()`,
`warp_perspective()` functions become allocator-agnostic dispatch points.
Currently the caller must explicitly call `launch_resize_bilinear_cuda()`.
The proposal wanted:

```rust
// Same call works for CPU and GPU tensors:
resize(&src_cpu, &mut dst_cpu, InterpolationMode::Bilinear)?;  // → CPU path
resize(&src_gpu, &mut dst_gpu, InterpolationMode::Bilinear)?;  // → GPU path
```

**Traits to add** (in `kornia-imgproc`):
```rust
pub trait ResizeKernel<const C: usize, A: ImageAllocator> {
    fn resize_f32(src: &Image<f32, C, A>, dst: &mut Image<f32, C, A>,
                  interpolation: InterpolationMode) -> Result<(), ImageError>;
}

pub trait WarpAffineKernel<const C: usize, A: ImageAllocator> { ... }
pub trait WarpPerspectiveKernel<const C: usize, A: ImageAllocator> { ... }
pub trait ColorConvertKernel<A: ImageAllocator> {
    fn rgb_to_gray_u8(...);
    fn gray_to_rgb_u8(...);
}
```

CPU impl: existing code behind `CpuAllocator` specialization.
GPU impl: dispatches to the existing CUDA launchers behind `CudaAllocator`
specialization.

Public entry points (`resize`, `warp_affine`, etc.) detect the allocator via
the domain tag and call the right impl — no user-facing API change.

**No silent CPU fallback** for GPU tensors — return `UnsupportedOpForBackend`
error instead.

---

## Order to implement

| # | Item | Why this order |
|---|------|----------------|
| 1 | VPI benchmarks | Standalone Python work, no code dependencies |
| 2 | Managed memory | Completes a missing MemoryDomain path |
| 3 | Benchmark suite (Layer C) | Can be done without dispatch traits; shows numbers |
| 4 | Tensor GPU dispatch traits | Core architectural deliverable from proposal |
| 5 | Imgproc backend dispatch | Depends on 4 for the pattern; final polish |

---

## PR workflow

After submitting each PR, run `/review` then `/simplify` as a self-review
pass before asking Edgar for review (reduces round-trips).
