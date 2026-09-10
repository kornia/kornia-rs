# cuda-oxide bilinear resize spike

Can a bilinear resize kernel written in **Rust** and compiled to PTX by NVlabs'
[cuda-oxide](https://github.com/NVlabs/cuda-oxide) hold kornia's byte-exactness
contract and match the NVRTC kernel's throughput on Jetson Orin?

**Both answers are yes.** The Rust kernel is bit-identical to the CPU path and
to the shipping CUDA C kernel across every size tested, and it lands within
noise of the CUDA C kernel on all five benchmark cases. Getting there required
three workarounds for backend gaps and one kernel-design fix that a CUDA C
author gets for free from `__restrict__`.

This is a spike. Nothing here is proposed for merge as-is.

## System

| | |
|---|---|
| Date | 2026-09-10 |
| Machine | NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super |
| Kernel / arch | 5.15.148-tegra aarch64, Cortex-A78AE x6, 7.4 GB |
| L4T | # R36 (release), REVISION: 4.3, GCID: 38968081, EABI: aarch64 |
| GPU | nvidia,ga10b (sm_87) |
| CUDA / driver | 12.6.68 / 540.4.0 |
| Power mode | MAXN_SUPER |
| rustc (kornia) | 1.98.0 (88d9e12ae 2026-08-18) |
| rustc (kernel crate) | 1.96.0-nightly (55e86c996 2026-04-02) |
| LLVM (`llc`) | 21.1.8 |
| cuda-oxide | `f819f23` (pins `nightly-2026-04-03`) |
| kornia-rs commit | `1fe3bda` |

`.cargo/config.toml` forces `-Copt-level=2` and `-Ctarget-feature=+fp16,+fhm`
on aarch64, overriding release `opt-level=3`. That applies to the host side of
every number below.

## Setup

The kernel lives in `oxide-kernels/`, a standalone crate (empty `[workspace]`
table) pinning `nightly-2026-04-03`. kornia's own crates stay on stable and
consume the committed PTX through `include_str!`, so nothing in the workspace
needs nightly. Build:

```
cargo oxide build resize_bilinear_oxide --arch sm_87
```

Emits `resize_bilinear_oxide.ptx` next to the source; it is copied to
`crates/kornia-imgproc/src/cuda/resize_bilinear_oxide_sm87.ptx`. `cudarc` loads
it with `Ptx::from_src` + `load_module` + `load_function`; cuda-oxide's PTX
entry name is the plain Rust function name, no mangling. Launchers are in
`crates/kornia-imgproc/src/cuda/resize_oxide.rs`.

cuda-oxide was already bootstrapped on this box (cached backend in
`~/.cargo/cuda-oxide`), and `cargo oxide doctor` passes on aarch64 with CUDA
12.6 once `/usr/local/cuda/bin` is on `PATH`. The README's CUDA 13.0+ / driver
R580+ requirement did not bite for PTX generation on this pinned snapshot.

## Arms

All four launch through `cudarc` with the same `make_config` geometry (32x8
block, `div_ceil` grid), the same `PixelMapping::coeffs`, and the same
`CU_FUNC_CACHE_PREFER_L1` cache config. The only variable is the kernel.

| Arm | Kernel |
|---|---|
| A | `BILINEAR_SRC` as shipped (NVRTC, `__ldg`) |
| B | same CUDA C with `__ldg` stripped to plain `src[i]` |
| C | cuda-oxide, raw-pointer parameters |
| D | cuda-oxide, `&[f32]` / `DisjointSlice<f32>` parameters |

Arm B exists because this cuda-oxide snapshot exposes **no `__ldg` intrinsic and
no inline PTX**. Without B, the missing intrinsic would have been charged to
cuda-oxide's codegen.

## Correctness

PTX gates on `resize_bilinear_oxide_sm87.ptx`:

| Gate | Result |
|---|---|
| `.target` | `sm_87` |
| `.version` | 7.4 (driver JITs it; well under R540's ceiling) |
| `__nv_` libdevice symbols | 0 — stays on the self-contained `llc -> .ptx` path |
| `fma.rn.f32` / `mad.f32` | **0** |

Parity tests in `cuda::resize_oxide::tests`, over 2x down, 2x up, and four
odd/non-dyadic sizes (`127x63->65x33`, `129x97->64x48`, `255x130->133x67`,
`63x129->127x255`):

```
test cuda::resize_oxide::tests::noldg_matches_cpu_bit_exact ... ok
test cuda::resize_oxide::tests::oxide_matches_cpu_bit_exact ... ok
test cuda::resize_oxide::tests::oxide_slice_matches_cpu_bit_exact ... ok
test cuda::resize_oxide::tests::arms_agree_with_shipping_kernel ... ok
```

`assert_eq!` on raw bits, no tolerance. `arms_agree_with_shipping_kernel`
compares each arm against arm A as well as against the CPU: a CPU-only
comparison cannot distinguish "cuda-oxide differs from NVRTC" from "both drifted
from the CPU together".

**The byte-exactness contract came for free.** kornia sets NVRTC
`fmad: Some(false)` deliberately so `ax * dst_x + bx` rounds twice like the CPU
LUT (`kornia-tensor/src/cuda.rs:774-789`). Rust never contracts float
operations implicitly, so the Rust kernel emits `mul.f32` + `add.f32` with no
flag at all. This is a genuine argument for authoring kernels in Rust, not just
parity with CUDA C: the property that CUDA C needs a compiler flag to preserve
is the Rust default. (Note the inverse risk: cuda-oxide `main` enables
contraction by default and needs `--no-fmad` to switch it off. This snapshot
predates that flag and its `llc` invocation passes no `-fp-contract`, so it
behaves as `--fmad=false`. Anyone repeating this on `main` must pass
`--no-fmad`.)

## Throughput

Wall clock, 200 iterations after 50 warmup, queue-all-then-synchronize —
the existing `bench_cuda_resize` methodology. `GB/s` counts one source read plus
one destination write per output pixel.

| case (src->dst) | A nvrtc `__ldg` | B nvrtc no-`__ldg` | C oxide raw ptr | D oxide slices |
|---|---|---|---|---|
| 1024x1024 -> 512x512 | 0.175 | 0.167 | **0.166** | 0.167 |
| 512x512 -> 1024x1024 | **0.197** | 0.198 | 0.199 | 0.221 |
| 1920x1080 -> 960x540 | 0.323 | 0.323 | 0.323 | 0.325 |
| 1920x1080 -> 3840x2160 | **1.526** | 1.530 | 1.536 | 1.720 |
| 3840x2160 -> 1920x1080 | 1.278 | **1.276** | 1.300 | 1.278 |

(ms/iter; lower is better.)

**Arm C is at parity with arm A** — within +-1.7% on every case, and marginally
ahead on the 1024x1024 downscale. Arm D costs 12-13% on the two upscale cases.

Arm order was verified not to be a variable: `KORNIA_ARM_REVERSE=1` runs the
arms back to front and the numbers track the kernel, not the position.

### nsys corroboration, and a trap in it

`nsys stats --report cuda_gpu_kern_sum` initially made arm A look 25-30% slower
than C with 60-100x the variance of the other arms. That is an artifact:
**nsys sees all 250 launches per case, including the 50 warmup launches the
wall-clock harness excludes**, so whichever arm runs first in a case absorbs the
cold-start (fresh 100 MB device buffers, H2D still draining, GPU clocks ramping).
Reversing the arm order moved the penalty to whichever arm was then first, and
arm A, running last, reported a median of 1.582 ms with a standard deviation of
3.6 us against a 16.8 ms maximum for the new first arm.

Taking each kernel from a run where it was not first, on 1920x1080 -> 3840x2160:

| arm | median (ms) | stddev |
|---|---|---|
| A nvrtc `__ldg` | 1.582 | 3.6 us |
| B nvrtc no-`__ldg` | 1.582 | 4.3 us |
| C oxide raw ptr | 1.587 | 89.9 us |
| D oxide slices | 1.809 | 6.7 us |

Consistent with the wall-clock table. Use `KORNIA_ARMS_ONLY=1
KORNIA_ARM_CASE=<idx>` to reproduce; profiling five cases of very different size
into one per-kernel average is not meaningful.

## Instruction mix

Per-kernel, from the emitted PTX; registers from `ptxas -v --gpu-name sm_87 -O3`.

| | A (nvcc, `--fmad=false`) | C (cuda-oxide raw ptr) | D (cuda-oxide slices) |
|---|---|---|---|
| `fma.rn.f32` / `mad.f32` | 0 | 0 | 0 |
| `ld.global.nc.f32` (`__ldg`) | 12 | 0 | 0 |
| `ld.global.b32` | 12 | 12 | 0 (generic `ld.b32`) |
| `min.f32` / `max.f32` | 4 | 0 | 0 |
| `bra` | 1 | 8 | 20 |
| `selp` | 0 | 4 | 4 |
| `cvt.rzi` | 2 | 2 | 2 |
| instructions (approx) | 143 | 134 | 182 |
| registers | 34 | 26 | 30 |

Arm C reaches parity with fewer instructions and fewer registers than arm A,
while carrying 8 branches against A's 1 and no read-only-cache loads. On this
part the kernel is bandwidth-bound, so neither the branches nor the missing
`__ldg` shows up in the time.

## Findings

### 1. `__ldg` is not load-bearing on Orin for this kernel

Arms A and B are indistinguishable (0.323 vs 0.323 ms at 1080p->540p; 1.582 vs
1.582 ms median at 1080p->4K). The `resize.rs` module doc attributes an
`__ldg`-vs-texture benefit to measurements on a GTX 1650, and the L1 preference
was reasoned about for Turing's 32 KB -> 64 KB carveout. That does not transfer
to sm_87. This is a kornia finding independent of cuda-oxide: the CUDA-C kernels
could drop `__ldg` on Orin at no measured cost, which incidentally removes the
only intrinsic the Rust port could not express.

### 2. Raw pointers lose `__restrict__`, and the cost is a 2x regression

The first working Rust kernel ran **0.384 ms against 0.197 ms** at 2x upscale —
95% slower. The cause was visible in the PTX as ordering, not instruction count:

```
ld.global.b32 x4;  st.global.b32;   // channel 0
ld.global.b32 x4;  st.global.b32;   // channel 1
ld.global.b32 x4;  st.global.b32;   // channel 2
```

Rust raw pointers carry no `noalias`, so the compiler cannot hoist a later
channel's loads above an earlier channel's store, and the kernel pays three
serialized memory-latency round trips. The CUDA C twin gets the batching for
free from `const float* __restrict__`. Reading all twelve source values into
locals before the first store fixes it — the arithmetic and its ordering are
untouched, so byte-exactness is unaffected — and arm C goes to parity.

Worth noting the shape of the failure: it hit the *latency*-bound upscale cases
(2x) far harder than the *bandwidth*-bound downscale cases (17%). A downscale-only
benchmark would have under-reported it by an order of magnitude.

If cuda-oxide can attach `noalias` to raw-pointer kernel parameters, that would
remove a sharp edge that costs 2x and is invisible in Rust source review.

### 3. Slices are the wrong choice here, for two compounding reasons

`&[f32]` / `DisjointSlice<f32>` were expected to *win*, since they carry
`readonly`/`noalias` and nvcc turns a bare `const float* __restrict__` into
`ld.global.nc.f32` on its own. Neither materialized:

- **No `ld.global.nc`.** The NVPTX backend did not infer the read-only cache
  load from `readonly noalias`. There is no other route on this snapshot.
- **Address-space inference is lost.** The raw-pointer kernel gets
  `cvta.to.global` and 12 `ld.global.b32`; the slice kernel gets **generic**
  `ld.b32` / `st.b32` with no `cvta` at all.
- Plus a bounds-check branch per access: 20 branches against 8.

Net 12-13% on upscale. The slice kernel is retained in
`resize_oxide.rs` as a measured negative result rather than deleted.

### 4. Three backend gaps, all worked around without losing exactness

Each of these failed the build or the verifier and forced a rewrite:

| Construct | Failure | Workaround |
|---|---|---|
| `Ord::min` on `u32` | `drop of RigidTy(Uint(U32)) is not supported on the device` | explicit `if a < b { a } else { b }` |
| `f32::min` / `f32::max` | LLVM verifier rejects `minimumnum`/`maximumnum` | explicit comparisons |
| `f32::to_int_unchecked::<u32>()` | verifier rejects `float_to_int_unchecked` | plain `as u32` |

All three preserve the *value*, so the byte-exact contract survived. Two cost
instructions: the clamps become branches instead of `min.f32`/`max.f32` (A has 4
such instructions and 1 branch; C has 0 and 8), and Rust's saturating `as u32`
emits a `setp`/`selp` pair around each `cvt.rzi` that CUDA C's cast does not
need. Neither is measurable on this bandwidth-bound kernel, but both would
matter for a compute-bound one — an integer-LUT u8 kernel, or the Lanczos warps
where restructuring transcendentals bought 22-28%.

Of these, `f32::min`/`max` is the one that would block real work: image kernels
clamp constantly, and losing `min.f32`/`max.f32` to a branch is both slower and
divergence-prone.

### 5. `#[kernel]` requires `cuda-host`, so a device-only crate is not possible

`#[kernel]` expands to code referencing `cuda_host`, so a PTX-only crate cannot
depend on `cuda-device` alone. That pulls `cuda-bindings`, `cuda-core`,
`libnvvm-sys` and `nvjitlink-sys` into a crate whose only output is a `.ptx`
file. It builds and it is confined to the kernel crate, but it is why the
CUDA 13 / R580 requirement looks alarming on first read for a consumer that
only wants PTX.

### 6. Committed PTX is a clean integration seam

The split works exactly as suggested: kornia's crates stay on stable and only
`include_str!` the artifact; nothing in the workspace needs nightly, and the
standalone `[workspace]` table means no root-manifest change at all. The PTX is
also reviewable — `.version`/`.target` assertions and a `grep -c '__nv_'` gate
are cheap CI checks, and diffing the PTX across a kernel change shows exactly
what moved.

One caveat: the PTX is pinned to `--arch sm_87`. Shipping for multiple compute
capabilities means committing one artifact per arch, or keeping NVRTC as a
fallback.

## Reproducing

```
# PTX (nightly, in oxide-kernels/)
export PATH=/usr/local/cuda/bin:$PATH
cargo oxide build resize_bilinear_oxide --arch sm_87
cp resize_bilinear_oxide.ptx ../crates/kornia-imgproc/src/cuda/resize_bilinear_oxide_sm87.ptx

# gates
grep -E '^\.version|^\.target' resize_bilinear_oxide.ptx
grep -c '__nv_' resize_bilinear_oxide.ptx                   # must be 0
grep -cE 'fma\.rn\.f32|mad\.f32' resize_bilinear_oxide.ptx  # must be 0

# parity
cargo test --release -p kornia-imgproc --features cuda --lib cuda::resize_oxide -j2

# throughput
cargo run -p kornia-imgproc --example bench_cuda_resize --features cuda --release -j2
KORNIA_ARM_REVERSE=1 ./target/release/examples/bench_cuda_resize   # order control

# per-kernel, one case at a time
KORNIA_ARMS_ONLY=1 KORNIA_ARM_CASE=3 nsys profile -t cuda -o arms ./target/release/examples/bench_cuda_resize
nsys stats --report cuda_gpu_kern_sum arms.nsys-rep
```

Build with `-j2`: 7.4 GB of RAM OOM-kills wider builds of this workspace.
