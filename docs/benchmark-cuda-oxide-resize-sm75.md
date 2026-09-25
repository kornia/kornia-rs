# cuda-oxide resize sweep: sm_75 (Turing / GTX 1650)

Follow-up to [benchmark-cuda-oxide-resize.md](benchmark-cuda-oxide-resize.md), which
covered bilinear resize on sm_87 (Jetson Orin Nano).  This report extends that
work to **Turing (sm_75)** and adds a **nearest-neighbour** sweep (arms E–F).

Both byte-exactness and throughput parity hold.  The slice regression Edgar
measured on Orin (12–13 % upscale penalty for arm D) **does not appear on
Turing** — the only qualitative difference between the two machines.

## System

| | |
|---|---|
| Date | 2026-09-26 |
| Machine | x86\_64, NVIDIA GeForce GTX 1650 |
| GPU arch | sm\_75 (Turing), 128 GB/s GDDR5 peak |
| CUDA / driver | 12.4 (Build cuda\_12.4.r12.4/compiler.34097967\_0) |
| rustc (kornia stable) | 1.92.0 (ded5c06cf 2025-12-08) |
| rustc (kernel crate) | 1.96.0-nightly (55e86c996 2026-04-02) |
| LLVM (`llc`) | 14.0.0 |
| cuda-oxide | `c8b3103` (pins `nightly-2026-04-03`; CUDA 12.4-compatible snapshot) |
| kornia-rs commit | `d0c9f3b6` |

**CUDA version note.** `cuda-oxide` main (≥ v0.2) requires CUDA 13.0+ via
`cuda-bindings v0.3.1`.  This machine has CUDA 12.4.  The workaround is to pin
snapshot `c8b3103` — the last commit before the `cuda-bindings` dependency
landed — and build the `rustc-codegen-cuda` backend locally:

```sh
git -C ~/.cargo/cuda-oxide/src checkout c8b3103
CUDA_OXIDE_BACKEND=~/.cargo/cuda-oxide/librustc_codegen_cuda.so \
  cargo oxide build resize_oxide --arch sm_75
```

`CUDA_OXIDE_BACKEND` overrides cargo-oxide's backend lookup entirely and
bypasses the cache-invalidation / re-clone loop that a version mismatch
triggers.  No `--no-fmad` flag is needed: `c8b3103` predates the default-FMA
change and its `llc` invocation never contracts floats.

## Setup

Kernels live in `oxide-kernels/`, a standalone crate with an empty `[workspace]`
table so it does not pull nightly into kornia's workspace.  `cargo oxide build`
emits `resize_oxide.ptx` containing all three kernel entry points:
`resize_bilinear_oxide_3c`, `resize_bilinear_oxide_slice_3c`,
`resize_nearest_oxide_3c`.

The PTX is committed to
`crates/kornia-imgproc/src/cuda/resize_bilinear_oxide_sm75.ptx` (and the
nearest twin shares the same file via a second `include_str!`).  kornia's
stable crates load it through `cudarc` `Ptx::from_src` + `OnceLock` — nothing
in the workspace needs nightly.

PTX gates (all pass):

| Gate | Result |
|---|---|
| `.target` | `sm_75` |
| `.version` | 6.3 (well under CUDA 12.4 ceiling) |
| `__nv_` libdevice symbols | 0 |
| `fma.rn.f32` / `mad.f32` | **0** |

## Arms

| Arm | Kernel | Notes |
|---|---|---|
| A | NVRTC bilinear with `__ldg` | shipping kernel |
| B | NVRTC bilinear, `__ldg` stripped | isolates `__ldg` contribution |
| C | cuda-oxide bilinear, raw-pointer params | load-hoisted (see Finding 2) |
| D | cuda-oxide bilinear, `&[f32]` / `DisjointSlice` params | measured negative result |
| E | NVRTC nearest-neighbour | shipping kernel |
| F | cuda-oxide nearest-neighbour, raw-pointer params | new on sm\_75 |

All arms use the same 32×8 block, `div_ceil` grid, `PixelMapping::coeffs`
coefficients, and `CU_FUNC_CACHE_PREFER_L1` cache config.

## Correctness

Parity tests in `cuda::resize_oxide::tests` (7/7 pass, `--release`, assert on
raw bits with no tolerance):

```
test cuda::resize_oxide::tests::bilinear_noldg_matches_cpu_bit_exact ... ok
test cuda::resize_oxide::tests::bilinear_oxide_matches_cpu_bit_exact ... ok
test cuda::resize_oxide::tests::bilinear_oxide_matches_nvrtc         ... ok
test cuda::resize_oxide::tests::bilinear_oxide_slice_matches_cpu_bit_exact ... ok
test cuda::resize_oxide::tests::nearest_oxide_matches_cpu_bit_exact  ... ok
test cuda::resize_oxide::tests::nearest_oxide_matches_nvrtc          ... ok
test cuda::resize_oxide::tests::all_bilinear_arms_agree              ... ok
```

`*_matches_nvrtc` compares oxide output against NVRTC output directly —
a CPU-only comparison cannot catch "both drifted from the CPU together."

## Throughput

Wall clock, 200 iterations after 50 warmup, queue-all-then-synchronize.
`GB/s` counts one source read plus one destination write per output pixel
(3 channels × 8 B/pixel).

### Bilinear sweep

| case (src→dst) | A nvrtc-ldg | B nvrtc-noldg | C oxide-ptr | D oxide-slice |
|---|---|---|---|---|
| 1024×1024→512×512   | 0.094 ms | 0.094 ms | 0.094 ms | 0.095 ms |
| 512×512→1024×1024   | 0.099 ms | 0.099 ms | 0.100 ms | 0.099 ms |
| 1920×1080→960×540   | 0.178 ms | 0.178 ms | 0.178 ms | 0.178 ms |
| 960×540→1920×1080   | 0.195 ms | 0.195 ms | 0.195 ms | 0.197 ms |
| 3840×2160→1920×1080 | 0.705 ms | 0.705 ms | 0.706 ms | 0.705 ms |

(lower is better; ±0.002 ms measurement noise)

### Nearest-neighbour sweep

| case (src→dst) | E nvrtc-nn | F oxide-nn |
|---|---|---|
| 1024×1024→512×512   | 0.059 ms | 0.059 ms |
| 512×512→1024×1024   | 0.098 ms | 0.098 ms |
| 1920×1080→960×540   | 0.109 ms | 0.109 ms |
| 960×540→1920×1080   | 0.194 ms | 0.194 ms |
| 3840×2160→1920×1080 | 0.429 ms | 0.427 ms |

Arm F is within ±0.002 ms of arm E on every case.  The 4K downscale shows a
0.5% lead for oxide; that is within noise but consistent across runs.

## Instruction mix

`ptxas -v --gpu-name sm_75 -O3`.

| | A (NVRTC, `--fmad=false`) | C (oxide raw ptr) | D (oxide slices) |
|---|---|---|---|
| `fma.rn.f32` / `mad.f32` | 0 | 0 | 0 |
| `ld.global.nc.f32` (`__ldg`) | 12 | 0 | 0 |
| `ld.global.b32` | 12 | 12 | 0 (generic `ld.b32`) |
| `st.global.b32` | 3 | 3 | 0 (generic `st.b32`) |
| `min.f32` / `max.f32` | 4 / 4 | 0 / 0 | 0 / 0 |
| `selp.b32` | 0 | 4 | 4 |
| `bra` | 1 | 8 | 20 |
| `cvt.rzi.u32.f32` | 2 | 2 | 2 |
| instructions (approx) | ~143 | 123 | — |
| registers | ~34 | **32** | 30 |
| spills | 0 | 0 | 0 |

Arm C compiles to fewer instructions and fewer registers than arm A while
producing bit-identical output.  The cost is 8 branches (vs 1) from the
backend-gap workarounds.

Nearest arm F: 16 registers, 56 instructions, 0 spills.  Minimal instruction
count reflects the simpler coordinate logic.

## Findings

### 1. `__ldg` is not load-bearing on sm_75

Arms A and B are indistinguishable across all five cases (max delta: 0.000 ms).
This corroborates Edgar's sm_87 finding.  The read-only cache path that `__ldg`
targets (`ld.global.nc`) provides no measurable benefit over the standard
`ld.global` path for this access pattern on either Turing or Ampere.

### 2. Load-hoisting is required; the 2× upscale regression is a real risk

This is unchanged from the sm_87 report.  Raw pointers carry no `noalias`, so
the compiler cannot hoist a later channel's loads above an earlier channel's
store without help.  Arm C explicitly reads all 12 source values into locals
before the first `st.global.b32`.  Without that, the interleaved
load-store pattern would serialize three memory-latency round trips on the
latency-bound upscale cases.  The 2× regression was measured on sm_87 and the
same PTX structure would produce it on sm_75.

The fix is correct and complete — all 7 parity tests pass with no tolerance.

### 3. Slice regression (arm D) does not appear on Turing

On sm_87 Edgar measured a 12–13 % upscale penalty for arm D versus arm A.
On sm_75:

| case | A (ms) | D (ms) | delta |
|---|---|---|---|
| 512×512→1024×1024 | 0.099 | 0.099 | 0 % |
| 960×540→1920×1080 | 0.195 | 0.197 | +1 % |

Both are within noise.  The cause of the sm_87 penalty — generic `ld.b32`
replacing `ld.global.b32` when address-space inference fails for slice
parameters — produces the same PTX on sm_75, but Turing's memory subsystem
absorbs the penalty whereas Orin's does not.  The slice variant is still a
measured negative result (more branches, no `__ldg` equivalent), but its cost
is architecture-dependent.

### 4. Nearest-neighbour arm F is at full parity

This was not tested in the sm_87 report.  The oxide nearest kernel matches NVRTC
on every case.  The kernel is simpler (one source read per output pixel, integer
truncation) and compiles to 56 instructions / 16 registers — half the bilinear
count.  The backend gaps that affect bilinear (clamp branches, `as u32`
saturation) are present here too but immeasurable given the bandwidth-bound
nature of the kernel.

### 5. Backend gaps are the same three as sm_87

| Construct | Failure | Workaround |
|---|---|---|
| `Ord::min` on `u32` | `drop of RigidTy(Uint(U32)) is not supported` | explicit `if a < b { a } else { b }` |
| `f32::min` / `f32::max` | LLVM verifier rejects `minimumnum`/`maximumnum` | explicit `setp` + `selp` |
| `f32::to_int_unchecked::<u32>()` | verifier rejects `float_to_int_unchecked` | plain `as u32` |

All three are workarounds in value, not in semantics.  `f32::min`/`max` is the
most consequential gap for future kernels: any kernel with frequent clamping
loses `min.f32`/`max.f32` to a branch pair.  This is measurable on compute-bound
kernels (bicubic, Lanczos) but invisible here.

### 6. Byte-exactness comes from the language default

NVRTC needs `--fmad=false` to suppress FMA contraction and match the CPU path.
Rust never contracts float operations implicitly.  Neither `fma.rn.f32` nor
`mad.f32` appears in the oxide PTX on either sm_87 or sm_75 without any flag.
The inverse risk applies to `cuda-oxide` ≥ v0.2, which enables FMA by default;
anyone regenerating on main must pass `--no-fmad`.

## Open questions

- **Bicubic**: the backend gaps (branch pairs for clamping, `setp`/`selp`
  instead of `min.f32`/`max.f32`) are invisible for bandwidth-bound kernels but
  would show up in bicubic, which does more arithmetic per output pixel and has
  a 4×4 tap clamp.  This is the next planned arm sweep.

- **Multi-arch PTX**: the committed `.ptx` is pinned to sm_75.  Shipping for
  both sm_75 and sm_87 means two committed artifacts, or NVRTC as fallback for
  arches not yet covered.

- **`noalias` on raw-pointer parameters**: if cuda-oxide can expose this, arm C
  would recover `ld.global.b32` scheduling without requiring the manual
  load-hoist pattern.

## Reproducing

```sh
# PTX generation (CUDA 12.4 machine)
cd oxide-kernels
export CUDA_OXIDE_BACKEND=~/.cargo/cuda-oxide/librustc_codegen_cuda.so
cargo oxide build resize_oxide --arch sm_75
cp resize_oxide.ptx ../crates/kornia-imgproc/src/cuda/resize_bilinear_oxide_sm75.ptx
cp resize_oxide.ptx ../crates/kornia-imgproc/src/cuda/resize_nearest_oxide_sm75.ptx

# PTX gates
grep -E '^\.version|^\.target' resize_bilinear_oxide_sm75.ptx
grep -c '__nv_'                 resize_bilinear_oxide_sm75.ptx  # 0
grep -cE 'fma\.rn\.f32|mad\.f32' resize_bilinear_oxide_sm75.ptx  # 0

# register counts
ptxas -v --gpu-name sm_75 -O3 resize_bilinear_oxide_sm75.ptx -o /dev/null

# parity tests
cd ..
cargo test --release -p kornia-imgproc --features cuda --lib cuda::resize_oxide

# throughput
cargo run -p kornia-imgproc --example bench_cuda_resize --features cuda --release
```
