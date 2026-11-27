Contract-judge result (step 0.5). This step did not write `tsdf/CONTRACT.md`: its tools are read-only, and the task asks for the full text here. The gate commits it. Flux scan: neither draft nor the merged contract contains a flux concept. No shell commands were run in this step. Every VERIFIED item cites a file:line read on this box. Numbers marked "draft-reported" were not reproduced.

# kornia-3d TSDF — bit-exact contract (FINAL v1, amended v1.1: §13 raycast back-off; v1.2: pixel cap 2^31, §14 PoolFull bound, 2026-11-14)

**Status.** This merges `contract-a` (numerics-first) and `contract-b` (API-first). The numerics come from A and the API from B. **[J]** marks a judge change.
- MUST, MUST NOT and MAY are normative.
- Risk tags **[R-…]** are defined in §19.
- **VERIFIED** items cite a file:line on this box. **PENDING** items are not verified. **draft-reported** numbers come from the drafts and were not reproduced.

**Abbreviations used throughout:**
- **oxide PTX** = `/mnt/data/kr-tsdf-targets/oxide-spike/kornia-oxide-kernels/ptx/kornia_kernels_oxide.ptx`
- **core PTX** = `/mnt/data/kr-tsdf-targets/core-share-spike/kornia-oxide-kernels/ptx/kornia_kernels_oxide.ptx`
- **gpu_test** = `/mnt/data/kr-tsdf-spikes/results/atomic_u32+atomic_u64+asm_math+plain_math+smem_static+smem_dynamic.nolibdevice/gpu_test.log`. Its sample size is 1048576 values per check.

## 0. Scope and principles

**Scope:**
- `crates/kornia-3d/src/tsdf/`, with three backends: scalar, NEON, and CUDA via cuda-oxide.
- The shared per-voxel core (§2).
- `SubmapTsdf` (multi-agent).
- `crates/kornia-3d/src/mesh.rs` and the `io/ply` mesh writer.
- Two constructors in `registration/rgbd.rs`.
- The `kornia-py` bindings.

Inputs are plain kornia types: `Image`, `DepthImage`, `PinholeCamera`, `Pose3d`.

**P1. One op sequence wherever rounding happens.**
- Every rounded f32 expression is written once below as normative Rust. Every backend evaluates exactly that sequence.
- Each `+ - * /` on `f32` is one IEEE-754 binary32 op with round-to-nearest-even. It is evaluated as parenthesised and never fused.
- `O::fma(a,b,c)` is RN(a·b+c), rounded once. `O::sqrt` is correctly rounded (§2).
- No op may be inserted, removed, reassociated, contracted or strength-reduced. Incremental stepping (`p += R·e_x·s`) does not conform [R-CONTRACT].

**P2. Comparison-only results are specified as functions.** Some integers come only from ordered compares: range gates, the LiDAR row, the cube index, and the floor of a range-checked value. Any algorithm that yields the same integer MAY compute them.

**P3. Host constants are computed once on the host and passed verbatim.**
- One `#[repr(C)]` params struct per sensor is shared by scalar, NEON and CUDA (pattern: `depth/kernels.rs:53-72` `UnprojectParams`).
- f32-sourced constants are derived in f32 (`1.0f32 / voxel_size`, like `depth/mod.rs:160-164` `metres_per_unit`).
- f64-sourced constants are derived in f64 and rounded once with `as f32` (like `depth/kernels.rs:160-162`). These are intrinsics, poses and LiDAR geometry.
- Kernels never compose, invert or re-derive a parameter [R-PARAM].

**P4. Unordered containers are specified by membership only.** This covers touched-key buffers, miss lists before sorting, visible lists and hash slots.

**P5. Every ordered output is in ascending key order:** submap id, block key, in-block index, edge index. Pool, hash, thread or arrival order never leaks (as `voxel/mod.rs:26-33`).

**P6. No float atomics, and no float reductions across voxels.** Integer atomics are used only in four ways [R-ATOMIC]:
- as sets;
- as counts;
- as the min of non-negative float bit patterns;
- followed by a sort.

**P7. No NaN is ever stored or emitted.** Every gate is an ordered compare that is false on NaN [R-NAN].

## 1. Arithmetic vocabulary

| Contract op | Scalar (`StdOps`) | NEON (`core::arch::aarch64`) | Device source (cuda-oxide, `PtxOps`) | PTX emitted (status) |
|---|---|---|---|---|
| `a+b`, `a-b`, `a*b` | f32 operators | `vaddq_f32`/`vsubq_f32`/`vmulq_f32` | f32 operators; build only via `build_ptx`, which forces `CUDA_OXIDE_NO_FMA=1` (`build_ptx.rs:179`) | `add.rn.f32`/`sub.rn.f32`/`mul.rn.f32`. VERIFIED: oxide PTX 125-127; gpu_test:14 0 mismatches |
| `a/b` | `/` | `vdivq_f32` | `/` | `div.rn.f32` VERIFIED (oxide PTX 130; gpu_test:15 0 mismatches). `1.0/x` lowers to `rcp.rn.f32`, which is IEEE RN with the same bits as `div.rn 1.0,x`. Emission VERIFIED (core PTX 122, 239); bit test PENDING |
| `O::fma(a,b,c)` | `f32::mul_add` | `vfmaq_f32(c, a, b)`, addend FIRST | `ptx_asm!("fma.rn.f32 …")` | `fma.rn.f32` between `// begin inline asm` markers. VERIFIED: oxide PTX 223-225; gpu_test:9 0 mismatches |
| `O::sqrt(a)` | `f32::sqrt` | `vsqrtq_f32` | `ptx_asm!("sqrt.rn.f32 …")` | `sqrt.rn.f32`. VERIFIED: oxide PTX 262; gpu_test:7 0 mismatches |
| `abs`, `-a` | mask `0x7fff_ffff`, unary `-` | `vabsq_f32`/`vnegq_f32` | same | `abs.f32`/`neg.f32`, never `.ftz` (core PTX 303) |
| `sel(m,a,b)` | `if m {a} else {b}` | `vbslq_f32` | same | `selp` or branch |
| ordered `< <= > >=` | f32 compare | `vcltq_f32` … | same | `setp.{lt,le,gt,ge}.f32` (false on NaN) |
| `trunc_i32(x)`, x range-checked | `x as i32` | `vcvtq_s32_f32` | `x as i32` | `cvt.rzi.s32.f32` plus saturation `selp`. VERIFIED: oxide PTX 173-179 |
| `floor_i32(x)`, x range-checked | `floor_i32` (§2) | `vcvtmq_s32_f32` | `floor_i32` | Same integer (P2). `x.floor()` gives `cvt.rmi.f32.f32` without libdevice. VERIFIED: oxide PTX 167; gpu_test:10-13 0 mismatches |
| `f32(n)`, \|n\| ≤ 2^24 | `n as f32` | `vcvtq_f32_s32`/`_u32` | same | `cvt.rn.f32.s32` (exact) |

**Forbidden in per-element code (scalar, NEON, device):**

**Float min/max.**
- `f32::min`, `f32::max` and `f32::clamp` are forbidden.
- Why: the CPU uses IEEE minNum/maxNum ("nsz": the sign of a zero result is unspecified). cuda-oxide lowers them to an ordered compare/select (`float_math.rs:100-107`, `call.rs:411-416, 1367-1372`). The two differ on ±0 and NaN.
- Use compare+select instead.
- One exception, R-MINMAX: NEON MAY use `vminq_f32`/`vmaxq_f32` for the two tsd clamp selects of §8.1 only.

**SIMD helpers.** `kornia_algebra::simd::F32x4::mul_add`/`F32x8::mul_add` fuse only on aarch64 or `+fma` builds (`simd.rs:100-113`).

**NEON intrinsics.** `vmlaq_f32` and `vmlsq_f32` are unfused despite the name. Also forbidden: `vrecpe*`, `vrecps*`, `vrsqrte*`, `vrsqrts*`.

**Device sources for fma and sqrt.**
- `f32::mul_add`, `f32::sqrt`, `core::f32::math::*` and `libm::*` lower to libdevice `__nv_fmaf`/`__nv_sqrtf` (`call.rs:374, 396`). They emit `.rn` only when libdevice is linked, and emit no PTX without it.
- `cuda_device::float::{fma,sqrt,div,rcp,mul,add}_rn_f32` need sm_80 or newer and fail at the gate's sm_75 (`results/fma_rn/build.log:44`, `results/sqrt_rn/build.log:44`).

**Fast-math intrinsics.** `core::intrinsics::{fadd_fast, fsub_fast, fmul_fast, fdiv_fast, frem_fast, fmuladdf32}` and `*_algebraic` are forbidden.

**Other forbidden code:**
- Any transcendental inside a kernel. Host tables use `libm::*` (§7.2).
- `f32::round`, and float `floor`/`ceil` in the shared core.
- A saturating `as u8`/`as u16` used as a clamp.
- Float atomics, and float reductions across voxels.
- Strength reduction: `x/c` becomes `x*inv_c` only for contract-defined host reciprocals.

**Forbidden PTX** (rejected by the gate, §18):
- `div.approx`, `div.full`, `rcp.approx*`, `sqrt.approx`, `rsqrt`, `sin`, `cos`, `ex2`, `lg2`, `tanh`
- any `.ftz`, and `.sat` on float ops
- float `min`/`max`
- `add/sub/mul.f32|f64` without a rounding modifier
- float `atom`/`red`

## 2. Shared core, backend shims, file layout

```rust
/// Float ops the shared core cannot spell itself (stable `no_std` has no
/// `f32::mul_add`/`sqrt`). Every impl MUST be correctly rounded.
pub trait F32Ops {
    /// RN(a·b + c), one rounding.
    fn fma(a: f32, b: f32, c: f32) -> f32;
    /// Correctly rounded square root.
    fn sqrt(x: f32) -> f32;
}
/// Exact floor for finite |x| < 2^31 (callers range-check to |x| < 2^24 first).
#[inline(always)]
pub fn floor_i32(x: f32) -> i32 { let t = x as i32; t - (((t as f32) > x) as i32) }
```

**Implementations:**
- **`StdOps`** (kornia-3d): `a.mul_add(b, c)`, `x.sqrt()`. Correctly rounded on every supported target:
  - aarch64: `fmadd`/`fsqrt`;
  - x86_64 with `+fma`: `vfmadd`/`sqrtss`;
  - x86_64 without `+fma`: the C library `fmaf`. Test `tsdf_stdops_fma_matches_libm` checks it against `libm::fmaf` on 1e8 random triples on every CI host.
- **`LibmOps`** (no_std check, tests): `libm::fmaf`, `libm::sqrtf`. libm 0.2.16 is in Cargo.lock; on aarch64 it uses `fmadd`/`fsqrt` (`libm-0.2.16/src/math/arch/aarch64.rs:19-31, 97-107`).
- **`PtxOps`** (device). Normative, taken from spike `core/oxide/device/src/tsdf_ops.rs:6-25`:
```rust
impl F32Ops for PtxOps {
    #[inline(always)] fn fma(a: f32, b: f32, c: f32) -> f32 { let y: f32;
        unsafe { ptx_asm!("fma.rn.f32 %0, %1, %2, %3;", out("=f") y, in("f") a, in("f") b, in("f") c, options(register_only)); } y }
    #[inline(always)] fn sqrt(x: f32) -> f32 { let y: f32;
        unsafe { ptx_asm!("sqrt.rn.f32 %0, %1;", out("=f") y, in("f") x, options(register_only)); } y }
}
```

**Core rules:**
- `no_std`, no `alloc`, no `#[kernel]`.
- Only slices, `&mut` and fixed arrays.
- Every fn is `#[inline(always)]`, because the gate rejects fma in a non-inlined `.func` (`ptx_gate.rs:91-97`).
- The code must compile under edition 2021 stable (host) and edition 2024 nightly-2026-10-11 (device).

**Core contents:**
- key packing, `floor_i32`, voxel/block maths
- `xform`/`rot`
- pinhole projection; spherical projection and `atan2_det`
- truncation, update and colour
- trilinear interpolation and gradient
- MC tables and edge interpolation
- hash probe

**File layout [J]** (PENDING final 0.1b sign-off at the gate):
- The shared files live in `crates/kornia-oxide-kernels/device/src/tsdf_core/`. That is inside the crate's packaged `device/src/**` (`kornia-oxide-kernels/Cargo.toml:14-26`).
- The device crate compiles them as `mod tsdf_core;`.
- The host lib compiles them as `#[doc(hidden)] #[path = "../device/src/tsdf_core/mod.rs"] pub mod tsdf_core;`. The host lib is `no_std` without features (`src/lib.rs:7`).
- kornia-3d depends on kornia-oxide-kernels with no features. `kornia-3d/src/tsdf/core.rs` is a re-export only.

**Evidence for the layout:**
- The v1 layout built (`core/results/v1/build.log:10-11`).
- A relative path dependency broke under build_ptx staging (`core/results/b-rel/build.log:1-8`).
- An absolute path built but cannot ship (`core/results/b-abs/build.log`).
- The CPU-vs-GPU parity of the shared core (`core/parity/src/main.rs`) has no run log: PENDING.

**Fallback if the gate rejects this layout:** a verbatim mirror, plus the `_bit_exact` tests, plus `tsdf_core_sources_identical` (which compares `include_str!` of both copies).

## 3. FP environment, sample validity, NaN/inf, errors

### 3.1 Environment [R-ENV] [J]

Every CPU thread that runs TSDF code needs round-to-nearest-even, no flush-to-zero and no denormals-are-zero:
- x86_64 MXCSR: FTZ (bit 15) = 0, DAZ (bit 6) = 0, RC (bits 13-14) = 00.
- aarch64 FPCR: FZ (bit 24) = 0, RMode (bits 22-23) = 00, AH (bit 1) = 0, FIZ (bit 0) = 0.

PTX never uses `.ftz` (gate G2).

Enforcement is read-only:
- `TsdfWorkspace` checks every rayon worker once, the first time it runs a kernel. This is the cold path, using `rayon::broadcast`.
- Every public call checks the calling thread, using `core::arch::asm!` `stmxcsr` / `mrs fpcr`.
- A non-default environment returns `TsdfError::FpEnvironment` before any mutation.
- The library never writes MXCSR or FPCR. Rust assumes the default environment, and `_mm_setcsr` is deprecated.

Residual risk: a worker whose environment changes after its first check is not detected (documented). 32-bit x86 without SSE2 is not a conformant target: add `compile_error!`.

### 3.2 Depth and range samples

These follow `depth/mod.rs:15-23, 160-230` and `kornia-imgproc/src/depth/mod.rs:56-58, 91-93`. `mpu = DepthImage::metres_per_unit() = 1.0f32 / meter` (`depth/mod.rs:415-418`).

```rust
#[inline(always)] fn sample_u16(raw: u16, mpu: f32, range: DepthRange) -> Option<f32> {
    if raw == 0 { return None; }                                   // is_valid_depth_u16
    let d = f32::from(raw) * mpu;
    if d >= range.min && d < range.max { Some(d) } else { None } }  // upper bound strict (VBG depth_max)
#[inline(always)] fn sample_f32(raw: f32, mpu: f32, range: DepthRange) -> Option<f32> {
    let d = raw * mpu;
    if !(d.is_finite() && d > 0.0) { return None; }                // is_valid_depth_f32 on metres
    if d >= range.min && d < range.max { Some(d) } else { None } }
```

### 3.3 Host validation before any mutation

- **Poses:** must be finite, with `max|RᵀR − I| ≤ 1e-6` and `det R > 0` (f64). Otherwise `InvalidTransform`.
- **Pinhole:** `fx, fy` finite and > 0; `cx, cy` finite; `inv_fx = (1.0/fx) as f32` finite and non-zero (rules of `depth/kernels.rs:156-181` and `registration/rgbd.rs:107-123`). Every distortion coefficient must be 0, otherwise `DistortedCamera`.
- **Sizes:** the depth and colour images must match the sensor size, otherwise `SizeMismatch`. Colour is allowed only on pinhole sensors, otherwise `ColorOnRangeSensor`.
- **Environment:** checked per §3.1.

### 3.4 No NaN or inf inside a kernel

- Voxel centres are bounded (§5), and transforms are finite.
- `1/p_z` is taken only for `p_z ≥ range.min > 0`.
- `sqrt` is taken only of a sum of squares.
- Every per-element divisor is proven non-zero at its site.
- All gates are ordered compares.
- The volume never stores NaN. `content_hash` hashes raw bits, so a stored NaN would show up.

### 3.5 Errors

- All validation happens before mutation. A failing `integrate` step leaves the volume unchanged: each step is all-or-nothing.
- `integrate_batch` first validates the whole batch statically, then applies steps in canonical order (§11).
  - If step k fails, the steps before k stay applied.
  - The call returns `TsdfError::Batch { index: k, source }`.
- A CUDA driver error leaves the device volume unspecified (`Cuda`). Recover by resetting, or with `download_into` from a host copy.

## 4. Frames, poses, host parameter derivation

**Pose naming.**
- Every pose is named `a_from_b: Pose3d` and means `p_a = R·p_b + t` (`Pose3d::transform_point`).
- `Pose3d` itself documents world→camera, i.e. camera_from_world (`pose/pose3d.rs:5-14`).
- `x.compose(&y)` is `x ∘ y`: `y` is applied first (`pose3d.rs:51-60`).
- `inverse()` is (Rᵀ, −Rᵀt) (`pose3d.rs:42-49`).

**Frames.**
- Pinhole: x right, y down, z forward. The centre of pixel (x,y) is at (x,y) (`depth/mod.rs:25-27`).
- LiDAR: x forward, y left, z up. Azimuth is counter-clockwise (CCW) from +x seen from +z; elevation is positive up.
- Volume and submap frames are in metres, with the grid origin at 0.

**Host derivation.** This runs in f64, once per sensor per step, with a fixed parenthesisation:
```rust
// TsdfVolume: volume_from_sensor comes from SensorObservation.
// SubmapTsdf:
let submap_from_body = world_from_submap.inverse().compose(&world_from_body);
let volume_from_sensor = submap_from_body.compose(&body_from_sensor);
// both:
let sensor_from_volume = volume_from_sensor.inverse();
// raycast: volume_from_camera given (TsdfVolume) or world_from_submap.inverse().compose(&world_from_camera);
//          camera_from_volume = volume_from_camera.inverse().
fn rows_f32(r: &Mat3F64) -> [f32; 9] { let m = r.to_cols_array(); let mut o = [0.0f32; 9];
    for i in 0..3 { for j in 0..3 { o[3 * i + j] = m[3 * j + i] as f32; } } o }   // as depth/kernels.rs:224-231
fn vec_f32(t: &Vec3F64) -> [f32; 3] { [t.x as f32, t.y as f32, t.z as f32] }
```

Properties of this derivation:
- Any non-finite entry gives `InvalidTransform`.
- The f32 pairs are never inverted.
- `inverse()` transposes exactly, so `rows_f32(Rᵀ)` is the exact transpose of `rows_f32(R)`.
- glam `DMat3::mul_vec3` uses plain mul/add (`glam-0.33.11/src/f64/dmat3.rs:795-800`), so the derivation is identical on x86_64 and aarch64 hosts.

**Transform op order** (normative) [R-FMA]. Voxel x, the fastest-varying index, is the OUTERMOST fma. A CPU x-row partial is therefore bit-identical to the full GPU expression:
```rust
#[inline(always)] pub fn xform<O: F32Ops>(r: &[f32; 9], t: &[f32; 3], p: [f32; 3]) -> [f32; 3] { [
    O::fma(r[0], p[0], O::fma(r[1], p[1], O::fma(r[2], p[2], t[0]))),
    O::fma(r[3], p[0], O::fma(r[4], p[1], O::fma(r[5], p[2], t[1]))),
    O::fma(r[6], p[0], O::fma(r[7], p[1], O::fma(r[8], p[2], t[2]))) ] }
#[inline(always)] pub fn rot<O: F32Ops>(r: &[f32; 9], v: [f32; 3]) -> [f32; 3] { [
    O::fma(r[0], v[0], O::fma(r[1], v[1], r[2] * v[2])),
    O::fma(r[3], v[0], O::fma(r[4], v[1], r[5] * v[2])),
    O::fma(r[6], v[0], O::fma(r[7], v[1], r[8] * v[2])) ] }
```
- Hoisting `A_i = O::fma(r[3i+1], c_y, O::fma(r[3i+2], c_z, t_i))` once per x-row conforms.
- This order deliberately differs from `depth/kernels.rs:530-532`.

**Python.**
- `extrinsic` (4×4 camera_from_world) is `sensor_from_volume`.
- The binding computes `volume_from_sensor = extrinsic.inverse()` in f64 once.
- A Rust caller gets identical bits only by passing that same f64 inverse.

## 5. Grid, keys, voxel centre, storage, hash, pool

**5.1 Block size.**
- `TsdfVolume<const B: usize = 8>` requires `BlockDim<B>: SupportedBlockDim`.
- The trait is sealed, with two impls: `BlockDim<8>` (`LOG2=3`, `VOXELS=512`) and `BlockDim<16>` (`LOG2=4`, `VOXELS=4096`).
- Every head-to-head bench runs both. For B=16 on CUDA, the launch geometry is free (§14), but the per-voxel ops are fixed.

**5.2 Voxel and block coordinates:**
```rust
pub const VOXEL_LIMIT: i32 = 1 << 22;          // |g| < 2^22 per axis
const G_F: f32 = 4194304.0;                    // 2^22, exact
let q = p[a] * inv_voxel;                      // inv_voxel = 1.0f32 / voxel_size (host), MUST be normal
if !(q >= -G_F && q < G_F) { /* outside the domain: not allocated, not sampled */ }
let g = floor_i32(q);
let b = g >> LOG2;                             // arithmetic shift = floor division
let l = g & (B as i32 - 1);                    // g == (b << LOG2) + l
```
- The block domain is ±2^19 for B=8 and ±2^18 for B=16.
- At 1 cm voxels the domain spans ±41.9 km.
- At |q| = 2^22 the f32 spacing is 0.5 voxel, so submaps keep coordinates local (§11).

**5.3 Key packing** (normative):
```rust
const KEY_BITS: u32 = 21; const KEY_BIAS: i32 = 1 << 20; const KEY_MASK: u64 = (1 << 21) - 1;
pub const EMPTY_KEY: u64 = u64::MAX;           // bit 63 set: never a valid key
#[inline(always)] pub fn pack_block_key(b: [i32; 3]) -> u64 {     // caller guarantees b in [-2^20, 2^20)
    let f = |v: i32| ((v + KEY_BIAS) as u32 as u64) & KEY_MASK;
    (f(b[0]) << 42) | (f(b[1]) << 21) | f(b[2]) }
#[inline(always)] pub fn unpack_block_key(k: u64) -> [i32; 3] {
    let g = |s: u32| (((k >> s) & KEY_MASK) as i32) - KEY_BIAS; [g(42), g(21), g(0)] }
```
- Ascending u64 order is lexicographic signed (x,y,z) order. The idea is the same as `voxel/downsample.rs:22` (biased, x most significant).
- `BlockKey::new` returns `KeyOutOfRange` outside [-2^20, 2^20).
- Inside kernels, an out-of-domain pixel or voxel is skipped and counted in `IntegrateStats::out_of_range`. It is never an error.

**5.4 Voxel centre** (normative).
- `c_a = O::fma(g_a as f32, voxel_size, half_voxel)`, with `half_voxel = voxel_size * 0.5` computed exactly on the host.
- Under |g| < 2^22 this equals `(g_a as f32 + 0.5) * voxel_size` bit for bit.

**5.5 In-block index.** `i = lx + B·(ly + B·lz)`, with x fastest.

**5.6 Storage.** Pools are SoA and block-major, with identical bytes on host and device: `field[pool·B³ + i]`.

| pool | dtype | init | present |
|---|---|---|---|
| `tsdf` | f32, normalised, [-1,1] | +0.0 | always |
| `weight` | u16, [0, max_weight] | 0 | always |
| `rgbw` | u32 = r \| g<<8 \| b<<16 \| color_weight<<24 | 0 | `TsdfParams::color` |

- 6 B/voxel without colour, 10 B/voxel with colour. Draft-reported comparisons: VBG 20 B, nvblox 16 B.
- `block_capacity·B³ ≤ u32::MAX`.
- A new or reused block MUST be zeroed before its first integrate.

**5.7 Hash** (layout non-normative).
- Open addressing with linear probing, and `EMPTY_KEY` as the empty marker.
- Capacity is a power of two, at least 2·block_capacity.
- The stored value is the u32 pool index.
- `slot0 = key.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> (64 - log2_cap)`.
- No deletion: compaction rebuilds the table.
- Slots and probe sequences MAY differ between backends; lookups MUST return the same pool index.

**5.8 Pool index** (normative, D1) [J].
- The pool is dense and append-only.
- A step's m new blocks get indices `num_blocks .. num_blocks+m`, in ascending key order.
- `sorted_keys: Vec<(u64, u32)>` (key → pool, ascending) is reserved to `block_capacity` at construction. New entries are merged in place from the back.
- `download_into`, parking and `content_hash` are byte scans of the pools.
- GPU: atomics build only the miss SET. Pool indices are assigned after ranking the sorted misses. An atomic counter is not deterministic: gpu_test:5 reads "key->pool index identical=false".

## 6. Pinhole sensor

**Params (host):**
- `fx, fy, cx, cy` (f64 → f32)
- `inv_fx = (1.0/fx) as f32`, `inv_fy` likewise
- `w, h: i32`, with 1 ≤ W,H ≤ 2^24 and W·H < 2^31 (v1.2: device counts pixels and strides grid loops in u32; 2^32 wrapped to 0). Same check on CPU and GPU, also for raycast and LiDAR sizes
- `w_f, h_f` (exact)
- `mpu`, `range.min`, `range.max`

**Per voxel** (integrate, normative):
```rust
let p = xform::<O>(&r_sv, &t_sv, c);                      // sensor_from_volume
if !(p[2] >= range_min) { skip }                          // cheirality, NaN-safe; range_min > 0
let iz = 1.0 / p[2];                                      // [R-DIV]
let u = O::fma(fx, p[0] * iz, cx);
let v = O::fma(fy, p[1] * iz, cy);
let uh = u + 0.5; let vh = v + 0.5;                       // THE pixel rule: pixel = floor(coord + 0.5)
if !(uh >= 0.0 && uh < w_f && vh >= 0.0 && vh < h_f) { skip }
let (px, py) = (uh as i32, vh as i32);                    // trunc == floor since uh, vh >= 0 [R-CVT]
let idx = (py as u32) * (w as u32) + (px as u32);
let d = match sample(depth, idx) { Some(d) => d, None => skip };   // §3.2
let sdf = d - p[2];                                       // projective distance (KinectFusion/VBG/nvblox)
```
This is the depth module's `floor(u+0.5)` (`depth/kernels.rs:534`, `depth/mod.rs:631`), with the range check moved before the conversion.

**Unprojection** (allocation and raycast; equals `depth/kernels.rs:298-307`):
- `xn = (px as f32 - cx) * inv_fx`
- `yn = (py as f32 - cy) * inv_fy`
- `p_s = [xn*d, yn*d, d]`

## 7. Spherical LiDAR

**7.1 Model.** `SphericalLidar` holds:
- `size`: W columns × H rows, with H ≥ 2 and W ≥ 2.
- `AzimuthGrid { start, step, wrap }` in f64 radians.
  - `start` is the azimuth of column 0's centre.
  - `step` is the signed column spacing; it is negative for clockwise spinners.
- Beam elevations `θ_0 > … > θ_{H−1}`, row 0 at the top.
- `range: DepthRange`.

Validation: `step` must be finite and non-zero, and `|start| ≤ 2π`. When `wrap` is set, `| |W·step| − 2π | ≤ 1e-9·2π`, otherwise `InvalidAzimuthGrid`. The range image is a W×H `DepthImage<u16|f32>` of Euclidean range.

**7.2 Host tables** [R-HOSTMATH].
- Computed in f64 and rounded once.
- Use `libm::sin`/`libm::cos` only: they are pure Rust and give the same bits on every host. Never use `f64::sin`/`cos`, which call the platform libm.

The tables:
- Ascending elevations: `e_k = θ_{H−1−k}`.
- Row boundaries: `bnd[j] = libm::sin((e_j + e_{j+1})/2) as f32` for j = 0..H−2.
- Field-of-view limits:
  - `lo = libm::sin(max(e_0 − (e_1 − e_0)/2, −π/2)) as f32`
  - `hi = libm::sin(min(e_{H−1} + (e_{H−1} − e_{H−2})/2, π/2)) as f32`
- After rounding, `lo < bnd[0] < … < bnd[H−2] < hi` MUST hold strictly, otherwise `InvalidBeamTable`.
- Column constants: `inv_az_step = (1.0/step) as f32` and `az_offset = (−start/step) as f32`.
- Unprojection tables: `cos_el[row], sin_el[row]` from θ_row, and `cos_az[col], sin_az[col]` from `a_col = start + col·step` (f64). The products `cos_el·cos_az` and `cos_el·sin_az` MAY be tabulated per pixel using the same single f32 multiply.
- `from_tables` accepts precomputed f32 tables. They are validated as finite, |v| ≤ 1, and strictly ordered.

**7.3 Per voxel** (normative):
```rust
let p = xform::<O>(&r_sv, &t_sv, c);
let rr = O::fma(p[0], p[0], O::fma(p[1], p[1], p[2] * p[2]));
let r = O::sqrt(rr);                                      // [R-SQRT]
if !(r >= range_min) { skip }                             // range gate BEFORE any division or atan2_det
let sz = p[2] / r;                                        // [R-DIV]; |sz| <= 1: sqrt(RN(z*z)) == |z|, rr >= RN(z*z)
if !(lo <= sz && sz < hi) { skip }                        // vertical FOV
let k = count_le(&bnd, sz);                               // #{ j in 0..H-1 : bnd[j] <= sz }  (P2)
let row = (h - 1) - k as i32;
let phi = atan2_det::<O>(p[1], p[0]);
let u = O::fma(phi, inv_az_step, az_offset);              // column coordinate, centres at integers
let uh = u + 0.5;                                         // THE pixel rule
if !(uh >= -16777216.0 && uh < 16777216.0) { skip }       // 2^24
let ci = floor_i32(uh);
let col = if wrap { ci.rem_euclid(w) } else if ci >= 0 && ci < w { ci } else { skip };
let idx = (row as u32) * (w as u32) + (col as u32);
let rho = match sample(range_image, idx) { Some(d) => d, None => skip };
let sdf = rho - r;                                        // range-space distance
```
Implementations of `count_le` that conform:
- a branchless binary search over `bnd`, padded with +inf to a power of two (fixed ⌈log2 H⌉ iterations);
- a linear scan;
- a uniform lookup table over `sz` that gives a lower bound, followed by fix-up compares `k += (bnd[k] <= sz)`.

The column wrap is integer arithmetic, so ±π land in one column.

**7.4 `atan2_det(y, x)`** (normative; callers pass finite values):
```rust
const C1: f32 = f32::from_bits(0xbeaa_a879); // -0.3333165943622589
const C2: f32 = f32::from_bits(0x3e4c_6b08); //  0.19962704181671143
const C3: f32 = f32::from_bits(0xbe0f_1ec6); // -0.13976582884788513
const C4: f32 = f32::from_bits(0x3dc8_9602); //  0.09794236719608307
const C5: f32 = f32::from_bits(0xbd6c_a408); // -0.05777361989021301
const C6: f32 = f32::from_bits(0x3cbc_beb6); //  0.023040156811475754
const C7: f32 = f32::from_bits(0xbb8e_b7d8); // -0.0043554119765758514
const HALF_PI: f32 = f32::from_bits(0x3fc9_0fdb); const PI: f32 = f32::from_bits(0x4049_0fdb);
#[inline(always)]
pub fn atan2_det<O: F32Ops>(y: f32, x: f32) -> f32 {
    let ax = f32::from_bits(x.to_bits() & 0x7fff_ffff);
    let ay = f32::from_bits(y.to_bits() & 0x7fff_ffff);
    let swap = ay > ax;
    let mx = if swap { ay } else { ax }; let mn = if swap { ax } else { ay };
    let den = if mx > 0.0 { mx } else { 1.0 };            // (0,0) -> t = 0
    let t = mn / den;                                     // [R-DIV], t in [0,1]
    let s = t * t;
    let mut q = C7;
    q = O::fma(q, s, C6); q = O::fma(q, s, C5); q = O::fma(q, s, C4);
    q = O::fma(q, s, C3); q = O::fma(q, s, C2); q = O::fma(q, s, C1);
    let mut r = O::fma(t * s, q, t);                      // atan(t) = t + t^3 Q(t^2), c0 = 1 exact
    if swap { r = HALF_PI - r; }
    if x < 0.0 { r = PI - r; }
    if y < 0.0 { r = -r; }
    r }
```

**Fit.** Remez minimax of the absolute error, with c0 fixed at 1; degree 15, 7 free coefficients.

**Draft-reported (contract-a; not reproduced):**
- Over all 1,065,353,217 f32 values in [0,1]: max error 1.017e-7 rad (3.82 ulp).
- Over 2e8 (y,x) pairs: 3.186e-7 rad (4.55 ulp).
- NEON: 0 mismatches over 2e8.
- 541 (W=1024) and 1058 (W=2048) of 2e8 points land in the neighbouring column compared with f64 atan2 (deterministic).

**Required on these bits** [J, grafted from contract-b]:
- `atan2_det_monotone`: the [0,1] core is non-decreasing over all f32 (0 violations).
- `atan2_det_error_bound`: core ≤ 1.1e-7, full ≤ 3.3e-7.
- PTX twin: PENDING.

**Special values:**
- `atan2_det(±0, x<0) = +π`. libm gives −π for −0.
- `atan2_det(0,0) = 0`.
- `atan2_det(±y,0) = ±π/2`.
- `atan2_det(1e-30, 1) = 1e-30`.

**Cost:** 1 div, 2 mul, 7 fma, ≤ 3 sub, 4 compares.

**7.5 `range_image_from_points_into(points_sensor, dst)`.**

Per point, compute exactly:
- `rr`, `r`;
- the gate `r >= range.min && r < range.max`;
- `sz` and the FOV gate;
- `count_le`, `atan2_det`, and the column rule.

Points that fail any gate (out of field of view, out of range, NaN) are dropped.

Each pixel keeps the minimum `r`, compared as `r.to_bits()`:
- On the device: `atomicMin` u32.
- On the host: `AtomicU32::fetch_min` with `dst` viewed as atomics (precedent: `depth/kernels.rs:9`), or a serial loop.
- A sentinel `u32::MAX` is turned into 0.0 in a final pass.

The output is `DepthImage<f32>` with meter 1. The result is order-independent.

## 8. Truncation, weighted update, colour

**8.1 Truncation.** On the host, `trunc = trunc_voxels * voxel_size` (f32) and `inv_trunc = 1.0f32 / trunc`.
```rust
if !(sdf >= -trunc) { skip }                              // behind the band, and NaN
let x = if sdf < trunc { sdf * inv_trunc } else { 1.0 };  // free space = exactly 1.0 (VBG)
let tsd = if x > 1.0 { 1.0 } else if x < -1.0 { -1.0 } else { x };   // trunc*inv_trunc may be 1 ± 1 ulp
```
`sdf` is never −0. The R-MINMAX exception from §1 (NEON `vminq`/`vmaxq`) applies to these two selects only.

**8.2 Update with cap** (u16 weights):
```rust
let w0 = weight[i] as u32;
let old = tsdf[i];                                        // +0.0 when w0 == 0 (zero-init invariant)
let n = w0 + 1;                                           // UNcapped count drives the mean
let rc = 1.0 / (n as f32);                                // n <= 65536 exact [R-DIV]
tsdf[i] = O::fma(tsd - old, rc, old);                     // first observation: fma(tsd, 1, +0) == tsd
weight[i] = if n > max_weight as u32 { max_weight } else { n as u16 };
```
- A host table `RCP[n] = 1.0f32/(n as f32)` gives the same bits and MAY replace the division.
- `(old·w + tsd)/(w+1)` does NOT conform.
- At the cap, the update becomes a fixed-window running average. `max_weight = 65535` is effectively uncapped.
- Only unit observation weight is supported; per-sensor weights would need a contract revision.

**8.3 Colour.** Applies to pinhole observations that carry a colour image, and only to voxels updated in 8.2. LiDAR never colours.
```rust
let s = rgbw[i]; let cw0 = s >> 24;
let rc = 1.0 / ((cw0 + 1) as f32);
let mut out = 0u32;
for ch in 0..3 {
    let c_old = ((s >> (8 * ch)) & 0xff) as f32;
    let c_obs = color[3 * idx + ch] as f32;              // registered colour: same size, pixel (px, py)
    let c = O::fma(c_obs - c_old, rc, c_old);             // exact subtraction; c in [0, 255]
    let hc = c + 0.5;                                     // round half up
    let qv = if hc >= 255.0 { 255 } else { hc as u32 };   // explicit clamp, never a saturating `as u8`
    out |= qv << (8 * ch);
}
let cw = if cw0 + 1 > max_color_weight as u32 { max_color_weight as u32 } else { cw0 + 1 };
rgbw[i] = out | (cw << 24);
```
Because colour is stored as u8, a channel freezes once `rc·|Δ| < 0.5`; `max_color_weight` defaults to 16 for that reason (D2).

## 9. Allocation (lookup first, sort only the misses)

**Scope.** Allocation covers exactly one step: one `TsdfVolume::integrate` call, or one `AgentStep`.
- It takes the union over all observations of the step.
- It runs BEFORE any of the step's integrate launches.
- `integrate_batch` MUST NOT union across steps.

**Touched set** per observation s, for each valid pixel with gated `d`:
```rust
// pinhole: ps = [xn(px) * d, yn(py) * d, d]
// lidar:   ps = [d * (cos_el[row] * cos_az[col]), d * (cos_el[row] * sin_az[col]), d * sin_el[row]]
let pv = xform::<O>(&r_vs, &t_vs, ps);                    // volume_from_sensor
for a in 0..3 {
    let lo = (pv[a] - trunc) * inv_voxel;
    let hi = (pv[a] + trunc) * inv_voxel;
    if !(lo >= -G_F && hi < G_F) { out_of_range += 1; continue 'pixel; }
    blo[a] = floor_i32(lo) >> LOG2; bhi[a] = floor_i32(hi) >> LOG2;
}
touched[s] |= { pack_block_key([bx, by, bz]) : bx in blo[0]..=bhi[0], by in blo[1]..=bhi[1], bz in blo[2]..=bhi[2] };
```
- `touched[s]` is a set: duplicates and order are free. Skipping a pixel whose box equals the previous pixel's box conforms.
- The workspace is sized for K³ blocks per pixel, with K = ⌈2·trunc_voxels/B⌉ + 2.
- Completeness note: the box contains the LiDAR band exactly. Pinhole voxels in the projective band beyond the box, where |(xn,yn,1)| > 1, are never allocated. This is deterministic and affects quality only.

**Insertion.** The CPU reference gives the normative result:
1. Probe every touched key.
   - Hit: `mark[pool] = step_epoch`.
   - Miss: push the key to `ws.misses`, in any order.
2. `ws.misses.sort_unstable(); ws.misses.dedup();`
3. If `num_blocks + |misses| > capacity`:
   - `Geometric`: reserve (cold frame).
   - `Fixed`: return `PoolFull` with nothing mutated.
4. Assign `pool(misses[j]) = num_blocks + j`. Zero-initialise the block, insert it into the hash, merge it into `sorted_keys`, and set `mark`.
5. v0 never frees blocks. If freeing is added, freed indices are reused in ascending order.

`step_epoch` is a u32; on wrap, all marks are reset.

**Equivalence test:** `tsdf_alloc_lookup_first_equals_full_sort`.

**GPU:**
- Allowed allocation paths: oxide atomics (u32/u64 CAS VERIFIED, gpu_test:1-4), `voxel/cuda.rs` radix sort plus unique, or host allocation.
- A path conforms only if both its key set and its key→pool map equal the CPU rule.

## 10. Visible sets, scheduling, multi-sensor order

- **Visible set [J decision V].** `visible[s] = { pool(k) : k ∈ touched[s] }`. An observation updates only the blocks it touched in this step (VBG semantics). Only membership is normative.
- **Fold order.**
  - In a SubmapTsdf rig: ascending `SensorId`. Each frame's sensor must be in the agent's rig, and each sensor at most once; otherwise `UnknownSensor` or `DuplicateSensor`.
  - In `TsdfVolume::integrate`: slice order.
- **Reference execution.** One integrate launch per observation, in fold order, on one stream. Within a launch, any schedule conforms.
- **Fused kernel.** PERMITTED if, for every voxel, observations apply in fold order, and observation s updates the voxel only when its block is in `visible[s]`. The visible sets are carried as a per-block u64 mask, so fusion is limited to 64 observations or fewer per step.
- **Culling.** Allowed only if exact.
- **Dirty tracking.** `modified_epoch[p] = volume_epoch` for every touched block. `last_step_blocks()` scans `sorted_keys`.

## 11. Multi-agent order and submaps

**Ids and disjoint writes.**
- `AgentId(u32)`, `SensorId(u32)`, `SubmapId { agent, seq }`.
- Each agent has one active submap and is its only writer.
- Each submap has its own hash, `sorted_keys` and page table.

**Order.** The normative order is (stamp, agent, sensor).
- `integrate_batch` sorts a workspace index buffer with `sort_unstable_by_key(|i| (stamp, agent))`. Keys are unique, so the order is unique.
- A duplicate (stamp, agent) gives `DuplicateStep`.
- Per agent, stamps MUST strictly increase across calls; otherwise `StaleStamp`.

**Invariance.** Take any interleaving or batching in which each agent's steps arrive in increasing stamp order. Every submap's bytes then equal a serial replay sorted by (stamp, agent).

**Spawn.** Decided on the host in f64, before step k, from the agent's own steps only. A new submap is spawned if there is no active submap, or if any of these holds:
- `steps_in_submap ≥ max_steps`;
- `|t_body − t_spawn|² ≥ max_translation²`;
- `(tr(R_spawnᵀR_body) − 1)/2 ≤ cos_max_rotation`, where `cos_max_rotation = libm::cos(max_rotation)` is computed once in `add_agent`;
- `num_blocks ≥ max_blocks`.

The step then goes into the new submap only. The new submap's `world_from_submap` is set by `SubmapFrame`:
- `TranslationAtSpawn` (default): identity rotation, origin at the body position.
- `BodyAtSpawn`: `world_from_body` of that step.

**Arena [J].** Two things are preallocated:
- `max_submaps` submap headers. Each has a fixed hash capacity of next_pow2(2·`volume.block_capacity`), a `sorted_keys` buffer reserved to `volume.block_capacity`, and a page table.
- A block arena of `arena_blocks` pages.

Spawning takes a free header and does not allocate. Local pool indices (normative) map to arena pages; page ids are not normative. When the pool or the headers run out:
- `Fixed`: `PoolFull` or `TooManySubmaps`.
- `Geometric`: grow on a cold frame.

**Loop closure.** `set_submap_poses` validates every pose before applying any. It changes poses only and never re-integrates. It acts as a barrier at call time. Mesh caches apply the new pose when they write.

**Residency (CUDA types).**
- Resident submaps always include the active submaps and the submaps the current query needs.
- `max_resident_bytes` is shared across agents; eviction is LRU by step counter.
- Parking is a lossless memcpy into preallocated host buffers. The first park of a submap allocates (cold).
- Eviction order MAY vary; content MUST NOT.

## 12. Marching cubes

**12.1 Tables.**
- `EDGE_TABLE: [u16; 256]`, `TRI_TABLE: [[i8; 16]; 256]` (Lorensen/Bourke, −1 terminated), `CORNER` and `EDGE` live once, in `tsdf_core/mc_tables.rs` [R-TABLE].
- Mirror fallback: `tsdf_mc_tables_identical`.
- Corners: c0 (0,0,0), c1 (1,0,0), c2 (1,1,0), c3 (0,1,0), c4 (0,0,1), c5 (1,0,1), c6 (1,1,1), c7 (0,1,1).

Each edge is given by an owner offset and an axis; the owner is the lower end along the axis:

| edge | corners | owner | axis |
|---|---|---|---|
| 0 | 0–1 | (0,0,0) | x |
| 1 | 1–2 | (1,0,0) | y |
| 2 | 2–3 | (0,1,0) | x |
| 3 | 3–0 | (0,0,0) | y |
| 4 | 4–5 | (0,0,1) | x |
| 5 | 5–6 | (1,0,1) | y |
| 6 | 6–7 | (0,1,1) | x |
| 7 | 7–4 | (0,0,1) | y |
| 8 | 0–4 | (0,0,0) | z |
| 9 | 1–5 | (1,0,0) | z |
| 10 | 2–6 | (1,1,0) | z |
| 11 | 3–7 | (0,1,0) | z |

**12.2 Validity and cube index.**
- A voxel is valid if its block is allocated and `weight ≥ weight_threshold` (u16, at least 1, default 1).
- A cube is emitted only if all 8 corners are valid.
- Bit c of the cube index is set if `f_c < 0.0`. This is strict (0 counts as outside), the same as the raycast hit test.

**12.3 Edge vertex** (normative). Always interpolate from the owner `a` to `b = a + e_k`:
```rust
let t = fa / (fa - fb);                                   // signs differ => non-zero; t in [0,1] [R-DIV]
let mut pos = centre(a); pos[k] = O::fma(t, voxel_size, centre(a)[k]);
// colour per channel: c = O::fma(t, cb as f32 - ca as f32, ca as f32); round half up + clamp as §8.3
// normal: grad(g)[k] = f(g+e_k) - f(g-e_k) if both valid; else f(g+e_k) - f(g) if g+e_k valid;
//         else f(g) - f(g-e_k) if g-e_k valid; else 0
let n = [O::fma(t, gb[0] - ga[0], ga[0]), O::fma(t, gb[1] - ga[1], ga[1]), O::fma(t, gb[2] - ga[2], ga[2])];
let l2 = O::fma(n[0], n[0], O::fma(n[1], n[1], n[2] * n[2]));
let n = if l2 > 0.0 { let inv = 1.0 / O::sqrt(l2); [n[0] * inv, n[1] * inv, n[2] * inv] } else { [0.0; 3] };  // toward +tsdf
```

**12.4 Per-block meshes** (normative). Block K's mesh is a pure function of its 3×3×3 block neighbourhood.
- **Cubes:** local min corner at l ∈ [0,B)³.
- **Vertices:** one per crossed edge (o,k) with o ∈ [0,B]³ that touches at least one emitted cube of K. They are ordered by `e = 3·(o_x + (B+1)·(o_y + (B+1)·o_z)) + k`.
- **Triangles:** cubes in ascending `l_x + B·(l_y + B·l_z)`, then TRI_TABLE order.
- **Winding:** counter-clockwise (CCW) seen from free space. If Bourke's order is the opposite, emit (i0, i2, i1). `tsdf_mc_winding_outward` decides: on a sphere, face normal · gradient > 0.
- Face vertices are duplicated in the neighbouring block's list (as nvblox). A weld pass is not part of v0.

**12.5 Global order.**
- Submaps by ascending `SubmapId`, then blocks by ascending key, then the 12.4 order.
- Vertex indices are global u32 values from an integer exclusive prefix sum of the per-block vertex counts. More than `u32::MAX` vertices gives `MeshTooLarge`.
- On the GPU: count, then scan, then emit.
- `extract_mesh_into` is byte-identical to `update_mesh_cache` run from empty followed by `write_into`.

**12.6 Incremental dirty set.**
- `J + {−1,0,1}³` with normals on, because normals read ±1 neighbours.
- `J + {−1,0}³` with normals off.

The set is taken for every block modified or allocated since the cache epoch.

**12.7 World frame.**
- `v_w = xform(R_ws, t_ws, v)` and `n_w = rot(R_ws, n)`, without renormalising.
- `world_from_submap` is rounded once per call.

**12.8 Point cloud.** For each block in ascending key, emit the edges it owns: o ∈ [0,B)³, both endpoints valid, a sign change. Order by ascending e and compute by 12.3. There are no duplicates.

**12.9 PLY.**
- `write_ply_mesh` writes binary little-endian.
- `element vertex` has `float x y z`, plus optional `float nx ny nz` and optional `uchar red green blue`.
- `element face` has `property list uchar uint vertex_indices`.
- The output bytes are deterministic.

## 13. Raycast

**Inputs.**
- A pinhole camera without distortion, of size W×H (the size of the outputs).
- `volume_from_camera`, giving `R_vc`, `t_vc` and `R_cv` (§4).
- `RaycastParams`:
  - `depth_min ≥ 0` and `depth_max > depth_min`, both camera z;
  - `weight_threshold ≥ 1`;
  - `max_steps`, defaulting to `2·ceil((depth_max−depth_min)/voxel_size) + 2`, computed in f64 on the host (the factor 2 pays for re-marching skipped blocks after a back-off);
  - `colors`.
- `block_step = (B as f32) * voxel_size`.

**March:**
```rust
let xn = (px as f32 - cx) * inv_fx; let yn = (py as f32 - cy) * inv_fy;
let dir = rot::<O>(&r_vc, [xn, yn, 1.0]);
let ray = |z: f32| [O::fma(dir[0], z, t_vc[0]), O::fma(dir[1], z, t_vc[1]), O::fma(dir[2], z, t_vc[2])];
let mut z = depth_min; let mut prev: Option<(f32, f32)> = None;     // (f, z)
let mut hit = None;
let mut fine_until = depth_min;                           // None below this steps one voxel, not one block
let mut skipped = false;                                  // the last step was a block skip
for _ in 0..max_steps {
    if !(z < depth_max) { break }
    match voxel_at(ray(z)) {                              // §5.2 floor lookup; None: outside domain or block unallocated
        None => {
            if z < fine_until { z = z + voxel_size; skipped = false; }
            else { z = z + block_step; skipped = true; }
            prev = None;
        }
        Some((_, w)) if w < weight_threshold => { z = z + voxel_size; prev = None; skipped = false; }
        Some((f, _)) if skipped && f < 0.0 => {           // BACK-OFF (v1.1)
            fine_until = z;
            z = (z - block_step) + voxel_size;            // plain sub then add, never fused
            skipped = false;
        }
        Some((f, _)) => {
            skipped = false;
            if let Some((fp, zp)) = prev { if fp >= 0.0 && f < 0.0 { hit = Some((fp, zp, f, z)); break } }
            prev = Some((f, z));
            let st = f * trunc;
            z = z + if st > voxel_size { st } else { voxel_size };   // >= voxel_size, no ±0 tie
        }
    }
}
// HIT (fp, zp, f, z):
let a = fp / (fp - f);                                    // fp >= 0 > f => denominator > 0 [R-DIV]
let z_hit = O::fma(a, z - zp, zp);
let vertex = [xn * z_hit, yn * z_hit, z_hit];             // camera frame; depth = z_hit
```
Each step except a back-off adds at least `voxel_size`, which is much larger than ulp(z). A back-off moves z back once, by less than one block, and the unallocated samples below `fine_until` then step one voxel each, so no second back-off can happen before z passes `fine_until`. `max_steps` is the hard cap.

**Why the back-off (v1.1).** v1 stepped a whole block over every unallocated sample. That step can land already behind the surface (`f < 0` with no previous sample), and then the crossing is never found. Reproducer: a fronto-parallel plane at 1.2 m, `depth_min` 0.1, voxel 2 cm, B = 8. The centre ray samples z = 1.06 (unallocated), then 1.22 (behind the plane), and v1 misses. Pinned by `tsdf_raycast_block_skip_back_off` and `tsdf_raycast_multi_submap_nearest`.

**Normal** at `ray(z_hit)`:
```rust
let p = ray(z_hit);
let q = [O::fma(p[0], inv_voxel, -0.5), O::fma(p[1], inv_voxel, -0.5), O::fma(p[2], inv_voxel, -0.5)];
// gate each q in [-G_F, G_F - 1) else normal = 0
let g0 = [floor_i32(q[0]), floor_i32(q[1]), floor_i32(q[2])];
let al = [q[0] - g0[0] as f32, q[1] - g0[1] as f32, q[2] - g0[2] as f32];   // may round to 1.0: harmless
// all 8 corners g0 + {0,1}^3 valid (allocated, weight >= threshold), else normal = [0,0,0]
let lerp = |a: f32, b: f32, t: f32| O::fma(t, b - a, a);
let gx = lerp(lerp(f100 - f000, f110 - f010, al[1]), lerp(f101 - f001, f111 - f011, al[1]), al[2]);
let gy = lerp(lerp(f010 - f000, f110 - f100, al[0]), lerp(f011 - f001, f111 - f101, al[0]), al[2]);
let gz = lerp(lerp(f001 - f000, f101 - f100, al[0]), lerp(f011 - f010, f111 - f110, al[0]), al[1]);
let gc = rot::<O>(&r_cv, [gx, gy, gz]);
let l2 = O::fma(gc[0], gc[0], O::fma(gc[1], gc[1], gc[2] * gc[2]));
let mut n = if l2 > 0.0 { let inv = 1.0 / O::sqrt(l2); [gc[0] * inv, gc[1] * inv, gc[2] * inv] } else { [0.0; 3] };
if O::fma(n[0], vertex[0], O::fma(n[1], vertex[1], n[2] * vertex[2])) > 0.0 { n = [-n[0], -n[1], -n[2]]; }  // face the camera
```

**Colour.** When `colors` is set, take the rgb of the voxel containing `ray(z_hit)` if that voxel is valid, otherwise 0.

**Output on a miss.** Depth 0, vertex [0,0,0], normal [0,0,0], colour 0. A hit whose normal is invalid keeps its vertex. ICP checks `is_valid_vertex`/`is_valid_normal` (`rgbd.rs:134-138, 250-254`).

**ICP hook (constructors only) [J].**
- New constructors: `RgbdLevel::zeroed(camera, size)` and `RgbdPyramid::zeroed(camera, size, num_levels)`.
  - Their level sizes and intrinsics match what `RgbdPyramid::from_depth_m` produces.
  - `depth_sigmas = None`.
  - Distortion is rejected as in `check_camera`.
- `raycast_level_into` and `raycast_pyramid_into` refill these in place, one raycast per level at that level's camera.
- This is needed because ICP consumes `&RgbdPyramid` (`icp.rs:504-525`) and requires equal level counts (`RgbdIcpError::PyramidLevelsMismatch`, `rgbd.rs:70-77`).

**Multi-submap.**
- Submaps are marched in ascending `SubmapId`, each in its own `submap_from_camera`.
- The smallest `z_hit` wins; on a tie, the lower id wins.
- Culling on the host in f64 once per call is allowed.
- Overlapping submaps are never fused voxel-wise.

## 14. What must be identical vs what may differ

**MUST be bit-identical** across scalar x86_64, scalar aarch64, NEON and CUDA; any thread count; any run; any interleaving that meets §11; and streaming vs batch:
- **Volume and submap state:**
  - the set of allocated keys;
  - per key: tsdf bits (including the sign of zero), weight and rgbw;
  - the key→pool map, `sorted_keys` and `modified_epoch`.
- **Counters and decisions:**
  - `IntegrateStats`, `StepReport`, `BatchReport` and `content_hash()`;
  - submap ids, spawn decisions and submap poses.
- **Outputs:**
  - mesh arrays and their order, PLY bytes, the point cloud;
  - raycast maps and `RgbdLevel`/`RgbdPyramid` maps;
  - `range_image_from_points`;
  - `atan2_det` for every finite input pair;
  - every call's Ok/Err outcome and error variant (one exception, v1.2: see below).

**MAY differ:**
- hash slots, probe sequences and hash capacity;
- the order of touched buffers, of miss lists before sorting, and of visible lists;
- thread and launch geometry; fused vs per-observation launches; exact culling;
- the residency set, LRU order, arena page ids, and host vs device placement;
- workspace capacities and growth timing;
- timing and addresses.
- **(v1.2)** `PoolFull::needed` when a step's miss set reaches ≥ 2^16 slots (`MAX_EXACT_MISS_SLOTS`) and the pool can never hold the misses: CUDA reports a lower bound, the CPU the exact count. The variant (`PoolFull`) and `capacity` still match; below 2^16 slots `needed` matches exactly.

## 15. Public Rust API

Kornia flavour; zero allocation per warmed frame (§17).

```rust
// ---- crates/kornia-3d/src/tsdf/mod.rs ----
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum TsdfError {
    #[error(transparent)] Depth(#[from] crate::depth::DepthError),
    #[error(transparent)] Image(#[from] kornia_image::ImageError),
    #[error("invalid {field}: {value}")] InvalidParameter { field: &'static str, value: f64 },
    #[error("invalid pinhole intrinsics fx={fx} fy={fy} cx={cx} cy={cy}")] InvalidIntrinsics { fx: f64, fy: f64, cx: f64, cy: f64 },
    #[error("camera has non-zero distortion; undistort first")] DistortedCamera,
    #[error("transform is not finite or its rotation is not orthonormal with det +1")] InvalidTransform,
    #[error("invalid LiDAR beam table: {0}")] InvalidBeamTable(&'static str),
    #[error("invalid azimuth grid start={start} step={step} columns={columns}")] InvalidAzimuthGrid { start: f64, step: f64, columns: usize },
    #[error("image is {got:?}, expected {expected:?}")] SizeMismatch { expected: ImageSize, got: ImageSize },
    #[error("colour given for a non-pinhole sensor")] ColorOnRangeSensor,
    #[error("block ({x}, {y}, {z}) is outside the key range")] KeyOutOfRange { x: i32, y: i32, z: i32 },
    #[error("sensor {0:?} is not in the rig")] UnknownSensor(SensorId),
    #[error("sensor {0:?} given twice")] DuplicateSensor(SensorId),
    #[error("agent {0:?} is not registered")] UnknownAgent(AgentId),
    #[error("agent {0:?} registered twice")] DuplicateAgent(AgentId),
    #[error("submap {0:?} not found")] UnknownSubmap(SubmapId),
    #[error("step ({agent:?}, {stamp}) given twice")] DuplicateStep { agent: AgentId, stamp: u64 },
    #[error("agent {agent:?}: stamp {got} <= last applied {last}")] StaleStamp { agent: AgentId, last: u64, got: u64 },
    #[error("block pool full: capacity {capacity}, needed {needed}")] PoolFull { capacity: usize, needed: usize },
    #[error("all {max} submap slots are in use")] TooManySubmaps { max: usize },
    #[error("mesh has {vertices} vertices, more than u32 indices address")] MeshTooLarge { vertices: usize },
    #[error("floating-point environment is not round-to-nearest without flush-to-zero")] FpEnvironment,
    #[error("batch step {index} failed: {source}")] Batch { index: usize, #[source] source: Box<TsdfError> },
    #[cfg(feature = "cuda")] #[error("CUDA error: {0}")] Cuda(String),
    #[cfg(feature = "cuda")] #[error("device buffer belongs to another stream")] StreamMismatch,
    #[cfg(feature = "cuda")] #[error("CUDA device has compute capability {major}.{minor}; the TSDF kernels need 7.5 or newer")] UnsupportedComputeCapability { major: i32, minor: i32 },
    #[cfg(feature = "cuda")] #[error("built without the oxide PTX (stub)")] KernelsUnavailable,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)] pub struct AgentId(pub u32);
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)] pub struct SensorId(pub u32);
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)] pub struct SubmapId { pub agent: AgentId, pub seq: u32 }

pub struct BlockDim<const B: usize>;
pub trait SupportedBlockDim: sealed::Sealed { const LOG2: u32; const VOXELS: usize; }      // BlockDim<8>, BlockDim<16>
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)] pub struct BlockKey(u64);
impl BlockKey { pub fn new(x: i32, y: i32, z: i32) -> Result<Self, TsdfError>; pub fn coords(self) -> [i32; 3]; pub fn bits(self) -> u64; }
#[derive(Debug, Clone, Copy, PartialEq)] pub struct TsdfLayout { /* voxel_size, half_voxel, inv_voxel, trunc_voxels, trunc, inv_trunc */ }
impl TsdfLayout {
    /// `voxel_size`: metres, finite, [1e-4, 10], 1/voxel_size normal. `trunc_voxels`: finite, [1, 64] (VBG trunc_voxel_multiplier).
    pub fn new(voxel_size: f32, trunc_voxels: f32) -> Result<Self, TsdfError>;
    pub fn voxel_size(&self) -> f32;
    pub fn truncation(&self) -> f32;                                                       // metres
    pub fn centre_from_voxel(&self, voxel: [i32; 3]) -> [f32; 3];
    pub fn voxel_from_point(&self, point: [f32; 3]) -> Option<[i32; 3]>;
}
#[derive(Debug, Clone, Copy, PartialEq)] pub struct DepthRange { pub min: f32, pub max: f32 }   // metres, [min, max), 0 < min < max, finite
impl DepthRange { pub fn new(min: f32, max: f32) -> Result<Self, TsdfError>; }

pub struct PinholeSensor { /* §6 params */ }
impl PinholeSensor { pub fn new(camera: &PinholeCamera, size: ImageSize, range: DepthRange) -> Result<Self, TsdfError>;
                     pub fn size(&self) -> ImageSize; pub fn camera(&self) -> &PinholeCamera; }
#[derive(Debug, Clone, Copy, PartialEq)] pub struct AzimuthGrid { pub start: f64, pub step: f64, pub wrap: bool }   // radians
pub struct LidarTables<'a> { pub fov_low: f32, pub fov_high: f32, pub row_boundaries: &'a [f32], pub cos_el: &'a [f32], pub sin_el: &'a [f32],
                             pub cos_az: &'a [f32], pub sin_az: &'a [f32], pub inv_az_step: f32, pub az_offset: f32 }
pub struct SphericalLidar { /* §7.2 tables */ }
impl SphericalLidar {
    pub fn new(size: ImageSize, azimuth: AzimuthGrid, beam_elevations: &[f64], range: DepthRange) -> Result<Self, TsdfError>;   // row 0 first, strictly decreasing
    pub fn uniform(size: ImageSize, azimuth: AzimuthGrid, top: f64, bottom: f64, range: DepthRange) -> Result<Self, TsdfError>;
    pub fn from_tables(size: ImageSize, azimuth: AzimuthGrid, tables: &LidarTables<'_>, range: DepthRange) -> Result<Self, TsdfError>;
    pub fn size(&self) -> ImageSize;
    pub fn range_image_from_points_into(&self, points_sensor: &[[f32; 3]], dst: &mut DepthImage<f32>) -> Result<(), TsdfError>;
}
pub enum RangeSensor { Pinhole(PinholeSensor), Spherical(SphericalLidar) }
pub fn atan2_det(y: f32, x: f32) -> f32;                                                    // §7.4 with StdOps
pub enum DepthRef<'a> { U16(&'a DepthImage<u16>), F32(&'a DepthImage<f32>) }                // + From impls
pub struct SensorObservation<'a> { pub sensor: &'a RangeSensor, pub volume_from_sensor: Pose3d,
                                   pub depth: DepthRef<'a>, pub color: Option<&'a Image<u8, 3>> }

#[derive(Debug, Clone, Copy, PartialEq, Eq)] pub enum PoolGrowth { Fixed, Geometric }
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)] #[non_exhaustive]
pub enum TsdfKernel { Scalar, #[default] Simd }        // as IcpKernel; Simd = NEON on aarch64, scalar elsewhere; bit-identical
#[derive(Debug, Clone, PartialEq)]
pub struct TsdfParams { pub max_weight: u16, pub max_color_weight: u8, pub color: bool, pub block_capacity: usize, pub growth: PoolGrowth }
impl Default for TsdfParams { /* max_weight 65535, max_color_weight 16, color true, block_capacity 1 << 16, Geometric */ }
#[derive(Debug, Default)] pub struct TsdfWorkspace { /* touched keys, misses, marks, merge scratch, batch order, sensor masks, MC counts/offsets, raycast scratch, FP-env state */ }
impl TsdfWorkspace { pub fn new(kernel: TsdfKernel) -> Self; pub fn kernel(&self) -> TsdfKernel; pub fn reserve(&mut self, hint: &WorkspaceHint); }
#[derive(Debug, Clone, Copy, Default)] pub struct WorkspaceHint { pub pixels_per_step: usize, pub observations_per_step: usize, pub blocks: usize, pub steps_per_batch: usize }
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct IntegrateStats { pub valid_pixels: u64, pub touched_blocks: u32, pub new_blocks: u32, pub updated_voxels: u64, pub out_of_range: u64 }
#[derive(Debug, Clone, Copy, PartialEq)] pub struct Voxel { pub tsdf: f32, pub weight: u16, pub rgb: [u8; 3], pub color_weight: u8 }
pub struct BlockView<'a> { pub key: BlockKey, pub tsdf: &'a [f32], pub weight: &'a [u16], pub rgbw: Option<&'a [u32]> }

pub struct TsdfVolume<const B: usize = 8> where BlockDim<B>: SupportedBlockDim { /* layout, params, pools, hash, sorted_keys, epochs */ }
impl<const B: usize> TsdfVolume<B> where BlockDim<B>: SupportedBlockDim {
    pub fn new(layout: TsdfLayout, params: TsdfParams) -> Result<Self, TsdfError>;
    pub fn layout(&self) -> &TsdfLayout;
    pub fn params(&self) -> &TsdfParams;
    /// One step: §9 allocation over all observations, then §10 per observation in slice order.
    pub fn integrate(&mut self, ws: &mut TsdfWorkspace, observations: &[SensorObservation<'_>]) -> Result<IntegrateStats, TsdfError>;
    pub fn raycast_into(&self, ws: &mut TsdfWorkspace, camera: &PinholeCamera, volume_from_camera: &Pose3d, params: &RaycastParams, maps: &mut RaycastMaps) -> Result<(), TsdfError>;
    pub fn raycast_level_into(&self, ws: &mut TsdfWorkspace, volume_from_camera: &Pose3d, params: &RaycastParams, level: &mut RgbdLevel) -> Result<(), TsdfError>;
    pub fn raycast_pyramid_into(&self, ws: &mut TsdfWorkspace, volume_from_camera: &Pose3d, params: &RaycastParams, pyramid: &mut RgbdPyramid) -> Result<(), TsdfError>;
    pub fn extract_mesh_into(&self, ws: &mut TsdfWorkspace, params: &MeshParams, mesh: &mut TriangleMesh) -> Result<(), TsdfError>;
    pub fn update_mesh_cache(&self, ws: &mut TsdfWorkspace, params: &MeshParams, cache: &mut BlockMeshCache) -> Result<MeshUpdateStats, TsdfError>;
    pub fn extract_point_cloud_into(&self, ws: &mut TsdfWorkspace, params: &MeshParams, points: &mut Vec<[f32; 3]>,
                                    colors: Option<&mut Vec<[u8; 3]>>, normals: Option<&mut Vec<[f32; 3]>>) -> Result<(), TsdfError>;
    pub fn reserve(&mut self, additional_blocks: usize) -> Result<(), TsdfError>;
    pub fn num_blocks(&self) -> usize;
    pub fn block_keys(&self) -> impl ExactSizeIterator<Item = BlockKey> + '_;               // ascending
    pub fn last_step_blocks(&self) -> impl Iterator<Item = BlockKey> + '_;                  // ascending
    pub fn block(&self, key: BlockKey) -> Option<BlockView<'_>>;
    pub fn voxel(&self, voxel: [i32; 3]) -> Option<Voxel>;
    pub fn memory_bytes(&self) -> usize;
    /// FNV-1a 64 over blocks in ascending key: key (u64 LE), then per voxel tsdf bits (u32 LE), weight (u16 LE), rgbw (u32 LE, if colour).
    pub fn content_hash(&self) -> u64;
}
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RaycastParams { pub depth_min: f32, pub depth_max: f32, pub weight_threshold: u16, pub max_steps: u32, pub colors: bool }
impl RaycastParams { pub fn new(layout: &TsdfLayout, depth_min: f32, depth_max: f32) -> Result<Self, TsdfError>; }   // threshold 1, default max_steps, no colour
pub struct RaycastMaps { /* depth: Image<f32,1>, vertices: Image<f32,3>, normals: Image<f32,3>, colors: Option<Image<u8,3>> */ }
impl RaycastMaps { pub fn new(size: ImageSize, colors: bool) -> Result<Self, TsdfError>; pub fn size(&self) -> ImageSize;
                   pub fn depth(&self) -> &Image<f32, 1>; pub fn vertices(&self) -> &Image<f32, 3>;
                   pub fn normals(&self) -> &Image<f32, 3>; pub fn colors(&self) -> Option<&Image<u8, 3>>; }
#[derive(Debug, Clone, Copy, PartialEq)] pub struct MeshParams { pub weight_threshold: u16, pub normals: bool, pub colors: bool }
pub struct BlockMeshCache { /* slab (§17) + cache epoch */ }
impl BlockMeshCache { pub fn new() -> Self; pub fn with_capacity(vertices: usize, triangles: usize, blocks: usize) -> Self;
                      pub fn write_into(&self, mesh: &mut TriangleMesh) -> Result<(), TsdfError>; }
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)] pub struct MeshUpdateStats { pub remeshed_blocks: u32, pub vertices: u64, pub triangles: u64 }

// ---- crates/kornia-3d/src/mesh.rs ----
#[derive(Debug, Clone, Default)]
pub struct TriangleMesh { pub vertices: Vec<[f32; 3]>, pub normals: Vec<[f32; 3]>, pub colors: Vec<[u8; 3]>, pub triangles: Vec<[u32; 3]> }
impl TriangleMesh { pub fn new() -> Self; pub fn clear(&mut self); }                         // clear keeps capacity
// ---- crates/kornia-3d/src/io/ply ----
pub fn write_ply_mesh(path: impl AsRef<std::path::Path>, mesh: &TriangleMesh) -> Result<(), PlyError>;
// ---- crates/kornia-3d/src/registration/rgbd.rs (constructors only, no ICP change) ----
impl RgbdLevel { pub fn zeroed(camera: &PinholeCamera, size: ImageSize) -> Result<RgbdLevel, RgbdIcpError>; }
impl RgbdPyramid { pub fn zeroed(camera: &PinholeCamera, size: ImageSize, num_levels: usize) -> Result<RgbdPyramid, RgbdIcpError>; }

// ---- crates/kornia-3d/src/tsdf/submap.rs ----
pub struct RigSensor { pub id: SensorId, pub sensor: RangeSensor, pub body_from_sensor: Pose3d }
pub struct SensorRig { /* sorted by id */ }
impl SensorRig { pub fn new(sensors: Vec<RigSensor>) -> Result<Self, TsdfError>; pub fn sensors(&self) -> &[RigSensor];
                 pub fn get(&self, id: SensorId) -> Option<&RigSensor>; }
pub struct SensorFrame<'a> { pub sensor: SensorId, pub depth: DepthRef<'a>, pub color: Option<&'a Image<u8, 3>> }
pub struct AgentStep<'a> { pub agent: AgentId, pub stamp: u64, pub world_from_body: Pose3d, pub frames: &'a [SensorFrame<'a>] }
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)] pub enum SubmapFrame { #[default] TranslationAtSpawn, BodyAtSpawn }
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpawnPolicy { pub max_steps: u32, pub max_translation: f64, pub max_rotation: f64, pub max_blocks: usize, pub frame: SubmapFrame }
// u32::MAX / f64::INFINITY / usize::MAX disable a criterion; max_rotation in [0, π]
#[derive(Debug, Clone, PartialEq)]
pub struct SubmapParams { pub layout: TsdfLayout, pub volume: TsdfParams /* block_capacity = per-submap cap */, pub arena_blocks: usize,
                          pub max_submaps: usize, pub max_resident_bytes: Option<usize> }
#[derive(Debug, Clone, Copy, PartialEq)] pub struct SubmapPose { pub id: SubmapId, pub world_from_submap: Pose3d }
#[derive(Debug, Clone, Copy)] pub enum SubmapFilter<'a> { All, Agents(&'a [AgentId]), Submaps(&'a [SubmapId]) }
#[derive(Debug, Clone, Copy, PartialEq, Eq)] pub struct StepReport { pub submap: SubmapId, pub spawned: bool, pub stats: IntegrateStats }
#[derive(Debug, Clone, Copy, PartialEq, Eq)] pub struct BatchReport { pub steps: usize, pub spawned: usize, pub stats: IntegrateStats }
#[derive(Debug, Clone, Copy, PartialEq, Eq)] pub struct MemoryReport { pub host_bytes: usize, pub device_bytes: usize, pub blocks: usize }
pub struct SubmapMeshCache { /* one BlockMeshCache slab per submap header */ }
impl SubmapMeshCache { pub fn new() -> Self;
    pub fn write_into<const B: usize>(&self, map: &SubmapTsdf<B>, filter: SubmapFilter<'_>, mesh: &mut TriangleMesh) -> Result<(), TsdfError>
        where BlockDim<B>: SupportedBlockDim; }
pub struct SubmapTsdf<const B: usize = 8> where BlockDim<B>: SupportedBlockDim { /* arena, headers, agents, poses */ }
impl<const B: usize> SubmapTsdf<B> where BlockDim<B>: SupportedBlockDim {
    pub fn new(params: SubmapParams) -> Result<Self, TsdfError>;
    pub fn add_agent(&mut self, agent: AgentId, rig: SensorRig, spawn: SpawnPolicy) -> Result<(), TsdfError>;
    pub fn integrate(&mut self, ws: &mut TsdfWorkspace, step: &AgentStep<'_>) -> Result<StepReport, TsdfError>;
    pub fn integrate_batch(&mut self, ws: &mut TsdfWorkspace, steps: &[AgentStep<'_>]) -> Result<BatchReport, TsdfError>;
    pub fn set_submap_poses(&mut self, poses: &[SubmapPose]) -> Result<(), TsdfError>;
    pub fn submap_ids(&self) -> impl Iterator<Item = SubmapId> + '_;                      // ascending
    pub fn active_submap(&self, agent: AgentId) -> Option<SubmapId>;
    pub fn submap_pose(&self, id: SubmapId) -> Option<Pose3d>;                           // world_from_submap
    pub fn extract_mesh_into(&self, ws: &mut TsdfWorkspace, filter: SubmapFilter<'_>, params: &MeshParams, mesh: &mut TriangleMesh) -> Result<(), TsdfError>;
    pub fn update_mesh_cache(&self, ws: &mut TsdfWorkspace, params: &MeshParams, cache: &mut SubmapMeshCache) -> Result<MeshUpdateStats, TsdfError>;
    pub fn raycast_into(&self, ws: &mut TsdfWorkspace, camera: &PinholeCamera, world_from_camera: &Pose3d, filter: SubmapFilter<'_>,
                        params: &RaycastParams, maps: &mut RaycastMaps) -> Result<(), TsdfError>;
    pub fn raycast_pyramid_into(&self, ws: &mut TsdfWorkspace, world_from_camera: &Pose3d, filter: SubmapFilter<'_>,
                                params: &RaycastParams, pyramid: &mut RgbdPyramid) -> Result<(), TsdfError>;
    pub fn memory_bytes(&self, agent: Option<AgentId>) -> MemoryReport;
    pub fn content_hash(&self, id: SubmapId) -> Option<u64>;
}
#[cfg(feature = "cuda")] pub struct CudaTsdfVolume<const B: usize = 8> where BlockDim<B>: SupportedBlockDim { /* device pools allocated once, owned stream */ }
#[cfg(feature = "cuda")] pub struct CudaSubmapTsdf<const B: usize = 8> where BlockDim<B>: SupportedBlockDim { /* + residency */ }
// Same methods as the host types, taking device or host-accessible images (zero-copy on unified memory),
// plus download_into(&self, &mut TsdfVolume<B>) and content_hash(). Below sm_75: UnsupportedComputeCapability;
// stub PTX: KernelsUnavailable; no NVRTC twin. Separate CUDA types follow repo precedent (CudaDepthProjector, CudaVoxelFilter).
```

**API conventions:**
- **Argument order:** the workspace comes first, as in `icp_projective_plane_with(&mut ws, ..)` (`icp.rs:504-525`).
- **Kernel choice** sits in `TsdfWorkspace::new(kernel)`, like `IcpWorkspace::new` (`icp.rs:491-496`).
- **Host types** reject device-resident images with `ImageError::UnsupportedDevice` (`depth/mod.rs:753-759`).
- **Docs:** every public item gets rustdoc with Arguments, Returns, Errors and Example, plus a README entry and a changelog fragment (kornia-api-flavor rule 10).

Usage:
```rust
let layout = TsdfLayout::new(0.01, 4.0)?;
let mut map = SubmapTsdf::<16>::new(SubmapParams { layout, volume: TsdfParams::default(), arena_blocks: 1 << 18, max_submaps: 256, max_resident_bytes: None })?;
map.add_agent(AgentId(0), rig_a, spawn)?;  map.add_agent(AgentId(1), rig_b, spawn)?;
let mut ws = TsdfWorkspace::new(TsdfKernel::Simd);
map.integrate_batch(&mut ws, &[AgentStep { agent: AgentId(1), stamp: 7, world_from_body, frames: &frames_b }])?;
map.set_submap_poses(&poses_from_pgo)?;            // kornia_3d::pgo, intra- and inter-agent edges
map.extract_mesh_into(&mut ws, SubmapFilter::All, &MeshParams { weight_threshold: 1, normals: true, colors: true }, &mut mesh)?;
```

## 16. Python surface

This lives in `kornia-py/src/tsdf.rs`, is registered in `k3d_mod` (`kornia-py/src/lib.rs:575`), and has stubs in `k3d.pyi`. It follows Open3D's VoxelBlockGrid.

| Open3D `t.geometry.VoxelBlockGrid` | kornia `k3d.TsdfVolume` |
|---|---|
| `VoxelBlockGrid(attr_names, attr_dtypes, attr_channels, voxel_size, block_resolution=16, block_count, device)` | `TsdfVolume(voxel_size, block_resolution=16, block_count, trunc_voxel_multiplier=8.0, max_weight=65535, max_color_weight=16, color=True, device="cpu"\|"cuda:0")` |
| `compute_unique_block_coordinates` + `integrate(block_coords, depth, color, intrinsic, extrinsic, depth_scale, depth_max, trunc_voxel_multiplier)` | `integrate(depth, color, intrinsic, extrinsic, depth_scale=1000.0, depth_max=3.0, depth_min=0.001)` (allocation fused, §9); `integrate_multi([...])` |
| `ray_cast(block_coords, intrinsic, extrinsic, width, height, ..., depth_scale, depth_min, depth_max, weight_threshold)` | `ray_cast(intrinsic, extrinsic, width, height, depth_scale=1000.0, depth_min=0.1, depth_max=3.0, weight_threshold=3.0)` → dict `depth`, `vertex`, `normal`, `color` (numpy; output depth = metres × depth_scale in the binding) |
| `extract_triangle_mesh(weight_threshold)` | `extract_triangle_mesh(weight_threshold=3.0)` → `TriangleMesh` (numpy views) |
| `extract_point_cloud(weight_threshold)` | `extract_point_cloud(weight_threshold=3.0)` |

**Semantics:**
- `extrinsic` is the 4×4 camera_from_world, which is the Open3D convention and `Pose3d`'s documented meaning.
- `depth_scale` = `meter`.
- Weights are integers, so a float `weight_threshold` t is applied as `weight >= ceil(t)`.
- `trunc_voxel_multiplier` is a property of the volume. `integrate(..., trunc_voxel_multiplier=x)` raises `ValueError` if x differs from it.
- `depth_min` must be > 0.

**Also exposed:** `SphericalLidar`, `PinholeSensor`, `SubmapTsdf` (with `submap_ids`, `submap_pose`, `set_submap_poses`), `save_ply`, `num_blocks`, `memory_bytes`.

**Interop:** numpy inputs are zero-copy, the GIL is released, and `cuda` builds accept DLPack. Python calls the same Rust, so it produces the same bits. Python allocates its output arrays on every call; this is outside the Rust zero-allocation contract.

## 17. Zero-allocation contract

**A warmed frame makes 0 heap allocations**, for both Scalar and Simd, and on the CUDA host side, for each of:
- `TsdfVolume::integrate` with one sensor and with several;
- `SubmapTsdf::integrate` and `integrate_batch` with several agents;
- `raycast_into`, `raycast_level_into`, `raycast_pyramid_into`;
- `update_mesh_cache` within the slab capacity;
- `extract_mesh_into` into a `TriangleMesh` that already has the capacity;
- spawning a submap within `max_submaps`;
- `range_image_from_points_into`.

**Allowed only on a cold frame:**
- workspace geometric growth;
- `PoolGrowth::Geometric` reserve plus the hash rehash;
- `TriangleMesh`/`Vec` growth;
- slab growth or compaction;
- the first park of a submap;
- rayon pool creation;
- CUDA module load.

Errors may allocate (`Batch` boxes its source).

**Storage rules [J]:**
- **`sorted_keys`:** reserved to capacity and merged in place from the back.
- **Sorting:** `sort_unstable*` only. Stable `sort` allocates.
- **Submaps:** use the preallocated arena and headers (§11).
- **`BlockMeshCache` is a slab:**
  - each block has an (offset, len) into shared vertex and triangle arrays;
  - a re-meshed block that still fits stays in place, otherwise it is appended;
  - compaction happens on cold frames only.
- **rayon:** use `par_iter`/`par_chunks`/`join` only (stack jobs). Never `scope`/`spawn` (boxed jobs) on a warm path.

**CUDA host [J].** cudarc 0.19's `LaunchArgs` holds three `Vec`s and pushes on every `.arg()` (`launch.rs:49-56, 94-96`), so every launch through `launch_builder` allocates. TSDF launchers MUST instead:
- launch through `cudarc::driver::result::launch_kernel(f.cu_function(), grid, block, smem, stream.cu_stream(), &mut params)` with a stack array of parameter pointers (`result.rs:1325`, `core.rs:2355, 787`);
- create device buffers with event tracking disabled (`CudaContext::disable_event_tracking`, `core.rs:493`);
- copy device-to-host (D2H) with `memcpy_dtoh` into preallocated host buffers, never `clone_dtoh`;
- never call `cudaMalloc` per frame.

CUDA-graph replay (plan ladder 3) also qualifies.

**Enforcement:** `crates/kornia-3d/tests/tsdf_alloc.rs`.
- It is its own binary with ONE test function and `--test-threads=1`.
- It uses a counting global allocator with a process-wide `AtomicUsize`. The counter in `icp_alloc.rs:19-21` is `thread_local` and would miss rayon workers.
- It snapshots the counter around the warmed call, with no printing in between.

## 18. PTX gate extension (`kornia-oxide-kernels/ptx_gate.rs`)

These rules apply to every float kernel in the crate. `device/src/color.rs` is integer-only and unaffected.

- **G0 (existing, unchanged):**
  - `.version 6.3` and `.target sm_75`;
  - no `.extern` except `.extern .shared`;
  - the `.entry` set equals the scanned `#[kernel]` set;
  - every `ld`/`ldu`/`st`/`atom`/`red` names a state space;
  - no `__nv_` symbol;
  - no `fma.` or float `mad.` outside an `.entry` listed in `FMA_ALLOWED` (`ptx_gate.rs:4-110`).
- **G1:** in every `.entry` and `.func`, reject `add|sub|mul` `.f32|.f64` without a rounding modifier.
  - The bare form appears whenever llc ran with `-fp-contract=fast` (`ptx.rs:761-763`).
  - ptxas or the driver JIT may fuse it. That fusion is draft-reported; the PTX ISA treats unrounded mul/add as fusable.
- **G2:** reject:
  - `.ftz`, `.approx`, `div.full`;
  - `rsqrt.`, `sin.`, `cos.`, `ex2.`, `lg2.`, `tanh.`;
  - float `min.`/`max.`;
  - `.sat` on float ops.

  Explicitly ALLOW `rcp.rn.f32`, which is how LLVM lowers `1.0/x` (core PTX 122, 239). Reject only `rcp.approx*`.
- **G3:** reject float `atom.*`/`red.*`.
- **G4:** `FMA_ALLOWED` is exactly these seven entries:
  - `tsdf_touch_pinhole`, `tsdf_touch_spherical`
  - `tsdf_integrate_pinhole`, `tsdf_integrate_spherical`
  - `tsdf_raycast`, `tsdf_mc_emit`
  - `tsdf_range_from_points`

  In these entries, every `fma.` must sit between `// begin inline asm` and `// end inline asm`: only `PtxOps` emits fma (oxide PTX 223-225). The current gate drops comments at `ptx_gate.rs:45`, so it must read these markers first. `tsdf_mc_count` and the hash/alloc kernels contain no fma.
- **G5 (runtime):** `tsdf_contraction_probe_bit_exact`, run per entry family, with inputs where RN(RN(a·b)+c) ≠ RN(a·b+c). Existing evidence: gpu_test:14 matched the CPU's unfused results on 1048576 random triples through the driver JIT.
- **Tests:** each rule gets fixtures in `tests/ptx_gate.rs`.

## 19. Risk register

| ID | Sites | Failure mode | Rule / enforcement | Status |
|---|---|---|---|---|
| R-CONTRACT | every unfused mul followed by add/sub | LLVM contraction without NO_FMA (`device_codegen.rs:1196`, `ptx.rs:761-763`); JIT fusion of bare forms | `build_ptx` only (`build_ptx.rs:179`); G1; G4; CPU Rust never contracts | NO_FMA → `.rn` VERIFIED; G1/G4 to implement |
| R-DIV | `1/p_z`, `z/r`, `mn/den`, `RCP(n)`, colour `rc`, MC `t`, raycast `fp/(fp−f)`, `1/sqrt(l2)` (MC, raycast) | approximate division | IEEE `/` per element; `div.rn`/`rcp.rn` only (G2); `vdivq_f32`; host reciprocals per P3 | `div.rn` VERIFIED; `rcp.rn` bit test PENDING |
| R-FMA | every `O::fma` | NEON addend slot, non-inlined `.func`, helper drift | `#[inline(always)]`; `_bit_exact` with fused≠unfused inputs; exact `FMA_ALLOWED` | asm fma VERIFIED |
| R-LIBDEVICE | device `f32::mul_add/sqrt`, `core::f32::math`, `libm` | libdevice route (`call.rs:374, 396`): `.rn` only with libdevice and PREC reflect (`ptx.rs:780-784`), no PTX without it | forbidden; `PtxOps` | VERIFIED |
| R-ARCH | `cuda_device::float::*_rn_f32` | need sm_80/PTX 7.0 | forbidden while ARCH = sm_75 | VERIFIED |
| R-SQRT | LiDAR `r`, normal lengths | approximate sqrt | `sqrt.rn` (`PtxOps`), `vsqrtq_f32`, `f32::sqrt`; G2 | VERIFIED |
| R-ENV | all CPU float code | FTZ/DAZ/RMode from a fast-math library | read-only check → `FpEnvironment` | test PENDING |
| R-MINMAX | clamps | ±0/NaN semantics (CPU minNum vs device select) | compare+select; NEON exception only for the tsd clamp; clippy `disallowed-methods`; G2 | rule |
| R-CVT | every float→int | saturation/NaN out of range | ordered range gate first | in-range VERIFIED (gpu_test:10-13) |
| R-INT2F | g, px, py, n, colour | inexact above 2^24 | \|g\|<2^22, W,H≤2^24, n≤65536 | host checks |
| R-PARAM | params | different rounding paths | one host fn, one `#[repr(C)]` struct | `tsdf_params_single_rounding` |
| R-HOSTMATH | LiDAR tables, spawn cos | platform libm last bit, x86 vs aarch64 | `libm::sin/cos`; `from_tables`; golden bits on both hosts | draft-reported risk |
| R-SIMD | lane helpers | `F32x4/8::mul_add` fused only on aarch64/+fma (`simd.rs:100-113`) | forbidden; `core::arch::aarch64` | VERIFIED source |
| R-X86FMA | scalar x86_64 without `+fma` | C library `fmaf` | `tsdf_stdops_fma_matches_libm` per host | rule |
| R-X87 | i686 without SSE2 | extended precision | `compile_error!` | rule |
| R-TABLE | MC and RCP tables | copies diverge | single source; fingerprint; RCP generated and tested | rule |
| R-ORDER | §§9-12 | arrival/hash/thread order leaks | sorted outputs; fold order | tests |
| R-ATOMIC | GPU alloc, counters, range image | atomic-counter pool index (gpu_test:5) | atomics as sets/counts/min-of-bits; rank misses | VERIFIED hazard |
| R-ZERO | init, sdf, atan2_det, sign tests | −0 vs +0 | +0 init; `< 0.0` tests; raw-bit hash | `tsdf_signed_zero_rules` |
| R-NAN | gates, outputs | NaN payloads differ CPU/GPU (`core/parity/src/main.rs:37-39`) | nothing NaN stored or emitted; range gate before div/atan2 | `tsdf_nan_inputs_never_stored` |
| R-RANGE | indices | u32 overflow | W·H < 2^31 (v1.2); cap·B³ ≤ u32::MAX; `MeshTooLarge` | host checks |
| R-NEONTAIL | NEON loops | masked lanes | no gathers in masked lanes; scalar tail | tails test |
| R-EDITION | shared core | 2021 host vs 2024 device | edition-neutral code; both builds in CI | v1 build VERIFIED |
| R-LAYOUT | core packaging | relative path dep breaks staging | file in `device/src`, `#[path]` | b-rel failure VERIFIED |
| R-RAYCAST | march | data-dependent path | exact voxel bits; host `max_steps` | `tsdf_raycast_bit_exact` |
| R-LAUNCH | CUDA launches | `LaunchArgs` Vec pushes | raw `launch_kernel` | VERIFIED hazard |
| R-COUNTER | alloc test | thread_local counter misses workers | global `AtomicUsize` | rule |

## 20. Tests

1. **atan2_det**
   - `atan2_det_bit_exact`: scalar vs NEON vs PTX, exhaustive core over [0,1], 2^24 random pairs, special values.
   - `atan2_det_error_bound`, `atan2_det_monotone`.
2. **Primitive ops**
   - `tsdf_floor_i32_matches_floor`: ±2^22, ±0, halves.
   - `tsdf_rcp_rn_bit_exact`: 1/x on device vs CPU over 2^24 values.
   - `tsdf_stdops_fma_matches_libm`.
3. **Kernel entries**
   - `tsdf_<entry>_bit_exact` for every entry: touch_pinhole, touch_spherical, integrate_pinhole, integrate_spherical, raycast, mc_count, mc_emit, range_from_points, alloc.
   - `tsdf_integrate_neon_tails_bit_exact`: all widths mod 4.
4. **Whole-volume parity**
   - `tsdf_content_hash_backends`: scalar = Simd = CUDA after N Replica frames, single camera and the multi-sensor rig, B=8 and B=16.
5. **Allocation and integration**
   - `tsdf_alloc_lookup_first_equals_full_sort`, `tsdf_pool_index_ascending` (including the GPU path).
   - `tsdf_fused_equals_per_observation`.
6. **Multi-agent**
   - `tsdf_multi_agent_interleave_bitwise`: shuffled arrival equals sorted serial replay.
   - `tsdf_submap_isolation`, `tsdf_inter_agent_pose_update_moves_only_target`.
   - `tsdf_stale_stamp_rejected`, `tsdf_duplicate_step_rejected`, `tsdf_batch_prefix_applied`.
7. **Mesh**
   - `tsdf_mesh_full_equals_incremental`, `tsdf_mc_shared_edge_vertex`, `tsdf_mc_winding_outward`, `tsdf_mc_tables_fingerprint`.
8. **Raycast and range image**
   - `tsdf_raycast_bit_exact`, `tsdf_raycast_rgbd_pyramid_layout`, `tsdf_raycast_multi_submap_nearest`.
   - `tsdf_range_from_points_bit_exact`.
9. **Numerics edge cases**
   - `tsdf_signed_zero_rules`.
   - `tsdf_nan_inputs_never_stored`: NaN, inf and subnormal depth and points.
   - `tsdf_color_round_u8`: the 255 boundary.
   - `tsdf_params_single_rounding`.
   - `tsdf_lidar_tables_golden_bits`: on both x86_64 and aarch64 CI.
   - `tsdf_fp_environment_rejected`.
10. **Gate:** `ptx_gate` fixtures for G1-G4, including `rcp.rn` allowed.
11. **Allocation:** `tsdf_alloc.rs` (§17).
12. **Mutations:** each backend must fail on at least one mutation:
   - replace one fma with mul+add;
   - swap two sensors;
   - swap the NEON fma addend slot.
13. **Output files and parity**
   - `tsdf_ply_bytes_deterministic`.
   - Python `test_tsdf.py` VBG parity: chamfer < 0.5 voxel at equal params.

## 21. Overrides of the plan (gate: do not silently revert)

- **O1.** `atan2_det` uses fma. Plan audit item 1 said "no FMA"; item 4b says FMA everywhere.
- **O2.** Storage uses u16 weights (the plan had f32) and packed `rgbw` u32 with a u8 colour weight (the plan had u8 rgb only). This gives 10 B/voxel.
- **O3.** The PTX gate gains G1-G5. Plan 4b assumed `FMA_ALLOWED` was enough.
- **O4.** The shared core uses the `F32Ops` trait, and the device shim is inline PTX asm. It does not use std, libm or `cuda_device::float`.
- **O5.** "Reciprocal multiply everywhere" applies to host constants only. Per-element divisions stay IEEE `/`.
- **O6.** Allocation scope is one step; `integrate_batch` never unions across steps.
- **O7.** The visible set is the per-observation touched set. Ladder rung 4 (culling) is allowed only if exact.
- **O8.** Multi-submap raycast: the nearest hit wins, and ties go to the lower id. The plan had left this open.
- **O9.** Per-agent stamps strictly increase across calls.
- **O10.** MC uses per-block meshes with duplicated face vertices, so full extraction equals incremental.
- **O11.** The shared core lives in `kornia-oxide-kernels/device/src/tsdf_core`, not `kornia-3d/src/tsdf/core/`. PENDING 0.1b sign-off.
- **O12.** Plan item 3 (one launch per sensor) is the reference; a fused loop is permitted when bit-identical.
- **O13.** The FP environment is checked read-only, not guarded by writing it.
- **O14.** CUDA host zero-alloc requires raw launches.
- **O15.** The ICP hook is based on `RgbdPyramid`. The plan named only the `RgbdLevel` layout.

## 22. Judge decisions (bit-changing conflicts between the drafts)

**Core numerics:**
- **Weight:** A's u16 with RCP is chosen over B's f32 with `w_obs`. It is exact integer counting, saves 2 B/voxel and keeps unit-weight semantics (plan: `w = min(w+1, max_w)`). B's `max_weight: f32`, `weight_threshold: f32`, `with_weight` and Python `max_weight=inf` are therefore replaced.
- **Transform nesting:** A's x-outermost order. It allows the CPU row hoist and matches the spike core.
- **Visible domain:** A's per-observation touched set, not B's union `V_step`. It is cheaper by construction, gives content invariant to how a step is grouped, and needs no culling for correctness.
- **atan2_det:** A's coefficient bits (they diverge from B's at C3), plus B's monotonicity test.
- **Truncation:** A's explicit 1.0 for `sdf ≥ trunc`.
- **Constants:** f32-sourced reciprocals are computed in f32 (B), not as f64 rounded to f32 (A).
- **Voxel domain:** A's \|g\| < 2^22. The centre uses B's fma form, which is bit-identical under that domain.
- **Pinhole cheirality:** A's `p_z ≥ range.min`.

**LiDAR:**
- **Elevation:** A's `sz = z/r` with a boundary count, not B's multiply-compare. It has fewer formulas and allows a lookup table.
- **Azimuth:** B's `AzimuthGrid` (signed step, wrap or partial FOV) with an integer `rem_euclid` wrap, plus an fma column formula [J].

**Raycast:**
- **Parameter:** A's camera-z parameter, consistent with the projective SDF and with unprojection.
- **Zero crossing:** B's `fp >= 0 && f < 0`, which matches MC inside = `f < 0`. A used `<= 0`, which contradicted its own MC.
- **Trilinear q:** B's single-fma `q`.
- **Colour:** B's nearest-voxel colour, which needs fewer ops.

**Storage and meshing:**
- **Colour storage:** packed `rgbw` u32 with the u8 colour weight in the top byte [J graft]. It needs one load/store per voxel.
- **Pool index:** normative (D1-A).
- **MC:** A's per-block scheme (full equals incremental). The point cloud uses B's owned edges (no duplicates). MC normals use A's one-sided fallbacks.

**Environment, device, gate:**
- **FP environment:** a read-only check [J].
- **Device shim:** spike `PtxOps` [J].
- **Gate rcp:** A's rule (allow `rcp.rn`).

**API:**
- B is the base, with A's workspace-first argument order and `TsdfKernel`/`TsdfWorkspace::new(kernel)`, following the `IcpKernel` precedent.
- Separate `Cuda*` types follow repo precedent.
- A's `content_hash`, `voxel()` and `IntegrateStats` counters are grafted in.

## 23. Spike results folded in, and what stays PENDING

- **P1 (lowering): RESOLVED.**
  - `/` lowers to `div.rn`.
  - `1.0/x` lowers to `rcp.rn`; the bit test is pending.
  - `floor` lowers to `cvt.rmi`.
  - `as i32` lowers to `cvt.rzi` plus `selp`.
  - std/libm sqrt and fma go through libdevice:
    - with libdevice they emit `.rn` (`results/sqrt_std/kernels.ptx:157`, `results/mul_add_std/kernels.ptx:166`);
    - without libdevice they emit no PTX (`results/mul_add_std.nolibdevice/build.log:45`).
  - `cuda_device::float::*_rn_f32` fail at sm_75.
  - `ptx_asm!` works at sm_75 with or without libdevice, with 0 mismatches.
- **P2: RESOLVED.** With NO_FMA, `a*b+c` lowers to `mul.rn` plus `add.rn` (oxide PTX 125-127), with 0 mismatches.
- **P3: PARTIAL.** The driver JIT honoured `.rn` on 1048576 random triples (gpu_test:14). G5 stays as a permanent test.
- **P4: PARTIAL.** The v1 layout builds. The parity harness has no run log: PENDING.
- **P5: RESOLVED, with a constraint.** u32/u64 CAS and atomic add work (gpu_test:1-4). Layout and the atomic-counter pool index vary between runs (line 5), so ranking is required.

**Still PENDING:**
- `atan2_det` PTX parity;
- the `rcp.rn` bit test;
- shared-core parity;
- the Open3D facts, which cannot be checked on this box:
  - the DepthTouch rule;
  - VBG pixel truncation vs rounding;
  - the `ray_cast` step and block-skip rule;
  - winding;
  - `>=` vs `>` for `weight_threshold`;
- the draft-reported claims:
  - ptxas FFMA fusion;
  - stable no_std lacking f32 math;
  - thumbv7em not installed.

## 24. Plan-step ownership

Each step has its own roster, copied from the plan (`/home/nvidia/.claude/plans/yes-prototype-it-in-vivid-robin.md:183-249`).

| Step | Agent(s) | agentType | model | effort | Contract sections |
|---|---|---|---|---|---|
| 0.1b | core-share-spike ×1 | oh-my-claudecode:executor-high | opus | high | §2 layout, R-EDITION, R-LAYOUT, P4 |
| 0.1 | oxide-spike ×1 | oh-my-claudecode:executor-high | opus | high | §1 PTX column, §18 facts, P1-P3, P5 |
| 0.2 | probe-o3d, probe-nvblox, probe-cudarc ×3 | oh-my-claudecode:researcher | sonnet | medium | §23 Open3D items, CUDA-graph option (§17) |
| 0.3 | data ×1 | oh-my-claudecode:executor | sonnet | low | test data for §20 |
| 0.4 | contract-a, contract-b ×2 | oh-my-claudecode:architect | opus | xhigh | drafts (inputs) |
| 0.5 | contract-judge ×1 | oh-my-claudecode:critic | opus | max | this document |
| gate | main loop | — | — | — | freeze; commit CONTRACT.md; settle §23 |
| 0.6 | measure-nvblox, measure-o3d (serial) | oh-my-claudecode:scientist-high | opus | high | — |
| 0.7 | baseline-skeptic ×1 | oh-my-claudecode:critic | opus | high | — |
| 1.1 | scalar ×1 (alone) | oh-my-claudecode:executor-high | opus | xhigh | §§2-16 core, scalar, `StdOps`, submaps, PLY, rgbd constructors |
| 1.1b | alloc-guard ×1 | oh-my-claudecode:tdd-guide | sonnet | high | §17 test, no_std check |
| 1.2 | tdd ×1 | oh-my-claudecode:tdd-guide | sonnet | high | §20 |
| 1.3 NEON | neon ×1, worktree | oh-my-claudecode:executor-high | opus | xhigh | §1 NEON column, R-NEONTAIL, R-MINMAX |
| 1.3 CUDA | cuda ×1, worktree | oh-my-claudecode:executor-high | opus | xhigh | `PtxOps`, kernels, §18 G1-G4, §17 raw launches, §5.8 ranking |
| 1.4 lens 1 | rev-exact-{neon,cuda} | oh-my-claudecode:architect | opus | high | refute vs §§1-14, §19 |
| 1.4 lens 2 | rev-safety-{neon,cuda} | pr-review-toolkit:silent-failure-hunter | sonnet | high | R-RANGE, R-CVT, R-NEONTAIL, unsafe |
| 1.5 | fix-{neon,cuda} | oh-my-claudecode:executor | sonnet | high | confirmed findings |
| 1.6 | gate-{neon,cuda} | oh-my-claudecode:qa-tester-high | opus | medium | §20 matrix and mutations |
| gate | main loop | — | — | — | 3 backends bit-identical |
| 2.1 | harness ×1 | oh-my-claudecode:executor | sonnet | high | respects §17 |
| 2.2 | measure | oh-my-claudecode:scientist | sonnet | medium | — |
| 2.3a | profile-cpu | oh-my-claudecode:scientist-high | opus | xhigh | — |
| 2.3b | profile-cuda | oh-my-claudecode:scientist-high | opus | xhigh | — |
| 2.4 | optimize | oh-my-claudecode:executor-high | opus | max | may change only §14 "may differ" |
| 2.5 | ab | oh-my-claudecode:scientist | sonnet | medium | — |
| loop | main loop | — | — | — | — |
| 2.6 | skeptic-a, skeptic-b ×2 | oh-my-claudecode:critic | opus | max | — |
| 2.7 | bench-doc ×1 | oh-my-claudecode:writer | sonnet | medium | cites §5.6, §22 |
| 3.1 python | python ×1, worktree | oh-my-claudecode:executor-high | opus | high | §16 |
| 3.1 example | example ×1, worktree | oh-my-claudecode:executor | sonnet | high | uses §15 |
| 3.1 docs | docs ×1 | oh-my-claudecode:writer | haiku | low | §15 rustdoc rule |
| 3.2 api | rev-api | pr-review-toolkit:type-design-analyzer | opus | high | §15-16 |
| 3.2 bindings | rev-bindings | oh-my-claudecode:code-reviewer | sonnet | high | §16 |
| 3.2 diff | rev-diff | pr-review-toolkit:code-reviewer | opus | xhigh | all |
| 3.2 tests | rev-tests | pr-review-toolkit:pr-test-analyzer | sonnet | medium | §20 |
| 3.3 | fix ×1 | oh-my-claudecode:executor | sonnet | high | findings |
| 3.4 | completeness ×1 | oh-my-claudecode:critic | opus | max | §23 PENDING |
| 3.5 | simplify ×1 | code-simplifier:code-simplifier | sonnet | medium | must keep §14 identical items |
| final | main loop | — | — | — | — |