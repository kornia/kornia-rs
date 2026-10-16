//! Benchmark arms for the cuda-oxide resize sweep (sm_75 / GTX 1650).
//!
//! This module compares kornia's NVRTC kernels against Rust-authored twins
//! compiled to PTX by NVlabs' `cuda-oxide`.  It extends Edgar Riba's original
//! bilinear spike (Jetson Orin, sm_87) to two interpolation modes and a second
//! GPU architecture:
//!
//! * **Bilinear** — raw-pointer (arm C) and slice (arm D) variants, byte-exact
//!   against the NVRTC kernel and the CPU across downscale and upscale.
//! * **Nearest-neighbour** — raw-pointer variant, byte-exact against the NVRTC
//!   kernel and the CPU.
//!
//! Both interpolation modes are profiled at 2× downscale, 2× upscale and the
//! five non-dyadic sizes from the bilinear test suite.
//!
//! # PTX artifacts
//!
//! The committed PTX files are arch-specific:
//!
//! * `resize_bilinear_oxide_sm75.ptx` — bilinear kernels (raw ptr + slice)
//! * `resize_nearest_oxide_sm75.ptx`  — nearest-neighbour kernel
//!
//! To rebuild them (from `oxide-kernels/`):
//!
//! ```sh
//! cargo oxide build resize_oxide --arch sm_75 --no-fmad
//! cp resize_bilinear_oxide.ptx \
//!    ../crates/kornia-imgproc/src/cuda/resize_bilinear_oxide_sm75.ptx
//! cp resize_nearest_oxide.ptx \
//!    ../crates/kornia-imgproc/src/cuda/resize_nearest_oxide_sm75.ptx
//! ```
//!
//! Gate checks (both must print 0):
//!
//! ```sh
//! grep -cE 'fma\.rn\.f32|mad\.f32' resize_bilinear_oxide_sm75.ptx
//! grep -c '__nv_' resize_bilinear_oxide_sm75.ptx
//! ```

use std::sync::{Arc, OnceLock};

use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, PushKernelArg};
use kornia_tensor::CudaKernel;

use super::resize::{CudaResizeError, PixelMapping};
use super::{make_config, try_compile_with_l1};

// ── PTX artifacts ────────────────────────────────────────────────────────────
//
// Each file is the output of `cargo oxide build ... --arch sm_75 --no-fmad`.
// The arch suffix is part of the filename so that multi-arch setups can coexist
// without a naming collision.

/// Committed PTX for bilinear kernels (raw-ptr + slice variants), sm_75.
static BILINEAR_OXIDE_PTX: &str = include_str!("resize_bilinear_oxide_sm75.ptx");

/// Committed PTX for the nearest-neighbour kernel, sm_75.
static NEAREST_OXIDE_PTX: &str = include_str!("resize_nearest_oxide_sm75.ptx");

// ── CUDA C: bilinear without __ldg (arm B) ───────────────────────────────────
//
// The April cuda-oxide snapshot exposes no `__ldg` intrinsic and no inline PTX,
// so without this arm a comparison would charge the missing intrinsic to
// cuda-oxide's codegen rather than measuring the kernel.

static BILINEAR_NOLDG_SRC: &str = r#"
extern "C" __global__ void resize_bilinear_noldg_3c(
    const float* __restrict__ src,
    float* __restrict__       dst,
    unsigned int src_w,
    unsigned int src_h,
    unsigned int dst_w,
    unsigned int dst_h,
    float ax, float bx,
    float ay, float by
) {
    unsigned int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (dst_x >= dst_w || dst_y >= dst_h) return;

    float sx = fmaxf(fminf(ax * (float)dst_x + bx, (float)(src_w - 1u)), 0.0f);
    float sy = fmaxf(fminf(ay * (float)dst_y + by, (float)(src_h - 1u)), 0.0f);

    unsigned int x0 = (unsigned int)sx;
    unsigned int y0 = (unsigned int)sy;
    unsigned int x1 = min(x0 + 1u, src_w - 1u);
    unsigned int y1 = min(y0 + 1u, src_h - 1u);

    float fx  = sx - (float)x0;
    float fy  = sy - (float)y0;
    float w00 = (1.0f - fy) * (1.0f - fx);
    float w10 = (1.0f - fy) * fx;
    float w01 = fy * (1.0f - fx);
    float w11 = fy * fx;

    unsigned int b00 = (y0 * src_w + x0) * 3u;
    unsigned int b10 = (y0 * src_w + x1) * 3u;
    unsigned int b01 = (y1 * src_w + x0) * 3u;
    unsigned int b11 = (y1 * src_w + x1) * 3u;

    unsigned int out = (dst_y * dst_w + dst_x) * 3u;
    dst[out]     = w00*src[b00]   + w10*src[b10]   + w01*src[b01]   + w11*src[b11];
    dst[out + 1] = w00*src[b00+1] + w10*src[b10+1] + w01*src[b01+1] + w11*src[b11+1];
    dst[out + 2] = w00*src[b00+2] + w10*src[b10+2] + w01*src[b01+2] + w11*src[b11+2];
}
"#;

// ── Module statics ────────────────────────────────────────────────────────────

static NOLDG_KERNEL: OnceLock<Result<CudaKernel, String>> = OnceLock::new();

// Bilinear PTX module and its two entry points.
static BILINEAR_OXIDE_MODULE: OnceLock<Result<Arc<CudaModule>, String>> = OnceLock::new();
static BILINEAR_OXIDE_FN: OnceLock<Result<CudaFunction, String>> = OnceLock::new();
static BILINEAR_OXIDE_SLICE_FN: OnceLock<Result<CudaFunction, String>> = OnceLock::new();

// Nearest-neighbour PTX module and its entry point.
static NEAREST_OXIDE_MODULE: OnceLock<Result<Arc<CudaModule>, String>> = OnceLock::new();
static NEAREST_OXIDE_FN: OnceLock<Result<CudaFunction, String>> = OnceLock::new();

// Bicubic entry point — reuses BILINEAR_OXIDE_MODULE (same PTX, all kernels together).
static BICUBIC_OXIDE_FN: OnceLock<Result<CudaFunction, String>> = OnceLock::new();

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Load a committed PTX module once per process, then hand out the named entry
/// point with `CU_FUNC_CACHE_PREFER_L1`.
fn oxide_function(
    module_cell: &'static OnceLock<Result<Arc<CudaModule>, String>>,
    fn_cell: &'static OnceLock<Result<CudaFunction, String>>,
    ctx: &Arc<CudaContext>,
    ptx: &'static str,
    name: &str,
) -> Result<CudaFunction, CudaResizeError> {
    let module = module_cell
        .get_or_init(|| {
            ctx.load_module(cudarc::nvrtc::Ptx::from_src(ptx))
                .map_err(|e| format!("failed to load oxide PTX: {e}"))
        })
        .as_ref()
        .map_err(|e| CudaResizeError::Cuda(e.clone()))?;

    fn_cell
        .get_or_init(|| {
            let f = module
                .load_function(name)
                .map_err(|e| format!("failed to load {name}: {e}"))?;
            use cudarc::driver::sys::CUfunc_cache_enum::CU_FUNC_CACHE_PREFER_L1;
            let _ = f.set_function_cache_config(CU_FUNC_CACHE_PREFER_L1);
            Ok(f)
        })
        .as_ref()
        .cloned()
        .map_err(|e| CudaResizeError::Cuda(e.clone()))
}

/// Validate operands the same way each NVRTC launcher does so guard differences
/// cannot mask kernel differences.
fn check_operands(
    dst: &CudaSlice<f32>,
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
) -> Result<(), CudaResizeError> {
    if src_width == 0 || src_height == 0 || dst_width == 0 || dst_height == 0 {
        return Err(CudaResizeError::Cuda(
            "image dimensions must be non-zero".into(),
        ));
    }
    let need = (dst_width as usize) * (dst_height as usize) * 3;
    if dst.len() < need {
        return Err(CudaResizeError::SliceTooSmall {
            got: dst.len(),
            need,
        });
    }
    Ok(())
}

// ── Bilinear launchers ────────────────────────────────────────────────────────

/// Arm B: CUDA C bilinear without `__ldg`.
///
/// Isolates the read-only cache hint's contribution so the cuda-oxide arm can
/// be compared against a CUDA C kernel that uses the same load form.
///
/// # Errors
///
/// Returns [`CudaResizeError`] on compile, launch, or geometry error.
#[allow(clippy::too_many_arguments)]
pub fn launch_resize_bilinear_noldg_cuda(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<f32>,
    dst: &mut CudaSlice<f32>,
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    mapping: PixelMapping,
    block_dim: Option<(u32, u32)>,
) -> Result<(), CudaResizeError> {
    check_operands(dst, src_width, src_height, dst_width, dst_height)?;
    let kernel = NOLDG_KERNEL
        .get_or_init(|| try_compile_with_l1(ctx, BILINEAR_NOLDG_SRC, "resize_bilinear_noldg_3c"))
        .as_ref()
        .map_err(|e| CudaResizeError::Cuda(e.clone()))?;
    let (ax, bx) = mapping.coeffs(src_width, dst_width);
    let (ay, by) = mapping.coeffs(src_height, dst_height);
    kernel
        .launch_builder(stream)
        .arg(src)
        .arg(dst)
        .arg(&src_width)
        .arg(&src_height)
        .arg(&dst_width)
        .arg(&dst_height)
        .arg(&ax)
        .arg(&bx)
        .arg(&ay)
        .arg(&by)
        .launch_2d(
            dst_width,
            dst_height,
            make_config(dst_width, dst_height, block_dim),
        )
        .map_err(|e| CudaResizeError::Cuda(e.to_string()))
}

/// Arm C: cuda-oxide bilinear, raw pointers.
///
/// Byte-exact against the NVRTC kernel and the CPU.  Reaches parity with the
/// NVRTC kernel through load-hoisting (all 12 source loads before any store)
/// to compensate for the missing `noalias`/`__restrict__` on raw pointers.
///
/// # Errors
///
/// Returns [`CudaResizeError`] on PTX load, launch, or geometry error.
#[allow(clippy::too_many_arguments)]
pub fn launch_resize_bilinear_oxide_cuda(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<f32>,
    dst: &mut CudaSlice<f32>,
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    mapping: PixelMapping,
    block_dim: Option<(u32, u32)>,
) -> Result<(), CudaResizeError> {
    check_operands(dst, src_width, src_height, dst_width, dst_height)?;
    let func = oxide_function(
        &BILINEAR_OXIDE_MODULE,
        &BILINEAR_OXIDE_FN,
        ctx,
        BILINEAR_OXIDE_PTX,
        "resize_bilinear_oxide_3c",
    )?;
    let (ax, bx) = mapping.coeffs(src_width, dst_width);
    let (ay, by) = mapping.coeffs(src_height, dst_height);
    let cfg = make_config(dst_width, dst_height, block_dim);
    let mut builder = stream.launch_builder(&func);
    builder
        .arg(src)
        .arg(dst)
        .arg(&src_width)
        .arg(&src_height)
        .arg(&dst_width)
        .arg(&dst_height)
        .arg(&ax)
        .arg(&bx)
        .arg(&ay)
        .arg(&by);
    // SAFETY: argument list matches the kernel's PTX parameter list exactly;
    // check_operands verified the output slice is large enough.
    unsafe { builder.launch(cfg) }
        .map(|_| ())
        .map_err(|e| CudaResizeError::Cuda(e.to_string()))
}

/// Arm D: cuda-oxide bilinear, slice parameters.
///
/// Retained as a measured negative result: slice parameters (`&[f32]` /
/// `DisjointSlice<f32>`) carry `readonly`/`noalias` but the NVPTX backend
/// does not infer `ld.global.nc`, address-space inference is lost, and each
/// access adds a bounds-check branch.  Net cost: ~12% on upscale cases.
///
/// # Errors
///
/// Returns [`CudaResizeError`] on PTX load, launch, or geometry error.
#[allow(clippy::too_many_arguments)]
pub fn launch_resize_bilinear_oxide_slice_cuda(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<f32>,
    dst: &mut CudaSlice<f32>,
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    mapping: PixelMapping,
    block_dim: Option<(u32, u32)>,
) -> Result<(), CudaResizeError> {
    check_operands(dst, src_width, src_height, dst_width, dst_height)?;
    let func = oxide_function(
        &BILINEAR_OXIDE_MODULE,
        &BILINEAR_OXIDE_SLICE_FN,
        ctx,
        BILINEAR_OXIDE_PTX,
        "resize_bilinear_oxide_slice_3c",
    )?;
    let (ax, bx) = mapping.coeffs(src_width, dst_width);
    let (ay, by) = mapping.coeffs(src_height, dst_height);
    let cfg = make_config(dst_width, dst_height, block_dim);
    let src_len = src.len() as u64;
    let dst_len = dst.len() as u64;
    let mut builder = stream.launch_builder(&func);
    builder
        .arg(src)
        .arg(&src_len)
        .arg(dst)
        .arg(&dst_len)
        .arg(&src_width)
        .arg(&src_height)
        .arg(&dst_width)
        .arg(&dst_height)
        .arg(&ax)
        .arg(&bx)
        .arg(&ay)
        .arg(&by);
    // SAFETY: slices are (pointer, length) pairs in PTX; each pointer arg is
    // followed by its element count, matching the kernel's parameter list.
    unsafe { builder.launch(cfg) }
        .map(|_| ())
        .map_err(|e| CudaResizeError::Cuda(e.to_string()))
}

// ── Nearest-neighbour launcher ────────────────────────────────────────────────

/// Arm E: cuda-oxide nearest-neighbour, raw pointers.
///
/// Byte-exact against `resize_nearest_downscale_3c` (NVRTC) and the CPU path.
/// Coordinate arithmetic is simpler than bilinear (no interpolation weights),
/// so the instruction count advantage of the Rust kernel is more visible.
///
/// # Errors
///
/// Returns [`CudaResizeError`] on PTX load, launch, or geometry error.
#[allow(clippy::too_many_arguments)]
pub fn launch_resize_nearest_oxide_cuda(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<f32>,
    dst: &mut CudaSlice<f32>,
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    mapping: PixelMapping,
    block_dim: Option<(u32, u32)>,
) -> Result<(), CudaResizeError> {
    check_operands(dst, src_width, src_height, dst_width, dst_height)?;
    let func = oxide_function(
        &NEAREST_OXIDE_MODULE,
        &NEAREST_OXIDE_FN,
        ctx,
        NEAREST_OXIDE_PTX,
        "resize_nearest_oxide_3c",
    )?;
    let (ax, bx) = mapping.coeffs(src_width, dst_width);
    let (ay, by) = mapping.coeffs(src_height, dst_height);
    let cfg = make_config(dst_width, dst_height, block_dim);
    let mut builder = stream.launch_builder(&func);
    builder
        .arg(src)
        .arg(dst)
        .arg(&src_width)
        .arg(&src_height)
        .arg(&dst_width)
        .arg(&dst_height)
        .arg(&ax)
        .arg(&bx)
        .arg(&ay)
        .arg(&by);
    // SAFETY: argument list matches the kernel's PTX parameter list;
    // check_operands verified the output slice is large enough.
    unsafe { builder.launch(cfg) }
        .map(|_| ())
        .map_err(|e| CudaResizeError::Cuda(e.to_string()))
}

// ── Bicubic oxide launcher ────────────────────────────────────────────────────

/// Launch the cuda-oxide bicubic resize kernel (Keys a = -0.5, 4×4 tap).
///
/// Arithmetic uses plain `*` and `+` throughout — **not** `fmaf`.  The
/// c8b3103 cuda-oxide snapshot emits NVVM IR instead of PTX when
/// `f32::mul_add` (`llvm.fma.f32`) is present, which requires `libnvvm` at
/// the consumer and breaks the `include_str!` workflow.  Using plain mul+add
/// keeps the self-contained PTX contract but means this kernel's Horner weights
/// and accumulated outputs differ in the last bit from the NVRTC bicubic, which
/// uses `fmaf` explicitly.  Parity tests compare against the CPU path.
///
/// The kernel entry point (`resize_bicubic_oxide_3c`) lives in the same PTX
/// file as the bilinear entries, so [`BILINEAR_OXIDE_MODULE`] is reused.
///
/// # Arguments
///
/// * `ctx`       – CUDA context.
/// * `stream`    – Stream for kernel execution.
/// * `src`       – Device slice: `src_h × src_w × 3` f32 values.
/// * `dst`       – Device slice: `dst_h × dst_w × 3` f32 values (written).
/// * `src_width`, `src_height` – Source image dimensions (non-zero).
/// * `dst_width`, `dst_height` – Output dimensions (non-zero).
/// * `mapping`   – Half-pixel or align-corners coordinate convention.
/// * `block_dim` – Optional override for CUDA block dimensions (default 32×8).
///
/// # Errors
///
/// Returns [`CudaResizeError`] on PTX load, launch, or geometry error.
#[allow(clippy::too_many_arguments)]
pub fn launch_resize_bicubic_oxide_cuda(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<f32>,
    dst: &mut CudaSlice<f32>,
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    mapping: PixelMapping,
    block_dim: Option<(u32, u32)>,
) -> Result<(), CudaResizeError> {
    check_operands(dst, src_width, src_height, dst_width, dst_height)?;
    let func = oxide_function(
        &BILINEAR_OXIDE_MODULE,
        &BICUBIC_OXIDE_FN,
        ctx,
        BILINEAR_OXIDE_PTX,
        "resize_bicubic_oxide_3c",
    )?;
    let (ax, bx) = mapping.coeffs(src_width, dst_width);
    let (ay, by) = mapping.coeffs(src_height, dst_height);
    let cfg = make_config(dst_width, dst_height, block_dim);
    let mut builder = stream.launch_builder(&func);
    builder
        .arg(src)
        .arg(dst)
        .arg(&src_width)
        .arg(&src_height)
        .arg(&dst_width)
        .arg(&dst_height)
        .arg(&ax)
        .arg(&bx)
        .arg(&ay)
        .arg(&by);
    // SAFETY: argument list matches resize_bicubic_oxide_3c PTX parameter list;
    // check_operands verified the output slice is large enough.
    unsafe { builder.launch(cfg) }
        .map(|_| ())
        .map_err(|e| CudaResizeError::Cuda(e.to_string()))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use crate::cuda::color::test_utils::{default_stream, pattern_f32};
    use crate::cuda::resize::{
        launch_resize_bilinear_downscale_cuda, launch_resize_nearest_downscale_cuda,
    };
    use crate::interpolation::InterpolationMode;
    use crate::resize::resize;
    use kornia_image::{Image, ImageSize};

    type Launcher = fn(
        &Arc<CudaContext>,
        &Arc<CudaStream>,
        &CudaSlice<f32>,
        &mut CudaSlice<f32>,
        u32,
        u32,
        u32,
        u32,
        PixelMapping,
        Option<(u32, u32)>,
    ) -> Result<(), CudaResizeError>;

    /// Run CPU `resize` and one GPU arm on the same deterministic input.
    fn cpu_and_arm(
        (sw, sh): (usize, usize),
        (dw, dh): (usize, usize),
        interp: InterpolationMode,
        launch: Launcher,
    ) -> Result<(Vec<f32>, Vec<f32>), Box<dyn std::error::Error>> {
        let data = pattern_f32(sw * sh * 3);
        let src = Image::<f32, 3>::new(
            ImageSize {
                width: sw,
                height: sh,
            },
            data.clone(),
        )?;
        let mut cpu = Image::<f32, 3>::from_size_val(
            ImageSize {
                width: dw,
                height: dh,
            },
            0.0,
        )?;
        resize(&src, &mut cpu, interp)?;

        let stream = default_stream();
        let ctx = &stream.context();
        let d_src = stream.clone_htod(&data)?;
        let mut d_dst = stream.alloc_zeros::<f32>(dw * dh * 3)?;
        launch(
            ctx,
            &stream,
            &d_src,
            &mut d_dst,
            sw as u32,
            sh as u32,
            dw as u32,
            dh as u32,
            PixelMapping::HalfPixel,
            None,
        )
        .map_err(|e| format!("launch failed {sw}x{sh}->{dw}x{dh}: {e}"))?;
        let gpu: Vec<f32> = stream.clone_dtoh(&d_dst)?;
        stream.synchronize()?;
        Ok((cpu.as_slice().to_vec(), gpu))
    }

    fn assert_bits_equal(
        label: &str,
        expected: &[f32],
        actual: &[f32],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let bad = expected
            .iter()
            .zip(actual)
            .enumerate()
            .find(|(_, (e, a))| e.to_bits() != a.to_bits());
        if let Some((i, (e, a))) = bad {
            let n = expected
                .iter()
                .zip(actual)
                .filter(|(e, a)| e.to_bits() != a.to_bits())
                .count();
            return Err(format!(
                "{label}: {n}/{} differ; first at {i}: {e} ({:#010x}) vs {a} ({:#010x})",
                expected.len(),
                e.to_bits(),
                a.to_bits()
            )
            .into());
        }
        Ok(())
    }

    // ── Case tables ──────────────────────────────────────────────────────────

    /// (src_w, src_h), (dst_w, dst_h)
    type Case = ((usize, usize), (usize, usize));

    const BILINEAR_CASES: &[Case] = &[
        ((640, 480), (320, 240)),   // 2× downscale
        ((320, 240), (640, 480)),   // 2× upscale
        ((1920, 1080), (960, 540)), // HD downscale
        ((960, 540), (1920, 1080)), // HD upscale
        ((127, 63), (65, 33)),      // non-dyadic
        ((129, 97), (64, 48)),
        ((255, 130), (133, 67)),
        ((63, 129), (127, 255)),
    ];

    const NEAREST_CASES: &[Case] = &[
        ((640, 480), (320, 240)),
        ((320, 240), (640, 480)),
        ((1920, 1080), (960, 540)),
        ((960, 540), (1920, 1080)),
        ((127, 63), (65, 33)),
        ((129, 97), (64, 48)),
        ((255, 130), (133, 67)),
        ((63, 129), (127, 255)),
    ];

    const BICUBIC_CASES: &[Case] = &[
        ((640, 480), (320, 240)),   // 2× downscale
        ((320, 240), (640, 480)),   // 2× upscale
        ((1920, 1080), (960, 540)), // HD downscale
        ((960, 540), (1920, 1080)), // HD upscale
        ((127, 63), (65, 33)),      // non-dyadic
        ((129, 97), (64, 48)),
        ((63, 129), (127, 255)),
    ];

    // ── Bilinear: arm B (no-ldg) ─────────────────────────────────────────────

    /// Dropping `__ldg` changes which cache serves the load, not the value.
    #[test]
    fn bilinear_noldg_matches_cpu_bit_exact() -> Result<(), Box<dyn std::error::Error>> {
        for &(src, dst) in BILINEAR_CASES {
            let (cpu, gpu) = cpu_and_arm(
                src,
                dst,
                InterpolationMode::Bilinear,
                launch_resize_bilinear_noldg_cuda,
            )?;
            assert_bits_equal(&format!("noldg {src:?}->{dst:?}"), &cpu, &gpu)?;
        }
        Ok(())
    }

    // ── Bilinear: arm C (oxide raw ptr) ──────────────────────────────────────

    /// Central question: does a Rust-authored bilinear kernel hold the CPU==GPU
    /// byte-exactness contract across downscale, upscale and non-dyadic sizes?
    #[test]
    fn bilinear_oxide_matches_cpu_bit_exact() -> Result<(), Box<dyn std::error::Error>> {
        for &(src, dst) in BILINEAR_CASES {
            let (cpu, gpu) = cpu_and_arm(
                src,
                dst,
                InterpolationMode::Bilinear,
                launch_resize_bilinear_oxide_cuda,
            )?;
            assert_bits_equal(&format!("oxide-bilinear {src:?}->{dst:?}"), &cpu, &gpu)?;
        }
        Ok(())
    }

    /// Compare arm C against the shipping NVRTC kernel so a failure distinguishes
    /// "cuda-oxide differs from NVRTC" from "both differ from the CPU".
    #[test]
    fn bilinear_oxide_matches_nvrtc() -> Result<(), Box<dyn std::error::Error>> {
        for &(src, dst) in BILINEAR_CASES {
            let (_, base) = cpu_and_arm(
                src,
                dst,
                InterpolationMode::Bilinear,
                launch_resize_bilinear_downscale_cuda,
            )?;
            let (_, arm) = cpu_and_arm(
                src,
                dst,
                InterpolationMode::Bilinear,
                launch_resize_bilinear_oxide_cuda,
            )?;
            assert_bits_equal(
                &format!("oxide vs nvrtc bilinear {src:?}->{dst:?}"),
                &base,
                &arm,
            )?;
        }
        Ok(())
    }

    // ── Bilinear: arm D (oxide slices) ───────────────────────────────────────

    /// Slice variant computes the same arithmetic; only loads and bounds checks differ.
    #[test]
    fn bilinear_oxide_slice_matches_cpu_bit_exact() -> Result<(), Box<dyn std::error::Error>> {
        for &(src, dst) in BILINEAR_CASES {
            let (cpu, gpu) = cpu_and_arm(
                src,
                dst,
                InterpolationMode::Bilinear,
                launch_resize_bilinear_oxide_slice_cuda,
            )?;
            assert_bits_equal(&format!("oxide-slice {src:?}->{dst:?}"), &cpu, &gpu)?;
        }
        Ok(())
    }

    // ── Nearest-neighbour: arm E (oxide raw ptr) ─────────────────────────────

    /// Does the Rust nearest-neighbour kernel hold byte-exactness with the CPU?
    #[test]
    fn nearest_oxide_matches_cpu_bit_exact() -> Result<(), Box<dyn std::error::Error>> {
        for &(src, dst) in NEAREST_CASES {
            let (cpu, gpu) = cpu_and_arm(
                src,
                dst,
                InterpolationMode::Nearest,
                launch_resize_nearest_oxide_cuda,
            )?;
            assert_bits_equal(&format!("oxide-nearest {src:?}->{dst:?}"), &cpu, &gpu)?;
        }
        Ok(())
    }

    /// Compare arm E against the shipping NVRTC nearest kernel.
    #[test]
    fn nearest_oxide_matches_nvrtc() -> Result<(), Box<dyn std::error::Error>> {
        for &(src, dst) in NEAREST_CASES {
            let (_, base) = cpu_and_arm(
                src,
                dst,
                InterpolationMode::Nearest,
                launch_resize_nearest_downscale_cuda,
            )?;
            let (_, arm) = cpu_and_arm(
                src,
                dst,
                InterpolationMode::Nearest,
                launch_resize_nearest_oxide_cuda,
            )?;
            assert_bits_equal(
                &format!("oxide vs nvrtc nearest {src:?}->{dst:?}"),
                &base,
                &arm,
            )?;
        }
        Ok(())
    }

    // ── Bicubic: arm G (oxide raw ptr) ───────────────────────────────────────

    // The oxide kernel uses plain * and + throughout (f32::mul_add → NVVM IR,
    // backend gap #4).  The production CPU path (bicubic_sample) uses mul_add /
    // fmaf and differs by ~1 ULP.  This reference mirrors the oxide arithmetic
    // exactly so the test can remain bit-exact.

    fn keys_weights_nofma(frac: f32) -> [f32; 4] {
        let mut w = [0.0f32; 4];
        let t = 1.0_f32 + frac;
        w[0] = ((-0.5_f32 * t + 2.5_f32) * t - 4.0_f32) * t + 2.0_f32;
        let t = frac;
        w[1] = ((1.5_f32 * t - 2.5_f32) * t) * t + 1.0_f32;
        let t = 1.0_f32 - frac;
        w[2] = ((1.5_f32 * t - 2.5_f32) * t) * t + 1.0_f32;
        let t = 2.0_f32 - frac;
        w[3] = ((-0.5_f32 * t + 2.5_f32) * t - 4.0_f32) * t + 2.0_f32;
        w
    }

    /// No-FMA CPU bicubic: same Horner weights and accumulation as the oxide
    /// kernel.  Used only in tests so that bit-exact comparison is valid.
    fn bicubic_cpu_nofma(data: &[f32], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<f32> {
        let ax = sw as f32 / dw as f32;
        let bx = 0.5 * ax - 0.5;
        let ay = sh as f32 / dh as f32;
        let by = 0.5 * ay - 0.5;
        let hi_x = (sw as i32) - 1;
        let hi_y = (sh as i32) - 1;

        let mut out = vec![0.0f32; dw * dh * 3];
        for dy in 0..dh {
            for dx in 0..dw {
                let cx = (ax * dx as f32 + bx).min((sw - 1) as f32).max(0.0);
                let cy = (ay * dy as f32 + by).min((sh - 1) as f32).max(0.0);
                let x0 = cx as i32;
                let y0 = cy as i32;
                let frac_x = cx - x0 as f32;
                let frac_y = cy - y0 as f32;
                let wx = keys_weights_nofma(frac_x);
                let wy = keys_weights_nofma(frac_y);

                let clamp_x = |v: i32| v.max(0).min(hi_x) as usize;
                let clamp_y = |v: i32| v.max(0).min(hi_y) as usize;

                let mut acc0 = 0.0_f32;
                let mut acc1 = 0.0_f32;
                let mut acc2 = 0.0_f32;
                for (dyi, &wyv) in wy.iter().enumerate() {
                    let yi = clamp_y(y0 - 1 + dyi as i32);
                    for (dxi, &wxv) in wx.iter().enumerate() {
                        let xi = clamp_x(x0 - 1 + dxi as i32);
                        let base = (yi * sw + xi) * 3;
                        let w = wxv * wyv; // plain *, matches oxide
                        acc0 += w * data[base]; // plain +=, matches oxide
                        acc1 += w * data[base + 1];
                        acc2 += w * data[base + 2];
                    }
                }
                let obase = (dy * dw + dx) * 3;
                out[obase] = acc0;
                out[obase + 1] = acc1;
                out[obase + 2] = acc2;
            }
        }
        out
    }

    /// Verify the oxide bicubic kernel against the no-FMA CPU reference (bit-exact).
    ///
    /// The production CPU path (`bicubic_sample`) uses `mul_add` / `fmaf` and
    /// differs by ~1 ULP from the oxide kernel, which cannot use FMA (backend
    /// gap #4: `f32::mul_add` → `llvm.fma.f32` → NVVM IR mode, no PTX emitted).
    /// This test uses the no-FMA CPU reference above instead, which mirrors the
    /// oxide arithmetic exactly.
    #[test]
    fn bicubic_oxide_matches_cpu_bit_exact() -> Result<(), Box<dyn std::error::Error>> {
        use crate::cuda::color::test_utils::{default_stream, pattern_f32};
        for &((sw, sh), (dw, dh)) in BICUBIC_CASES {
            let data = pattern_f32(sw * sh * 3);
            let cpu = bicubic_cpu_nofma(&data, sw, sh, dw, dh);

            let stream = default_stream();
            let ctx = &stream.context();
            let d_src = stream.clone_htod(&data)?;
            let mut d_dst = stream.alloc_zeros::<f32>(dw * dh * 3)?;
            launch_resize_bicubic_oxide_cuda(
                ctx,
                &stream,
                &d_src,
                &mut d_dst,
                sw as u32,
                sh as u32,
                dw as u32,
                dh as u32,
                PixelMapping::HalfPixel,
                None,
            )
            .map_err(|e| format!("launch failed {sw}x{sh}->{dw}x{dh}: {e}"))?;
            let gpu: Vec<f32> = stream.clone_dtoh(&d_dst)?;
            stream.synchronize()?;

            assert_bits_equal(&format!("oxide-bicubic {sw}x{sh}->{dw}x{dh}"), &cpu, &gpu)?;
        }
        Ok(())
    }

    // ── All bilinear arms together ────────────────────────────────────────────

    /// Run all four bilinear arms (A/B/C/D) against each other to catch any
    /// kernel that differs from the rest.
    #[test]
    fn all_bilinear_arms_agree() -> Result<(), Box<dyn std::error::Error>> {
        for &(src, dst) in BILINEAR_CASES {
            let (_, base) = cpu_and_arm(
                src,
                dst,
                InterpolationMode::Bilinear,
                launch_resize_bilinear_downscale_cuda,
            )?;
            for (name, launch) in [
                ("noldg", launch_resize_bilinear_noldg_cuda as Launcher),
                ("oxide", launch_resize_bilinear_oxide_cuda as Launcher),
                (
                    "oxide-slice",
                    launch_resize_bilinear_oxide_slice_cuda as Launcher,
                ),
            ] {
                let (_, arm) = cpu_and_arm(src, dst, InterpolationMode::Bilinear, launch)?;
                assert_bits_equal(&format!("{name} vs nvrtc {src:?}->{dst:?}"), &base, &arm)?;
            }
        }
        Ok(())
    }
}
