//! Benchmark arms for the cuda-oxide bilinear resize spike.
//!
//! This module exists to compare kornia's NVRTC bilinear kernel against a
//! Rust-authored twin compiled to PTX by NVlabs' `cuda-oxide`. It holds two
//! extra launchers alongside the shipping one in [`super::resize`]:
//!
//! * [`launch_resize_bilinear_noldg_cuda`] — the same CUDA C source as
//!   `BILINEAR_SRC` with `__ldg` stripped. The April cuda-oxide snapshot
//!   exposes no `__ldg` intrinsic and no inline PTX, so without this arm a
//!   comparison would blame cuda-oxide's codegen for a missing intrinsic.
//! * [`launch_resize_bilinear_oxide_cuda`] — loads committed PTX emitted by
//!   `oxide-kernels/`. Its PTX parameter list is identical to the CUDA C
//!   kernel's, so the argument sequence below matches
//!   `launch_resize_bilinear_downscale_cuda` one for one.
//!
//! Both use [`make_config`] and [`PixelMapping::coeffs`], and both set
//! `CU_FUNC_CACHE_PREFER_L1`, so geometry, coefficients and the L1 carveout are
//! identical across arms and the only variable is the kernel itself.

use std::sync::{Arc, OnceLock};

use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, PushKernelArg};
use kornia_tensor::CudaKernel;

use super::resize::{CudaResizeError, PixelMapping};
use super::{make_config, try_compile_with_l1};

/// `BILINEAR_SRC` with every `__ldg(&src[i])` rewritten to `src[i]`.
///
/// Isolates what the read-only cache hint contributes, so the cuda-oxide arm
/// can be compared against a CUDA C kernel with the same load form.
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

/// PTX emitted by `oxide-kernels/` via `cargo oxide build --arch sm_87`.
///
/// Committed as an artifact so kornia's crates stay on stable Rust: only the
/// kernel crate needs the nightly cuda-oxide toolchain.
static OXIDE_PTX: &str = include_str!("resize_bilinear_oxide_sm87.ptx");

static NOLDG_KERNEL: OnceLock<Result<CudaKernel, String>> = OnceLock::new();
static OXIDE_MODULE: OnceLock<Result<Arc<CudaModule>, String>> = OnceLock::new();
static OXIDE_FN: OnceLock<Result<CudaFunction, String>> = OnceLock::new();
static OXIDE_SLICE_FN: OnceLock<Result<CudaFunction, String>> = OnceLock::new();

/// Load the committed PTX once per process and hand out the named entry point.
///
/// `prefer_l1_cache` has no module-level equivalent in cudarc, so the cache
/// config is set per function here, mirroring [`try_compile_with_l1`].
fn oxide_function(
    cell: &'static OnceLock<Result<CudaFunction, String>>,
    ctx: &Arc<CudaContext>,
    name: &str,
) -> Result<CudaFunction, CudaResizeError> {
    let module = OXIDE_MODULE
        .get_or_init(|| {
            ctx.load_module(cudarc::nvrtc::Ptx::from_src(OXIDE_PTX))
                .map_err(|e| format!("failed to load cuda-oxide PTX: {e}"))
        })
        .as_ref()
        .map_err(|e| CudaResizeError::Cuda(e.clone()))?;

    cell.get_or_init(|| {
        let f = module
            .load_function(name)
            .map_err(|e| format!("failed to load {name} from cuda-oxide PTX: {e}"))?;
        use cudarc::driver::sys::CUfunc_cache_enum::CU_FUNC_CACHE_PREFER_L1;
        let _ = f.set_function_cache_config(CU_FUNC_CACHE_PREFER_L1);
        Ok(f)
    })
    .as_ref()
    .cloned()
    .map_err(|e| CudaResizeError::Cuda(e.clone()))
}

/// Shared operand validation: mirrors the checks in
/// [`super::resize::launch_resize_bilinear_downscale_cuda`] so a mismatch in
/// any arm is a kernel difference and never a guard difference.
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

/// Arm B: the CUDA C kernel without `__ldg`.
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

/// Arm C: the cuda-oxide raw-pointer kernel.
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

    let func = oxide_function(&OXIDE_FN, ctx, "resize_bilinear_oxide_3c")?;
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
    // SAFETY: the argument sequence above matches the kernel's PTX parameter
    // list, and `check_operands` has verified the output slice is large enough.
    unsafe { builder.launch(cfg) }
        .map(|_| ())
        .map_err(|e| CudaResizeError::Cuda(e.to_string()))
}

/// Arm C2: the cuda-oxide slice kernel.
///
/// Kept because it is a measured negative result rather than an oversight:
/// `&[f32]`/`DisjointSlice` lose the address-space inference the raw-pointer
/// kernel gets (generic `ld.b32` instead of `ld.global.b32`) and add a
/// bounds-check branch per access, while buying no `ld.global.nc`. Slices are
/// fat pointers in PTX, so each one contributes an extra length parameter.
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

    let func = oxide_function(&OXIDE_SLICE_FN, ctx, "resize_bilinear_oxide_slice_3c")?;
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
    // SAFETY: slices are (pointer, length) pairs in the emitted PTX, so each
    // pointer argument is followed by its element count, matching the kernel's
    // parameter list.
    unsafe { builder.launch(cfg) }
        .map(|_| ())
        .map_err(|e| CudaResizeError::Cuda(e.to_string()))
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use crate::cuda::color::test_utils::{default_stream, pattern_f32};
    use crate::cuda::resize::launch_resize_bilinear_downscale_cuda;
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
    ///
    /// Same shape as `cuda::resize::tests::cpu_and_gpu`, but takes the launcher
    /// as a parameter so every arm is exercised through one code path — the
    /// comparison must not depend on which arm is under test.
    fn cpu_and_arm(
        (sw, sh): (usize, usize),
        (dw, dh): (usize, usize),
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
        resize(&src, &mut cpu, InterpolationMode::Bilinear)?;

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
                "{label}: {n}/{} elements differ; first at {i}: expected {e} ({:#010x}) got {a} ({:#010x})",
                expected.len(),
                e.to_bits(),
                a.to_bits()
            )
            .into());
        }
        Ok(())
    }

    /// The cases the NVRTC kernel is held to in
    /// `cuda::resize::tests`: dyadic down, dyadic up, and the odd/non-dyadic
    /// sizes that need the uncontracted-coordinate contract to stay exact.
    /// `(src_w, src_h), (dst_w, dst_h)` pairs.
    type Case = ((usize, usize), (usize, usize));

    const CASES: &[Case] = &[
        ((640, 480), (320, 240)),
        ((320, 240), (640, 480)),
        ((127, 63), (65, 33)),
        ((129, 97), (64, 48)),
        ((255, 130), (133, 67)),
        ((63, 129), (127, 255)),
    ];

    /// Arm B must keep the byte-exact contract: dropping `__ldg` changes which
    /// cache serves the load, never the value.
    #[test]
    fn noldg_matches_cpu_bit_exact() -> Result<(), Box<dyn std::error::Error>> {
        for &(src, dst) in CASES {
            let (cpu, gpu) = cpu_and_arm(src, dst, launch_resize_bilinear_noldg_cuda)?;
            assert_bits_equal(&format!("noldg {src:?}->{dst:?}"), &cpu, &gpu)?;
        }
        Ok(())
    }

    /// The spike's central question: does a Rust-authored kernel compiled by
    /// cuda-oxide hold kornia's CPU==GPU byte-exactness contract?
    #[test]
    fn oxide_matches_cpu_bit_exact() -> Result<(), Box<dyn std::error::Error>> {
        for &(src, dst) in CASES {
            let (cpu, gpu) = cpu_and_arm(src, dst, launch_resize_bilinear_oxide_cuda)?;
            assert_bits_equal(&format!("oxide {src:?}->{dst:?}"), &cpu, &gpu)?;
        }
        Ok(())
    }

    /// The slice variant computes the same arithmetic, so it must agree too;
    /// only its loads and bounds checks differ.
    #[test]
    fn oxide_slice_matches_cpu_bit_exact() -> Result<(), Box<dyn std::error::Error>> {
        for &(src, dst) in CASES {
            let (cpu, gpu) = cpu_and_arm(src, dst, launch_resize_bilinear_oxide_slice_cuda)?;
            assert_bits_equal(&format!("oxide-slice {src:?}->{dst:?}"), &cpu, &gpu)?;
        }
        Ok(())
    }

    /// Compare the arms against the shipping kernel directly, so a failure
    /// distinguishes "cuda-oxide differs from NVRTC" from "both differ from the
    /// CPU" — a CPU-only comparison cannot tell those apart.
    #[test]
    fn arms_agree_with_shipping_kernel() -> Result<(), Box<dyn std::error::Error>> {
        for &(src, dst) in CASES {
            let (_, base) = cpu_and_arm(src, dst, launch_resize_bilinear_downscale_cuda)?;
            for (name, launch) in [
                ("noldg", launch_resize_bilinear_noldg_cuda as Launcher),
                ("oxide", launch_resize_bilinear_oxide_cuda as Launcher),
                (
                    "oxide-slice",
                    launch_resize_bilinear_oxide_slice_cuda as Launcher,
                ),
            ] {
                let (_, arm) = cpu_and_arm(src, dst, launch)?;
                assert_bits_equal(&format!("{name} vs nvrtc {src:?}->{dst:?}"), &base, &arm)?;
            }
        }
        Ok(())
    }
}
