//! Scale-space construction for CUDA SIFT: base-image upsample, separable
//! Gaussian blur, and the nearest-neighbour octave subsample.
//!
//! Every launcher validates the geometry and slice lengths it depends on
//! rather than trusting its caller: these are `pub` entry points reachable from
//! outside the pipeline, so a kernel precondition checked only by the caller is
//! a memory-safety hazard for anyone else.
//!
//! # These are not the generic filters, despite the names
//!
//! Investigated 2026-07-27 and rejected: routing this module's blurs through
//! [`crate::cuda::filter::launch_separable_filter_f32`], or the octave
//! subsample through [`crate::cuda::pyramid::launch_pyrdown_f32`], **cannot**
//! preserve the bit-exactness contract. Four independent mismatches, any one of
//! which is disqualifying:
//!
//! | | `cuda/filter.rs` | here |
//! |---|---|---|
//! | border | constant-zero, out-of-range taps skipped | `BORDER_REFLECT_101` |
//! | accumulation | ascending taps, plain `acc += v * k` | column pass sums the **symmetric pair first**, then one FMA |
//! | taps | a device `CudaSlice` | baked as literals per shape (`__constant__` measured ~3x worse) |
//! | fusion | none | the DoG subtraction is folded into the vertical pass |
//!
//! The border difference alone changes every pixel within `ksize / 2` of an
//! edge, which at this module's largest kernel (27 taps) is most of the frame's
//! margin. The accumulation order is not stylistic either: the pair-sum-then-FMA
//! shape was established by diffing against the reference's own dumped
//! intermediates, and getting it wrong shifts every layer of the pyramid.
//!
//! `launch_pyrdown_f32` additionally blurs before decimating; the octave
//! subsample here is a bare stride-2 pick, because an extra blur breaks parity
//! with the reference.
//!
//! So SIFT's blur is not a specialisation of the generic separable filter — it
//! is a different filter with a similar name. Teaching the generic path
//! reflect-101 *and* pair-summing *and* baked taps would leave two
//! implementations behind one signature, which is worse than two honest ones.
//!
//! [`launch_sift_downsample_nearest_cuda_view`] should stay here too, even
//! though its *arithmetic* is general — a bare stride-2 pick. Arithmetic is the
//! wrong test; the contract is what binds:
//!
//! * Beside `pyrdown`/`pyrup` it would alias the very thing it is not. Those
//!   blur before decimating, which is the Gaussian-pyramid contract a caller
//!   reaching into that module expects. Skipping the blur is deliberate here and
//!   breaks parity with the reference, so a neighbour that silently omits it is
//!   the same name-collision trap this section exists to warn about.
//! * It is coupled to the octave geometry: the pipeline's `cw / 2` integer
//!   truncation, which the CPU twin mirrors and the oracle tests pin. Publishing
//!   it means publishing that truncation as a contract.
//! * It has exactly one caller and no prospect of a second.
//!
//! Moving it would widen the public surface for no consumer, days after that
//! surface was deliberately narrowed from 30 names to 16.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, CudaView, CudaViewMut};

use super::kernels::{
    blur_h_src, blur_h_tiled_src, blur_hv_fused_src, blur_v_tiled_src, dog_src,
    downsample_nearest_src, get_or_compile, upsample2x_src,
};
use super::SiftCudaError;
use crate::cuda::make_config;

/// Cheap, order-sensitive digest of the coefficient bits, so two kernels with
/// the same tap count but different sigmas never collide in the kernel cache.
fn kernel_digest(kernel: &[f32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &c in kernel {
        h ^= c.to_bits() as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}

fn check_dims(w: u32, h: u32) -> Result<(), SiftCudaError> {
    if w == 0 || h == 0 {
        return Err(SiftCudaError::Geometry(
            "image dimensions must be non-zero".into(),
        ));
    }
    Ok(())
}

fn check_len(slice_len: usize, need: usize) -> Result<(), SiftCudaError> {
    if slice_len < need {
        return Err(SiftCudaError::SliceTooSmall {
            got: slice_len,
            need,
        });
    }
    Ok(())
}

/// Double the image in both axes to form the base of octave 0.
///
/// `dst` must hold at least `4 * src_width * src_height` elements.
pub fn launch_sift_upsample2x_cuda_view(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src: &CudaView<'_, f32>,
    dst: &mut CudaViewMut<'_, f32>,
    src_width: u32,
    src_height: u32,
) -> Result<(), SiftCudaError> {
    check_dims(src_width, src_height)?;
    check_len(src.len(), (src_width as usize) * (src_height as usize))?;
    let (dw, dh) = (src_width * 2, src_height * 2);
    check_len(dst.len(), (dw as usize) * (dh as usize))?;

    let kernel = get_or_compile(ctx, "sift_upsample2x", upsample2x_src, "sift_upsample2x")?;
    kernel
        .launch_builder(stream)
        .arg(src)
        .arg(dst)
        .arg(&src_width)
        .arg(&src_height)
        .launch_2d(dw, dh, make_config(dw, dh, None))
        .map_err(|e| SiftCudaError::Cuda(e.to_string()))
}

/// Horizontal (row) pass of the separable Gaussian.
pub fn launch_sift_blur_h_cuda(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<f32>,
    dst: &mut CudaSlice<f32>,
    width: u32,
    height: u32,
    gauss_kernel: &[f32],
) -> Result<(), SiftCudaError> {
    check_dims(width, height)?;
    if gauss_kernel.is_empty() || gauss_kernel.len().is_multiple_of(2) {
        return Err(SiftCudaError::Geometry(format!(
            "gaussian kernel length must be odd and non-zero, got {}",
            gauss_kernel.len()
        )));
    }
    let need = (width as usize) * (height as usize);
    check_len(src.len(), need)?;
    check_len(dst.len(), need)?;

    let key = format!(
        "sift_blur_h:{}:{:016x}",
        gauss_kernel.len(),
        kernel_digest(gauss_kernel)
    );
    let kernel = get_or_compile(ctx, &key, || blur_h_src(gauss_kernel), "sift_blur_h")?;
    kernel
        .launch_builder(stream)
        .arg(src)
        .arg(dst)
        .arg(&width)
        .arg(&height)
        .launch_2d(width, height, make_config(width, height, None))
        .map_err(|e| SiftCudaError::Cuda(e.to_string()))
}

/// Vertical (symmetric column) pass of the separable Gaussian.
pub fn launch_sift_blur_v_cuda_view(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src: &CudaView<'_, f32>,
    dst: &mut CudaViewMut<'_, f32>,
    width: u32,
    height: u32,
    gauss_kernel: &[f32],
) -> Result<(), SiftCudaError> {
    check_dims(width, height)?;
    if gauss_kernel.is_empty() || gauss_kernel.len().is_multiple_of(2) {
        return Err(SiftCudaError::Geometry(format!(
            "gaussian kernel length must be odd and non-zero, got {}",
            gauss_kernel.len()
        )));
    }
    let need = (width as usize) * (height as usize);
    check_len(src.len(), need)?;
    check_len(dst.len(), need)?;

    let key = format!(
        "sift_blur_v_tiled:{}:{}:{:016x}",
        gauss_kernel.len(),
        tile_q(),
        kernel_digest(gauss_kernel)
    );
    let kernel = get_or_compile(
        ctx,
        &key,
        || blur_v_tiled_src(gauss_kernel, tile_q(), false),
        "sift_blur_v",
    )?;
    let gh = height.div_ceil(tile_q() as u32);
    kernel
        .launch_builder(stream)
        .arg(src)
        .arg(dst)
        .arg(&width)
        .arg(&height)
        .launch_2d(width, height, make_config(width, gh, None))
        .map_err(|e| SiftCudaError::Cuda(e.to_string()))
}

/// Nearest-neighbour subsample forming the base layer of the next octave.
#[allow(clippy::too_many_arguments)]
pub fn launch_sift_downsample_nearest_cuda_view(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src: &CudaView<'_, f32>,
    dst: &mut CudaViewMut<'_, f32>,
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
) -> Result<(), SiftCudaError> {
    check_dims(src_width, src_height)?;
    check_dims(dst_width, dst_height)?;
    check_len(src.len(), (src_width as usize) * (src_height as usize))?;
    check_len(dst.len(), (dst_width as usize) * (dst_height as usize))?;

    let kernel = get_or_compile(
        ctx,
        "sift_downsample_nearest",
        downsample_nearest_src,
        "sift_downsample_nearest",
    )?;
    kernel
        .launch_builder(stream)
        .arg(src)
        .arg(dst)
        .arg(&src_width)
        .arg(&src_height)
        .arg(&dst_width)
        .arg(&dst_height)
        .launch_2d(
            dst_width,
            dst_height,
            make_config(dst_width, dst_height, None),
        )
        .map_err(|e| SiftCudaError::Cuda(e.to_string()))
}

/// Difference of two adjacent Gaussian layers: `dst = upper - lower`.
pub fn launch_sift_dog_cuda(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    lower: &CudaSlice<f32>,
    upper: &CudaSlice<f32>,
    dst: &mut CudaSlice<f32>,
    width: u32,
    height: u32,
) -> Result<(), SiftCudaError> {
    check_dims(width, height)?;
    let need = (width as usize) * (height as usize);
    check_len(lower.len(), need)?;
    check_len(upper.len(), need)?;
    check_len(dst.len(), need)?;

    let kernel = get_or_compile(ctx, "sift_dog", dog_src, "sift_dog")?;
    kernel
        .launch_builder(stream)
        .arg(lower)
        .arg(upper)
        .arg(dst)
        .arg(&width)
        .arg(&height)
        .launch_2d(width, height, make_config(width, height, None))
        .map_err(|e| SiftCudaError::Cuda(e.to_string()))
}

/// Vertical Gaussian pass fused with the DoG subtract against `lower`.
///
/// Equivalent to [`launch_sift_blur_v_cuda`] followed by
/// [`launch_sift_dog_cuda`], but writes the difference from the register the
/// blur already holds instead of re-reading the layer in a second pass.
#[allow(clippy::too_many_arguments)]
pub fn launch_sift_blur_v_dog_cuda_view(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src: &CudaView<'_, f32>,
    dst: &mut CudaViewMut<'_, f32>,
    lower: &CudaView<'_, f32>,
    dog: &mut CudaViewMut<'_, f32>,
    width: u32,
    height: u32,
    gauss_kernel: &[f32],
) -> Result<(), SiftCudaError> {
    check_dims(width, height)?;
    if gauss_kernel.is_empty() || gauss_kernel.len().is_multiple_of(2) {
        return Err(SiftCudaError::Geometry(format!(
            "gaussian kernel length must be odd and non-zero, got {}",
            gauss_kernel.len()
        )));
    }
    let need = (width as usize) * (height as usize);
    check_len(src.len(), need)?;
    check_len(dst.len(), need)?;
    check_len(lower.len(), need)?;
    check_len(dog.len(), need)?;

    let key = format!(
        "sift_blur_v_tiled_dog:{}:{}:{:016x}",
        gauss_kernel.len(),
        tile_q(),
        kernel_digest(gauss_kernel)
    );
    let kernel = get_or_compile(
        ctx,
        &key,
        || blur_v_tiled_src(gauss_kernel, tile_q(), true),
        "sift_blur_v_dog",
    )?;
    let gh = height.div_ceil(tile_q() as u32);
    kernel
        .launch_builder(stream)
        .arg(src)
        .arg(dst)
        .arg(lower)
        .arg(dog)
        .arg(&width)
        .arg(&height)
        .launch_2d(width, height, make_config(width, gh, None))
        .map_err(|e| SiftCudaError::Cuda(e.to_string()))
}

/// Horizontal Gaussian pass with `TILE_P` outputs per thread.
///
/// Bit-identical to [`launch_sift_blur_h_cuda`]; see [`blur_h_tiled_src`].
/// Outputs per thread in the tiled horizontal blur.
///
/// Swept end to end on mh01 (`KORNIA_SIFT_TILE_P`, min of 4 medians):
/// 1 -> 17.66 ms, **2 -> 16.99**, 3 -> 16.98, 4 -> 17.33, 8 -> 20.5. The
/// original 4 was tuned at 1080p against a very different pipeline; at these
/// sizes the wider tile costs more in redundant edge taps than it saves in
/// loads.
pub const TILE_P: usize = 2;

/// `KORNIA_SIFT_TILE_P` overrides [`TILE_P`] for sweeps.
///
/// Read once: this sits on the per-launch path (twice per horizontal blur, ~74
/// environment scans a frame before the audit), and every sibling knob in the
/// module already gets the `OnceLock` treatment for exactly that reason.
fn tile_p() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("KORNIA_SIFT_TILE_P")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|v| *v >= 1 && *v <= 16)
            .unwrap_or(TILE_P)
    })
}

/// Outputs per thread in the tiled vertical blur.
pub const TILE_Q: usize = 2;

/// `KORNIA_SIFT_TILE_Q` overrides [`TILE_Q`] for sweeps.
fn tile_q() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("KORNIA_SIFT_TILE_Q")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|v| *v >= 1 && *v <= 16)
            .unwrap_or(TILE_Q)
    })
}

/// Register-tiled horizontal pass. Same contract as [`launch_sift_blur_h_cuda`].
pub fn launch_sift_blur_h_tiled_cuda_view(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src: &CudaView<'_, f32>,
    dst: &mut CudaViewMut<'_, f32>,
    width: u32,
    height: u32,
    gauss_kernel: &[f32],
) -> Result<(), SiftCudaError> {
    check_dims(width, height)?;
    if gauss_kernel.is_empty() || gauss_kernel.len().is_multiple_of(2) {
        return Err(SiftCudaError::Geometry(format!(
            "gaussian kernel length must be odd and non-zero, got {}",
            gauss_kernel.len()
        )));
    }
    let need = (width as usize) * (height as usize);
    check_len(src.len(), need)?;
    check_len(dst.len(), need)?;

    let key = format!(
        "sift_blur_h_tiled:{}:{}:{:016x}",
        gauss_kernel.len(),
        tile_p(),
        kernel_digest(gauss_kernel)
    );
    let kernel = get_or_compile(
        ctx,
        &key,
        || blur_h_tiled_src(gauss_kernel, tile_p()),
        "sift_blur_h_tiled",
    )?;
    let gw = width.div_ceil(tile_p() as u32);
    kernel
        .launch_builder(stream)
        .arg(src)
        .arg(dst)
        .arg(&width)
        .arg(&height)
        .launch_2d(gw, height, make_config(gw, height, None))
        .map_err(|e| SiftCudaError::Cuda(e.to_string()))
}

/// Block shape for the fused blur-HV kernel.
pub const HV_BLOCK_W: usize = 32;
/// Rows per block for the fused blur-HV kernel.
pub const HV_BLOCK_H: usize = 8;

/// Fused horizontal+vertical Gaussian in one launch. See [`blur_hv_fused_src`]
/// for why this is only profitable below a tap-count threshold.
pub fn launch_sift_blur_hv_cuda(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<f32>,
    dst: &mut CudaSlice<f32>,
    width: u32,
    height: u32,
    gauss_kernel: &[f32],
) -> Result<(), SiftCudaError> {
    check_dims(width, height)?;
    if gauss_kernel.is_empty() || gauss_kernel.len().is_multiple_of(2) {
        return Err(SiftCudaError::Geometry(format!(
            "gaussian kernel length must be odd and non-zero, got {}",
            gauss_kernel.len()
        )));
    }
    let need = (width as usize) * (height as usize);
    check_len(src.len(), need)?;
    check_len(dst.len(), need)?;

    let key = format!(
        "sift_blur_hv:{}:{}x{}:{:016x}",
        gauss_kernel.len(),
        HV_BLOCK_W,
        HV_BLOCK_H,
        kernel_digest(gauss_kernel)
    );
    let kernel = get_or_compile(
        ctx,
        &key,
        || blur_hv_fused_src(gauss_kernel, HV_BLOCK_W, HV_BLOCK_H),
        "sift_blur_hv",
    )?;
    kernel
        .launch_builder(stream)
        .arg(src)
        .arg(dst)
        .arg(&width)
        .arg(&height)
        .launch_2d(
            width,
            height,
            make_config(width, height, Some((HV_BLOCK_W as u32, HV_BLOCK_H as u32))),
        )
        .map_err(|e| SiftCudaError::Cuda(e.to_string()))
}

/// Convenience wrapper over [`launch_sift_upsample2x_cuda_view`] for whole buffers.
#[allow(clippy::too_many_arguments)]
pub fn launch_sift_upsample2x_cuda(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<f32>,
    dst: &mut CudaSlice<f32>,
    src_width: u32,
    src_height: u32,
) -> Result<(), SiftCudaError> {
    launch_sift_upsample2x_cuda_view(
        ctx,
        stream,
        &src.as_view(),
        &mut dst.as_view_mut(),
        src_width,
        src_height,
    )
}

/// Convenience wrapper over [`launch_sift_blur_h_tiled_cuda_view`] for whole buffers.
#[allow(clippy::too_many_arguments)]
pub fn launch_sift_blur_h_tiled_cuda(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<f32>,
    dst: &mut CudaSlice<f32>,
    width: u32,
    height: u32,
    gauss_kernel: &[f32],
) -> Result<(), SiftCudaError> {
    launch_sift_blur_h_tiled_cuda_view(
        ctx,
        stream,
        &src.as_view(),
        &mut dst.as_view_mut(),
        width,
        height,
        gauss_kernel,
    )
}

/// Convenience wrapper over [`launch_sift_blur_v_cuda_view`] for whole buffers.
#[allow(clippy::too_many_arguments)]
pub fn launch_sift_blur_v_cuda(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<f32>,
    dst: &mut CudaSlice<f32>,
    width: u32,
    height: u32,
    gauss_kernel: &[f32],
) -> Result<(), SiftCudaError> {
    launch_sift_blur_v_cuda_view(
        ctx,
        stream,
        &src.as_view(),
        &mut dst.as_view_mut(),
        width,
        height,
        gauss_kernel,
    )
}

/// Convenience wrapper over [`launch_sift_downsample_nearest_cuda_view`] for whole buffers.
#[allow(clippy::too_many_arguments)]
pub fn launch_sift_downsample_nearest_cuda(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<f32>,
    dst: &mut CudaSlice<f32>,
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
) -> Result<(), SiftCudaError> {
    launch_sift_downsample_nearest_cuda_view(
        ctx,
        stream,
        &src.as_view(),
        &mut dst.as_view_mut(),
        src_width,
        src_height,
        dst_width,
        dst_height,
    )
}

/// Convenience wrapper over [`launch_sift_blur_v_dog_cuda_view`] for whole buffers.
#[allow(clippy::too_many_arguments)]
pub fn launch_sift_blur_v_dog_cuda(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<f32>,
    dst: &mut CudaSlice<f32>,
    lower: &CudaSlice<f32>,
    dog: &mut CudaSlice<f32>,
    width: u32,
    height: u32,
    gauss_kernel: &[f32],
) -> Result<(), SiftCudaError> {
    launch_sift_blur_v_dog_cuda_view(
        ctx,
        stream,
        &src.as_view(),
        &mut dst.as_view_mut(),
        &lower.as_view(),
        &mut dog.as_view_mut(),
        width,
        height,
        gauss_kernel,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::color::test_utils::default_stream;
    use crate::cuda::sift::{gaussian_kernel_f32, SiftCudaConfig};

    /// Load an oracle dump written by the reference dumper:
    /// `[i32 rows][i32 cols][rows*cols f32]`.
    fn load_dump(path: &str) -> Option<(usize, usize, Vec<f32>)> {
        let bytes = std::fs::read(path).ok()?;
        let rows = i32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
        let cols = i32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
        let data = bytes[8..]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .take(rows * cols)
            .collect::<Vec<f32>>();
        assert_eq!(data.len(), rows * cols, "truncated dump: {path}");
        Some((rows, cols, data))
    }

    /// Directory of reference dumps; tests that need it are skipped when unset
    /// so the suite still runs on machines without the oracle.
    /// All oracle directories to verify against, colon-separated in
    /// `KORNIA_SIFT_ORACLE`. Verifying a single image hides bugs that only
    /// appear at odd dimensions, tiny sizes, or saturated/low-texture content.
    fn oracle_dirs() -> Vec<String> {
        std::env::var("KORNIA_SIFT_ORACLE")
            .ok()
            .map(|v| {
                v.split(':')
                    .filter(|s| !s.is_empty())
                    .map(String::from)
                    .collect()
            })
            .unwrap_or_default()
    }

    fn assert_bit_equal(got: &[f32], want: &[f32], label: &str) {
        assert_eq!(got.len(), want.len(), "{label}: length mismatch");
        let mismatches: Vec<usize> = got
            .iter()
            .zip(want.iter())
            .enumerate()
            .filter(|(_, (g, w))| g.to_bits() != w.to_bits())
            .map(|(i, _)| i)
            .collect();
        if !mismatches.is_empty() {
            let first = &mismatches[..mismatches.len().min(5)];
            let detail: Vec<String> = first
                .iter()
                .map(|&i| format!("[{i}] got {:?} want {:?}", got[i], want[i]))
                .collect();
            panic!(
                "{label}: {} of {} values differ; first: {}",
                mismatches.len(),
                got.len(),
                detail.join(", ")
            );
        }
    }

    #[test]
    fn upsample2x_matches_reference_bitwise() {
        let dirs = oracle_dirs();
        if dirs.is_empty() {
            eprintln!("KORNIA_SIFT_ORACLE unset; skipping");
            return;
        }
        for dir in dirs {
            eprintln!("  [upsample2x_matches_reference_bitwise] {dir}");
            upsample2x_matches_reference_bitwise_for(&dir);
        }
    }

    fn upsample2x_matches_reference_bitwise_for(dir: &str) {
        let (sh, sw, src) = load_dump(&format!("{dir}/gray_fpt.f32")).expect("gray_fpt dump");
        let (dh, dw, want) = load_dump(&format!("{dir}/dbl.f32")).expect("dbl dump");
        assert_eq!((dh, dw), (sh * 2, sw * 2));

        let stream = default_stream();
        let ctx = &stream.context();
        let d_src = stream.clone_htod(&src).unwrap();
        let mut d_dst = stream.alloc_zeros::<f32>(dh * dw).unwrap();
        launch_sift_upsample2x_cuda(ctx, &stream, &d_src, &mut d_dst, sw as u32, sh as u32)
            .unwrap();
        let got = stream.clone_dtoh(&d_dst).unwrap();

        assert_bit_equal(&got, &want, "upsample2x");
    }

    #[test]
    fn separable_blur_matches_reference_bitwise() {
        let dirs = oracle_dirs();
        if dirs.is_empty() {
            eprintln!("KORNIA_SIFT_ORACLE unset; skipping");
            return;
        }
        for dir in dirs {
            eprintln!("  [separable_blur_matches_reference_bitwise] {dir}");
            separable_blur_matches_reference_bitwise_for(&dir);
        }
    }

    fn separable_blur_matches_reference_bitwise_for(dir: &str) {
        let (h, w, src) = load_dump(&format!("{dir}/dbl.f32")).expect("dbl dump");
        let (bh, bw, want) = load_dump(&format!("{dir}/base.f32")).expect("base dump");
        assert_eq!((bh, bw), (h, w));

        let cfg = SiftCudaConfig::default();
        let sigma = cfg.base_sig_diff() as f64;
        let ksize = crate::cuda::sift::gaussian_ksize(sigma);
        let gk = gaussian_kernel_f32(ksize, sigma);

        let stream = default_stream();
        let ctx = &stream.context();
        let d_src = stream.clone_htod(&src).unwrap();
        let mut d_tmp = stream.alloc_zeros::<f32>(h * w).unwrap();
        let mut d_dst = stream.alloc_zeros::<f32>(h * w).unwrap();

        launch_sift_blur_h_cuda(ctx, &stream, &d_src, &mut d_tmp, w as u32, h as u32, &gk).unwrap();
        launch_sift_blur_v_cuda(ctx, &stream, &d_tmp, &mut d_dst, w as u32, h as u32, &gk).unwrap();
        let got = stream.clone_dtoh(&d_dst).unwrap();

        assert_bit_equal(&got, &want, "separable gaussian blur");
    }

    #[test]
    fn octave_downsample_matches_reference_bitwise() {
        let dirs = oracle_dirs();
        if dirs.is_empty() {
            eprintln!("KORNIA_SIFT_ORACLE unset; skipping");
            return;
        }
        for dir in dirs {
            eprintln!("  [octave_downsample_matches_reference_bitwise] {dir}");
            octave_downsample_matches_reference_bitwise_for(&dir);
        }
    }

    fn octave_downsample_matches_reference_bitwise_for(dir: &str) {
        // The base of octave 1 is the nearest-subsample of octave 0's layer
        // `n_octave_layers`.
        let n = SiftCudaConfig::default().n_octave_layers;
        let (sh, sw, src) = load_dump(&format!("{dir}/gauss_o0_l{n}.f32")).expect("src layer");
        let (dh, dw, want) = load_dump(&format!("{dir}/gauss_o1_l0.f32")).expect("dst layer");

        let stream = default_stream();
        let ctx = &stream.context();
        let d_src = stream.clone_htod(&src).unwrap();
        let mut d_dst = stream.alloc_zeros::<f32>(dh * dw).unwrap();
        launch_sift_downsample_nearest_cuda(
            ctx, &stream, &d_src, &mut d_dst, sw as u32, sh as u32, dw as u32, dh as u32,
        )
        .unwrap();
        let got = stream.clone_dtoh(&d_dst).unwrap();

        assert_bit_equal(&got, &want, "octave downsample");
    }

    #[test]
    fn full_gaussian_pyramid_matches_reference_bitwise() {
        let dirs = oracle_dirs();
        if dirs.is_empty() {
            eprintln!("KORNIA_SIFT_ORACLE unset; skipping");
            return;
        }
        for dir in dirs {
            eprintln!("  [full_gaussian_pyramid_matches_reference_bitwise] {dir}");
            full_gaussian_pyramid_matches_reference_bitwise_for(&dir);
        }
    }

    fn full_gaussian_pyramid_matches_reference_bitwise_for(dir: &str) {
        let cfg = SiftCudaConfig::default();
        let sigmas = cfg.layer_sigmas();
        let layers = cfg.n_octave_layers + 3;
        let stream = default_stream();
        let ctx = &stream.context();

        // Build the base image from the raw f32 input rather than loading it,
        // so the upsample and the base-image blur sigma are both under test.
        // Loading `base.f32` here previously masked a wrong `base_sig_diff`.
        let (gh, gw, gray) = load_dump(&format!("{dir}/gray_fpt.f32")).expect("gray_fpt dump");
        let (bh, bw) = (gh * 2, gw * 2);
        let base = {
            let d_gray = stream.clone_htod(&gray).unwrap();
            let mut d_dbl = stream.alloc_zeros::<f32>(bh * bw).unwrap();
            launch_sift_upsample2x_cuda(ctx, &stream, &d_gray, &mut d_dbl, gw as u32, gh as u32)
                .unwrap();

            let sigma = cfg.base_sig_diff() as f64;
            let gk = gaussian_kernel_f32(crate::cuda::sift::gaussian_ksize(sigma), sigma);
            let mut d_tmp = stream.alloc_zeros::<f32>(bh * bw).unwrap();
            let mut d_base = stream.alloc_zeros::<f32>(bh * bw).unwrap();
            launch_sift_blur_h_cuda(ctx, &stream, &d_dbl, &mut d_tmp, bw as u32, bh as u32, &gk)
                .unwrap();
            launch_sift_blur_v_cuda(ctx, &stream, &d_tmp, &mut d_base, bw as u32, bh as u32, &gk)
                .unwrap();
            stream.clone_dtoh(&d_base).unwrap()
        };
        let (_, _, want_base) = load_dump(&format!("{dir}/base.f32")).expect("base dump");
        assert_bit_equal(&base, &want_base, "base image");

        let n_octaves = cfg.n_octaves(bh.min(bw));

        let mut cur = base;
        let (mut ch, mut cw) = (bh, bw);

        for o in 0..n_octaves {
            if o > 0 {
                // Octave base: subsample layer `n_octave_layers` of the previous octave.
                let (ph, pw, prev) = load_dump(&format!(
                    "{dir}/gauss_o{}_l{}.f32",
                    o - 1,
                    cfg.n_octave_layers
                ))
                .expect("prev layer");
                let (nh, nw) = (ph / 2, pw / 2);
                let d_src = stream.clone_htod(&prev).unwrap();
                let mut d_dst = stream.alloc_zeros::<f32>(nh * nw).unwrap();
                launch_sift_downsample_nearest_cuda(
                    ctx, &stream, &d_src, &mut d_dst, pw as u32, ph as u32, nw as u32, nh as u32,
                )
                .unwrap();
                cur = stream.clone_dtoh(&d_dst).unwrap();
                ch = nh;
                cw = nw;
            }

            let (wh, ww, want) = load_dump(&format!("{dir}/gauss_o{o}_l0.f32")).expect("layer 0");
            assert_eq!((wh, ww), (ch, cw), "octave {o} layer 0 size");
            assert_bit_equal(&cur, &want, &format!("gauss_o{o}_l0"));

            for (i, &sigma_i) in sigmas.iter().enumerate().take(layers).skip(1) {
                let ksize = crate::cuda::sift::gaussian_ksize(sigma_i);
                let gk = gaussian_kernel_f32(ksize, sigma_i);

                let d_src = stream.clone_htod(&cur).unwrap();
                let mut d_tmp = stream.alloc_zeros::<f32>(ch * cw).unwrap();
                let mut d_dst = stream.alloc_zeros::<f32>(ch * cw).unwrap();
                launch_sift_blur_h_cuda(
                    ctx, &stream, &d_src, &mut d_tmp, cw as u32, ch as u32, &gk,
                )
                .unwrap();
                launch_sift_blur_v_cuda(
                    ctx, &stream, &d_tmp, &mut d_dst, cw as u32, ch as u32, &gk,
                )
                .unwrap();
                cur = stream.clone_dtoh(&d_dst).unwrap();

                let (_, _, want) =
                    load_dump(&format!("{dir}/gauss_o{o}_l{i}.f32")).expect("layer dump");
                assert_bit_equal(&cur, &want, &format!("gauss_o{o}_l{i}"));
            }
        }
    }

    #[test]
    fn dog_pyramid_matches_reference_bitwise() {
        let dirs = oracle_dirs();
        if dirs.is_empty() {
            eprintln!("KORNIA_SIFT_ORACLE unset; skipping");
            return;
        }
        for dir in dirs {
            eprintln!("  [dog_pyramid_matches_reference_bitwise] {dir}");
            dog_pyramid_matches_reference_bitwise_for(&dir);
        }
    }

    fn dog_pyramid_matches_reference_bitwise_for(dir: &str) {
        let cfg = SiftCudaConfig::default();
        let stream = default_stream();
        let ctx = &stream.context();

        let base_dim = {
            let (h, w, _) = load_dump(&format!("{dir}/base.f32")).expect("base dump");
            h.min(w)
        };
        let mut checked = 0usize;
        for o in 0..cfg.n_octaves(base_dim) {
            for i in 0..cfg.n_octave_layers + 2 {
                let (h, w, lower) =
                    load_dump(&format!("{dir}/gauss_o{o}_l{i}.f32")).expect("lower layer");
                let (_, _, upper) =
                    load_dump(&format!("{dir}/gauss_o{o}_l{}.f32", i + 1)).expect("upper layer");
                let (_, _, want) =
                    load_dump(&format!("{dir}/dog_o{o}_l{i}.f32")).expect("dog dump");

                let d_lo = stream.clone_htod(&lower).unwrap();
                let d_hi = stream.clone_htod(&upper).unwrap();
                let mut d_dst = stream.alloc_zeros::<f32>(h * w).unwrap();
                launch_sift_dog_cuda(ctx, &stream, &d_lo, &d_hi, &mut d_dst, w as u32, h as u32)
                    .unwrap();
                let got = stream.clone_dtoh(&d_dst).unwrap();
                assert_bit_equal(&got, &want, &format!("dog_o{o}_l{i}"));
                checked += 1;
            }
        }
        // Layer count is image-dependent (octaves scale with resolution), so
        // derive it rather than hard-coding one image's value.
        let expected = cfg.n_octaves(base_dim) * (cfg.n_octave_layers + 2);
        assert_eq!(
            checked, expected,
            "expected {expected} DoG layers for this oracle image"
        );
    }

    /// Wall-clock timing of the scale-space stages at 1080p. Enabled with
    /// `KORNIA_SIFT_BENCH=1`; skipped otherwise so CI stays fast.
    /// Roofline probe: same image, same buffers, only the TAP COUNT varies.
    /// Traffic per pass is identical (one read stream + one write); only the
    /// arithmetic and the number of loads change. If time scales with taps the
    /// kernel is compute/issue-bound; if it is flat it is DRAM-bound.
    #[test]
    fn bench_roofline_taps() {
        if std::env::var("KORNIA_SIFT_BENCH").is_err() {
            eprintln!("KORNIA_SIFT_BENCH unset; skipping");
            return;
        }
        let stream = default_stream();
        let ctx = &stream.context();
        let (w, h) = (3840usize, 2160usize); // octave-0 size, the dominant cost
        let src: Vec<f32> = (0..w * h)
            .map(|i| ((i * 2654435761usize) % 255) as f32)
            .collect();
        let d_src = stream.clone_htod(&src).unwrap();
        let mut d_dst = stream.alloc_zeros::<f32>(w * h).unwrap();

        for ksize in [5usize, 11, 17, 27, 41] {
            let sigma = (ksize as f64 - 1.0) / 8.0;
            let gk = gaussian_kernel_f32(ksize, sigma.max(0.5));
            for _ in 0..2 {
                launch_sift_blur_h_cuda(ctx, &stream, &d_src, &mut d_dst, w as u32, h as u32, &gk)
                    .unwrap();
            }
            stream.synchronize().unwrap();
            let mut ts = Vec::new();
            for _ in 0..5 {
                let t0 = std::time::Instant::now();
                launch_sift_blur_h_cuda(ctx, &stream, &d_src, &mut d_dst, w as u32, h as u32, &gk)
                    .unwrap();
                stream.synchronize().unwrap();
                ts.push(t0.elapsed().as_secs_f64() * 1e3);
            }
            ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let ms = ts[ts.len() / 2];
            let gbs = (2.0 * 4.0 * (w * h) as f64) / (ms * 1e-3) / 1e9;

            // Two-pass (H then V) vs the fused single-pass kernel.
            let mut d_b = stream.alloc_zeros::<f32>(w * h).unwrap();
            for _ in 0..2 {
                launch_sift_blur_h_cuda(ctx, &stream, &d_src, &mut d_dst, w as u32, h as u32, &gk)
                    .unwrap();
                launch_sift_blur_v_cuda(ctx, &stream, &d_dst, &mut d_b, w as u32, h as u32, &gk)
                    .unwrap();
                launch_sift_blur_hv_cuda(ctx, &stream, &d_src, &mut d_b, w as u32, h as u32, &gk)
                    .unwrap();
            }
            stream.synchronize().unwrap();

            let t0 = std::time::Instant::now();
            for _ in 0..5 {
                launch_sift_blur_h_cuda(ctx, &stream, &d_src, &mut d_dst, w as u32, h as u32, &gk)
                    .unwrap();
                launch_sift_blur_v_cuda(ctx, &stream, &d_dst, &mut d_b, w as u32, h as u32, &gk)
                    .unwrap();
            }
            stream.synchronize().unwrap();
            let t2 = t0.elapsed().as_secs_f64() * 1e3 / 5.0;

            let t1 = std::time::Instant::now();
            for _ in 0..5 {
                launch_sift_blur_hv_cuda(ctx, &stream, &d_src, &mut d_b, w as u32, h as u32, &gk)
                    .unwrap();
            }
            stream.synchronize().unwrap();
            let tf = t1.elapsed().as_secs_f64() * 1e3 / 5.0;

            eprintln!(
                "  ksize={ksize:2}: H {ms:6.3} ({gbs:4.1} GB/s) | two-pass {t2:6.3} | fused {tf:6.3} | speedup {:.2}x",
                t2 / tf
            );
        }
    }

    #[test]
    fn bench_scale_space_1080p() {
        if std::env::var("KORNIA_SIFT_BENCH").is_err() {
            eprintln!("KORNIA_SIFT_BENCH unset; skipping");
            return;
        }
        let cfg = SiftCudaConfig::default();
        let stream = default_stream();
        let ctx = &stream.context();
        // KORNIA_SIFT_WH=WxH overrides the input size. Resolution is the only
        // lever that moves a bandwidth-bound pyramid: time tracks pixel count.
        let (w, h) = std::env::var("KORNIA_SIFT_WH")
            .ok()
            .and_then(|v| {
                let (a, b) = v.split_once('x')?;
                Some((a.parse().ok()?, b.parse().ok()?))
            })
            .unwrap_or((1920usize, 1080usize));

        let src: Vec<f32> = (0..w * h)
            .map(|i| ((i * 2654435761usize) % 255) as f32)
            .collect();
        let d_src = stream.clone_htod(&src).unwrap();

        let sigmas = cfg.layer_sigmas();
        let layers = cfg.n_octave_layers + 3;
        let base_sigma = cfg.base_sig_diff() as f64;
        let base_gk =
            gaussian_kernel_f32(crate::cuda::sift::gaussian_ksize(base_sigma), base_sigma);
        let layer_gks: Vec<Vec<f32>> = (1..layers)
            .map(|i| gaussian_kernel_f32(crate::cuda::sift::gaussian_ksize(sigmas[i]), sigmas[i]))
            .collect();

        // `first_octave`: -1 (the OpenCV/COLMAP/VLFeat default) doubles the base
        // image and so quadruples every downstream pixel count; 0 starts at
        // native resolution. Set KORNIA_SIFT_FIRSTOCT=0 to measure the latter.
        let upsample = std::env::var("KORNIA_SIFT_FIRSTOCT").as_deref() != Ok("0");

        // Preallocate once and ping-pong, mirroring what the plan object does:
        // no per-octave allocation and no device-to-device copies per layer.
        let (bw, bh) = if upsample { (w * 2, h * 2) } else { (w, h) };
        let mut buf_a = stream.alloc_zeros::<f32>(bw * bh).unwrap();
        let mut buf_b = stream.alloc_zeros::<f32>(bw * bh).unwrap();
        let mut buf_c = stream.alloc_zeros::<f32>(bw * bh).unwrap();
        let mut buf_d = stream.alloc_zeros::<f32>(bw * bh).unwrap();

        let run = |stream: &std::sync::Arc<cudarc::driver::CudaStream>,
                   a: &mut cudarc::driver::CudaSlice<f32>,
                   b: &mut cudarc::driver::CudaSlice<f32>,
                   c: &mut cudarc::driver::CudaSlice<f32>,
                   d: &mut cudarc::driver::CudaSlice<f32>| {
            if upsample {
                launch_sift_upsample2x_cuda(ctx, stream, &d_src, a, w as u32, h as u32).unwrap();
            } else {
                stream.memcpy_dtod(&d_src, a).unwrap();
            }
            // The base blur runs on the doubled image -- the single largest
            // plane in the pyramid -- so it gets the same tiled kernel the
            // octave loop uses, not the plain one.
            // KORNIA_SIFT_BASE=1 stops after the upsample, 2 after blur-H.
            let base_stage = std::env::var("KORNIA_SIFT_BASE")
                .ok()
                .and_then(|v| v.parse::<u32>().ok())
                .unwrap_or(3);
            if base_stage >= 2 {
                launch_sift_blur_h_tiled_cuda(ctx, stream, a, b, bw as u32, bh as u32, &base_gk)
                    .unwrap();
            }
            if base_stage >= 3 {
                launch_sift_blur_v_cuda(ctx, stream, b, c, bw as u32, bh as u32, &base_gk).unwrap();
            }
            // `c` now holds the octave base.
            let (mut cw, mut ch) = (bw, bh);
            // Cumulative-octave timing: running with KORNIA_SIFT_MAXOCT=K for
            // K = 1..n and differencing gives each octave's cost with ONE sync,
            // instead of a sync per octave that inflates the first one.
            let max_oct = std::env::var("KORNIA_SIFT_MAXOCT")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(usize::MAX);
            for _o in 0..cfg.n_octaves(bh.min(bw)).min(max_oct) {
                if cw < 16 || ch < 16 {
                    break;
                }
                for gk in layer_gks.iter() {
                    // c -> (blur h) b -> (blur v) a ; DoG(c, a) -> d ; then a becomes c
                    launch_sift_blur_h_tiled_cuda(ctx, stream, c, b, cw as u32, ch as u32, gk)
                        .unwrap();
                    launch_sift_blur_v_dog_cuda(ctx, stream, b, a, c, d, cw as u32, ch as u32, gk)
                        .unwrap();
                    std::mem::swap(a, c);
                }
                let (nw, nh) = (cw / 2, ch / 2);
                if nw == 0 || nh == 0 {
                    break;
                }
                launch_sift_downsample_nearest_cuda(
                    ctx, stream, c, b, cw as u32, ch as u32, nw as u32, nh as u32,
                )
                .unwrap();
                std::mem::swap(b, c);
                cw = nw;
                ch = nh;
            }
            stream.synchronize().unwrap();
        };

        for _ in 0..2 {
            run(&stream, &mut buf_a, &mut buf_b, &mut buf_c, &mut buf_d);
        }
        let mut ts = Vec::new();
        for _ in 0..5 {
            let t0 = std::time::Instant::now();
            run(&stream, &mut buf_a, &mut buf_b, &mut buf_c, &mut buf_d);
            ts.push(t0.elapsed().as_secs_f64() * 1e3);
        }
        ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        eprintln!(
            "GPU scale-space (upsample+blur+DoG+downsample, all octaves) @1080p \
             first_octave={}: median {:.2} ms",
            if upsample { -1 } else { 0 },
            ts[ts.len() / 2]
        );
    }

    #[test]
    fn fused_blur_v_dog_matches_unfused_bitwise() {
        let dirs = oracle_dirs();
        if dirs.is_empty() {
            eprintln!("KORNIA_SIFT_ORACLE unset; skipping");
            return;
        }
        for dir in dirs {
            eprintln!("  [fused_blur_v_dog_matches_unfused_bitwise] {dir}");
            fused_blur_v_dog_matches_unfused_bitwise_for(&dir);
        }
    }

    fn fused_blur_v_dog_matches_unfused_bitwise_for(dir: &str) {
        let cfg = SiftCudaConfig::default();
        let sigmas = cfg.layer_sigmas();
        let stream = default_stream();
        let ctx = &stream.context();

        // Octave 0, layer 1: blur layer 0 and difference against it.
        let (h, w, lower) = load_dump(&format!("{dir}/gauss_o0_l0.f32")).expect("layer 0");
        let (_, _, want_layer) = load_dump(&format!("{dir}/gauss_o0_l1.f32")).expect("layer 1");
        let (_, _, want_dog) = load_dump(&format!("{dir}/dog_o0_l0.f32")).expect("dog 0");

        let ksize = crate::cuda::sift::gaussian_ksize(sigmas[1]);
        let gk = gaussian_kernel_f32(ksize, sigmas[1]);

        let d_src = stream.clone_htod(&lower).unwrap();
        let mut d_tmp = stream.alloc_zeros::<f32>(h * w).unwrap();
        let mut d_layer = stream.alloc_zeros::<f32>(h * w).unwrap();
        let mut d_dog = stream.alloc_zeros::<f32>(h * w).unwrap();

        launch_sift_blur_h_cuda(ctx, &stream, &d_src, &mut d_tmp, w as u32, h as u32, &gk).unwrap();
        launch_sift_blur_v_dog_cuda(
            ctx,
            &stream,
            &d_tmp,
            &mut d_layer,
            &d_src,
            &mut d_dog,
            w as u32,
            h as u32,
            &gk,
        )
        .unwrap();

        let got_layer = stream.clone_dtoh(&d_layer).unwrap();
        let got_dog = stream.clone_dtoh(&d_dog).unwrap();
        assert_bit_equal(&got_layer, &want_layer, "fused blur-V layer");
        assert_bit_equal(&got_dog, &want_dog, "fused DoG");
    }

    #[test]
    fn tiled_blur_h_matches_plain_bitwise() {
        let dirs = oracle_dirs();
        if dirs.is_empty() {
            eprintln!("KORNIA_SIFT_ORACLE unset; skipping");
            return;
        }
        for dir in dirs {
            eprintln!("  [tiled_blur_h_matches_plain_bitwise] {dir}");
            tiled_blur_h_matches_plain_bitwise_for(&dir);
        }
    }

    fn tiled_blur_h_matches_plain_bitwise_for(dir: &str) {
        let cfg = SiftCudaConfig::default();
        let sigmas = cfg.layer_sigmas();
        let stream = default_stream();
        let ctx = &stream.context();
        let (h, w, src) = load_dump(&format!("{dir}/gauss_o0_l0.f32")).expect("layer 0");

        for sigma in [sigmas[1], sigmas[5]] {
            let gk = gaussian_kernel_f32(crate::cuda::sift::gaussian_ksize(sigma), sigma);
            let d_src = stream.clone_htod(&src).unwrap();
            let mut d_plain = stream.alloc_zeros::<f32>(h * w).unwrap();
            let mut d_tiled = stream.alloc_zeros::<f32>(h * w).unwrap();
            launch_sift_blur_h_cuda(ctx, &stream, &d_src, &mut d_plain, w as u32, h as u32, &gk)
                .unwrap();
            launch_sift_blur_h_tiled_cuda(
                ctx,
                &stream,
                &d_src,
                &mut d_tiled,
                w as u32,
                h as u32,
                &gk,
            )
            .unwrap();
            let a = stream.clone_dtoh(&d_plain).unwrap();
            let b = stream.clone_dtoh(&d_tiled).unwrap();
            assert_bit_equal(&b, &a, &format!("tiled blur-H (sigma={sigma})"));
        }
    }

    #[test]
    fn fused_blur_hv_matches_two_pass_bitwise() {
        let dirs = oracle_dirs();
        if dirs.is_empty() {
            eprintln!("KORNIA_SIFT_ORACLE unset; skipping");
            return;
        }
        let cfg = SiftCudaConfig::default();
        let sigmas = cfg.layer_sigmas();
        let stream = default_stream();
        let ctx = &stream.context();
        let (h, w, src) = load_dump(&format!("{}/gauss_o0_l0.f32", dirs[0])).expect("layer 0");

        for sigma in [sigmas[1], sigmas[3], sigmas[5]] {
            let gk = gaussian_kernel_f32(crate::cuda::sift::gaussian_ksize(sigma), sigma);
            let d_src = stream.clone_htod(&src).unwrap();
            let mut d_tmp = stream.alloc_zeros::<f32>(h * w).unwrap();
            let mut d_two = stream.alloc_zeros::<f32>(h * w).unwrap();
            let mut d_fused = stream.alloc_zeros::<f32>(h * w).unwrap();

            launch_sift_blur_h_cuda(ctx, &stream, &d_src, &mut d_tmp, w as u32, h as u32, &gk)
                .unwrap();
            launch_sift_blur_v_cuda(ctx, &stream, &d_tmp, &mut d_two, w as u32, h as u32, &gk)
                .unwrap();
            launch_sift_blur_hv_cuda(ctx, &stream, &d_src, &mut d_fused, w as u32, h as u32, &gk)
                .unwrap();

            let a = stream.clone_dtoh(&d_two).unwrap();
            let b = stream.clone_dtoh(&d_fused).unwrap();
            assert_bit_equal(&b, &a, &format!("fused blur-HV (sigma={sigma})"));
        }
    }

    /// Per-octave cost breakdown. OpenCV's octave count formula mandates 8 for a
    /// doubled 1080p base, but the last octaves are tiny and launch-bound —
    /// this measures what they actually cost.
    #[test]
    fn bench_per_octave() {
        if std::env::var("KORNIA_SIFT_BENCH").is_err() {
            eprintln!("KORNIA_SIFT_BENCH unset; skipping");
            return;
        }
        let cfg = SiftCudaConfig::default();
        let stream = default_stream();
        let ctx = &stream.context();
        let (w, h) = (1920usize, 1080usize);
        let src: Vec<f32> = (0..w * h)
            .map(|i| ((i * 2654435761usize) % 255) as f32)
            .collect();
        let d_src = stream.clone_htod(&src).unwrap();

        let sigmas = cfg.layer_sigmas();
        let layers = cfg.n_octave_layers + 3;
        let base_sigma = cfg.base_sig_diff() as f64;
        let base_gk =
            gaussian_kernel_f32(crate::cuda::sift::gaussian_ksize(base_sigma), base_sigma);
        let layer_gks: Vec<Vec<f32>> = (1..layers)
            .map(|i| gaussian_kernel_f32(crate::cuda::sift::gaussian_ksize(sigmas[i]), sigmas[i]))
            .collect();

        let (bw, bh) = (w * 2, h * 2);
        let mut a = stream.alloc_zeros::<f32>(bw * bh).unwrap();
        let mut b = stream.alloc_zeros::<f32>(bw * bh).unwrap();
        let mut c = stream.alloc_zeros::<f32>(bw * bh).unwrap();
        let mut d = stream.alloc_zeros::<f32>(bw * bh).unwrap();

        let n_oct = cfg.n_octaves(bh.min(bw));
        let mut per_oct = vec![0.0f64; n_oct];
        let reps = 5;

        for _ in 0..(reps + 2) {
            launch_sift_upsample2x_cuda(ctx, &stream, &d_src, &mut a, w as u32, h as u32).unwrap();
            launch_sift_blur_h_cuda(ctx, &stream, &a, &mut b, bw as u32, bh as u32, &base_gk)
                .unwrap();
            launch_sift_blur_v_cuda(ctx, &stream, &b, &mut c, bw as u32, bh as u32, &base_gk)
                .unwrap();
            let (mut cw, mut ch) = (bw, bh);
            for slot in per_oct.iter_mut() {
                if cw < 16 || ch < 16 {
                    break;
                }
                stream.synchronize().unwrap();
                let t0 = std::time::Instant::now();
                for gk in layer_gks.iter() {
                    launch_sift_blur_h_tiled_cuda(
                        ctx, &stream, &c, &mut b, cw as u32, ch as u32, gk,
                    )
                    .unwrap();
                    launch_sift_blur_v_dog_cuda(
                        ctx, &stream, &b, &mut a, &c, &mut d, cw as u32, ch as u32, gk,
                    )
                    .unwrap();
                    std::mem::swap(&mut a, &mut c);
                }
                let (nw, nh) = (cw / 2, ch / 2);
                if nw > 0 && nh > 0 {
                    launch_sift_downsample_nearest_cuda(
                        ctx, &stream, &c, &mut b, cw as u32, ch as u32, nw as u32, nh as u32,
                    )
                    .unwrap();
                    std::mem::swap(&mut b, &mut c);
                }
                stream.synchronize().unwrap();
                *slot += t0.elapsed().as_secs_f64() * 1e3;
                cw = nw;
                ch = nh;
            }
        }

        let total: f64 = per_oct.iter().sum::<f64>() / (reps + 2) as f64;
        eprintln!("  per-octave cost @1080p (total {total:.2} ms):");
        let mut cum = 0.0;
        for (o, t) in per_oct.iter().enumerate() {
            let ms = t / (reps + 2) as f64;
            cum += ms;
            eprintln!(
                "    octave {o}: {ms:7.3} ms  ({:5.1}% of total, cumulative {:5.1}%)",
                100.0 * ms / total,
                100.0 * cum / total
            );
        }
    }
}
