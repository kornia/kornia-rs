//! Device adapters for [`resize`](super::resize) and
//! [`resize_fast_u8_aa`](super::resize_fast_u8_aa): route a checked device
//! pair to the native CUDA resize kernels.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use cudarc::driver::{CudaSlice, CudaStream};
use kornia_image::{Image, ImageError};

use crate::cuda::dispatch::{device_slices, dims_u32, no_gpu_kernel_err, untyped_device_err};
use crate::cuda::resize::{
    launch_resize_bicubic_cuda, launch_resize_bilinear_downscale_cuda, launch_resize_lanczos_cuda,
    launch_resize_nearest_downscale_cuda, PixelMapping,
};
use crate::cuda::resize_u8::{
    launch_resize_u8_bilinear_cuda, launch_resize_u8_nearest_cuda,
    launch_resize_u8_pyrdown2x_rgb_cuda, launch_resize_u8_pyrup2x_rgb_cuda,
    launch_resize_u8_separable_cuda,
};
use crate::interpolation::InterpolationMode;

use super::bilinear::bilinear_axis_lut;
use super::common::{build_xsrc_lut, pack_xw_i16, precompute_contribs, FilterKind};
use super::nearest::nearest_axis_lut;

/// Run the CUDA resize for a device-resident f32 pair.
///
/// The kernels are 3-channel only, so `C != 3` errors (never a silent CPU
/// fallback — see `cuda::dispatch`). Output is bit-identical to the CPU
/// [`resize`](super::resize) — the byte-exact contract
/// established with the half-pixel grid and `--fmad=false`, asserted by the
/// parity tests in `cuda::resize`.
pub(super) fn resize_f32_cuda<const C: usize>(
    src: &Image<f32, C>,
    dst: &mut Image<f32, C>,
    interpolation: InterpolationMode,
    stream: &Arc<CudaStream>,
) -> Result<(), ImageError> {
    if C != 3 {
        return Err(no_gpu_kernel_err("resize", "3-channel f32 images"));
    }
    let (src_w, src_h) = dims_u32(src)?;
    let (dst_w, dst_h) = dims_u32(dst)?;
    let ctx = stream.context();
    let (s, d) = device_slices!(src, dst);

    match interpolation {
        InterpolationMode::Bilinear => launch_resize_bilinear_downscale_cuda(
            ctx,
            stream,
            s,
            d,
            src_w,
            src_h,
            dst_w,
            dst_h,
            PixelMapping::HalfPixel,
            None,
        ),
        InterpolationMode::Nearest => launch_resize_nearest_downscale_cuda(
            ctx,
            stream,
            s,
            d,
            src_w,
            src_h,
            dst_w,
            dst_h,
            PixelMapping::HalfPixel,
            None,
        ),
        InterpolationMode::Bicubic => launch_resize_bicubic_cuda(
            ctx,
            stream,
            s,
            d,
            src_w,
            src_h,
            dst_w,
            dst_h,
            PixelMapping::HalfPixel,
            None,
        ),
        InterpolationMode::Lanczos => launch_resize_lanczos_cuda(
            ctx,
            stream,
            s,
            d,
            src_w,
            src_h,
            dst_w,
            dst_h,
            PixelMapping::HalfPixel,
            None,
        ),
    }
    .map_err(|e| ImageError::Cuda(e.to_string()))
}

/// Device-resident coordinate/weight tables for one u8 resize geometry.
///
/// Cached because Jetson pageable H2D uploads have a catastrophic latency
/// tail (~250 µs average per `cuMemcpyHtoDAsync` measured under nsys), and
/// the lanczos host table build re-evaluates f64 `sin` per tap — together
/// they dominated the GPU op several times over. A cache hit makes a repeat
/// resize pure kernel launches, which also keeps it CUDA-Graph-capturable
/// (a pageable upload inside a capture would fail); warm the cache with one
/// call before capturing.
enum U8Luts {
    Nearest {
        xmap: CudaSlice<i32>,
        ymap: CudaSlice<i32>,
    },
    Bilinear {
        xofs: CudaSlice<u32>,
        xfx: CudaSlice<u32>,
        yofs: CudaSlice<u32>,
        yfy: CudaSlice<u32>,
    },
    Separable {
        xsrc: CudaSlice<u16>,
        xw: CudaSlice<i16>,
        kx: u32,
        yofs: CudaSlice<i32>,
        yw: CudaSlice<i16>,
        ky: u32,
    },
}

/// Cache key: device ordinal + everything that determines table contents.
/// (`antialias` only matters for the separable modes; the other tags store
/// `false`.) Channel count does not affect the tables.
#[derive(PartialEq, Eq, Hash, Clone, Copy)]
struct U8LutKey {
    dev: usize,
    mode: InterpolationMode,
    antialias: bool,
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
}

static U8_LUT_CACHE: OnceLock<Mutex<HashMap<U8LutKey, Arc<U8Luts>>>> = OnceLock::new();

/// Bound on cached geometries. Each entry is a few KB of device memory
/// (worst case ~100 KB for a wide antialiased lanczos); on overflow the
/// cache is cleared wholesale — resize-target sets are small and stable in
/// practice, so eviction is a non-event.
const U8_LUT_CACHE_CAP: usize = 128;

fn cached_luts(
    key: U8LutKey,
    build: impl FnOnce() -> Result<U8Luts, ImageError>,
) -> Result<Arc<U8Luts>, ImageError> {
    let cache = U8_LUT_CACHE.get_or_init(Default::default);
    if let Some(hit) = cache.lock().expect("u8 LUT cache poisoned").get(&key) {
        return Ok(hit.clone());
    }
    // Build + upload outside the lock; a racing duplicate upload is harmless
    // and the entry API keeps the first insertion.
    let built = Arc::new(build()?);
    let mut map = cache.lock().expect("u8 LUT cache poisoned");
    if map.len() >= U8_LUT_CACHE_CAP {
        map.clear();
    }
    Ok(map.entry(key).or_insert(built).clone())
}

fn htod<T: cudarc::driver::DeviceRepr>(
    stream: &Arc<CudaStream>,
    host: &[T],
) -> Result<CudaSlice<T>, ImageError> {
    stream
        .clone_htod(host)
        .map_err(|e| ImageError::Cuda(e.to_string()))
}

/// Run the CUDA u8 resize for a device-resident pair, byte-exact with
/// [`resize_fast_u8_aa`](super::resize_fast_u8_aa).
///
/// The dispatch cascade below mirrors the CPU cascade in
/// `resize_fast_u8_aa` BRANCH FOR BRANCH — same guards, same order — so the
/// kernel a device pair takes is always the twin of the CPU path a host pair
/// would take. Do not reorder one side without the other.
///
/// All coordinate/weight tables are built by the same host functions the CPU
/// uses (`bilinear_axis_lut`, `nearest_axis_lut`, `precompute_contribs`),
/// uploaded once per geometry (see [`U8Luts`]); the kernels are integer-only.
/// Output is bit-identical to the CPU, which the parity tests assert with
/// `assert_eq!`.
pub(super) fn resize_fast_u8_cuda<const C: usize>(
    src: &Image<u8, C>,
    dst: &mut Image<u8, C>,
    interpolation: InterpolationMode,
    antialias: bool,
    stream: &Arc<CudaStream>,
) -> Result<(), ImageError> {
    if !(C == 1 || C == 3 || C == 4) {
        return Err(no_gpu_kernel_err(
            "resize_fast_u8",
            "1/3/4-channel u8 images",
        ));
    }
    let (src_w, src_h) = dims_u32(src)?;
    let (dst_w, dst_h) = dims_u32(dst)?;
    let ctx = stream.context();
    let (s, d) = device_slices!(src, dst);
    let err = |e: crate::cuda::resize::CudaResizeError| ImageError::Cuda(e.to_string());

    // Branch 1/2: exact-2× RGB fast paths (bilinear only).
    if interpolation == InterpolationMode::Bilinear
        && C == 3
        && src_w == dst_w * 2
        && src_h == dst_h * 2
        && src_w >= 2
        && src_h >= 2
    {
        return launch_resize_u8_pyrdown2x_rgb_cuda(ctx, stream, s, d, dst_w, dst_h, None)
            .map_err(err);
    }
    if interpolation == InterpolationMode::Bilinear
        && C == 3
        && dst_w == src_w * 2
        && dst_h == src_h * 2
        && src_w >= 2
        && src_h >= 2
    {
        return launch_resize_u8_pyrup2x_rgb_cuda(ctx, stream, s, d, src_w, src_h, None)
            .map_err(err);
    }

    let key = U8LutKey {
        dev: ctx.ordinal(),
        mode: interpolation,
        // Only the separable modes read the flag; normalize so nearest and
        // bilinear share one entry regardless of the caller's antialias.
        antialias: antialias
            && matches!(
                interpolation,
                InterpolationMode::Bicubic | InterpolationMode::Lanczos
            ),
        src_w,
        src_h,
        dst_w,
        dst_h,
    };

    match interpolation {
        InterpolationMode::Nearest => {
            let luts = cached_luts(key, || {
                Ok(U8Luts::Nearest {
                    xmap: htod(stream, &nearest_axis_lut(src_w as usize, dst_w as usize))?,
                    ymap: htod(stream, &nearest_axis_lut(src_h as usize, dst_h as usize))?,
                })
            })?;
            let U8Luts::Nearest { xmap, ymap } = &*luts else {
                unreachable!("cache key encodes the mode")
            };
            launch_resize_u8_nearest_cuda(
                ctx, stream, s, d, src_w, src_h, dst_w, dst_h, C as u32, xmap, ymap, None,
            )
            .map_err(err)
        }
        InterpolationMode::Bilinear => {
            let luts = cached_luts(key, || {
                let (xofs, xfx, _) = bilinear_axis_lut(src_w as usize, dst_w as usize);
                let (yofs, yfy, _) = bilinear_axis_lut(src_h as usize, dst_h as usize);
                Ok(U8Luts::Bilinear {
                    xofs: htod(stream, &xofs)?,
                    xfx: htod(stream, &xfx)?,
                    yofs: htod(stream, &yofs)?,
                    yfy: htod(stream, &yfy)?,
                })
            })?;
            let U8Luts::Bilinear {
                xofs,
                xfx,
                yofs,
                yfy,
            } = &*luts
            else {
                unreachable!("cache key encodes the mode")
            };
            launch_resize_u8_bilinear_cuda(
                ctx, stream, s, d, src_w, src_h, dst_w, dst_h, C as u32, xofs, xfx, yofs, yfy, None,
            )
            .map_err(err)
        }
        InterpolationMode::Bicubic | InterpolationMode::Lanczos => {
            let luts = cached_luts(key, || {
                let filt = if interpolation == InterpolationMode::Bicubic {
                    FilterKind::Cubic
                } else {
                    FilterKind::Lanczos3
                };
                let (xofs, xw_q14, kx) =
                    precompute_contribs(src_w as usize, dst_w as usize, filt, antialias);
                let (yofs, yw_q14, ky) =
                    precompute_contribs(src_h as usize, dst_h as usize, filt, antialias);
                let xsrc = build_xsrc_lut(&xofs, dst_w as usize, kx, src_w as usize);
                Ok(U8Luts::Separable {
                    xsrc: htod(stream, &xsrc)?,
                    xw: htod(stream, &pack_xw_i16(&xw_q14))?,
                    kx: kx as u32,
                    yofs: htod(stream, &yofs)?,
                    // Same i32 → i16 cast the CPU vertical pass applies per tap.
                    yw: htod(stream, &pack_xw_i16(&yw_q14))?,
                    ky: ky as u32,
                })
            })?;
            let U8Luts::Separable {
                xsrc,
                xw,
                kx,
                yofs,
                yw,
                ky,
            } = &*luts
            else {
                unreachable!("cache key encodes the mode")
            };
            launch_resize_u8_separable_cuda(
                ctx, stream, s, d, src_w, src_h, dst_w, dst_h, C as u32, xsrc, xw, *kx, yofs, yw,
                *ky, None,
            )
            .map_err(err)
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use crate::cuda::color::test_utils::{default_stream, pattern_f32, pattern_u8};
    use crate::interpolation::InterpolationMode;
    use crate::resize::{resize, resize_fast_u8_aa};
    use kornia_image::{Image, ImageError, ImageSize};

    fn sized(w: usize, h: usize) -> ImageSize {
        ImageSize {
            width: w,
            height: h,
        }
    }

    /// Run `resize_fast_u8_aa` on host and device for the same input and
    /// assert the outputs are bit-identical.
    fn assert_u8_device_equals_host<const C: usize>(
        (sw, sh): (usize, usize),
        (dw, dh): (usize, usize),
        mode: InterpolationMode,
        antialias: bool,
    ) {
        let stream = default_stream();
        let src = Image::<u8, C>::new(sized(sw, sh), pattern_u8(sw * sh * C)).unwrap();

        let mut cpu_dst = Image::<u8, C>::from_size_val(sized(dw, dh), 0).unwrap();
        resize_fast_u8_aa(&src, &mut cpu_dst, mode, antialias).unwrap();

        let d_src = src.to_cuda(&stream).unwrap();
        let mut d_dst = Image::<u8, C>::zeros_cuda(sized(dw, dh), &stream).unwrap();
        resize_fast_u8_aa(&d_src, &mut d_dst, mode, antialias).unwrap();

        let back = d_dst.to_host_owned().unwrap();
        assert_eq!(
            back.as_slice(),
            cpu_dst.as_slice(),
            "{sw}x{sh}->{dw}x{dh} C={C} {mode:?} aa={antialias}: device must be bit-identical to host"
        );
    }

    /// Branches 1+2 of the cascade: exact-2× RGB pyrdown/pyrup fast paths.
    #[test]
    fn u8_pyr2x_fast_paths_device_equal_host() {
        // Odd-ish dst dims exercise the tail columns of the CPU NEON kernels.
        assert_u8_device_equals_host::<3>((130, 98), (65, 49), InterpolationMode::Bilinear, true);
        assert_u8_device_equals_host::<3>((65, 49), (130, 98), InterpolationMode::Bilinear, true);
    }

    /// Branch 3: nearest, all supported channel counts, odd sizes, up + down.
    #[test]
    fn u8_nearest_device_equals_host() {
        for &(s, d) in &[((129, 97), (64, 48)), ((63, 41), (127, 90))] {
            assert_u8_device_equals_host::<1>(s, d, InterpolationMode::Nearest, true);
            assert_u8_device_equals_host::<3>(s, d, InterpolationMode::Nearest, true);
            assert_u8_device_equals_host::<4>(s, d, InterpolationMode::Nearest, true);
        }
    }

    /// Branch 4: generic Q14 bilinear, non-2× ratios.
    #[test]
    fn u8_bilinear_device_equals_host() {
        for &(s, d) in &[
            ((129, 97), (64, 48)),
            ((63, 41), (127, 90)),
            ((33, 21), (33, 21)),
        ] {
            assert_u8_device_equals_host::<1>(s, d, InterpolationMode::Bilinear, true);
            assert_u8_device_equals_host::<3>(s, d, InterpolationMode::Bilinear, true);
            assert_u8_device_equals_host::<4>(s, d, InterpolationMode::Bilinear, true);
        }
    }

    /// Branch 5: Q14 separable bicubic/lanczos, antialiased (PIL semantics,
    /// wide kernels) and non-antialiased (OpenCV semantics, fixed taps).
    #[test]
    fn u8_separable_device_equals_host() {
        for mode in [InterpolationMode::Bicubic, InterpolationMode::Lanczos] {
            for antialias in [true, false] {
                assert_u8_device_equals_host::<3>((129, 97), (64, 48), mode, antialias);
                assert_u8_device_equals_host::<1>((63, 41), (127, 90), mode, antialias);
                assert_u8_device_equals_host::<4>((100, 80), (47, 33), mode, antialias);
            }
        }
    }

    /// Extreme antialiased downscale: tall kernels (ksize grows with the
    /// scale factor) stress the i32 accumulator bound and the tap loops.
    #[test]
    fn u8_separable_extreme_downscale_device_equals_host() {
        assert_u8_device_equals_host::<3>((1024, 64), (50, 40), InterpolationMode::Lanczos, true);
        assert_u8_device_equals_host::<3>((1024, 64), (50, 40), InterpolationMode::Bicubic, true);
    }

    /// u8 device pairs with unsupported channel counts error; mixed
    /// residency errors — never a silent CPU fallback or transfer.
    #[test]
    fn u8_resize_error_semantics() {
        let stream = default_stream();

        let src = Image::<u8, 2>::new(sized(16, 16), pattern_u8(16 * 16 * 2)).unwrap();
        let d_src = src.to_cuda(&stream).unwrap();
        let mut d_dst = Image::<u8, 2>::zeros_cuda(sized(8, 8), &stream).unwrap();
        let err =
            resize_fast_u8_aa(&d_src, &mut d_dst, InterpolationMode::Bilinear, true).unwrap_err();
        assert!(
            matches!(&err, ImageError::Cuda(msg) if msg.contains("1/3/4-channel")),
            "got {err:?}"
        );

        let src3 = Image::<u8, 3>::new(sized(16, 16), pattern_u8(16 * 16 * 3)).unwrap();
        let d_src3 = src3.to_cuda(&stream).unwrap();
        let mut host_dst = Image::<u8, 3>::from_size_val(sized(8, 8), 0).unwrap();
        let err = resize_fast_u8_aa(&d_src3, &mut host_dst, InterpolationMode::Bilinear, true)
            .unwrap_err();
        assert!(matches!(err, ImageError::MixedResidency), "got {err:?}");
    }

    /// The public `resize`, called with device images, must produce
    /// bit-identical output to the same call with host images — for both
    /// modes, downscale and upscale, at non-dyadic sizes.
    #[test]
    fn public_resize_device_equals_host() {
        let stream = default_stream();
        for &((sw, sh), (dw, dh)) in &[((129, 97), (64, 48)), ((63, 41), (127, 90))] {
            let src = Image::<f32, 3>::new(sized(sw, sh), pattern_f32(sw * sh * 3)).unwrap();
            for mode in [InterpolationMode::Bilinear, InterpolationMode::Nearest] {
                let mut cpu_dst = Image::<f32, 3>::from_size_val(sized(dw, dh), 0.0).unwrap();
                resize(&src, &mut cpu_dst, mode).unwrap();

                let d_src = src.to_cuda(&stream).unwrap();
                let mut d_dst = Image::<f32, 3>::zeros_cuda(sized(dw, dh), &stream).unwrap();
                resize(&d_src, &mut d_dst, mode).unwrap();

                let back = d_dst.to_host_owned().unwrap();
                assert_eq!(
                    back.as_slice(),
                    cpu_dst.as_slice(),
                    "{sw}x{sh}->{dw}x{dh} {mode:?}: device must be bit-identical to host"
                );
            }
        }
    }

    /// Bicubic and Lanczos through the public resize: device == host, bit for
    /// bit — bicubic is direct 4×4 on both sides; lanczos is separable with
    /// shared host-built weight tables.
    #[test]
    fn public_resize_bicubic_lanczos_device_equals_host() {
        let stream = default_stream();
        for &((sw, sh), (dw, dh)) in &[((129, 97), (64, 48)), ((63, 41), (127, 90))] {
            let src = Image::<f32, 3>::new(sized(sw, sh), pattern_f32(sw * sh * 3)).unwrap();
            for mode in [InterpolationMode::Bicubic, InterpolationMode::Lanczos] {
                let mut cpu_dst = Image::<f32, 3>::from_size_val(sized(dw, dh), 0.0).unwrap();
                resize(&src, &mut cpu_dst, mode).unwrap();

                let d_src = src.to_cuda(&stream).unwrap();
                let mut d_dst = Image::<f32, 3>::zeros_cuda(sized(dw, dh), &stream).unwrap();
                resize(&d_src, &mut d_dst, mode).unwrap();

                let back = d_dst.to_host_owned().unwrap();
                for (i, (c, g)) in cpu_dst.as_slice().iter().zip(back.as_slice()).enumerate() {
                    assert!(
                        c.to_bits() == g.to_bits(),
                        "{sw}x{sh}->{dw}x{dh} {mode:?} element {i}: cpu {c} ({:#010x}) gpu {g} ({:#010x})",
                        c.to_bits(),
                        g.to_bits()
                    );
                }
            }
        }
    }

    /// Same-size device resize must run on the device (an identity-coefficient
    /// kernel pass), NOT fall into the host memcpy short-circuit — which would
    /// dereference device pointers on the host.
    #[test]
    fn same_size_device_resize_is_safe_and_identity() {
        let stream = default_stream();
        let src = Image::<f32, 3>::new(sized(33, 21), pattern_f32(33 * 21 * 3)).unwrap();
        let d_src = src.to_cuda(&stream).unwrap();
        let mut d_dst = Image::<f32, 3>::zeros_cuda(sized(33, 21), &stream).unwrap();
        resize(&d_src, &mut d_dst, InterpolationMode::Bilinear).unwrap();
        let back = d_dst.to_host_owned().unwrap();
        assert_eq!(back.as_slice(), src.as_slice(), "identity resize");
    }

    /// Mixed residency errors; C != 3 device pairs error instead of silently
    /// falling back to the CPU.
    #[test]
    fn resize_error_semantics() {
        let stream = default_stream();
        let src = Image::<f32, 3>::new(sized(16, 16), pattern_f32(16 * 16 * 3)).unwrap();
        let d_src = src.to_cuda(&stream).unwrap();
        let mut host_dst = Image::<f32, 3>::from_size_val(sized(8, 8), 0.0).unwrap();
        let err = resize(&d_src, &mut host_dst, InterpolationMode::Bilinear).unwrap_err();
        assert!(matches!(err, ImageError::MixedResidency), "got {err:?}");

        let src1 = Image::<f32, 1>::new(sized(16, 16), pattern_f32(16 * 16)).unwrap();
        let d_src1 = src1.to_cuda(&stream).unwrap();
        let mut d_dst1 = Image::<f32, 1>::zeros_cuda(sized(8, 8), &stream).unwrap();
        let err = resize(&d_src1, &mut d_dst1, InterpolationMode::Bilinear).unwrap_err();
        assert!(
            matches!(&err, ImageError::Cuda(msg) if msg.contains("3-channel")),
            "got {err:?}"
        );
    }
}
