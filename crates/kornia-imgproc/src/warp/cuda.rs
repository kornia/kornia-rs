//! Device adapters for [`warp_affine`](super::warp_affine) and
//! [`warp_perspective`](super::warp_perspective): route checked device pairs
//! to the native CUDA warp kernels.

use std::sync::Arc;

use cudarc::driver::CudaStream;
use kornia_image::{Image, ImageError};

use crate::cuda::dispatch::{device_slices, dims_u32, no_gpu_kernel_err, untyped_device_err};
use crate::cuda::warp_affine::{
    launch_warp_affine_bicubic_cuda, launch_warp_affine_bilinear_cuda,
    launch_warp_affine_lanczos_cuda, launch_warp_affine_nearest_cuda,
};
use crate::cuda::warp_affine_u8::launch_warp_affine_u8_bilinear_cuda;
use crate::cuda::warp_perspective::{
    launch_warp_perspective_bicubic_cuda, launch_warp_perspective_bilinear_cuda,
    launch_warp_perspective_lanczos_cuda, launch_warp_perspective_nearest_cuda,
};
use crate::cuda::warp_perspective_u8::launch_warp_perspective_u8_bilinear_cuda;
use crate::interpolation::InterpolationMode;

/// Run the CUDA warp-affine for a device-resident f32 pair.
///
/// `m` is the forward matrix — the launchers invert internally, exactly like
/// the CPU [`warp_affine`](super::warp_affine), so it passes straight
/// through. Output is bit-identical to the CPU path (the byte-exact contract
/// asserted by the parity tests in `cuda::warp_affine`).
pub(super) fn warp_affine_f32_cuda<const C: usize>(
    src: &Image<f32, C>,
    dst: &mut Image<f32, C>,
    m: &[f32; 6],
    interpolation: InterpolationMode,
    stream: &Arc<CudaStream>,
) -> Result<(), ImageError> {
    if C != 3 {
        return Err(no_gpu_kernel_err("warp_affine", "3-channel f32 images"));
    }
    let (src_w, src_h) = dims_u32(src)?;
    let (dst_w, dst_h) = dims_u32(dst)?;
    let ctx = stream.context();
    let (s, d) = device_slices!(src, dst);

    match interpolation {
        InterpolationMode::Bilinear => {
            launch_warp_affine_bilinear_cuda(ctx, stream, s, d, src_w, src_h, dst_w, dst_h, m, None)
        }
        InterpolationMode::Nearest => {
            launch_warp_affine_nearest_cuda(ctx, stream, s, d, src_w, src_h, dst_w, dst_h, m, None)
        }
        InterpolationMode::Bicubic => {
            launch_warp_affine_bicubic_cuda(ctx, stream, s, d, src_w, src_h, dst_w, dst_h, m, None)
        }
        InterpolationMode::Lanczos => {
            launch_warp_affine_lanczos_cuda(ctx, stream, s, d, src_w, src_h, dst_w, dst_h, m, None)
        }
    }
    .map_err(|e| ImageError::Cuda(e.to_string()))
}

/// Run the CUDA warp-perspective for a device-resident f32 pair.
///
/// `m` is the forward homography — inverted internally like the CPU path.
/// Bit-identical to CPU [`warp_perspective`](super::warp_perspective).
pub(super) fn warp_perspective_f32_cuda<const C: usize>(
    src: &Image<f32, C>,
    dst: &mut Image<f32, C>,
    m: &[f32; 9],
    interpolation: InterpolationMode,
    stream: &Arc<CudaStream>,
) -> Result<(), ImageError> {
    if C != 3 {
        return Err(no_gpu_kernel_err(
            "warp_perspective",
            "3-channel f32 images",
        ));
    }
    let (src_w, src_h) = dims_u32(src)?;
    let (dst_w, dst_h) = dims_u32(dst)?;
    let ctx = stream.context();
    let (s, d) = device_slices!(src, dst);

    match interpolation {
        InterpolationMode::Bilinear => launch_warp_perspective_bilinear_cuda(
            ctx, stream, s, d, src_w, src_h, dst_w, dst_h, m, None,
        ),
        InterpolationMode::Nearest => launch_warp_perspective_nearest_cuda(
            ctx, stream, s, d, src_w, src_h, dst_w, dst_h, m, None,
        ),
        InterpolationMode::Bicubic => launch_warp_perspective_bicubic_cuda(
            ctx, stream, s, d, src_w, src_h, dst_w, dst_h, m, None,
        ),
        InterpolationMode::Lanczos => launch_warp_perspective_lanczos_cuda(
            ctx, stream, s, d, src_w, src_h, dst_w, dst_h, m, None,
        ),
    }
    .map_err(|e| ImageError::Cuda(e.to_string()))
}

/// Run the CUDA u8 warp-affine (bilinear) for a device-resident pair,
/// byte-exact with [`warp_affine_u8`](super::warp_affine_u8).
///
/// The forward matrix is inverted here with the same
/// [`invert_affine_transform`](super::invert_affine_transform) the CPU path
/// calls — shared code, not mirrored code — and the kernel reproduces the
/// CPU's per-row span + Q16/Q10 fixed-point arithmetic exactly (see
/// `cuda::warp_affine_u8`).
pub(super) fn warp_affine_u8_cuda<const C: usize>(
    src: &Image<u8, C>,
    dst: &mut Image<u8, C>,
    m: &[f32; 6],
    stream: &Arc<CudaStream>,
) -> Result<(), ImageError> {
    let (src_w, src_h) = dims_u32(src)?;
    let (dst_w, dst_h) = dims_u32(dst)?;
    let ctx = stream.context();
    let (s, d) = device_slices!(src, dst);
    let m_inv = super::invert_affine_transform(m);

    launch_warp_affine_u8_bilinear_cuda(
        ctx, stream, s, d, &m_inv, src_w, src_h, dst_w, dst_h, C as u32, None,
    )
    .map_err(|e| ImageError::Cuda(e.to_string()))
}

/// Run the CUDA u8 warp-perspective (bilinear) for a device-resident pair,
/// byte-exact with [`warp_perspective_u8`](super::warp_perspective_u8) —
/// every CPU backend evaluates the coordinate directly per column, and the
/// kernel mirrors the same expression tree (see `cuda::warp_perspective_u8`).
pub(super) fn warp_perspective_u8_cuda<const C: usize>(
    src: &Image<u8, C>,
    dst: &mut Image<u8, C>,
    m: &[f32; 9],
    stream: &Arc<CudaStream>,
) -> Result<(), ImageError> {
    let (src_w, src_h) = dims_u32(src)?;
    let (dst_w, dst_h) = dims_u32(dst)?;
    let ctx = stream.context();
    let (s, d) = device_slices!(src, dst);
    let m_inv = super::perspective::inverse_perspective_matrix(m)?;

    launch_warp_perspective_u8_bilinear_cuda(
        ctx, stream, s, d, &m_inv, src_w, src_h, dst_w, dst_h, C as u32, None,
    )
    .map_err(|e| ImageError::Cuda(e.to_string()))
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use crate::cuda::color::test_utils::{default_stream, pattern_f32};
    use crate::interpolation::InterpolationMode;
    use crate::warp::{warp_affine, warp_perspective};
    use kornia_image::{Image, ImageError, ImageSize};

    fn host_pair(w: usize, h: usize) -> (Image<f32, 3>, Image<f32, 3>) {
        let size = ImageSize {
            width: w,
            height: h,
        };
        (
            Image::<f32, 3>::new(size, pattern_f32(w * h * 3)).unwrap(),
            Image::<f32, 3>::from_size_val(size, 0.0).unwrap(),
        )
    }

    /// The public `warp_affine_u8`, called with device images, must be
    /// bit-identical to the host path — spans, Q16 anchors, Q10 sampler and
    /// all. Rotation+scale exercises fractional spans; the flip and the 90°
    /// rotation pin the exact-integer span boundaries `warp::span` exists
    /// for; translation by half a pixel exercises the sampler everywhere.
    #[test]
    fn public_warp_affine_u8_device_equals_host() {
        use crate::cuda::color::test_utils::pattern_u8;
        let stream = default_stream();

        fn run<const C: usize>(
            w: usize,
            h: usize,
            m: &[f32; 6],
            stream: &std::sync::Arc<cudarc::driver::CudaStream>,
        ) {
            let size = ImageSize {
                width: w,
                height: h,
            };
            let src = Image::<u8, C>::new(size, pattern_u8(w * h * C)).unwrap();
            let mut cpu_dst = Image::<u8, C>::from_size_val(size, 0).unwrap();
            crate::warp::warp_affine_u8(&src, &mut cpu_dst, m).unwrap();

            let d_src = src.to_cuda(stream).unwrap();
            let mut d_dst = Image::<u8, C>::zeros_cuda(size, stream).unwrap();
            crate::warp::warp_affine_u8(&d_src, &mut d_dst, m).unwrap();

            assert_eq!(
                d_dst.to_host_owned().unwrap().as_slice(),
                cpu_dst.as_slice(),
                "{w}x{h} C={C} m={m:?}: device must be bit-identical to host"
            );
        }

        let rot = crate::warp::get_rotation_matrix2d((64.0, 48.0), 37.0, 1.3);
        let rot90 = crate::warp::get_rotation_matrix2d((64.0, 48.0), 90.0, 1.0);
        let flip: [f32; 6] = [-1.0, 0.0, 128.0, 0.0, 1.0, 0.0];
        let half: [f32; 6] = [1.0, 0.0, 0.5, 0.0, 1.0, 0.5];
        for m in [&rot, &rot90, &flip, &half] {
            run::<1>(129, 97, m, &stream);
            run::<3>(129, 97, m, &stream);
            run::<4>(129, 97, m, &stream);
        }
        // Odd small size exercises span-clamps and single-warp blocks.
        run::<3>(33, 21, &rot, &stream);
    }

    /// The public `warp_perspective_u8`, device vs host, bit-identical —
    /// including the negative-denominator branch and an affine-equivalent
    /// homography that must take the uniform-nd span path.
    #[test]
    fn public_warp_perspective_u8_device_equals_host() {
        use crate::cuda::color::test_utils::pattern_u8;
        let stream = default_stream();

        fn run<const C: usize>(
            w: usize,
            h: usize,
            m: &[f32; 9],
            stream: &std::sync::Arc<cudarc::driver::CudaStream>,
        ) {
            let size = ImageSize {
                width: w,
                height: h,
            };
            let src = Image::<u8, C>::new(size, pattern_u8(w * h * C)).unwrap();
            let mut cpu_dst = Image::<u8, C>::from_size_val(size, 0).unwrap();
            crate::warp::warp_perspective_u8(&src, &mut cpu_dst, m).unwrap();

            let d_src = src.to_cuda(stream).unwrap();
            let mut d_dst = Image::<u8, C>::zeros_cuda(size, stream).unwrap();
            crate::warp::warp_perspective_u8(&d_src, &mut d_dst, m).unwrap();

            assert_eq!(
                d_dst.to_host_owned().unwrap().as_slice(),
                cpu_dst.as_slice(),
                "{w}x{h} C={C} m={m:?}: device must be bit-identical to host"
            );
        }

        // Genuine projective transform (non-zero perspective terms).
        let proj: [f32; 9] = [
            0.9, 0.12, 4.0, //
            -0.08, 1.05, -2.0, //
            6.0e-4, -4.5e-4, 1.0,
        ];
        // Affine-equivalent homography — uniform-nd span path, nd == 1.
        let affine_h: [f32; 9] = [1.1, 0.1, -3.0, -0.05, 0.95, 2.0, 0.0, 0.0, 1.0];
        // Negated homography — same geometry, exercises the nd < 0 branch
        // (xf = nx/nd is invariant, and IEEE negation keeps it bit-exact).
        let mut neg = proj;
        for v in neg.iter_mut() {
            *v = -*v;
        }
        for m in [&proj, &affine_h, &neg] {
            run::<1>(129, 97, m, &stream);
            run::<3>(129, 97, m, &stream);
            run::<4>(129, 97, m, &stream);
        }
        run::<3>(33, 21, &proj, &stream);
    }

    /// The public `warp_affine`, called with device images, must produce
    /// bit-identical output to the same call with host images — the whole
    /// point of the byte-exact program.
    #[test]
    fn public_warp_affine_device_equals_host() {
        let stream = default_stream();
        let (src, mut cpu_dst) = host_pair(129, 97);
        let m = crate::warp::get_rotation_matrix2d((64.0, 48.0), 37.0, 1.3);

        warp_affine(&src, &mut cpu_dst, &m, InterpolationMode::Bilinear).unwrap();

        let d_src = src.to_cuda(&stream).unwrap();
        let mut d_dst = Image::<f32, 3>::zeros_cuda(src.size(), &stream).unwrap();
        warp_affine(&d_src, &mut d_dst, &m, InterpolationMode::Bilinear).unwrap();

        let back = d_dst.to_host_owned().unwrap();
        assert_eq!(
            back.as_slice(),
            cpu_dst.as_slice(),
            "device warp_affine must be bit-identical to host"
        );
    }

    /// Same for the public `warp_perspective`, with a genuinely projective
    /// homography.
    #[test]
    fn public_warp_perspective_device_equals_host() {
        let stream = default_stream();
        let (src, mut cpu_dst) = host_pair(129, 97);
        let hm = [
            1.03,
            0.05,
            -3.0,
            -0.02,
            0.97,
            4.0,
            2.0 / (97.0 * 129.0),
            1.5 / (129.0 * 97.0),
            1.0,
        ];

        warp_perspective(&src, &mut cpu_dst, &hm, InterpolationMode::Bilinear).unwrap();

        let d_src = src.to_cuda(&stream).unwrap();
        let mut d_dst = Image::<f32, 3>::zeros_cuda(src.size(), &stream).unwrap();
        warp_perspective(&d_src, &mut d_dst, &hm, InterpolationMode::Bilinear).unwrap();

        let back = d_dst.to_host_owned().unwrap();
        assert_eq!(
            back.as_slice(),
            cpu_dst.as_slice(),
            "device warp_perspective must be bit-identical to host"
        );
    }

    /// Bicubic and Lanczos through the public functions: device == host,
    /// bit for bit — the last two modes to join the byte-exact contract.
    #[test]
    fn public_warp_bicubic_lanczos_device_equals_host() {
        let stream = default_stream();
        let (src, _) = host_pair(129, 97);
        let ma = crate::warp::get_rotation_matrix2d((64.0, 48.0), 37.0, 1.3);
        let hm = [0.9, 0.15, 10.0, -0.1, 1.1, -6.0, 1e-5, -2e-5, 1.0];

        for mode in [InterpolationMode::Bicubic, InterpolationMode::Lanczos] {
            let mut cpu_dst = Image::<f32, 3>::from_size_val(src.size(), 0.0).unwrap();
            warp_affine(&src, &mut cpu_dst, &ma, mode).unwrap();
            let d_src = src.to_cuda(&stream).unwrap();
            let mut d_dst = Image::<f32, 3>::zeros_cuda(src.size(), &stream).unwrap();
            warp_affine(&d_src, &mut d_dst, &ma, mode).unwrap();
            let back = d_dst.to_host_owned().unwrap();
            for (i, (c, g)) in cpu_dst.as_slice().iter().zip(back.as_slice()).enumerate() {
                assert!(
                    c.to_bits() == g.to_bits(),
                    "affine {mode:?} element {i}: cpu {c} ({:#010x}) gpu {g} ({:#010x})",
                    c.to_bits(),
                    g.to_bits()
                );
            }

            let mut cpu_dst = Image::<f32, 3>::from_size_val(src.size(), 0.0).unwrap();
            warp_perspective(&src, &mut cpu_dst, &hm, mode).unwrap();
            let mut d_dst = Image::<f32, 3>::zeros_cuda(src.size(), &stream).unwrap();
            warp_perspective(&d_src, &mut d_dst, &hm, mode).unwrap();
            let back = d_dst.to_host_owned().unwrap();
            for (i, (c, g)) in cpu_dst.as_slice().iter().zip(back.as_slice()).enumerate() {
                assert!(
                    c.to_bits() == g.to_bits(),
                    "perspective {mode:?} element {i}: cpu {c} ({:#010x}) gpu {g} ({:#010x})",
                    c.to_bits(),
                    g.to_bits()
                );
            }
        }
    }

    /// Mixed residency is a typed error, not an implicit transfer.
    #[test]
    fn mixed_residency_is_a_typed_error() {
        let stream = default_stream();
        let (src, mut host_dst) = host_pair(16, 16);
        let d_src = src.to_cuda(&stream).unwrap();
        let m = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let err = warp_affine(&d_src, &mut host_dst, &m, InterpolationMode::Nearest).unwrap_err();
        assert!(matches!(err, ImageError::MixedResidency), "got {err:?}");
    }

    /// A device pair with a channel count the kernels don't support errors —
    /// never a silent CPU fallback.
    #[test]
    fn unsupported_channels_error_not_fallback() {
        let stream = default_stream();
        let size = ImageSize {
            width: 16,
            height: 16,
        };
        let src = Image::<f32, 1>::new(size, pattern_f32(16 * 16)).unwrap();
        let d_src = src.to_cuda(&stream).unwrap();
        let mut d_dst = Image::<f32, 1>::zeros_cuda(size, &stream).unwrap();
        let m = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let err = warp_affine(&d_src, &mut d_dst, &m, InterpolationMode::Nearest).unwrap_err();
        assert!(
            matches!(&err, ImageError::Cuda(msg) if msg.contains("3-channel")),
            "got {err:?}"
        );
    }
}

#[cfg(test)]
mod bench_probe {
    /// Release-build device-throughput probe (min of 5 vs unlocked DVFS):
    /// `cargo test --release ... warp::cuda::bench_probe -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn probe_warps_1080p() {
        use crate::cuda::color::test_utils::{default_stream, pattern_u8};
        use kornia_image::{Image, ImageSize};

        let stream = default_stream();
        let size = ImageSize {
            width: 1920,
            height: 1080,
        };
        let src = Image::<u8, 3>::new(size, pattern_u8(1920 * 1080 * 3)).unwrap();
        let d_src = src.to_cuda(&stream).unwrap();
        let mut d_dst = Image::<u8, 3>::zeros_cuda(size, &stream).unwrap();
        let aff = crate::warp::get_rotation_matrix2d((960.0, 540.0), 20.0, 1.1);
        let psp: [f32; 9] = [0.9, 0.12, 40.0, -0.08, 1.05, -20.0, 6.0e-5, -4.5e-5, 1.0];

        for _ in 0..200 {
            crate::warp::warp_affine_u8(&d_src, &mut d_dst, &aff).unwrap();
        }
        stream.synchronize().unwrap();

        let mut best_a = f64::MAX;
        let mut best_p = f64::MAX;
        for _ in 0..5 {
            for _ in 0..50 {
                crate::warp::warp_affine_u8(&d_src, &mut d_dst, &aff).unwrap();
            }
            stream.synchronize().unwrap();
            let t0 = std::time::Instant::now();
            for _ in 0..200 {
                crate::warp::warp_affine_u8(&d_src, &mut d_dst, &aff).unwrap();
            }
            stream.synchronize().unwrap();
            best_a = best_a.min(t0.elapsed().as_secs_f64() * 1000.0 / 200.0);

            for _ in 0..50 {
                crate::warp::warp_perspective_u8(&d_src, &mut d_dst, &psp).unwrap();
            }
            stream.synchronize().unwrap();
            let t0 = std::time::Instant::now();
            for _ in 0..200 {
                crate::warp::warp_perspective_u8(&d_src, &mut d_dst, &psp).unwrap();
            }
            stream.synchronize().unwrap();
            best_p = best_p.min(t0.elapsed().as_secs_f64() * 1000.0 / 200.0);
        }
        println!("affine: {best_a:.3} ms/op (min of 5)");
        println!("perspective: {best_p:.3} ms/op (min of 5)");
    }

    /// The u8 warp kernels are per-byte generic over C — a 2-channel warp
    /// must work on BOTH residencies and agree byte-for-byte (the adapter
    /// previously gated {1,3,4} while the CPU path accepted any C).
    #[test]
    fn warp_u8_c2_device_matches_cpu() {
        use kornia_image::{Image, ImageSize};
        let stream = crate::cuda::color::test_utils::default_stream();
        let src = Image::<u8, 2>::new(
            ImageSize {
                width: 37,
                height: 29,
            },
            crate::cuda::color::test_utils::pattern_u8(37 * 29 * 2),
        )
        .unwrap();
        let m = [0.9f32, 0.1, 2.0, -0.05, 1.05, -1.0];
        let mut cpu_dst = Image::<u8, 2>::from_size_val(
            ImageSize {
                width: 31,
                height: 23,
            },
            0,
        )
        .unwrap();
        crate::warp::warp_affine_u8(&src, &mut cpu_dst, &m).unwrap();

        let d_src = src.to_cuda(&stream).unwrap();
        let mut d_dst = Image::<u8, 2>::zeros_cuda(
            ImageSize {
                width: 31,
                height: 23,
            },
            &stream,
        )
        .unwrap();
        crate::warp::warp_affine_u8(&d_src, &mut d_dst, &m).unwrap();
        assert_eq!(
            d_dst.to_host_owned().unwrap().as_slice(),
            cpu_dst.as_slice()
        );
    }
}
