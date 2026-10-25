//! Host vs device parity: the CUDA kernels must equal the CPU exactly.

use std::sync::Arc;

use super::*;
use cudarc::driver::{CudaContext, CudaStream};
use kornia_algebra::{Mat3F64, Vec3F64};
use kornia_image::ImageSize;

type TestResult = Result<(), Box<dyn std::error::Error>>;

const MM: f32 = 0.001;

/// A single pixel, a row wider than one 256-pixel tile, a height above one
/// scan run per thread (1024 rows), and the benchmark size.
const SIZES: [(usize, usize); 6] = [(1, 1), (300, 7), (513, 3), (37, 1500), (91, 67), (640, 480)];

fn stream() -> Result<Arc<CudaStream>, Box<dyn std::error::Error>> {
    Ok(CudaContext::new(0)?.default_stream())
}

fn sz(width: usize, height: usize) -> ImageSize {
    ImageSize { width, height }
}

fn pinhole(fx: f64, fy: f64, cx: f64, cy: f64) -> PinholeCamera {
    PinholeCamera {
        fx,
        fy,
        cx,
        cy,
        k1: 0.0,
        k2: 0.0,
        p1: 0.0,
        p2: 0.0,
    }
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
}

fn depth_u16(w: usize, h: usize, seed: u64) -> Vec<u16> {
    let mut rng = Lcg(seed);
    (0..w * h)
        .map(|_| match rng.next() % 6 {
            0 => 0,
            _ => 300 + (rng.next() % 9_000) as u16,
        })
        .collect()
}

fn depth_f32(w: usize, h: usize, seed: u64) -> Vec<f32> {
    let mut rng = Lcg(seed);
    (0..w * h)
        .map(|_| match rng.next() % 8 {
            0 => 0.0,
            1 => f32::NAN,
            _ => 0.3 + (rng.next() % 9_000) as f32 * 1e-3,
        })
        .collect()
}

fn mask(w: usize, h: usize, seed: u64) -> Vec<u8> {
    let mut rng = Lcg(seed);
    (0..w * h)
        .map(|_| (!rng.next().is_multiple_of(3)) as u8)
        .collect()
}

fn camera_for(w: usize, h: usize) -> PinholeCamera {
    let f = 0.8 * w.max(h) as f64 + 1.0;
    pinhole(f, 1.01 * f, 0.5 * w as f64 - 0.3, 0.5 * h as f64 + 0.2)
}

/// A small rotation about y then x plus a few centimetres of translation:
/// exercises every rotation entry, parallax and occlusions.
fn pose() -> Pose3d {
    let (a, b) = (0.07f64, -0.04f64);
    let ry =
        Mat3F64::from_cols_array(&[a.cos(), 0.0, -a.sin(), 0.0, 1.0, 0.0, a.sin(), 0.0, a.cos()]);
    let rx =
        Mat3F64::from_cols_array(&[1.0, 0.0, 0.0, 0.0, b.cos(), b.sin(), 0.0, -b.sin(), b.cos()]);
    Pose3d::new(
        Mat3F64::from(rx.0 * ry.0),
        Vec3F64::new(0.035, -0.012, 0.02),
    )
}

fn check_unproject<T>(
    proj: &mut CudaDepthProjector,
    st: &Arc<CudaStream>,
    data: Vec<T>,
    scale: f32,
    w: usize,
    h: usize,
) -> TestResult
where
    T: DepthSample + cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits + Default,
{
    let camera = camera_for(w, h);
    let depth = Image::<T, 1>::new(sz(w, h), data)?;
    let m = Image::<u8, 1>::new(sz(w, h), mask(w, h, (w + h) as u64))?;
    let depth_dev = depth.to_cuda(st)?;
    let m_dev = m.to_cuda(st)?;
    for use_mask in [false, true] {
        let (host_mask, dev_mask) = if use_mask {
            (Some(&m), Some(&m_dev))
        } else {
            (None, None)
        };
        let (mut pts, mut ix) = (Vec::new(), Vec::new());
        unproject_depth_into(&depth, scale, &camera, host_mask, &mut pts, Some(&mut ix))?;

        let mut cloud = proj.alloc_point_cloud(w * h, true)?;
        proj.unproject_depth(&depth_dev, scale, &camera, dev_mask, &mut cloud)?;
        let dev_pts = cloud.to_host()?;
        assert_eq!(dev_pts.points(), &pts, "{w}x{h} mask={use_mask}");
        assert_eq!(
            cloud.pixel_index_to_host()?,
            Some(ix),
            "{w}x{h} mask={use_mask}"
        );

        let mut plain = proj.alloc_point_cloud(w * h, false)?;
        proj.unproject_depth(&depth_dev, scale, &camera, dev_mask, &mut plain)?;
        assert_eq!(plain.to_host()?.points(), &pts);
        assert_eq!(plain.pixel_index_to_host()?, None);
    }
    Ok(())
}

#[test]
fn cuda_unproject_matches_cpu() -> TestResult {
    let st = stream()?;
    let mut proj = CudaDepthProjector::new(&st)?;
    for (w, h) in SIZES {
        check_unproject(&mut proj, &st, depth_u16(w, h, 1), MM, w, h)?;
        check_unproject(&mut proj, &st, depth_f32(w, h, 2), 1.0, w, h)?;
    }
    Ok(())
}

fn check_register<T>(
    proj: &mut CudaDepthProjector,
    st: &Arc<CudaStream>,
    data: Vec<T>,
    scale: f32,
    (w, h): (usize, usize),
    (dw, dh): (usize, usize),
) -> TestResult
where
    T: DepthSample
        + cudarc::driver::DeviceRepr
        + cudarc::driver::ValidAsZeroBits
        + Default
        + PartialEq
        + std::fmt::Debug,
{
    let src_cam = camera_for(w, h);
    let dst_cam = camera_for(dw, dh);
    let src = Image::<T, 1>::new(sz(w, h), data)?;
    let src_dev = src.to_cuda(st)?;
    for radius in [0, 1, 2] {
        let mut want = Image::<T, 1>::from_size_val(sz(dw, dh), T::default())?;
        register_depth(&src, scale, &src_cam, &pose(), &dst_cam, radius, &mut want)?;
        let mut got_dev = Image::<T, 1>::zeros_cuda(sz(dw, dh), st)?;
        proj.register_depth(
            &src_dev,
            scale,
            &src_cam,
            &pose(),
            &dst_cam,
            radius,
            &mut got_dev,
        )?;
        let got = got_dev.to_host_owned()?;
        assert_eq!(
            got.as_slice(),
            want.as_slice(),
            "{w}x{h} -> {dw}x{dh} r={radius}"
        );
        // Not a vacuous comparison: when both sides span 64+ pixels the
        // rotation of `pose()` moves points by a few pixels only, so they land
        // in the destination. Narrow images have a field of view below that
        // rotation and may legitimately see every point leave.
        if w.min(h) >= 64 {
            assert!(
                want.as_slice().iter().any(|v| *v != T::default()),
                "{w}x{h} -> {dw}x{dh} r={radius} is empty"
            );
        }
    }
    Ok(())
}

#[test]
fn cuda_register_matches_cpu() -> TestResult {
    let st = stream()?;
    let mut proj = CudaDepthProjector::new(&st)?;
    for (w, h) in SIZES {
        // Same grid, a 2x upsample and a downsample.
        for (dw, dh) in [(w, h), (2 * w, 2 * h), (w.div_ceil(2), h.div_ceil(2))] {
            check_register(&mut proj, &st, depth_u16(w, h, 3), MM, (w, h), (dw, dh))?;
            let f = depth_f32(w, h, 4);
            check_register(&mut proj, &st, f, 1.0, (w, h), (dw, dh))?;
        }
    }
    Ok(())
}

/// Identity transform on the device: valid samples come back bit-exact,
/// invalid ones as `0.0`.
#[test]
fn cuda_register_f32_identity_reproduces_input() -> TestResult {
    let st = stream()?;
    let mut proj = CudaDepthProjector::new(&st)?;
    let (w, h) = (129, 77);
    let data = depth_f32(w, h, 5);
    let camera = camera_for(w, h);
    let src = Image::<f32, 1>::new(sz(w, h), data.clone())?.to_cuda(&st)?;
    let mut dst = Image::<f32, 1>::zeros_cuda(sz(w, h), &st)?;
    proj.register_depth(&src, 1.0, &camera, &Pose3d::IDENTITY, &camera, 0, &mut dst)?;
    let got = dst.to_host_owned()?;
    for (&o, &d) in got.as_slice().iter().zip(&data) {
        let want = if d > 0.0 && d.is_finite() { d } else { 0.0 };
        assert_eq!(o.to_bits(), want.to_bits());
    }
    Ok(())
}

#[test]
fn cuda_rejects_residency_and_capacity_errors() -> TestResult {
    let st = stream()?;
    let mut proj = CudaDepthProjector::new(&st)?;
    let camera = pinhole(10.0, 10.0, 2.0, 2.0);
    let host = Image::<u16, 1>::from_size_val(sz(4, 4), 1000)?;
    let dev = host.to_cuda(&st)?;

    let mut cloud = proj.alloc_point_cloud(16, false)?;
    assert!(matches!(
        proj.unproject_depth(&host, MM, &camera, None, &mut cloud),
        Err(DepthError::Image(ImageError::HostResident))
    ));
    let mut small = proj.alloc_point_cloud(15, false)?;
    assert!(matches!(
        proj.unproject_depth(&dev, MM, &camera, None, &mut small),
        Err(DepthError::CapacityTooSmall {
            capacity: 15,
            needed: 16
        })
    ));
    // Host functions refuse device images instead of panicking in as_slice.
    assert!(matches!(
        unproject_depth(&dev, MM, &camera, None),
        Err(DepthError::Image(ImageError::UnsupportedDevice))
    ));
    let mut dst_dev = Image::<u16, 1>::zeros_cuda(sz(4, 4), &st)?;
    assert!(matches!(
        register_depth(
            &dev,
            MM,
            &camera,
            &Pose3d::IDENTITY,
            &camera,
            0,
            &mut dst_dev
        ),
        Err(DepthError::Image(ImageError::UnsupportedDevice))
    ));
    let mut dst_host = Image::<u16, 1>::from_size_val(sz(4, 4), 0)?;
    assert!(matches!(
        proj.register_depth(
            &dev,
            MM,
            &camera,
            &Pose3d::IDENTITY,
            &camera,
            0,
            &mut dst_host
        ),
        Err(DepthError::Image(ImageError::HostResident))
    ));
    // A buffer from another stream is rejected.
    let other = st.context().new_stream()?;
    let mut foreign = CudaDepthProjector::new(&other)?.alloc_point_cloud(16, false)?;
    assert!(matches!(
        proj.unproject_depth(&dev, MM, &camera, None, &mut foreign),
        Err(DepthError::StreamMismatch)
    ));
    Ok(())
}
