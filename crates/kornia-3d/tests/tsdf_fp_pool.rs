//! `TsdfKernel::Simd` called from outside a rayon pool stays on the calling thread, so a
//! rounding mode set on the global pool's workers after a warm call cannot change its bits.
//!
//! Its own test binary with ONE test function, because it changes the FP control register of
//! every global-pool worker (and restores it).

#![cfg(target_arch = "aarch64")]
#![allow(unsafe_code)]

use kornia_3d::camera::PinholeCamera;
use kornia_3d::depth::DepthImage;
use kornia_3d::pose::Pose3d;
use kornia_3d::tsdf::{
    DepthRange, PinholeSensor, RangeSensor, SensorObservation, TsdfKernel, TsdfLayout, TsdfParams,
    TsdfVolume, TsdfWorkspace,
};
use kornia_algebra::{Mat3F64, Vec3F64};
use kornia_image::{Image, ImageSize};

const RMODE: u64 = 0b11 << 22;

fn fpcr() -> u64 {
    let v: u64;
    // SAFETY: `mrs` only reads the FP control register.
    unsafe { core::arch::asm!("mrs {0}, fpcr", out(reg) v) };
    v
}

fn set_fpcr(v: u64) {
    // SAFETY: changes this thread's rounding mode; the test restores it.
    unsafe { core::arch::asm!("msr fpcr, {0}", in(reg) v) };
}

/// Rotation about `axis` by `ang` (not a multiple of a right angle, so fused and
/// unfused results differ).
fn rot(axis: [f64; 3], ang: f64) -> Mat3F64 {
    let n = axis.iter().map(|a| a * a).sum::<f64>().sqrt();
    let [x, y, z] = axis.map(|a| a / n);
    let (s, c) = ang.sin_cos();
    let t = 1.0 - c;
    Mat3F64::from_cols_array(&[
        t * x * x + c,
        t * x * y + s * z,
        t * x * z - s * y,
        t * x * y - s * z,
        t * y * y + c,
        t * y * z + s * x,
        t * x * z + s * y,
        t * y * z - s * x,
        t * z * z + c,
    ])
}

#[test]
fn simd_outside_a_pool_ignores_the_global_workers_rounding_mode() -> Result<(), String> {
    let e = |e: kornia_3d::tsdf::TsdfError| format!("{e:?}");
    let size = ImageSize {
        width: 96,
        height: 72,
    };
    let cam = PinholeCamera {
        fx: 83.3,
        fy: 81.7,
        cx: 47.3,
        cy: 35.1,
        ..PinholeCamera::IDENTITY
    };
    let range = DepthRange::new(0.1, 4.0).map_err(e)?;
    let sensor = RangeSensor::from(PinholeSensor::new(&cam, size, range).map_err(e)?);
    let raw: Vec<u16> = (0..size.width * size.height)
        .map(|i| (1000 + 3 * (i % size.width) + 2 * (i / size.width)) as u16)
        .collect();
    let depth = DepthImage::new(Image::new(size, raw).unwrap(), 1000.0).unwrap();
    let rgb: Vec<u8> = (0..size.width * size.height * 3)
        .map(|i| (i * 37 % 251) as u8)
        .collect();
    let color = Image::<u8, 3>::new(size, rgb).unwrap();
    let p1 = Pose3d::new(
        rot([0.3, -0.2, 0.9], 0.31),
        Vec3F64::new(12.3456, -7.89, 3.21),
    );
    let p2 = Pose3d::new(
        rot([0.3, -0.2, 0.9], 0.33),
        Vec3F64::new(12.3556, -7.88, 3.22),
    );
    let obs = |p: Pose3d| {
        [SensorObservation {
            sensor: &sensor,
            volume_from_sensor: p,
            depth: (&depth).into(),
            color: Some(&color),
        }]
    };
    let layout = TsdfLayout::new(0.0137, 3.7).map_err(e)?;
    let volume = || TsdfVolume::<8>::new(layout, TsdfParams::default()).map_err(e);

    let mut reference = volume()?;
    let mut ws = TsdfWorkspace::new(TsdfKernel::Scalar);
    for p in [p1, p2] {
        reference.integrate(&mut ws, &obs(p)).map_err(e)?;
    }

    let mut vol = volume()?;
    let mut ws = TsdfWorkspace::new(TsdfKernel::Simd);
    vol.integrate(&mut ws, &obs(p1)).map_err(e)?;

    // every global worker now rounds towards zero
    let old = rayon::broadcast(|_| {
        let old = fpcr();
        set_fpcr(old | RMODE);
        old
    });
    assert!(rayon::broadcast(|_| fpcr() & RMODE == RMODE)
        .iter()
        .all(|&b| b));
    let r = vol.integrate(&mut ws, &obs(p2));
    // restore before any assertion can unwind
    rayon::broadcast(|c| set_fpcr(old[c.index()]));
    let stats = r.map_err(e)?;

    assert_eq!(
        vol.content_hash(),
        reference.content_hash(),
        "bits changed with the workers' rounding mode ({stats:?})"
    );
    Ok(())
}
