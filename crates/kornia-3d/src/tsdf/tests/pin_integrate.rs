//! Pins of the integration rules (contract sections 3, 5, 6, 7, 9, 10, 14):
//! host validation and errors, LiDAR tables, allocation, per-observation
//! visibility, colour registration, counters and the hash format.
#![allow(clippy::neg_cmp_op_on_partial_ord, clippy::needless_range_loop)]

use std::f64::consts::{FRAC_PI_2, TAU};

use super::*;
use crate::tsdf::alloc::{prepare, ObsKind};

const ID: tc::Rigid = tc::Rigid {
    r: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    t: [0.0; 3],
};

fn size(width: usize, height: usize) -> ImageSize {
    ImageSize { width, height }
}

fn range(min: f32, max: f32) -> DepthRange {
    DepthRange::new(min, max).unwrap()
}

fn fnv_words(words: &[u32]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for w in words {
        for b in w.to_le_bytes() {
            h = (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
        }
    }
    h
}

fn mirror() -> Pose3d {
    Pose3d::new(
        Mat3F64::from_cols_array(&[-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
        Vec3F64::new(0.0, 0.0, 0.0),
    )
}

fn at_z(z: f64) -> Pose3d {
    Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(0.0, 0.0, z))
}

// ------------------------------------------------------------ LiDAR tables ----

/// Every table value, in a fixed order, as words.
fn table_words(l: &SphericalLidar) -> Vec<u32> {
    let t = l.tables();
    let mut w = vec![t.fov_low.to_bits(), t.fov_high.to_bits()];
    for s in [
        t.row_boundaries,
        t.cos_el,
        t.sin_el,
        t.cos_az,
        t.sin_az,
        &[t.inv_az_step, t.az_offset],
    ] {
        w.extend(s.iter().map(|v| v.to_bits()));
    }
    w
}

/// Irregular beams, a clockwise spinner and no wrap.
fn lidar_irregular() -> SphericalLidar {
    SphericalLidar::new(
        size(12, 6),
        AzimuthGrid {
            start: 0.1,
            step: -0.1,
            wrap: false,
        },
        &[0.30, 0.17, 0.05, -0.02, -0.19, -0.33],
        range(0.2, 30.0),
    )
    .unwrap()
}

/// Uniform beams over a full counter-clockwise turn.
fn lidar_uniform() -> SphericalLidar {
    SphericalLidar::uniform(
        size(8, 4),
        AzimuthGrid {
            start: -3.0,
            step: TAU / 8.0,
            wrap: true,
        },
        0.4,
        -0.4,
        range(0.2, 30.0),
    )
    .unwrap()
}

#[test]
fn tsdf_lidar_tables_golden_bits() {
    let (a, b) = (lidar_irregular(), lidar_uniform());
    let (ta, tb) = (a.tables(), b.tables());
    // spelled out: (1 / -0.1) and (-0.1 / -0.1) are exactly -10 and 1
    assert_eq!(ta.inv_az_step.to_bits(), 0xc120_0000);
    assert_eq!(ta.az_offset.to_bits(), 0x3f80_0000);
    // the beams are listed top to bottom, the boundaries ascend (bottom to top)
    assert_eq!(ta.row_boundaries.len(), 5);
    assert!(ta.row_boundaries.windows(2).all(|w| w[0] < w[1]));
    assert!(ta.fov_low < ta.row_boundaries[0] && *ta.row_boundaries.last().unwrap() < ta.fov_high);
    assert_eq!(ta.sin_el[0].to_bits(), (libm::sin(0.30) as f32).to_bits());
    assert_eq!(ta.cos_el[5].to_bits(), (libm::cos(-0.33) as f32).to_bits());
    assert_eq!(
        ta.cos_az[3].to_bits(),
        (libm::cos(0.1 + 3.0 * -0.1) as f32).to_bits()
    );
    assert_eq!(
        ta.row_boundaries[0].to_bits(),
        (libm::sin((-0.33 + -0.19) / 2.0) as f32).to_bits()
    );
    // the field of view extends half a beam spacing past the outer beams
    assert_eq!(
        ta.fov_high.to_bits(),
        (libm::sin(0.30 + (0.30 - 0.17) / 2.0) as f32).to_bits()
    );
    assert_eq!(
        ta.fov_low.to_bits(),
        (libm::sin(-0.33 - (-0.19 - -0.33) / 2.0) as f32).to_bits()
    );
    // uniform: the full turn, the +-0.4 rad beams
    assert_eq!(
        tb.inv_az_step.to_bits(),
        ((1.0 / (TAU / 8.0)) as f32).to_bits()
    );
    assert_eq!(
        tb.az_offset.to_bits(),
        ((3.0 / (TAU / 8.0)) as f32).to_bits()
    );
    assert_eq!(tb.sin_el[0].to_bits(), (libm::sin(0.4) as f32).to_bits());
    let (ha, hb) = (fnv_words(&table_words(&a)), fnv_words(&table_words(&b)));
    assert_eq!(ha, LIDAR_IRREGULAR_HASH, "irregular {ha:#x}");
    assert_eq!(hb, LIDAR_UNIFORM_HASH, "uniform {hb:#x}");
}

const LIDAR_IRREGULAR_HASH: u64 = 0xacb2_6e72_661b_b41f;
const LIDAR_UNIFORM_HASH: u64 = 0xbecd_c348_4f02_4d21;

#[test]
fn tsdf_lidar_from_tables_validation() -> R {
    let a = lidar_irregular();
    let t = a.tables();
    let grid = a.azimuth();
    let sz = a.size();
    let r = a.range();
    // a sensor rebuilt from its own tables is bit-identical
    let b = SphericalLidar::from_tables(sz, grid, &t, r)?;
    assert_eq!(table_words(&a), table_words(&b));
    let bad = |f: &dyn Fn(&mut Vec<f32>, &mut LidarTablesOwned)| {
        let mut own = LidarTablesOwned::of(&t);
        let mut bnd = t.row_boundaries.to_vec();
        f(&mut bnd, &mut own);
        let lt = LidarTables {
            row_boundaries: &bnd,
            cos_el: &own.cos_el,
            sin_el: &own.sin_el,
            cos_az: &own.cos_az,
            sin_az: &own.sin_az,
            fov_low: own.fov_low,
            fov_high: own.fov_high,
            inv_az_step: own.inv_az_step,
            az_offset: own.az_offset,
        };
        SphericalLidar::from_tables(sz, grid, &lt, r).err()
    };
    let beam = |e: Option<TsdfError>| matches!(e, Some(TsdfError::InvalidBeamTable(_)));
    let azim = |e: Option<TsdfError>| matches!(e, Some(TsdfError::InvalidAzimuthGrid { .. }));
    // boundaries must stay strictly ascending after rounding
    assert!(beam(bad(&|b, _| b[1] = b[0])), "equal boundaries");
    assert!(beam(bad(&|b, _| b.swap(1, 2))), "descending boundaries");
    assert!(
        beam(bad(&|b, o| o.fov_low = b[0])),
        "fov_low == first boundary"
    );
    assert!(
        beam(bad(&|b, o| o.fov_high = b[4])),
        "fov_high == last boundary"
    );
    assert!(beam(bad(&|_, o| o.fov_high = 1.5)), "value above 1");
    assert!(beam(bad(&|_, o| o.cos_el[2] = f32::NAN)), "NaN");
    assert!(
        beam(bad(&|_, o| {
            o.sin_az.pop();
        })),
        "table length"
    );
    assert!(
        beam(bad(&|b, _| {
            b.pop();
        })),
        "boundary count"
    );
    assert!(azim(bad(&|_, o| o.inv_az_step = 0.0)));
    assert!(azim(bad(&|_, o| o.az_offset = f32::NAN)));
    // elevations closer than the f32 rounding of their sines
    let e = [1.0 + 3e-9, 1.0 + 2e-9, 1.0 + 1e-9, 1.0];
    let r = SphericalLidar::new(size(8, 4), grid_wrap(8), &e, range(0.1, 5.0));
    assert!(matches!(r, Err(TsdfError::InvalidBeamTable(_))));
    Ok(())
}

/// Owned copies of the tables of a sensor, for building bad variants.
struct LidarTablesOwned {
    cos_el: Vec<f32>,
    sin_el: Vec<f32>,
    cos_az: Vec<f32>,
    sin_az: Vec<f32>,
    fov_low: f32,
    fov_high: f32,
    inv_az_step: f32,
    az_offset: f32,
}

impl LidarTablesOwned {
    fn of(t: &LidarTables<'_>) -> Self {
        Self {
            cos_el: t.cos_el.to_vec(),
            sin_el: t.sin_el.to_vec(),
            cos_az: t.cos_az.to_vec(),
            sin_az: t.sin_az.to_vec(),
            fov_low: t.fov_low,
            fov_high: t.fov_high,
            inv_az_step: t.inv_az_step,
            az_offset: t.az_offset,
        }
    }
}

fn grid_wrap(columns: usize) -> AzimuthGrid {
    AzimuthGrid {
        start: 0.0,
        step: TAU / columns as f64,
        wrap: true,
    }
}

#[test]
fn tsdf_lidar_and_sensor_validation() -> R {
    let dr = range(0.1, 5.0);
    let ok = [0.3, 0.1, -0.1];
    let new = |sz, grid, el: &[f64]| SphericalLidar::new(sz, grid, el, dr);
    let beam =
        |r: Result<SphericalLidar, TsdfError>| matches!(r, Err(TsdfError::InvalidBeamTable(_)));
    let azim = |r: Result<SphericalLidar, TsdfError>| {
        matches!(r, Err(TsdfError::InvalidAzimuthGrid { .. }))
    };
    let par =
        |r: Result<SphericalLidar, TsdfError>| matches!(r, Err(TsdfError::InvalidParameter { .. }));
    assert!(new(size(8, 3), grid_wrap(8), &ok).is_ok());
    // beams: one per row, strictly decreasing, finite, within +-pi/2
    assert!(beam(new(size(8, 3), grid_wrap(8), &ok[..2])));
    assert!(
        beam(new(size(8, 3), grid_wrap(8), &[0.3, 0.3, -0.1])),
        "equal"
    );
    assert!(
        beam(new(size(8, 3), grid_wrap(8), &[-0.1, 0.1, 0.3])),
        "increasing"
    );
    assert!(beam(new(size(8, 3), grid_wrap(8), &[0.3, f64::NAN, -0.1])));
    assert!(
        beam(new(size(8, 3), grid_wrap(8), &[1.6, 0.1, -0.1])),
        "above pi/2"
    );
    assert!(new(size(8, 3), grid_wrap(8), &[FRAC_PI_2, 0.1, -FRAC_PI_2]).is_ok());
    // size: at least 2 x 2
    assert!(par(new(size(1, 3), grid_wrap(8), &ok)));
    assert!(par(SphericalLidar::uniform(
        size(8, 1),
        grid_wrap(8),
        0.1,
        -0.1,
        dr
    )));
    // azimuth grid
    let g = |start: f64, step: f64, wrap: bool| AzimuthGrid { start, step, wrap };
    let w = size(8, 3);
    assert!(azim(new(w, g(0.0, 0.0, false), &ok)), "zero step");
    assert!(azim(new(w, g(f64::NAN, 0.1, false), &ok)));
    assert!(azim(new(w, g(0.0, f64::INFINITY, false), &ok)));
    assert!(
        azim(new(w, g(TAU + 0.01, 0.1, false), &ok)),
        "|start| > 2 pi"
    );
    assert!(new(w, g(TAU, 0.1, false), &ok).is_ok());
    assert!(
        azim(new(w, g(0.0, 0.1, true), &ok)),
        "wrap needs a full turn"
    );
    // the full turn holds to 1e-9 relative, either direction
    let step = TAU / 8.0;
    assert!(new(w, g(0.0, step * (1.0 + 0.5e-9), true), &ok).is_ok());
    assert!(azim(new(w, g(0.0, step * (1.0 + 2.0e-9), true), &ok)));
    assert!(
        new(w, g(0.0, -step, true), &ok).is_ok(),
        "clockwise full turn"
    );
    assert!(
        new(w, g(0.0, 0.1, false), &ok).is_ok(),
        "partial field of view"
    );

    // pinhole sensors
    let cam = |fx: f64, fy: f64, cx: f64, cy: f64| PinholeCamera {
        fx,
        fy,
        cx,
        cy,
        ..PinholeCamera::IDENTITY
    };
    let s = size(4, 4);
    let intr = |c: &PinholeCamera| {
        matches!(
            PinholeSensor::new(c, s, dr),
            Err(TsdfError::InvalidIntrinsics { .. })
        )
    };
    assert!(PinholeSensor::new(&cam(10.0, 10.0, 2.0, 2.0), s, dr).is_ok());
    assert!(intr(&cam(0.0, 10.0, 2.0, 2.0)));
    assert!(intr(&cam(10.0, -1.0, 2.0, 2.0)));
    assert!(intr(&cam(f64::NAN, 10.0, 2.0, 2.0)));
    assert!(intr(&cam(10.0, 10.0, f64::INFINITY, 2.0)));
    assert!(intr(&cam(10.0, 10.0, 2.0, f64::NAN)));
    assert!(intr(&cam(1e-300, 10.0, 2.0, 2.0)), "1 / fx overflows f32");
    assert!(intr(&cam(1e300, 10.0, 2.0, 2.0)), "1 / fx underflows to 0");
    for (k1, k2, p1, p2) in [
        (1e-9, 0.0, 0.0, 0.0),
        (0.0, -1e-9, 0.0, 0.0),
        (0.0, 0.0, 1e-9, 0.0),
        (0.0, 0.0, 0.0, f64::NAN),
    ] {
        let c = PinholeCamera {
            k1,
            k2,
            p1,
            p2,
            ..cam(10.0, 10.0, 2.0, 2.0)
        };
        assert!(matches!(
            PinholeSensor::new(&c, s, dr),
            Err(TsdfError::DistortedCamera)
        ));
    }
    // image size limits: W, H <= 2^24 and W * H < 2^31 (u32 on the device)
    let c = cam(10.0, 10.0, 2.0, 2.0);
    let sized = |w: usize, h: usize| PinholeSensor::new(&c, size(w, h), dr);
    assert!(sized(1 << 24, 1).is_ok());
    assert!(sized(1, 1 << 24).is_ok());
    assert!(sized((1 << 24) + 1, 1).is_err());
    assert!(sized(1 << 24, 127).is_ok());
    assert!(sized(1 << 24, 128).is_err());
    assert!(sized(65536, 65536).is_err());
    assert!(sized(0, 4).is_err() && sized(4, 0).is_err());
    // depth ranges
    for (a, b) in [
        (0.0, 1.0),
        (-1.0, 1.0),
        (1.0, 1.0),
        (2.0, 1.0),
        (0.1, f32::INFINITY),
        (f32::NAN, 1.0),
        (0.1, f32::NAN),
    ] {
        assert!(
            matches!(
                DepthRange::new(a, b),
                Err(TsdfError::InvalidParameter { .. })
            ),
            "{a} {b}"
        );
    }
    // layouts
    let lay = |v: f32, t: f32| TsdfLayout::new(v, t).is_ok();
    assert!(lay(1e-4, 1.0) && lay(10.0, 64.0));
    assert!(!lay(9.9e-5, 4.0) && !lay(10.001, 4.0) && !lay(0.02, 0.999) && !lay(0.02, 64.01));
    assert!(!lay(f32::NAN, 4.0) && !lay(0.02, f32::NAN) && !lay(f32::INFINITY, 4.0));
    // volume parameters
    let lay = TsdfLayout::new(0.02, 4.0)?;
    let p = |f: &dyn Fn(&mut TsdfParams)| {
        let mut q = TsdfParams {
            block_capacity: 16,
            ..TsdfParams::default()
        };
        f(&mut q);
        TsdfVolume::<8>::new(lay, q).is_err()
    };
    assert!(p(&|q| q.max_weight = 0) && p(&|q| q.max_color_weight = 0));
    assert!(p(&|q| q.block_capacity = 0));
    assert!(
        p(&|q| q.block_capacity = (u32::MAX as usize / 512) + 1),
        "indices must fit u32"
    );
    // rigs: ascending ids, no duplicates, valid mounts
    let mk = |id: u32, p: Pose3d| RigSensor {
        id: SensorId(id),
        sensor: PinholeSensor::new(&c, s, dr).unwrap().into(),
        body_from_sensor: p,
    };
    let rig = SensorRig::new(vec![
        mk(5, Pose3d::IDENTITY),
        mk(2, Pose3d::IDENTITY),
        mk(9, Pose3d::IDENTITY),
    ])?;
    assert_eq!(
        rig.sensors().iter().map(|r| r.id.0).collect::<Vec<_>>(),
        [2, 5, 9]
    );
    assert!(rig.get(SensorId(5)).is_some() && rig.get(SensorId(4)).is_none());
    assert!(matches!(
        SensorRig::new(vec![mk(1, Pose3d::IDENTITY), mk(1, Pose3d::IDENTITY)]),
        Err(TsdfError::DuplicateSensor(SensorId(1)))
    ));
    assert!(matches!(
        SensorRig::new(vec![mk(1, mirror())]),
        Err(TsdfError::InvalidTransform)
    ));
    // poses: finite, orthonormal to 1e-6, det > 0
    let rot = |e: f64| {
        Pose3d::new(
            Mat3F64::from_cols_array(&[1.0 + e, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
            Vec3F64::ZERO,
        )
    };
    assert!(check_pose(&rot(0.0)).is_ok() && check_pose(&rot(4e-7)).is_ok());
    assert!(check_pose(&rot(1e-5)).is_err());
    let nan_t = Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(0.0, f64::NAN, 0.0));
    let inf_t = Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(f64::INFINITY, 0.0, 0.0));
    assert!(check_pose(&nan_t).is_err() && check_pose(&inf_t).is_err());
    let huge = Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(1e300, 0.0, 0.0));
    assert!(
        rigid_from_pose(&huge).is_err(),
        "f32 overflow of a translation"
    );
    Ok(())
}

// ------------------------------------------------------- all or nothing ----

#[test]
fn tsdf_integrate_all_or_nothing() -> R {
    let cam = pinhole(32, 24, 30.0, (0.1, 4.0));
    let lid = lidar(64, 8, 0.2, -0.2, (0.2, 5.0));
    let d = DepthImage::from_size_val(cam.sensor.size(), 1.0f32, 1.0)?;
    let dl = DepthImage::from_size_val(lid.sensor.size(), 1.0f32, 1.0)?;
    let small = DepthImage::from_size_val(size(8, 6), 1.0f32, 1.0)?;
    let colour = color_image(cam.sensor.size(), 0);
    let wrong_colour = color_image(size(8, 6), 0);
    let nan_pose = Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(f64::NAN, 0.0, 0.0));
    let mut vol = volume::<8>(1024);
    let mut ws = TsdfWorkspace::default();
    let empty = vol.content_hash();
    let good = obs(&cam, Pose3d::IDENTITY, &d);
    let cases: Vec<(&str, SensorObservation<'_>)> = vec![
        ("mirror pose", obs(&cam, mirror(), &d)),
        ("NaN pose", obs(&cam, nan_pose, &d)),
        ("depth size", obs(&cam, Pose3d::IDENTITY, &small)),
        (
            "colour on a lidar",
            SensorObservation {
                color: Some(&colour),
                ..obs(&lid, Pose3d::IDENTITY, &dl)
            },
        ),
        (
            "colour size",
            SensorObservation {
                color: Some(&wrong_colour),
                ..obs(&cam, Pose3d::IDENTITY, &d)
            },
        ),
    ];
    for (name, bad) in cases {
        // the valid observation comes FIRST: it must not be applied
        let e = vol.integrate(&mut ws, &[good, bad]).unwrap_err();
        match name {
            "mirror pose" | "NaN pose" => {
                assert!(matches!(e, TsdfError::InvalidTransform), "{name}: {e}")
            }
            "depth size" => assert!(
                matches!(e, TsdfError::SizeMismatch { expected, got } if expected == cam.sensor.size() && got == size(8, 6)),
                "{name}: {e}"
            ),
            "colour on a lidar" => {
                assert!(matches!(e, TsdfError::ColorOnRangeSensor), "{name}: {e}")
            }
            _ => assert!(
                matches!(e, TsdfError::SizeMismatch { expected, got } if expected == cam.sensor.size() && got == size(8, 6)),
                "{name}: {e}"
            ),
        }
        assert_eq!(vol.num_blocks(), 0, "{name}");
        assert_eq!(vol.content_hash(), empty, "{name}");
        assert_eq!(vol.last_step_blocks().count(), 0);
    }
    // an empty step is fine and changes nothing
    assert_eq!(vol.integrate(&mut ws, &[])?, IntegrateStats::default());
    assert_eq!(vol.content_hash(), empty);
    // and the volume still works afterwards
    assert!(vol.integrate(&mut ws, &[good])?.updated_voxels > 0);
    // colour is ignored by a volume without colour storage
    let mut plain = TsdfVolume::<8>::new(
        TsdfLayout::new(VOXEL, 4.0)?,
        TsdfParams {
            color: false,
            ..params(256)
        },
    )?;
    let with_colour = SensorObservation {
        color: Some(&colour),
        ..good
    };
    plain.integrate(&mut ws, &[with_colour])?;
    assert!(plain
        .block_keys()
        .all(|k| plain.block(k).unwrap().rgbw.is_none()));
    assert_eq!(plain.voxel([0, 0, 49]).unwrap().color_weight, 0);
    Ok(())
}

// --------------------------------------------------------- visibility ----

#[test]
fn tsdf_integrate_visible_set_per_observation() -> R {
    // decision V: an observation updates only the blocks it touched. Depth 1.0
    // and depth 3.0 along the same rays: the far observation would write free
    // space (+1) into the near band if visibility were the union of the step.
    let cam = pinhole(64, 48, 60.0, (0.1, 4.0));
    let near = DepthImage::from_size_val(cam.sensor.size(), 1.0f32, 1.0)?;
    let far = DepthImage::from_size_val(cam.sensor.size(), 3.0f32, 1.0)?;
    let mut ws = TsdfWorkspace::default();
    let mut fused = volume::<8>(4096);
    fused.integrate(
        &mut ws,
        &[
            obs(&cam, Pose3d::IDENTITY, &near),
            obs(&cam, Pose3d::IDENTITY, &far),
        ],
    )?;
    let mut sequential = volume::<8>(4096);
    sequential.integrate(&mut ws, &[obs(&cam, Pose3d::IDENTITY, &near)])?;
    sequential.integrate(&mut ws, &[obs(&cam, Pose3d::IDENTITY, &far)])?;
    assert_eq!(fused.content_hash(), sequential.content_hash());
    let l = fused.layout();
    let (g_near, g_far) = ([0, 0, 49], [0, 0, 149]);
    let (vn, vf) = (fused.voxel(g_near).unwrap(), fused.voxel(g_far).unwrap());
    assert_eq!((vn.weight, vf.weight), (1, 1));
    let tn = tc::truncate(
        1.0 - l.centre_from_voxel(g_near)[2],
        l.truncation(),
        1.0 / l.truncation(),
    );
    assert_eq!(vn.tsdf.to_bits(), tn.unwrap().to_bits());
    assert!(
        vn.tsdf < 1.0,
        "the near voxel is inside the truncation band"
    );
    Ok(())
}

// ---------------------------------------------------- colour registration ----

#[test]
fn tsdf_color_pixel_end_to_end() -> R {
    // even W x H: the principal point (31.5, 23.5) is a pixel corner, so the
    // axis voxels project to a fractional coordinate and floor(u) and
    // floor(u + 0.5) pick different pixels
    let cam = pinhole(64, 48, 60.0, (0.1, 4.0));
    let sz = cam.sensor.size();
    let d = DepthImage::from_size_val(sz, 1.0f32, 1.0)?;
    let mut data = Vec::new();
    for y in 0..sz.height {
        for x in 0..sz.width {
            data.extend([x as u8, y as u8, (7 * x + y) as u8]);
        }
    }
    let colour = Image::<u8, 3>::new(sz, data)?;
    let mut vol = volume::<8>(2048);
    let o = SensorObservation {
        color: Some(&colour),
        ..obs(&cam, Pose3d::IDENTITY, &d)
    };
    vol.integrate(&mut TsdfWorkspace::default(), &[o])?;
    let (mut u_differs, mut v_differs, mut checked) = (0, 0, 0);
    for gx in -3..=3 {
        for gy in -3..=3 {
            for gz in [48, 49, 50] {
                let g = [gx, gy, gz];
                let c = vol.layout().centre_from_voxel(g);
                let (x, y, z) = (c[0] as f64, c[1] as f64, c[2] as f64);
                let (u, v) = (60.0 * x / z + 31.5, 60.0 * y / z + 23.5);
                // skip coordinates within 1e-3 of a pixel boundary (f64 spec vs f32 ops)
                let near = |t: f64| {
                    let f = (t + 0.5) - (t + 0.5).floor();
                    f.min(1.0 - f) < 1e-3
                };
                if near(u) || near(v) {
                    continue;
                }
                let (px, py) = ((u + 0.5).floor() as i32, (v + 0.5).floor() as i32);
                u_differs += (u.floor() as i32 != px) as usize;
                v_differs += (v.floor() as i32 != py) as usize;
                let vx = vol.voxel(g).unwrap();
                assert_eq!(vx.weight, 1, "{g:?}");
                assert_eq!(vx.color_weight, 1);
                assert_eq!(
                    vx.rgb,
                    [px as u8, py as u8, (7 * px + py) as u8],
                    "{g:?} u {u} v {v}"
                );
                checked += 1;
            }
        }
    }
    assert!(checked > 100 && u_differs > 10 && v_differs > 10);
    // behind the band (sdf < -trunc): allocated, but neither tsdf nor colour touched
    let behind = vol.voxel([0, 0, 54]).unwrap();
    assert_eq!(
        (behind.weight, behind.rgb, behind.color_weight),
        (0, [0; 3], 0)
    );
    assert_eq!(behind.tsdf.to_bits(), 0);
    Ok(())
}

// ------------------------------------------------------------- allocation ----

#[test]
fn tsdf_out_of_range_counted() -> R {
    let cam = pinhole(16, 12, 20.0, (0.1, 4.0));
    let d = DepthImage::from_size_val(cam.sensor.size(), 1.0f32, 1.0)?;
    let mut ws = TsdfWorkspace::default();
    let n = 16 * 12;
    // far beyond |g| < 2^22 voxels (83886 m at 2 cm), both signs: counted, never allocated
    for z in [90_000.0, -90_000.0] {
        let mut vol = volume::<8>(256);
        let s = vol.integrate(&mut ws, &[obs(&cam, at_z(z), &d)])?;
        assert_eq!(
            (
                s.valid_pixels,
                s.out_of_range,
                s.new_blocks,
                s.updated_voxels
            ),
            (n, n, 0, 0)
        );
        assert_eq!(vol.num_blocks(), 0);
    }
    // the band edge at x = 83886.0: columns 0..7 are in, 8..15 are out
    let mut vol = volume::<8>(256);
    let pose = Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(83_886.0, 0.0, 0.0));
    let s = vol.integrate(&mut ws, &[obs(&cam, pose, &d)])?;
    assert_eq!((s.valid_pixels, s.out_of_range), (n, n / 2));
    assert!(vol.num_blocks() > 0 && s.updated_voxels > 0);
    // the blocks sit at the end of the key range, below 2^19
    assert!(vol.block_keys().all(|k| k.coords()[0] < (1 << 19)));
    Ok(())
}

#[test]
fn tsdf_mark_epoch_wraps_without_stale_marks() -> R {
    let cam = pinhole(48, 36, 45.0, (0.1, 4.0));
    let d = DepthImage::from_size_val(cam.sensor.size(), 1.0f32, 1.0)?;
    let none = DepthImage::from_size_val(cam.sensor.size(), 0.0f32, 1.0)?;
    let (mut a, mut b) = (volume::<8>(1024), volume::<8>(1024));
    let mut ws = TsdfWorkspace::default();
    for v in [&mut a, &mut b] {
        v.integrate(&mut ws, &[obs(&cam, Pose3d::IDENTITY, &d)])?;
    }
    // b's next mark is u32::MAX, then the counter wraps inside the step; the
    // blocks still carry mark 1 from the first step
    b.index.mark_epoch = u32::MAX - 1;
    for v in [&mut a, &mut b] {
        v.integrate(
            &mut ws,
            &[
                obs(&cam, Pose3d::IDENTITY, &none),
                obs(&cam, Pose3d::IDENTITY, &d),
            ],
        )?;
        v.integrate(&mut ws, &[obs(&cam, Pose3d::IDENTITY, &d)])?;
    }
    assert!(b.index.mark_epoch < 10, "the counter wrapped");
    assert_eq!(a.content_hash(), b.content_hash());
    assert!(a.voxel([0, 0, 49]).unwrap().weight == 3);
    Ok(())
}

/// Blocks of a pinhole + lidar step, from the contract's touched-set rule.
fn spec_blocks(
    pinhole: (&PinholeSensor, &Pose3d, &DepthImage<f32>),
    lidar: (&SphericalLidar, &Pose3d, &DepthImage<f32>),
    layout: &TsdfLayout,
    log2: u32,
) -> std::collections::BTreeSet<u64> {
    let (trunc, inv) = (layout.truncation(), 1.0f32 / layout.voxel_size());
    let mut set = std::collections::BTreeSet::new();
    let mut add = |pv: [f32; 3]| {
        let mut lo = [0i32; 3];
        let mut hi = [0i32; 3];
        for a in 0..3 {
            let (l, h) = ((pv[a] - trunc) * inv, (pv[a] + trunc) * inv);
            assert!(l >= -4_194_304.0 && h < 4_194_304.0);
            (lo[a], hi[a]) = ((l.floor() as i32) >> log2, (h.floor() as i32) >> log2);
        }
        for x in lo[0]..=hi[0] {
            for y in lo[1]..=hi[1] {
                for z in lo[2]..=hi[2] {
                    set.insert(tc::pack_block_key([x, y, z]));
                }
            }
        }
    };
    let xf = |pose: &Pose3d, p: [f32; 3]| {
        let t = rigid_from_pose(pose).unwrap();
        let f = |i: usize| {
            libm::fmaf(
                t.r[3 * i],
                p[0],
                libm::fmaf(
                    t.r[3 * i + 1],
                    p[1],
                    libm::fmaf(t.r[3 * i + 2], p[2], t.t[i]),
                ),
            )
        };
        [f(0), f(1), f(2)]
    };
    let (ps, pp, pd) = pinhole;
    let k = ps.params(ID, ID, 1.0);
    for (i, &z) in pd.image().as_slice().iter().enumerate() {
        if z > 0.0 && z >= k.range_min && z < k.range_max {
            let (px, py) = ((i % 16) as f32, (i / 16) as f32);
            let (xn, yn) = ((px - k.cx) * k.inv_fx, (py - k.cy) * k.inv_fy);
            add(xf(pp, [xn * z, yn * z, z]));
        }
    }
    let (ls, lp, ld) = lidar;
    let t = ls.tables();
    for (i, &d) in ld.image().as_slice().iter().enumerate() {
        if d > 0.0 && d >= ls.range().min && d < ls.range().max {
            let (row, col) = (i / ls.size().width, i % ls.size().width);
            let (ce, se, ca, sa) = (t.cos_el[row], t.sin_el[row], t.cos_az[col], t.sin_az[col]);
            add(xf(lp, [d * (ce * ca), d * (ce * sa), d * se]));
        }
    }
    set
}

#[test]
fn tsdf_alloc_union_of_observations_equals_spec() -> R {
    let cam = pinhole(16, 12, 14.0, (0.1, 4.0));
    let lid = lidar(90, 6, 0.25, -0.25, (0.2, 4.0));
    let scene = [
        tilted_plane(),
        Surface::Sphere {
            centre: [0.0, 0.1, 1.0],
            radius: 0.3,
        },
    ];
    let pc = Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(0.01, -0.02, 0.0));
    let pl = lidar_mount([0.0, 0.05, 0.0]);
    let dc = render(&cam, &pc, &scene);
    let dl = render(&lid, &pl, &scene);
    let (RangeSensor::Pinhole(ps), RangeSensor::Spherical(ls)) = (&cam.sensor, &lid.sensor) else {
        unreachable!()
    };
    let want = spec_blocks(
        (ps, &pc, &dc),
        (ls, &pl, &dl),
        &TsdfLayout::new(VOXEL, 4.0)?,
        3,
    );
    assert!(want.len() > 20);
    // within a step the union of both observations is allocated (lidar first
    // or last changes nothing)
    for order in [[0, 1], [1, 0]] {
        let all = [obs(&cam, pc, &dc), obs(&lid, pl, &dl)];
        let mut vol = volume::<8>(4096);
        let s = vol.integrate(
            &mut TsdfWorkspace::default(),
            &[all[order[0]], all[order[1]]],
        )?;
        let got: std::collections::BTreeSet<u64> = vol.block_keys().map(|k| k.bits()).collect();
        assert_eq!(got, want, "{order:?}");
        assert_eq!(s.new_blocks as usize, want.len());
        assert_eq!(s.touched_blocks as usize, want.len(), "distinct blocks");
    }
    Ok(())
}

#[test]
fn tsdf_step_counters_and_last_step_blocks() -> R {
    let cam = pinhole(48, 36, 45.0, (0.1, 4.0));
    let scene = [tilted_plane()];
    let p0 = Pose3d::IDENTITY;
    let p1 = pose(0.0, 0.5, 0.0, [0.4, 0.0, 0.0]);
    let (d0, d1) = (render(&cam, &p0, &scene), render(&cam, &p1, &scene));
    let valid =
        |d: &DepthImage<f32>| d.image().as_slice().iter().filter(|&&v| v > 0.0).count() as u64;
    let mut ws = TsdfWorkspace::default();
    let keys_of = |o: &[SensorObservation<'_>]| -> Result<Vec<u64>, TsdfError> {
        let mut v = volume::<8>(4096);
        v.integrate(&mut TsdfWorkspace::default(), o)?;
        Ok(v.block_keys().map(|k| k.bits()).collect())
    };
    let (k0, k1) = (
        keys_of(&[obs(&cam, p0, &d0)])?,
        keys_of(&[obs(&cam, p1, &d1)])?,
    );
    let overlap = k0.iter().filter(|k| k1.contains(k)).count();
    assert!(
        overlap > 0 && overlap < k1.len(),
        "{overlap} of {}",
        k1.len()
    );
    let mut vol = volume::<8>(4096);
    let s0 = vol.integrate(&mut ws, &[obs(&cam, p0, &d0)])?;
    assert_eq!(s0.valid_pixels, valid(&d0));
    assert_eq!(
        (s0.new_blocks as usize, s0.touched_blocks as usize),
        (k0.len(), k0.len())
    );
    assert!(vol
        .last_step_blocks()
        .map(|k| k.bits())
        .eq(k0.iter().copied()));
    // the second step touches its own set (new and old blocks), not the first's
    let s1 = vol.integrate(&mut ws, &[obs(&cam, p1, &d1)])?;
    assert_eq!(s1.touched_blocks as usize, k1.len());
    assert_eq!(s1.new_blocks as usize, k1.len() - overlap);
    assert_eq!(s1.valid_pixels, valid(&d1));
    assert!(vol
        .last_step_blocks()
        .map(|k| k.bits())
        .eq(k1.iter().copied()));
    assert_eq!(vol.num_blocks(), k0.len() + k1.len() - overlap);
    // two observations of one step count a shared block once
    let both = vol.integrate(&mut ws, &[obs(&cam, p0, &d0), obs(&cam, p1, &d1)])?;
    let union = k0.len() + k1.len() - overlap;
    assert_eq!((both.touched_blocks as usize, both.new_blocks), (union, 0));
    assert_eq!(both.valid_pixels, valid(&d0) + valid(&d1));
    // an empty step touches nothing
    vol.integrate(&mut ws, &[])?;
    assert_eq!(vol.last_step_blocks().count(), 0);
    Ok(())
}

// --------------------------------------------------------- hash format ----

#[test]
fn tsdf_content_hash_format() -> R {
    let spec = |vol: &TsdfVolume<8>, colour: bool| {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        let mut eat = |b: &[u8]| {
            for &x in b {
                h = (h ^ u64::from(x)).wrapping_mul(0x0100_0000_01b3);
            }
        };
        for k in vol.block_keys() {
            let b = vol.block(k).unwrap();
            eat(&k.bits().to_le_bytes());
            for i in 0..512 {
                eat(&b.tsdf[i].to_bits().to_le_bytes());
                eat(&b.weight[i].to_le_bytes());
                if colour {
                    eat(&b.rgbw.unwrap()[i].to_le_bytes());
                }
            }
        }
        h
    };
    let with = scene_two_views::<8>()?;
    assert!(with
        .block(with.block_keys().next().unwrap())
        .unwrap()
        .rgbw
        .is_some());
    assert_eq!(with.content_hash(), spec(&with, true));
    // without colour storage the rgbw word is not hashed
    let cam = pinhole(48, 36, 45.0, (0.1, 4.0));
    let d = render(&cam, &Pose3d::IDENTITY, &[tilted_plane()]);
    let mut plain = TsdfVolume::<8>::new(
        TsdfLayout::new(VOXEL, 4.0)?,
        TsdfParams {
            color: false,
            ..params(1024)
        },
    )?;
    plain.integrate(
        &mut TsdfWorkspace::default(),
        &[obs(&cam, Pose3d::IDENTITY, &d)],
    )?;
    assert!(plain.num_blocks() > 0);
    assert_eq!(plain.content_hash(), spec(&plain, false));
    // raw bits: -0.0 and +0.0 hash differently, so a stored -0 would show
    let mut neg = TsdfVolume::<8>::new(
        TsdfLayout::new(VOXEL, 4.0)?,
        TsdfParams {
            color: false,
            ..params(1024)
        },
    )?;
    neg.integrate(
        &mut TsdfWorkspace::default(),
        &[obs(&cam, Pose3d::IDENTITY, &d)],
    )?;
    assert_eq!(neg.content_hash(), plain.content_hash());
    let k = neg.block_keys().next().unwrap();
    let i = neg.view().block_base(k.bits()).unwrap();
    assert_eq!(neg.pools.tsdf[i].to_bits(), 0);
    neg.pools.tsdf[i] = -0.0;
    assert_ne!(neg.content_hash(), plain.content_hash());
    Ok(())
}

// ------------------------------------------------------ range image gates ----

#[test]
fn tsdf_range_image_gates() -> R {
    let l = lidar(64, 8, 0.2, -0.2, (0.2, 20.0));
    let RangeSensor::Spherical(sl) = &l.sensor else {
        unreachable!()
    };
    let (min, max) = (sl.range().min, sl.range().max);
    let run = |pts: &[[f32; 3]]| -> Result<Vec<(usize, u32)>, TsdfError> {
        let mut dst = DepthImage::from_size_val(sl.size(), 7.0f32, 1.0)?;
        sl.range_image_from_points_into(pts, &mut dst)?;
        Ok(dst
            .image()
            .as_slice()
            .iter()
            .enumerate()
            .filter(|(_, v)| **v != 0.0)
            .map(|(i, v)| (i, v.to_bits()))
            .collect())
    };
    // along +x: sqrt(RN(x * x)) == x, so r == range.min is kept and the float below is not
    let one = |r: f32| run(&[[r, 0.0, 0.0]]);
    assert_eq!(
        one(min)?.iter().map(|p| p.1).collect::<Vec<_>>(),
        [min.to_bits()]
    );
    assert!(one(pred_f(min))?.is_empty(), "below range.min");
    assert!(one(max)?.is_empty(), "range.max is excluded");
    assert_eq!(one(pred_f(max))?.len(), 1);
    // the empty pixels are 0, whatever the buffer held, and bad points drop
    let bad = [
        [f32::NAN, 0.0, 0.0],
        [f32::INFINITY, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [5.0, 0.0, 9.0],
        [5.0, 0.0, -9.0],
    ];
    assert!(run(&bad)?.is_empty());
    // the nearest return of a pixel wins, in either order
    let a = run(&[[3.0, 0.0, 0.0], [2.0, 0.0, 0.0], [9.0, 0.0, 0.0]])?;
    let b = run(&[[9.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])?;
    assert_eq!(a, b);
    assert_eq!(
        a.iter().map(|p| p.1).collect::<Vec<_>>(),
        [2.0f32.to_bits()]
    );
    // a size mismatch and a metre scale are rejected before any write
    let mut wrong = DepthImage::from_size_val(size(8, 8), 7.0f32, 1.0)?;
    let e = sl
        .range_image_from_points_into(&[[3.0, 0.0, 0.0]], &mut wrong)
        .unwrap_err();
    assert!(
        matches!(e, TsdfError::SizeMismatch { expected, got } if expected == sl.size() && got == size(8, 8))
    );
    let mut scaled = DepthImage::from_size_val(sl.size(), 7.0f32, 1000.0)?;
    let e = sl
        .range_image_from_points_into(&[[3.0, 0.0, 0.0]], &mut scaled)
        .unwrap_err();
    assert!(matches!(
        e,
        TsdfError::InvalidParameter { field: "meter", .. }
    ));
    assert!(scaled.image().as_slice().iter().all(|&v| v == 7.0));
    Ok(())
}

fn pred_f(x: f32) -> f32 {
    f32::from_bits(x.to_bits() - 1)
}

#[test]
fn tsdf_prepare_derives_rows_from_pose() -> R {
    // an axis permutation: the rows applied to a basis vector give the columns of R
    let cam = pinhole(8, 6, 8.0, (0.1, 4.0));
    let d = DepthImage::from_size_val(cam.sensor.size(), 1.0f32, 1.0)?;
    let r = Mat3F64::from_cols_array(&[0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
    let p = Pose3d::new(r, Vec3F64::new(1.0, 2.0, 3.0));
    let o = prepare(&cam.sensor, &p, (&d).into(), None)?;
    let ObsKind::Pinhole(k) = o.kind else {
        unreachable!()
    };
    // R * (1, 0, 0) = (0, 1, 0), plus t
    assert_eq!(
        tc::xform::<StdOps>(
            &k.volume_from_sensor.r,
            &k.volume_from_sensor.t,
            [1.0, 0.0, 0.0]
        ),
        [1.0, 3.0, 3.0]
    );
    // the inverse maps back
    assert_eq!(
        tc::xform::<StdOps>(
            &k.sensor_from_volume.r,
            &k.sensor_from_volume.t,
            [1.0, 3.0, 3.0]
        ),
        [1.0, 0.0, 0.0]
    );
    Ok(())
}

#[test]
fn tsdf_pool_growth_keeps_content() -> R {
    let cam = pinhole(64, 48, 60.0, (0.1, 4.0));
    let scene = [tilted_plane()];
    let poses = [Pose3d::IDENTITY, pose(0.0, 0.2, 0.0, [0.2, 0.0, 0.0])];
    let renders: Vec<_> = poses.iter().map(|p| render(&cam, p, &scene)).collect();
    let mut ws = TsdfWorkspace::default();
    let (mut tiny, mut roomy) = (volume::<8>(1), volume::<8>(4096));
    for (p, d) in poses.iter().zip(&renders) {
        for v in [&mut tiny, &mut roomy] {
            v.integrate(&mut ws, &[obs(&cam, *p, d)])?;
        }
    }
    assert!(tiny.num_blocks() > 8);
    assert_eq!(tiny.content_hash(), roomy.content_hash());
    assert!(tiny.block_keys().eq(roomy.block_keys()));
    // Fixed fails whole, an explicit reserve lifts the cap, and the cap is the u32 index range
    let mut fixed = TsdfVolume::<8>::new(
        TsdfLayout::new(VOXEL, 4.0)?,
        TsdfParams {
            growth: PoolGrowth::Fixed,
            ..params(1)
        },
    )?;
    let e = fixed
        .integrate(&mut ws, &[obs(&cam, poses[0], &renders[0])])
        .unwrap_err();
    assert!(matches!(e, TsdfError::PoolFull { capacity: 1, .. }));
    assert_eq!(fixed.num_blocks(), 0);
    fixed.reserve(4096)?;
    fixed.integrate(&mut ws, &[obs(&cam, poses[0], &renders[0])])?;
    let mut one_step = volume::<8>(4096);
    one_step.integrate(&mut ws, &[obs(&cam, poses[0], &renders[0])])?;
    assert_eq!(fixed.content_hash(), one_step.content_hash());
    let e = fixed.reserve(usize::MAX).unwrap_err();
    assert!(matches!(e, TsdfError::PoolFull { .. }));
    Ok(())
}
