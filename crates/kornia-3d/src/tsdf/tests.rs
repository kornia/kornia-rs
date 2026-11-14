//! TSDF tests: analytic renders (pinhole and spherical), determinism,
//! weight cap, invalid input, marching cubes, incremental meshing, raycast,
//! multi-sensor and multi-agent order, submaps, numerics of the core.

use std::collections::{BTreeSet, HashMap};

use kornia_algebra::{Mat3F64, Vec3F64};
use kornia_image::{Image, ImageSize};
use kornia_oxide_kernels::tsdf_core::{self as tc, F32Ops};

use super::*;
use crate::camera::PinholeCamera;
use crate::depth::DepthImage;
use crate::mesh::TriangleMesh;
use crate::pose::Pose3d;
use crate::registration::RgbdPyramid;

type R = Result<(), Box<dyn std::error::Error>>;

mod neon;
mod pin_core;
mod pin_integrate;
mod pin_ray_mesh;
mod pin_submap;

const VOXEL: f32 = 0.02;

// ------------------------------------------------------------ fixtures ----

/// Analytic surfaces; `sdf > 0` on the sensors' side.
#[derive(Clone, Copy, Debug)]
enum Surface {
    /// `n . p = c`, unit `n` pointing into free space.
    Plane { n: [f64; 3], c: f64 },
    /// Sphere, sensors outside.
    Sphere { centre: [f64; 3], radius: f64 },
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn unit(a: [f64; 3]) -> [f64; 3] {
    let l = dot(a, a).sqrt();
    [a[0] / l, a[1] / l, a[2] / l]
}

fn f64x3(p: [f32; 3]) -> [f64; 3] {
    [p[0] as f64, p[1] as f64, p[2] as f64]
}

impl Surface {
    fn sdf(&self, p: [f64; 3]) -> f64 {
        match *self {
            Self::Plane { n, c } => dot(n, p) - c,
            Self::Sphere { centre, radius } => dot(sub(p, centre), sub(p, centre)).sqrt() - radius,
        }
    }

    fn normal(&self, p: [f64; 3]) -> [f64; 3] {
        match *self {
            Self::Plane { n, .. } => n,
            Self::Sphere { centre, .. } => unit(sub(p, centre)),
        }
    }

    /// Ray parameter of the first hit of `o + t d`, `t > 0`.
    fn hit(&self, o: [f64; 3], d: [f64; 3]) -> Option<f64> {
        match *self {
            Self::Plane { n, c } => {
                let den = dot(n, d);
                let t = (c - dot(n, o)) / den;
                (den != 0.0 && t > 0.0).then_some(t)
            }
            Self::Sphere { centre, radius } => {
                let oc = sub(o, centre);
                let (a, b, cc) = (dot(d, d), 2.0 * dot(oc, d), dot(oc, oc) - radius * radius);
                let disc = b * b - 4.0 * a * cc;
                if disc < 0.0 {
                    return None;
                }
                let t = (-b - disc.sqrt()) / (2.0 * a);
                (t > 0.0).then_some(t)
            }
        }
    }
}

fn mat(rx: f64, ry: f64, rz: f64) -> Mat3F64 {
    let (sx, cx) = rx.sin_cos();
    let (sy, cy) = ry.sin_cos();
    let (sz, cz) = rz.sin_cos();
    let x = Mat3F64::from_cols_array(&[1.0, 0.0, 0.0, 0.0, cx, sx, 0.0, -sx, cx]);
    let y = Mat3F64::from_cols_array(&[cy, 0.0, -sy, 0.0, 1.0, 0.0, sy, 0.0, cy]);
    let z = Mat3F64::from_cols_array(&[cz, sz, 0.0, -sz, cz, 0.0, 0.0, 0.0, 1.0]);
    z * y * x
}

fn pose(rx: f64, ry: f64, rz: f64, t: [f64; 3]) -> Pose3d {
    Pose3d::new(mat(rx, ry, rz), Vec3F64::new(t[0], t[1], t[2]))
}

/// LiDAR mounted looking along the camera z axis (x forward, z up).
fn lidar_mount(t: [f64; 3]) -> Pose3d {
    Pose3d::new(
        Mat3F64::from_cols_array(&[0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0]),
        Vec3F64::new(t[0], t[1], t[2]),
    )
}

/// A sensor plus the f64 ray direction of each pixel (pinhole: `z = 1`;
/// LiDAR: unit).
struct TestSensor {
    sensor: RangeSensor,
    dirs: Vec<[f64; 3]>,
}

fn pinhole(w: usize, h: usize, f: f64, range: (f32, f32)) -> TestSensor {
    let (cx, cy) = ((w as f64 - 1.0) / 2.0, (h as f64 - 1.0) / 2.0);
    let camera = PinholeCamera {
        fx: f,
        fy: f,
        cx,
        cy,
        ..PinholeCamera::IDENTITY
    };
    let size = ImageSize {
        width: w,
        height: h,
    };
    let s = PinholeSensor::new(&camera, size, DepthRange::new(range.0, range.1).unwrap()).unwrap();
    let dirs = (0..w * h)
        .map(|i| [((i % w) as f64 - cx) / f, ((i / w) as f64 - cy) / f, 1.0])
        .collect();
    TestSensor {
        sensor: s.into(),
        dirs,
    }
}

fn lidar(w: usize, h: usize, top: f64, bottom: f64, range: (f32, f32)) -> TestSensor {
    let step = std::f64::consts::TAU / w as f64;
    let az = AzimuthGrid {
        start: 0.0,
        step,
        wrap: true,
    };
    let size = ImageSize {
        width: w,
        height: h,
    };
    let r = DepthRange::new(range.0, range.1).unwrap();
    let s = SphericalLidar::uniform(size, az, top, bottom, r).unwrap();
    let dirs = (0..w * h)
        .map(|i| {
            let el = top + (bottom - top) * ((i / w) as f64) / (h as f64 - 1.0);
            let a = (i % w) as f64 * step;
            [el.cos() * a.cos(), el.cos() * a.sin(), el.sin()]
        })
        .collect();
    TestSensor {
        sensor: s.into(),
        dirs,
    }
}

/// Renders the nearest hit (camera z or range) of every pixel; misses and
/// out-of-range hits are `0`.
fn render(s: &TestSensor, volume_from_sensor: &Pose3d, scene: &[Surface]) -> DepthImage<f32> {
    let size = s.sensor.size();
    let r = volume_from_sensor.rotation;
    let t = volume_from_sensor.translation;
    let o = [t.x, t.y, t.z];
    let (lo, hi) = match &s.sensor {
        RangeSensor::Pinhole(p) => (p.range().min, p.range().max),
        RangeSensor::Spherical(l) => (l.range().min, l.range().max),
    };
    let data = s
        .dirs
        .iter()
        .map(|&d| {
            let dw = r * Vec3F64::new(d[0], d[1], d[2]);
            let dw = [dw.x, dw.y, dw.z];
            let best = scene
                .iter()
                .filter_map(|sf| sf.hit(o, dw))
                .fold(f64::INFINITY, f64::min);
            let v = best as f32;
            if v >= lo && v < hi {
                v
            } else {
                0.0
            }
        })
        .collect();
    DepthImage::new(Image::new(size, data).unwrap(), 1.0).unwrap()
}

fn color_image(size: ImageSize, seed: usize) -> Image<u8, 3> {
    let data = (0..size.width * size.height)
        .flat_map(|i| {
            let (x, y) = (i % size.width, i / size.width);
            [
                ((x * 3 + seed) % 256) as u8,
                ((y * 5 + 2 * seed) % 256) as u8,
                ((x + y + seed) % 256) as u8,
            ]
        })
        .collect();
    Image::new(size, data).unwrap()
}

fn params(capacity: usize) -> TsdfParams {
    TsdfParams {
        block_capacity: capacity,
        ..TsdfParams::default()
    }
}

fn volume<const B: usize>(capacity: usize) -> TsdfVolume<B>
where
    BlockDim<B>: SupportedBlockDim,
{
    TsdfVolume::<B>::new(TsdfLayout::new(VOXEL, 4.0).unwrap(), params(capacity)).unwrap()
}

fn obs_d<'a>(s: &'a TestSensor, p: Pose3d, depth: DepthRef<'a>) -> SensorObservation<'a> {
    SensorObservation {
        sensor: &s.sensor,
        volume_from_sensor: p,
        depth,
        color: None,
    }
}

fn obs<'a>(s: &'a TestSensor, p: Pose3d, d: &'a DepthImage<f32>) -> SensorObservation<'a> {
    obs_d(s, p, d.into())
}

fn tilted_plane() -> Surface {
    let a = 30f64.to_radians();
    let n = [a.sin(), 0.0, -a.cos()];
    Surface::Plane {
        n,
        c: dot(n, [0.0, 0.0, 1.2]),
    }
}

fn mesh_bits(m: &TriangleMesh) -> (Vec<u32>, Vec<u32>, Vec<u8>, Vec<u32>) {
    let f = |v: &Vec<[f32; 3]>| v.iter().flatten().map(|x| x.to_bits()).collect();
    (
        f(&m.vertices),
        f(&m.normals),
        m.colors.iter().flatten().copied().collect(),
        m.triangles.iter().flatten().copied().collect(),
    )
}

/// Every sign-changing edge between valid voxels whose surface normal is
/// within `max_incidence` of the ray from `origin` lies within voxel/4 of
/// the analytic surface. Returns the worst distance.
fn check_zero_crossings<const B: usize>(
    vol: &TsdfVolume<B>,
    surface: &Surface,
    origin: [f64; 3],
    max_incidence_deg: f64,
    min_points: usize,
) -> Result<f64, Box<dyn std::error::Error>>
where
    BlockDim<B>: SupportedBlockDim,
{
    let mut ws = TsdfWorkspace::new(TsdfKernel::Scalar);
    let mut pts = Vec::new();
    vol.extract_point_cloud_into(&mut ws, &MeshParams::default(), &mut pts, None, None)?;
    let cos_max = max_incidence_deg.to_radians().cos();
    let (mut checked, mut worst) = (0usize, 0f64);
    for p in &pts {
        let p = f64x3(*p);
        let ray = unit(sub(p, origin));
        if dot(ray, surface.normal(p)).abs() < cos_max {
            continue;
        }
        worst = worst.max(surface.sdf(p).abs());
        checked += 1;
    }
    let tol = VOXEL as f64 / 4.0;
    assert!(checked >= min_points, "only {checked} crossings checked");
    assert!(
        worst <= tol,
        "zero crossing {worst} m off, > voxel/4 = {tol}"
    );
    eprintln!("zero crossings: {checked} checked, worst {worst:.5} m");
    Ok(worst)
}

// ---------------------------------------------------- analytic renders ----

fn pinhole_plane<const B: usize>() -> R
where
    BlockDim<B>: SupportedBlockDim,
{
    let cam = pinhole(240, 180, 225.0, (0.1, 4.0));
    let surf = tilted_plane();
    let d = render(&cam, &Pose3d::IDENTITY, &[surf]);
    let mut vol = volume::<B>(4096);
    let mut ws = TsdfWorkspace::new(TsdfKernel::Scalar);
    let stats = vol.integrate(&mut ws, &[obs(&cam, Pose3d::IDENTITY, &d)])?;
    assert_eq!(stats.valid_pixels, 240 * 180);
    assert_eq!(stats.new_blocks as usize, vol.num_blocks());
    assert_eq!(stats.touched_blocks as usize, vol.num_blocks());
    check_zero_crossings(&vol, &surf, [0.0; 3], 45.0, 500)?;
    Ok(())
}

#[test]
fn tsdf_pinhole_plane_zero_crossing_b8() -> R {
    pinhole_plane::<8>()
}

#[test]
fn tsdf_pinhole_plane_zero_crossing_b16() -> R {
    pinhole_plane::<16>()
}

fn pinhole_sphere<const B: usize>() -> R
where
    BlockDim<B>: SupportedBlockDim,
{
    let cam = pinhole(240, 180, 225.0, (0.1, 4.0));
    let surf = Surface::Sphere {
        centre: [0.05, -0.03, 1.3],
        radius: 0.35,
    };
    let d = render(&cam, &Pose3d::IDENTITY, &[surf]);
    let mut vol = volume::<B>(4096);
    let mut ws = TsdfWorkspace::new(TsdfKernel::Scalar);
    vol.integrate(&mut ws, &[obs(&cam, Pose3d::IDENTITY, &d)])?;
    check_zero_crossings(&vol, &surf, [0.0; 3], 45.0, 300)?;
    Ok(())
}

#[test]
fn tsdf_pinhole_sphere_zero_crossing_b8() -> R {
    pinhole_sphere::<8>()
}

#[test]
fn tsdf_pinhole_sphere_zero_crossing_b16() -> R {
    pinhole_sphere::<16>()
}

/// Dense LiDAR (2048 x 128 over +-15 deg): at 1.5 m the beams are 4.6 mm
/// (azimuth) and 4.1 mm (elevation) apart, so within 30 deg of normal
/// incidence the nearest-beam error stays well under voxel/4 = 5 mm.
fn lidar_scene<const B: usize>(surf: Surface) -> R
where
    BlockDim<B>: SupportedBlockDim,
{
    let l = lidar(
        2048,
        128,
        15f64.to_radians(),
        -15f64.to_radians(),
        (0.2, 2.5),
    );
    let d = render(&l, &Pose3d::IDENTITY, &[surf]);
    let mut vol = volume::<B>(8192);
    let mut ws = TsdfWorkspace::new(TsdfKernel::Scalar);
    let stats = vol.integrate(&mut ws, &[obs(&l, Pose3d::IDENTITY, &d)])?;
    assert!(stats.updated_voxels > 0);
    check_zero_crossings(&vol, &surf, [0.0; 3], 30.0, 150)?;
    Ok(())
}

#[test]
fn tsdf_lidar_plane_zero_crossing_b8() -> R {
    lidar_scene::<8>(Surface::Plane {
        n: [-1.0, 0.0, 0.0],
        c: -1.5,
    })
}

#[test]
fn tsdf_lidar_plane_zero_crossing_b16() -> R {
    lidar_scene::<16>(Surface::Plane {
        n: [-1.0, 0.0, 0.0],
        c: -1.5,
    })
}

#[test]
fn tsdf_lidar_sphere_zero_crossing_b8() -> R {
    lidar_scene::<8>(Surface::Sphere {
        centre: [1.5, 0.2, 0.05],
        radius: 0.4,
    })
}

// ------------------------------------------- determinism and allocation ----

fn scene_two_views<const B: usize>() -> Result<TsdfVolume<B>, Box<dyn std::error::Error>>
where
    BlockDim<B>: SupportedBlockDim,
{
    let cam = pinhole(160, 120, 150.0, (0.1, 4.0));
    let scene = [
        tilted_plane(),
        Surface::Sphere {
            centre: [0.1, 0.0, 1.0],
            radius: 0.2,
        },
    ];
    let mut vol = volume::<B>(4096);
    let mut ws = TsdfWorkspace::new(TsdfKernel::Scalar);
    for k in 0..2 {
        let f = k as f64;
        let p = pose(0.02 * f, -0.05 * f, 0.0, [0.05 * f, 0.0, 0.0]);
        let d = render(&cam, &p, &scene);
        let c = color_image(cam.sensor.size(), k);
        let o = SensorObservation {
            color: Some(&c),
            ..obs(&cam, p, &d)
        };
        vol.integrate(&mut ws, &[o])?;
    }
    Ok(vol)
}

fn determinism<const B: usize>() -> R
where
    BlockDim<B>: SupportedBlockDim,
{
    let a = scene_two_views::<B>()?;
    let b = scene_two_views::<B>()?;
    assert_eq!(a.content_hash(), b.content_hash());
    assert!(a.block_keys().eq(b.block_keys()));
    let mut ws = TsdfWorkspace::default();
    let (mut ma, mut mb) = (TriangleMesh::new(), TriangleMesh::new());
    let mp = MeshParams {
        colors: true,
        ..MeshParams::default()
    };
    a.extract_mesh_into(&mut ws, &mp, &mut ma)?;
    b.extract_mesh_into(&mut ws, &mp, &mut mb)?;
    assert!(!ma.triangles.is_empty());
    assert_eq!(mesh_bits(&ma), mesh_bits(&mb));
    Ok(())
}

#[test]
fn tsdf_bitwise_determinism_two_runs_b8() -> R {
    determinism::<8>()
}

#[test]
fn tsdf_bitwise_determinism_two_runs_b16() -> R {
    determinism::<16>()
}

#[test]
fn tsdf_pool_index_ascending() -> R {
    let cam = pinhole(80, 60, 75.0, (0.1, 4.0));
    let scene = [tilted_plane()];
    let mut vol = volume::<8>(4096);
    let mut ws = TsdfWorkspace::default();
    let d = render(&cam, &Pose3d::IDENTITY, &scene);
    vol.integrate(&mut ws, &[obs(&cam, Pose3d::IDENTITY, &d)])?;
    let n1 = vol.num_blocks();
    // first step: pool index = rank of the key
    assert!(vol
        .index
        .sorted
        .iter()
        .enumerate()
        .all(|(i, &(_, p))| p as usize == i));
    let p = pose(0.0, 0.3, 0.0, [0.3, 0.0, 0.0]);
    let d = render(&cam, &p, &scene);
    vol.integrate(&mut ws, &[obs(&cam, p, &d)])?;
    let n2 = vol.num_blocks();
    assert!(n2 > n1);
    let new: Vec<u64> = (n1..n2).map(|p| vol.index.pool_keys[p]).collect();
    assert!(new.windows(2).all(|w| w[0] < w[1]));
    assert!(vol.index.sorted.windows(2).all(|w| w[0].0 < w[1].0));
    for &(k, p) in &vol.index.sorted {
        assert_eq!(vol.index.pool_keys[p as usize], k);
    }
    Ok(())
}

#[test]
fn tsdf_alloc_lookup_first_equals_full_sort() -> R {
    let cam = pinhole(80, 60, 75.0, (0.1, 4.0));
    let scene = [tilted_plane()];
    let mut vol = volume::<8>(4096);
    let mut ws = TsdfWorkspace::new(TsdfKernel::Scalar);
    let mut want = BTreeSet::new();
    let grid = vol.layout().grid(3);
    let RangeSensor::Pinhole(ps) = &cam.sensor else {
        unreachable!()
    };
    for k in 0..2 {
        let f = k as f64;
        let p = pose(0.0, 0.1 * f, 0.0, [0.1 * f, 0.0, 0.0]);
        let d = render(&cam, &p, &scene);
        vol.integrate(&mut ws, &[obs(&cam, p, &d)])?;
        // brute force: every key of every valid pixel's box
        let vfs = rigid_from_pose(&p)?;
        let pp = ps.params(vfs, vfs, 1.0);
        for (i, &z) in d.image().as_slice().iter().enumerate() {
            if z == 0.0 {
                continue;
            }
            let s = tc::pinhole_point(&pp, (i % 80) as u32, (i / 80) as u32, z);
            let pv = tc::xform::<StdOps>(&vfs.r, &vfs.t, s);
            let (lo, hi) = tc::touch_box(pv, &grid).unwrap();
            for x in lo[0]..=hi[0] {
                for y in lo[1]..=hi[1] {
                    for z in lo[2]..=hi[2] {
                        want.insert(tc::pack_block_key([x, y, z]));
                    }
                }
            }
        }
        let got: BTreeSet<u64> = vol.block_keys().map(|k| k.bits()).collect();
        assert_eq!(got, want);
    }
    Ok(())
}

#[test]
fn tsdf_weight_cap_running_mean() -> R {
    let cam = pinhole(64, 48, 60.0, (0.1, 4.0));
    let size = cam.sensor.size();
    let mut vol = TsdfVolume::<8>::new(
        TsdfLayout::new(VOXEL, 4.0)?,
        TsdfParams {
            max_weight: 3,
            ..params(1024)
        },
    )?;
    let mut ws = TsdfWorkspace::new(TsdfKernel::Scalar);
    // voxel on the optical axis, just in front of the planes
    let g = [0, 0, 59];
    let c = vol.layout().centre_from_voxel(g);
    let trunc = vol.layout().truncation();
    let (mut t, mut w) = (0.0f32, 0u16);
    for k in 0..7 {
        let z = if k % 2 == 0 { 1.2f32 } else { 1.23 };
        let d = DepthImage::from_size_val(size, z, 1.0)?;
        vol.integrate(&mut ws, &[obs(&cam, Pose3d::IDENTITY, &d)])?;
        let tsd = tc::truncate(z - c[2], trunc, 1.0 / trunc).unwrap();
        (t, w) = tc::update_tsdf::<StdOps>(t, w, tsd, 3);
        let v = vol.voxel(g).unwrap();
        assert_eq!(v.tsdf.to_bits(), t.to_bits(), "frame {k}");
        assert_eq!(v.weight, w);
    }
    assert_eq!(w, 3);
    let max_w = vol
        .block_keys()
        .flat_map(|k| vol.block(k).unwrap().weight.to_vec())
        .max();
    assert_eq!(max_w, Some(3));
    Ok(())
}

#[test]
fn tsdf_missing_and_invalid_depth_skipped() -> R {
    let cam = pinhole(32, 24, 30.0, (0.3, 3.0));
    let size = cam.sensor.size();
    let mut vol = volume::<8>(1024);
    let mut ws = TsdfWorkspace::new(TsdfKernel::Scalar);
    // all missing: nothing allocated or updated
    let zero = DepthImage::from_size_val(size, 0u16, 1000.0)?;
    let s = vol.integrate(&mut ws, &[obs_d(&cam, Pose3d::IDENTITY, (&zero).into())])?;
    assert_eq!(s, IntegrateStats::default());
    assert_eq!(vol.num_blocks(), 0);
    // u16: zeros and out-of-range samples are skipped
    let raw: Vec<u16> = (0..32 * 24)
        .map(|i| match i % 4 {
            0 => 0,
            1 => 200,  // 0.2 m < range.min
            2 => 3000, // == range.max (exclusive)
            _ => 1500,
        })
        .collect();
    let d = DepthImage::new(Image::new(size, raw)?, 1000.0)?;
    let s = vol.integrate(&mut ws, &[obs_d(&cam, Pose3d::IDENTITY, (&d).into())])?;
    assert_eq!(s.valid_pixels, 32 * 24 / 4);
    // f32: NaN, +-inf, negative, subnormal and zero are skipped
    let bad = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -1.0, 1e-40, 0.0];
    let raw: Vec<f32> = (0..32 * 24)
        .map(|i| if i % 7 == 0 { 1.4 } else { bad[i % 6] })
        .collect();
    let n_valid = (0..32 * 24).filter(|i| i % 7 == 0).count() as u64;
    let d = DepthImage::new(Image::new(size, raw)?, 1.0)?;
    let s = vol.integrate(&mut ws, &[obs(&cam, Pose3d::IDENTITY, &d)])?;
    assert_eq!(s.valid_pixels, n_valid);
    Ok(())
}

#[test]
fn tsdf_nan_inputs_never_stored() -> R {
    let cam = pinhole(48, 36, 45.0, (0.1, 4.0));
    let size = cam.sensor.size();
    let specials = [
        f32::NAN,
        f32::INFINITY,
        -f32::INFINITY,
        -0.0,
        1e-42,
        f32::MAX,
        1.1,
        1.3,
    ];
    let raw: Vec<f32> = (0..48 * 36)
        .map(|i| specials[(i * 7 + i / 48) % specials.len()])
        .collect();
    let d = DepthImage::new(Image::new(size, raw)?, 1.0)?;
    let l = lidar(256, 16, 0.2, -0.2, (0.2, 5.0));
    let RangeSensor::Spherical(sl) = &l.sensor else {
        unreachable!()
    };
    let mut pts: Vec<[f32; 3]> = (0..2000)
        .map(|i| {
            let a = i as f32 * 0.01;
            [2.0 * a.cos(), 2.0 * a.sin(), 0.1 * (i % 5) as f32 - 0.2]
        })
        .collect();
    pts.extend([
        [f32::NAN, 0.0, 0.0],
        [f32::INFINITY, 1.0, 0.0],
        [1.0, f32::NAN, f32::NAN],
    ]);
    let mut range = DepthImage::from_size_val(sl.size(), 0.0, 1.0)?;
    sl.range_image_from_points_into(&pts, &mut range)?;
    assert!(range.image().as_slice().iter().all(|v| v.is_finite()));
    assert!(range.image().as_slice().iter().any(|&v| v > 0.0));
    let mut vol = volume::<8>(4096);
    let mut ws = TsdfWorkspace::default();
    let o = [
        obs(&cam, Pose3d::IDENTITY, &d),
        obs(&l, Pose3d::IDENTITY, &range),
    ];
    vol.integrate(&mut ws, &o)?;
    assert!(vol.num_blocks() > 0);
    for k in vol.block_keys() {
        let b = vol.block(k).unwrap();
        assert!(b
            .tsdf
            .iter()
            .all(|t| t.is_finite() && (-1.0..=1.0).contains(t)));
    }
    Ok(())
}

#[test]
fn tsdf_pool_full_leaves_volume_unchanged() -> R {
    let cam = pinhole(64, 48, 60.0, (0.1, 4.0));
    let d = render(&cam, &Pose3d::IDENTITY, &[tilted_plane()]);
    let mut vol = TsdfVolume::<8>::new(
        TsdfLayout::new(VOXEL, 4.0)?,
        TsdfParams {
            growth: PoolGrowth::Fixed,
            ..params(8)
        },
    )?;
    let before = vol.content_hash();
    let mut ws = TsdfWorkspace::default();
    let r = vol.integrate(&mut ws, &[obs(&cam, Pose3d::IDENTITY, &d)]);
    assert!(matches!(r, Err(TsdfError::PoolFull { capacity: 8, .. })));
    assert_eq!(vol.num_blocks(), 0);
    assert_eq!(vol.content_hash(), before);
    // Geometric growth integrates the same frame
    let mut vol = TsdfVolume::<8>::new(TsdfLayout::new(VOXEL, 4.0)?, params(8))?;
    vol.integrate(&mut ws, &[obs(&cam, Pose3d::IDENTITY, &d)])?;
    assert!(vol.num_blocks() > 8);
    Ok(())
}

// ------------------------------------------------------ marching cubes ----

type Filled<const B: usize> = (TsdfVolume<B>, [f64; 3]);

fn filled_sphere<const B: usize>(r: f64) -> Result<Filled<B>, Box<dyn std::error::Error>>
where
    BlockDim<B>: SupportedBlockDim,
{
    let centre = [0.0123, -0.0071, 0.0049];
    let mut vol = volume::<B>(4096);
    let trunc = vol.layout().truncation() as f64;
    let bsz = B as f64 * VOXEL as f64;
    let reach = trunc + bsz * 3f64.sqrt() + 2.0 * VOXEL as f64;
    let n = ((r + reach) / bsz).ceil() as i32 + 1;
    let mut keys = Vec::new();
    for x in -n..=n {
        for y in -n..=n {
            for z in -n..=n {
                let c = [
                    (x as f64 + 0.5) * bsz,
                    (y as f64 + 0.5) * bsz,
                    (z as f64 + 0.5) * bsz,
                ];
                let d = (dot(sub(c, centre), sub(c, centre)).sqrt() - r).abs();
                if d < reach {
                    keys.push(BlockKey::new(x, y, z)?.bits());
                }
            }
        }
    }
    vol.fill_for_test(&mut keys, |c| {
        let p = f64x3(c);
        let sd = dot(sub(p, centre), sub(p, centre)).sqrt() - r;
        Some((sd / trunc).clamp(-1.0, 1.0) as f32)
    })?;
    Ok((vol, centre))
}

fn mc_sphere_closed<const B: usize>() -> R
where
    BlockDim<B>: SupportedBlockDim,
{
    let r = 0.3;
    let (vol, centre) = filled_sphere::<B>(r)?;
    let mut ws = TsdfWorkspace::default();
    let mut mesh = TriangleMesh::new();
    vol.extract_mesh_into(&mut ws, &MeshParams::default(), &mut mesh)?;
    assert_eq!(mesh.normals.len(), mesh.vertices.len());
    // weld the per-block duplicates by exact position bits
    let mut ids: HashMap<[u32; 3], u32> = HashMap::new();
    let weld: Vec<u32> = mesh
        .vertices
        .iter()
        .map(|v| {
            let key = [v[0].to_bits(), v[1].to_bits(), v[2].to_bits()];
            let n = ids.len() as u32;
            *ids.entry(key).or_insert(n)
        })
        .collect();
    let mut edges: HashMap<(u32, u32), u32> = HashMap::new();
    let (mut area, mut outward, mut faces) = (0f64, 0usize, 0usize);
    for t in &mesh.triangles {
        let w = [
            weld[t[0] as usize],
            weld[t[1] as usize],
            weld[t[2] as usize],
        ];
        assert!(
            w[0] != w[1] && w[1] != w[2] && w[0] != w[2],
            "degenerate triangle"
        );
        for (a, b) in [(w[0], w[1]), (w[1], w[2]), (w[2], w[0])] {
            *edges.entry((a.min(b), a.max(b))).or_default() += 1;
        }
        let v = |i: usize| f64x3(mesh.vertices[t[i] as usize]);
        let (e1, e2) = (sub(v(1), v(0)), sub(v(2), v(0)));
        let cr = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ];
        area += 0.5 * dot(cr, cr).sqrt();
        // counter-clockwise seen from outside
        outward += (dot(cr, sub(v(0), centre)) > 0.0) as usize;
        faces += 1;
    }
    assert!(edges.values().all(|&c| c == 2), "mesh not closed");
    let euler = ids.len() as i64 - edges.len() as i64 + faces as i64;
    assert_eq!(euler, 2, "sphere must have genus 0");
    assert_eq!(
        outward, faces,
        "winding is counter-clockwise from free space"
    );
    let want = 4.0 * std::f64::consts::PI * r * r;
    let rel = (area - want).abs() / want;
    assert!(rel < 0.02, "area {area} vs {want}");
    eprintln!(
        "MC sphere B={B}: {faces} faces, area error {:.3}%",
        100.0 * rel
    );
    // vertex normals point outward
    for (v, n) in mesh.vertices.iter().zip(&mesh.normals) {
        let radial = unit(sub(f64x3(*v), centre));
        assert!(dot(f64x3(*n), radial) > 0.9);
    }
    Ok(())
}

#[test]
fn tsdf_mc_sphere_closed_area_b8() -> R {
    mc_sphere_closed::<8>()
}

#[test]
fn tsdf_mc_sphere_closed_area_b16() -> R {
    mc_sphere_closed::<16>()
}

#[test]
fn tsdf_mc_tables_consistent() {
    for c in 0..256 {
        let used = tc::TRI_TABLE[c]
            .iter()
            .filter(|&&e| e >= 0)
            .fold(0u16, |m, &e| m | (1 << e));
        assert_eq!(used, tc::EDGE_TABLE[c], "case {c}");
        let n = tc::TRI_TABLE[c].iter().take_while(|&&e| e >= 0).count();
        assert_eq!(n % 3, 0);
        assert!(tc::TRI_TABLE[c][n..].iter().all(|&e| e == -1));
        // complementary configurations cross the same edges
        assert_eq!(tc::EDGE_TABLE[c], tc::EDGE_TABLE[255 - c]);
    }
    // edge owners are the lower end along the axis
    for k in 0..12 {
        assert_eq!(tc::EDGE_OWNER[k][tc::EDGE_AXIS[k] as usize], 0);
    }
}

/// FNV-1a 64 of the tables (EDGE_TABLE as u16 LE, then TRI_TABLE bytes),
/// pinned so a copy or an edit cannot drift silently.
#[test]
fn tsdf_mc_tables_fingerprint() {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    let mut eat = |b: &[u8]| {
        for &x in b {
            h = (h ^ u64::from(x)).wrapping_mul(0x0100_0000_01b3);
        }
    };
    for e in tc::EDGE_TABLE {
        eat(&e.to_le_bytes());
    }
    for row in tc::TRI_TABLE {
        for v in row {
            eat(&v.to_le_bytes());
        }
    }
    assert_eq!(h, 0x5a98_ec56_0bb1_4b45);
}

fn incremental_equals_full(mp: MeshParams) -> R {
    let cam = pinhole(120, 90, 110.0, (0.1, 4.0));
    let scene = [
        tilted_plane(),
        Surface::Sphere {
            centre: [-0.1, 0.05, 0.9],
            radius: 0.15,
        },
    ];
    let mut vol = volume::<8>(4096);
    let mut ws = TsdfWorkspace::default();
    let mut cache = BlockMeshCache::new();
    let (mut inc, mut full) = (TriangleMesh::new(), TriangleMesh::new());
    for k in 0..3 {
        let f = k as f64;
        let p = pose(0.03 * f, 0.04 * f, 0.0, [0.03 * f, 0.0, -0.02 * f]);
        let d = render(&cam, &p, &scene);
        let c = color_image(cam.sensor.size(), 7 * k);
        let o = SensorObservation {
            color: Some(&c),
            ..obs(&cam, p, &d)
        };
        vol.integrate(&mut ws, &[o])?;
        vol.update_mesh_cache(&mut ws, &mp, &mut cache)?;
        cache.write_into(&mut inc)?;
        vol.extract_mesh_into(&mut ws, &mp, &mut full)?;
        // threshold 2 needs a second view
        assert!(k + 1 < mp.weight_threshold as usize || !full.triangles.is_empty());
        assert_eq!(mesh_bits(&inc), mesh_bits(&full), "frame {k}");
    }
    // a small far object touches few blocks: few are re-meshed
    let small = pinhole(8, 6, 8.0, (0.1, 4.0));
    let d = DepthImage::from_size_val(small.sensor.size(), 3.0f32, 1.0)?;
    vol.integrate(&mut ws, &[obs(&small, Pose3d::IDENTITY, &d)])?;
    let st = vol.update_mesh_cache(&mut ws, &mp, &mut cache)?;
    assert!((st.remeshed_blocks as usize) < vol.num_blocks() / 2);
    cache.write_into(&mut inc)?;
    vol.extract_mesh_into(&mut ws, &mp, &mut full)?;
    assert_eq!(mesh_bits(&inc), mesh_bits(&full));
    assert_eq!(st.vertices as usize, full.vertices.len());
    Ok(())
}

#[test]
fn tsdf_mesh_incremental_equals_full() -> R {
    incremental_equals_full(MeshParams {
        weight_threshold: 1,
        normals: true,
        colors: true,
    })
}

#[test]
fn tsdf_mesh_incremental_equals_full_no_normals() -> R {
    incremental_equals_full(MeshParams {
        weight_threshold: 2,
        normals: false,
        colors: false,
    })
}

#[test]
fn tsdf_point_cloud_matches_mesh_vertices() -> R {
    let vol = scene_two_views::<8>()?;
    let mut ws = TsdfWorkspace::default();
    let mut mesh = TriangleMesh::new();
    vol.extract_mesh_into(&mut ws, &MeshParams::default(), &mut mesh)?;
    let mut pts = Vec::new();
    let mut nrm = Vec::new();
    vol.extract_point_cloud_into(
        &mut ws,
        &MeshParams::default(),
        &mut pts,
        None,
        Some(&mut nrm),
    )?;
    let bits = |v: &[f32; 3]| [v[0].to_bits(), v[1].to_bits(), v[2].to_bits()];
    let mesh_set: BTreeSet<[u32; 3]> = mesh.vertices.iter().map(bits).collect();
    let pt_set: BTreeSet<[u32; 3]> = pts.iter().map(bits).collect();
    assert_eq!(pt_set.len(), pts.len(), "no duplicate points");
    assert_eq!(nrm.len(), pts.len());
    // every mesh vertex is a zero crossing of an owned edge
    assert!(mesh_set.is_subset(&pt_set));
    Ok(())
}

// -------------------------------------------------------------- raycast ----

#[test]
fn tsdf_raycast_depth_matches_render() -> R {
    let cam = pinhole(160, 120, 150.0, (0.1, 4.0));
    let surf = tilted_plane();
    let p = pose(0.0, 0.05, 0.0, [0.02, 0.0, 0.0]);
    let d = render(&cam, &p, &[surf]);
    let mut vol = volume::<8>(4096);
    let mut ws = TsdfWorkspace::default();
    for _ in 0..3 {
        vol.integrate(&mut ws, &[obs(&cam, p, &d)])?;
    }
    let RangeSensor::Pinhole(ps) = &cam.sensor else {
        unreachable!()
    };
    let mut maps = RaycastMaps::new(cam.sensor.size(), false)?;
    let rp = RaycastParams::new(vol.layout(), 0.1, 4.0)?;
    vol.raycast_into(&mut ws, ps.camera(), &p, &rp, &mut maps)?;
    let n_cam = {
        let n = surf.normal([0.0; 3]);
        let rt = p.rotation.transpose() * Vec3F64::new(n[0], n[1], n[2]);
        [rt.x, rt.y, rt.z]
    };
    let (mut hits, mut close, mut good_n, mut valid_n, mut valid) = (0, 0, 0, 0, 0);
    let tol = VOXEL as f64;
    let depth = maps.depth().as_slice();
    for (i, (&z, &zt)) in depth.iter().zip(d.image().as_slice()).enumerate() {
        if zt == 0.0 {
            continue;
        }
        valid += 1;
        if z == 0.0 {
            // only frustum-edge rays may miss: their voxels behind the
            // surface project outside the image and are never integrated
            let (px, py) = (i % 160, i / 160);
            assert!(
                px <= 1 || px >= 158 || py <= 1 || py >= 118,
                "interior miss at ({px}, {py})"
            );
            continue;
        }
        hits += 1;
        let e = (z as f64 - zt as f64).abs();
        assert!(e <= tol, "pixel {i}: raycast {z} vs {zt}");
        close += (e <= tol / 2.0) as usize;
        let v = &maps.vertices().as_slice()[3 * i..3 * i + 3];
        assert_eq!(v[2], z);
        let nrm = &maps.normals().as_slice()[3 * i..3 * i + 3];
        if nrm != [0.0; 3] {
            valid_n += 1;
            let nn = [nrm[0] as f64, nrm[1] as f64, nrm[2] as f64];
            good_n += (dot(nn, n_cam) > 0.95) as usize;
            assert!(dot(nn, [v[0] as f64, v[1] as f64, v[2] as f64]) <= 0.0);
        }
    }
    eprintln!("raycast: {hits}/{valid} hits, {close} within voxel/2, {good_n}/{valid_n} normals");
    assert!(hits * 4 >= valid * 3, "{hits} hits of {valid}");
    // samples are the containing voxel (no trilinear interpolation), so a
    // hit is up to ~half a voxel off along the normal
    assert!(close * 100 >= hits * 85, "{close} of {hits} within voxel/2");
    assert!(good_n * 10 >= valid_n * 9 && valid_n * 2 >= hits);
    Ok(())
}

#[test]
fn tsdf_raycast_rgbd_pyramid_layout() -> R {
    let cam = pinhole(64, 48, 60.0, (0.1, 4.0));
    let d = render(&cam, &Pose3d::IDENTITY, &[tilted_plane()]);
    let mut vol = volume::<8>(2048);
    let mut ws = TsdfWorkspace::default();
    vol.integrate(&mut ws, &[obs(&cam, Pose3d::IDENTITY, &d)])?;
    let RangeSensor::Pinhole(ps) = &cam.sensor else {
        unreachable!()
    };
    let mut pyr = RgbdPyramid::zeroed(ps.camera(), ps.size(), 3)?;
    let reference = RgbdPyramid::from_depth_m(d.image(), ps.camera(), 3)?;
    let rp = RaycastParams::new(vol.layout(), 0.1, 4.0)?;
    vol.raycast_pyramid_into(&mut ws, &Pose3d::IDENTITY, &rp, &mut pyr)?;
    let mut maps = RaycastMaps::new(ps.size(), false)?;
    vol.raycast_into(&mut ws, ps.camera(), &Pose3d::IDENTITY, &rp, &mut maps)?;
    for (a, b) in pyr.levels().iter().zip(reference.levels()) {
        assert_eq!(a.size(), b.size());
        assert_eq!(a.camera().fx, b.camera().fx);
        assert_eq!(a.camera().cx, b.camera().cx);
        assert!(a.depth_sigmas().is_none());
    }
    let l0 = &pyr.levels()[0];
    assert_eq!(l0.vertices().as_slice(), maps.vertices().as_slice());
    assert_eq!(l0.normals().as_slice(), maps.normals().as_slice());
    assert!(l0.vertices().as_slice().iter().any(|&v| v != 0.0));
    Ok(())
}

// ------------------------------------------------- multi-sensor order ----

#[test]
fn tsdf_multi_sensor_step_equals_sequential() -> R {
    let cam_a = pinhole(96, 72, 90.0, (0.1, 4.0));
    let cam_b = pinhole(80, 60, 70.0, (0.1, 4.0));
    let lid = lidar(512, 32, 0.3, -0.3, (0.2, 4.0));
    let scene = [
        tilted_plane(),
        Surface::Sphere {
            centre: [0.0, 0.1, 1.0],
            radius: 0.2,
        },
    ];
    let pa = Pose3d::IDENTITY;
    let pb = pose(0.0, 0.15, 0.0, [-0.1, 0.0, 0.05]);
    let pl = lidar_mount([0.0, 0.05, 0.0]);
    let da = render(&cam_a, &pa, &scene);
    let db = render(&cam_b, &pb, &scene);
    let dl = render(&lid, &pl, &scene);
    let ca = color_image(cam_a.sensor.size(), 1);
    let all = [
        SensorObservation {
            color: Some(&ca),
            ..obs(&cam_a, pa, &da)
        },
        obs(&lid, pl, &dl),
        obs(&cam_b, pb, &db),
    ];
    let mut ws = TsdfWorkspace::default();
    let mut fused = volume::<8>(8192);
    let s = fused.integrate(&mut ws, &all)?;
    let mut seq = volume::<8>(8192);
    let mut total = IntegrateStats::default();
    for o in &all {
        total.accumulate(&seq.integrate(&mut ws, std::slice::from_ref(o))?);
    }
    assert!(fused.num_blocks() > 0);
    assert_eq!(fused.num_blocks(), seq.num_blocks());
    assert_eq!(fused.content_hash(), seq.content_hash());
    assert_eq!(s.updated_voxels, total.updated_voxels);
    assert_eq!(s.valid_pixels, total.valid_pixels);
    assert_eq!(s.new_blocks as usize, fused.num_blocks());
    Ok(())
}

// ------------------------------------------------- submaps and agents ----

fn submap_params(color: bool) -> SubmapParams {
    SubmapParams {
        layout: TsdfLayout::new(VOXEL, 4.0).unwrap(),
        volume: TsdfParams {
            color,
            ..params(2048)
        },
        arena_blocks: 16384,
        max_submaps: 16,
        max_resident_bytes: None,
    }
}

type Rig = Vec<(SensorId, TestSensor, Pose3d)>;

struct World {
    rig_a: Rig,
    rig_b: Rig,
    scene: Vec<Surface>,
}

fn world() -> World {
    World {
        rig_a: vec![
            (
                SensorId(3),
                lidar(256, 16, 0.25, -0.25, (0.2, 4.0)),
                lidar_mount([0.0, -0.05, 0.0]),
            ),
            (
                SensorId(1),
                pinhole(64, 48, 60.0, (0.1, 4.0)),
                Pose3d::IDENTITY,
            ),
        ],
        rig_b: vec![(
            SensorId(0),
            pinhole(48, 36, 45.0, (0.1, 4.0)),
            pose(0.0, -0.1, 0.0, [0.0; 3]),
        )],
        scene: vec![
            tilted_plane(),
            Surface::Sphere {
                centre: [0.2, 0.0, 1.0],
                radius: 0.25,
            },
        ],
    }
}

fn rig(sensors: &Rig) -> SensorRig {
    SensorRig::new(
        sensors
            .iter()
            .map(|(id, s, p)| RigSensor {
                id: *id,
                sensor: s.sensor.clone(),
                body_from_sensor: *p,
            })
            .collect(),
    )
    .unwrap()
}

/// Rendered frames of one agent step, owned.
struct StepData {
    agent: AgentId,
    stamp: u64,
    body: Pose3d,
    depth: Vec<(SensorId, DepthImage<f32>)>,
}

/// Agent 0: stamps 10..50 (indices 0..5); agent 1: 15, 20, 35, 45, 55
/// (indices 5..10); stamp 20 is shared.
fn steps(w: &World) -> Vec<StepData> {
    let mut out = Vec::new();
    let a_stamps = [10u64, 20, 30, 40, 50];
    let b_stamps = [15u64, 20, 35, 45, 55];
    for (agent, stamps, rig) in [
        (AgentId(0), a_stamps, &w.rig_a),
        (AgentId(1), b_stamps, &w.rig_b),
    ] {
        for (k, &stamp) in stamps.iter().enumerate() {
            let f = k as f64;
            let off = if agent.0 == 0 { 0.0 } else { 0.15 };
            let body = pose(
                0.0,
                0.03 * f - off,
                0.0,
                [0.02 * f + off, 0.01 * f, -0.01 * f],
            );
            let depth = rig
                .iter()
                .map(|(id, s, mount)| (*id, render(s, &body.compose(mount), &w.scene)))
                .collect();
            out.push(StepData {
                agent,
                stamp,
                body,
                depth,
            });
        }
    }
    out
}

fn new_map(w: &World) -> SubmapTsdf<8> {
    let mut map = SubmapTsdf::<8>::new(submap_params(false)).unwrap();
    let spawn = SpawnPolicy {
        max_steps: 2,
        ..SpawnPolicy::default()
    };
    map.add_agent(AgentId(0), rig(&w.rig_a), spawn).unwrap();
    map.add_agent(AgentId(1), rig(&w.rig_b), spawn).unwrap();
    map
}

fn frames(s: &StepData) -> Vec<SensorFrame<'_>> {
    // reversed: the map applies frames by ascending sensor id anyway
    s.depth
        .iter()
        .rev()
        .map(|(id, d)| SensorFrame {
            sensor: *id,
            depth: d.into(),
            color: None,
        })
        .collect()
}

fn agent_step<'a>(d: &StepData, f: &'a [SensorFrame<'a>]) -> AgentStep<'a> {
    AgentStep {
        agent: d.agent,
        stamp: d.stamp,
        world_from_body: d.body,
        frames: f,
    }
}

fn snapshot(map: &SubmapTsdf<8>) -> Vec<(SubmapId, u64, Pose3d)> {
    map.submap_ids()
        .map(|id| {
            (
                id,
                map.content_hash(id).unwrap(),
                map.submap_pose(id).unwrap(),
            )
        })
        .collect()
}

#[test]
fn tsdf_multi_agent_interleave_bitwise() -> R {
    let w = world();
    let data = steps(&w);
    let frames: Vec<Vec<SensorFrame<'_>>> = data.iter().map(frames).collect();
    let step = |i: usize| agent_step(&data[i], &frames[i]);
    let mut ws = TsdfWorkspace::default();
    // serial replay in (stamp, agent) order
    let mut order: Vec<usize> = (0..data.len()).collect();
    order.sort_by_key(|&i| (data[i].stamp, data[i].agent));
    let mut serial = new_map(&w);
    let mut spawned = 0;
    for &i in &order {
        spawned += serial.integrate(&mut ws, &step(i))?.spawned as usize;
    }
    assert_eq!(spawned, 6, "max_steps 2: 3 submaps per agent");
    // shuffled arrival in three batches, each agent's steps in order
    let batches: [&[usize]; 3] = [&[6, 1, 0, 5], &[8, 3, 2, 7], &[9, 4]];
    let mut shuffled = new_map(&w);
    let mut ws2 = TsdfWorkspace::default();
    let mut n = 0;
    for b in batches {
        let s: Vec<AgentStep<'_>> = b.iter().map(|&i| step(i)).collect();
        n += shuffled.integrate_batch(&mut ws2, &s)?.steps;
    }
    assert_eq!(n, 10);
    // agent 1 runs ahead of agent 0 (legal: stamps only increase per
    // agent), so the global order, arena pages and slots all differ
    let mut ahead = new_map(&w);
    for i in (5..10).chain(0..5) {
        ahead.integrate(&mut ws2, &step(i))?;
    }
    let (a, b) = (snapshot(&serial), snapshot(&shuffled));
    assert_eq!(a.len(), 6);
    assert_eq!(a, b);
    assert_eq!(a, snapshot(&ahead));
    let mut mesh_a = TriangleMesh::new();
    let mut mesh_b = TriangleMesh::new();
    let mp = MeshParams::default();
    serial.extract_mesh_into(&mut ws, SubmapFilter::All, &mp, &mut mesh_a)?;
    shuffled.extract_mesh_into(&mut ws, SubmapFilter::All, &mp, &mut mesh_b)?;
    assert!(!mesh_a.triangles.is_empty());
    assert_eq!(mesh_bits(&mesh_a), mesh_bits(&mesh_b));
    ahead.extract_mesh_into(&mut ws, SubmapFilter::All, &mp, &mut mesh_b)?;
    assert_eq!(mesh_bits(&mesh_a), mesh_bits(&mesh_b));
    Ok(())
}

#[test]
fn tsdf_submap_pose_update_moves_only_target_agent() -> R {
    let w = world();
    let data = steps(&w);
    let frames: Vec<Vec<SensorFrame<'_>>> = data.iter().map(frames).collect();
    let all: Vec<AgentStep<'_>> = (0..data.len())
        .map(|i| agent_step(&data[i], &frames[i]))
        .collect();
    let mut map = new_map(&w);
    let mut ws = TsdfWorkspace::default();
    map.integrate_batch(&mut ws, &all)?;
    let mp = MeshParams::default();
    let mut cache = SubmapMeshCache::new();
    map.update_mesh_cache(&mut ws, &mp, &mut cache)?;
    let mesh = |map: &SubmapTsdf<8>, ws: &mut TsdfWorkspace, a: u32| {
        let mut m = TriangleMesh::new();
        map.extract_mesh_into(ws, SubmapFilter::Agents(&[AgentId(a)]), &mp, &mut m)
            .unwrap();
        m
    };
    let (a0, b0) = (mesh(&map, &mut ws, 0), mesh(&map, &mut ws, 1));
    assert!(!a0.triangles.is_empty() && !b0.triangles.is_empty());
    let hashes: Vec<_> = snapshot(&map).iter().map(|s| s.1).collect();
    // move agent 1's submaps by +1 m in x
    let moved: Vec<SubmapPose> = map
        .submap_ids()
        .filter(|id| id.agent == AgentId(1))
        .map(|id| {
            let mut p = map.submap_pose(id).unwrap();
            p.translation.x += 1.0;
            SubmapPose {
                id,
                world_from_submap: p,
            }
        })
        .collect();
    assert_eq!(moved.len(), 3);
    // an unknown id rejects the whole update
    let mut bad = moved.clone();
    bad.push(SubmapPose {
        id: SubmapId {
            agent: AgentId(1),
            seq: 99,
        },
        world_from_submap: Pose3d::IDENTITY,
    });
    assert!(matches!(
        map.set_submap_poses(&bad),
        Err(TsdfError::UnknownSubmap(_))
    ));
    assert_eq!(mesh_bits(&mesh(&map, &mut ws, 1)), mesh_bits(&b0));
    map.set_submap_poses(&moved)?;
    let (a1, b1) = (mesh(&map, &mut ws, 0), mesh(&map, &mut ws, 1));
    assert_eq!(mesh_bits(&a0), mesh_bits(&a1));
    assert_eq!(b0.vertices.len(), b1.vertices.len());
    assert!(b0.vertices.iter().zip(&b1.vertices).all(|(p, q)| {
        (q[0] - p[0] - 1.0).abs() < 1e-5 && (q[1] - p[1]).abs() < 1e-5 && (q[2] - p[2]).abs() < 1e-5
    }));
    // poses never touch the voxels
    let after: Vec<_> = snapshot(&map).iter().map(|s| s.1).collect();
    assert_eq!(hashes, after);
    // the cached meshes follow the new poses without re-meshing
    let st = map.update_mesh_cache(&mut ws, &mp, &mut cache)?;
    assert_eq!(st.remeshed_blocks, 0);
    let mut cached = TriangleMesh::new();
    cache.write_into(&map, SubmapFilter::Agents(&[AgentId(1)]), &mut cached)?;
    assert_eq!(mesh_bits(&cached), mesh_bits(&b1));
    Ok(())
}

#[test]
fn tsdf_submap_cache_with_capacity_matches_new() -> R {
    let w = world();
    let data = steps(&w);
    let frames: Vec<Vec<SensorFrame<'_>>> = data.iter().map(frames).collect();
    let all: Vec<AgentStep<'_>> = (0..data.len())
        .map(|i| agent_step(&data[i], &frames[i]))
        .collect();
    let mut map = new_map(&w);
    let mut ws = TsdfWorkspace::default();
    map.integrate_batch(&mut ws, &all)?;
    let mp = MeshParams::default();
    let mut plain = SubmapMeshCache::new();
    let mut reserved = SubmapMeshCache::with_capacity(16, 1 << 12, 1 << 12, 64);
    let (mut a, mut b) = (TriangleMesh::new(), TriangleMesh::new());
    for cache in [&mut plain, &mut reserved] {
        let st = map.update_mesh_cache(&mut ws, &mp, cache)?;
        assert!(st.remeshed_blocks > 0);
    }
    plain.write_into(&map, SubmapFilter::All, &mut a)?;
    reserved.write_into(&map, SubmapFilter::All, &mut b)?;
    assert!(!a.triangles.is_empty());
    assert_eq!(mesh_bits(&a), mesh_bits(&b));
    // a second update finds nothing to re-mesh
    let st = map.update_mesh_cache(&mut ws, &mp, &mut reserved)?;
    assert_eq!(st.remeshed_blocks, 0);
    Ok(())
}

#[test]
fn tsdf_raycast_multi_submap_nearest() -> R {
    let cam = pinhole(48, 36, 45.0, (0.1, 4.0));
    let size = cam.sensor.size();
    let mut map = SubmapTsdf::<8>::new(submap_params(false))?;
    let one = vec![(
        SensorId(0),
        pinhole(48, 36, 45.0, (0.1, 4.0)),
        Pose3d::IDENTITY,
    )];
    map.add_agent(AgentId(0), rig(&one), SpawnPolicy::default())?;
    map.add_agent(AgentId(1), rig(&one), SpawnPolicy::default())?;
    // 1.2: the centre ray samples z = 1.06 (unallocated), then 1.22, already
    // behind the plane; the march must back off and re-march that block
    let far = DepthImage::from_size_val(size, 1.2f32, 1.0)?;
    let near = DepthImage::from_size_val(size, 1.0f32, 1.0)?;
    let mut ws = TsdfWorkspace::default();
    for (agent, d) in [(AgentId(0), &far), (AgentId(1), &near)] {
        let f = [SensorFrame {
            sensor: SensorId(0),
            depth: d.into(),
            color: None,
        }];
        let step = AgentStep {
            agent,
            stamp: 1,
            world_from_body: Pose3d::IDENTITY,
            frames: &f,
        };
        map.integrate(&mut ws, &step)?;
    }
    let RangeSensor::Pinhole(ps) = &cam.sensor else {
        unreachable!()
    };
    let rp = RaycastParams::new(&submap_params(false).layout, 0.1, 4.0)?;
    let centre = (size.height / 2) * size.width + size.width / 2;
    let mut cast = |filter: SubmapFilter<'_>| -> Result<f32, TsdfError> {
        let mut maps = RaycastMaps::new(size, false)?;
        map.raycast_into(
            &mut ws,
            ps.camera(),
            &Pose3d::IDENTITY,
            filter,
            &rp,
            &mut maps,
        )?;
        Ok(maps.depth().as_slice()[centre])
    };
    let all = cast(SubmapFilter::All)?;
    let a0 = cast(SubmapFilter::Agents(&[AgentId(0)]))?;
    assert!((all - 1.0).abs() < VOXEL, "nearest submap wins: {all}");
    assert!((a0 - 1.2).abs() < VOXEL, "filtered to agent 0: {a0}");
    let unknown = SubmapId {
        agent: AgentId(0),
        seq: 7,
    };
    assert!(matches!(
        cast(SubmapFilter::Submaps(&[unknown])),
        Err(TsdfError::UnknownSubmap(_))
    ));
    Ok(())
}

#[test]
fn tsdf_submap_isolation_and_spawn() -> R {
    let w = world();
    let data = steps(&w);
    let frames: Vec<Vec<SensorFrame<'_>>> = data.iter().map(frames).collect();
    let mut map = new_map(&w);
    let mut ws = TsdfWorkspace::default();
    // only agent 0 integrates
    for i in 0..5 {
        let r = map.integrate(&mut ws, &agent_step(&data[i], &frames[i]))?;
        assert_eq!(r.spawned, i % 2 == 0);
        assert_eq!(r.submap.seq as usize, i / 2);
        assert_eq!(r.submap.agent, AgentId(0));
    }
    assert_eq!(map.active_submap(AgentId(1)), None);
    assert_eq!(map.memory_bytes(Some(AgentId(1))).blocks, 0);
    assert!(map.memory_bytes(Some(AgentId(0))).blocks > 0);
    // TranslationAtSpawn: identity rotation at the spawning body position
    let id = SubmapId {
        agent: AgentId(0),
        seq: 1,
    };
    let p = map.submap_pose(id).unwrap();
    assert_eq!(p.rotation, Mat3F64::IDENTITY);
    assert_eq!(p.translation, data[2].body.translation);
    let mut resident = Vec::new();
    map.resident_submaps_into(&mut resident);
    assert_eq!(resident.len(), 3);
    Ok(())
}

#[test]
fn tsdf_stale_and_duplicate_steps_rejected() -> R {
    let w = world();
    let data = steps(&w);
    let frames: Vec<Vec<SensorFrame<'_>>> = data.iter().map(frames).collect();
    let step = |i: usize, stamp: u64| AgentStep {
        stamp,
        ..agent_step(&data[i], &frames[i])
    };
    let mut map = new_map(&w);
    let mut ws = TsdfWorkspace::default();
    map.integrate(&mut ws, &step(0, 10))?;
    let before = snapshot(&map);
    assert!(matches!(
        map.integrate(&mut ws, &step(1, 10)),
        Err(TsdfError::StaleStamp {
            last: 10,
            got: 10,
            ..
        })
    ));
    let r = map.integrate_batch(&mut ws, &[step(1, 30), step(2, 30)]);
    assert!(
        matches!(r, Err(TsdfError::Batch { source, .. }) if matches!(*source, TsdfError::DuplicateStep { stamp: 30, .. }))
    );
    assert_eq!(snapshot(&map), before, "a rejected batch applies nothing");
    // unknown sensor and duplicated sensor
    let d = &data[0].depth[0].1;
    let f = [SensorFrame {
        sensor: SensorId(9),
        depth: d.into(),
        color: None,
    }];
    let r = map.integrate(
        &mut ws,
        &AgentStep {
            frames: &f,
            ..step(0, 99)
        },
    );
    assert!(matches!(r, Err(TsdfError::UnknownSensor(SensorId(9)))));
    let f = [frames[0][0], frames[0][0]];
    let r = map.integrate(
        &mut ws,
        &AgentStep {
            frames: &f,
            ..step(0, 99)
        },
    );
    assert!(matches!(r, Err(TsdfError::DuplicateSensor(_))));
    assert_eq!(snapshot(&map), before);
    Ok(())
}

#[test]
fn tsdf_batch_prefix_applied() -> R {
    let w = world();
    let data = steps(&w);
    let frames: Vec<Vec<SensorFrame<'_>>> = data.iter().map(frames).collect();
    let mut p = submap_params(false);
    p.max_submaps = 1;
    p.volume.growth = PoolGrowth::Fixed;
    let mut map = SubmapTsdf::<8>::new(p)?;
    let spawn = SpawnPolicy {
        max_steps: 1,
        ..SpawnPolicy::default()
    };
    map.add_agent(AgentId(0), rig(&w.rig_a), spawn)?;
    let s0 = agent_step(&data[0], &frames[0]);
    let s1 = agent_step(&data[1], &frames[1]);
    // input order reversed: the later step (by stamp) fails, the earlier stays
    let r = map.integrate_batch(&mut TsdfWorkspace::default(), &[s1, s0]);
    assert!(
        matches!(r, Err(TsdfError::Batch { index: 0, source }) if matches!(*source, TsdfError::TooManySubmaps { max: 1 }))
    );
    assert_eq!(map.submap_ids().count(), 1);
    Ok(())
}

// ------------------------------------------------------------ LiDAR ----

#[test]
fn tsdf_range_image_from_points_deterministic() -> R {
    let l = lidar(1024, 32, 0.3, -0.3, (0.2, 20.0));
    let RangeSensor::Spherical(sl) = &l.sensor else {
        unreachable!()
    };
    let scene = [
        Surface::Plane {
            n: [-1.0, 0.0, 0.0],
            c: -3.0,
        },
        Surface::Sphere {
            centre: [-2.0, 1.0, 0.0],
            radius: 0.8,
        },
    ];
    let rendered = render(&l, &Pose3d::IDENTITY, &scene);
    // points from the render, pseudo-random farther ones, and NaN
    let mut pts: Vec<[f32; 3]> = Vec::new();
    for (i, &r) in rendered.image().as_slice().iter().enumerate() {
        if r > 0.0 {
            let d = l.dirs[i];
            let r = r as f64;
            pts.push([(d[0] * r) as f32, (d[1] * r) as f32, (d[2] * r) as f32]);
        }
    }
    let mut s = 0x9e37_79b9u32;
    for _ in 0..5000 {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        let a = (s % 6283) as f32 * 1e-3;
        pts.push([12.0 * a.cos(), 12.0 * a.sin(), (s % 7) as f32 - 3.0]);
    }
    pts.push([f32::NAN; 3]);
    let mut a = DepthImage::from_size_val(sl.size(), 0.0, 1.0)?;
    let mut b = DepthImage::from_size_val(sl.size(), 0.0, 1.0)?;
    sl.range_image_from_points_into(&pts, &mut a)?;
    pts.reverse();
    sl.range_image_from_points_into(&pts, &mut b)?;
    let bits = |d: &DepthImage<f32>| {
        d.image()
            .as_slice()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    };
    assert_eq!(bits(&a), bits(&b));
    // each rendered beam bins back to its own pixel
    let (mut same, mut total) = (0, 0);
    for (&r, &q) in rendered.image().as_slice().iter().zip(a.image().as_slice()) {
        if r > 0.0 {
            total += 1;
            same += ((r - q).abs() <= 1e-5 * r) as usize;
        }
    }
    assert!(same * 100 >= total * 99, "{same} of {total}");
    // a range beyond max is dropped
    let mut c = DepthImage::from_size_val(sl.size(), 0.0, 1.0)?;
    sl.range_image_from_points_into(&[[25.0, 0.0, 0.0]], &mut c)?;
    assert!(c.image().as_slice().iter().all(|&v| v == 0.0));
    Ok(())
}

// -------------------------------------------------------- core numerics ----

fn rng(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

/// Exhaustive over `[0, 1]` in optimised builds, strided under debug.
#[test]
fn tsdf_atan2_det_error_bound() {
    let stride: u32 = if cfg!(debug_assertions) { 997 } else { 1 };
    let (mut worst_core, mut prev, mut monotone) = (0f64, f32::NEG_INFINITY, true);
    let mut bits = 0u32;
    while bits <= 1.0f32.to_bits() {
        let t = f32::from_bits(bits);
        let r = atan2_det(t, 1.0);
        worst_core = worst_core.max((r as f64 - (t as f64).atan()).abs());
        monotone &= r >= prev;
        prev = r;
        bits += stride;
    }
    assert!(monotone, "atan2_det core must be non-decreasing on [0, 1]");
    let n = if cfg!(debug_assertions) {
        1 << 20
    } else {
        1 << 26
    };
    let mut s = 0x2545_f491_4f6c_dd1du64;
    let mut worst = 0f64;
    let scale = (1u32 << 24) as f32;
    for _ in 0..n {
        let v = rng(&mut s);
        let y = (((v >> 40) as f32) / scale - 0.5) * 2e3;
        let x = ((((v >> 16) & 0xff_ffff) as f32) / scale - 0.5) * 2e3;
        worst = worst.max((atan2_det(y, x) as f64 - (y as f64).atan2(x as f64)).abs());
    }
    eprintln!("atan2_det max error: core {worst_core:e}, full {worst:e}");
    assert!(worst_core <= 1.1e-7, "core error {worst_core}");
    assert!(worst <= 3.3e-7, "full error {worst}");
    use std::f32::consts::{FRAC_PI_2, PI};
    assert_eq!(atan2_det(0.0, 0.0), 0.0);
    assert_eq!(atan2_det(0.0, -1.0), PI);
    assert_eq!(atan2_det(-0.0, -1.0), PI);
    assert_eq!(atan2_det(1.0, 0.0), FRAC_PI_2);
    assert_eq!(atan2_det(-1.0, 0.0), -FRAC_PI_2);
    assert_eq!(atan2_det(1e-30, 1.0), 1e-30);
}

#[test]
fn tsdf_stdops_fma_matches_libm() {
    let n = if cfg!(debug_assertions) {
        1 << 20
    } else {
        100_000_000
    };
    let mut s = 0x9e37_79b9_7f4a_7c15u64;
    let mut checked = 0u64;
    for _ in 0..n {
        let v = rng(&mut s);
        let w = rng(&mut s);
        let a = f32::from_bits(v as u32);
        let b = f32::from_bits((v >> 32) as u32);
        let c = f32::from_bits(w as u32);
        if !(a.is_finite() && b.is_finite() && c.is_finite()) {
            continue;
        }
        let got = StdOps::fma(a, b, c);
        let want = libm::fmaf(a, b, c);
        assert!(
            got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan()),
            "{a:e} {b:e} {c:e}"
        );
        assert_eq!(
            StdOps::sqrt(a.abs()).to_bits(),
            libm::sqrtf(a.abs()).to_bits()
        );
        checked += 1;
    }
    assert!(checked > n / 2);
}

#[test]
fn tsdf_floor_i32_matches_floor() {
    let mut vals = vec![
        0.0f32, -0.0, 0.5, -0.5, 1.5, -1.5, 4194303.5, -4194303.5, -4194304.0,
    ];
    let mut s = 7u64;
    for _ in 0..100_000 {
        let v = rng(&mut s);
        vals.push(((v >> 40) as f32 / (1u32 << 24) as f32 - 0.5) * 8.0e6);
    }
    for v in vals {
        assert_eq!(tc::floor_i32(v), v.floor() as i32, "{v}");
    }
}

#[test]
fn tsdf_signed_zero_and_colour_rules() {
    // sdf == 0 stores +0; the first observation keeps tsd exactly
    let tsd = tc::truncate(0.0, 0.08, 12.5).unwrap();
    let (t, w) = tc::update_tsdf::<StdOps>(0.0, 0, tsd, 10);
    assert_eq!(t.to_bits(), 0.0f32.to_bits());
    assert_eq!(w, 1);
    assert_eq!(tc::truncate(0.08, 0.08, 12.5), Some(1.0));
    assert_eq!(tc::truncate(-0.08, 0.08, 12.5), Some(-1.0));
    assert_eq!(tc::truncate(-0.081, 0.08, 12.5), None);
    assert_eq!(tc::truncate(f32::NAN, 0.08, 12.5), None);
    // colour rounds half up and saturates explicitly at 255
    assert_eq!(tc::round_u8(254.5), 255);
    assert_eq!(tc::round_u8(254.49), 254);
    let s = tc::update_color::<StdOps>(0, [255, 0, 128], 16);
    assert_eq!(tc::unpack_rgb(s), [255, 0, 128]);
    assert_eq!(s >> 24, 1);
    let s = tc::update_color::<StdOps>(s, [0, 255, 129], 16);
    assert_eq!(tc::unpack_rgb(s), [128, 128, 129]);
}

#[test]
fn tsdf_key_packing_round_trip() -> R {
    for c in [[0, 0, 0], [-1, 2, -3], [(1 << 20) - 1, -(1 << 20), 7]] {
        assert_eq!(BlockKey::new(c[0], c[1], c[2])?.coords(), c);
    }
    assert!(BlockKey::new(1 << 20, 0, 0).is_err());
    // ascending keys = lexicographic (x, y, z)
    assert!(BlockKey::new(-1, 5, 5)? < BlockKey::new(0, -5, -5)?);
    Ok(())
}

#[test]
fn tsdf_params_validated() {
    assert!(TsdfLayout::new(0.0, 4.0).is_err());
    assert!(TsdfLayout::new(f32::NAN, 4.0).is_err());
    assert!(TsdfLayout::new(0.02, 0.5).is_err());
    assert!(DepthRange::new(0.0, 1.0).is_err());
    assert!(DepthRange::new(1.0, 1.0).is_err());
    let bad = TsdfParams {
        max_weight: 0,
        ..params(16)
    };
    assert!(TsdfVolume::<8>::new(TsdfLayout::new(VOXEL, 4.0).unwrap(), bad).is_err());
    let camera = PinholeCamera {
        k1: 0.1,
        ..PinholeCamera::IDENTITY
    };
    let size = ImageSize {
        width: 4,
        height: 4,
    };
    let r = PinholeSensor::new(&camera, size, DepthRange::new(0.1, 1.0).unwrap());
    assert!(matches!(r, Err(TsdfError::DistortedCamera)));
    // a reflection is not a rotation
    let mirror = Pose3d::new(
        Mat3F64::from_cols_array(&[-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
        Vec3F64::new(0.0, 0.0, 0.0),
    );
    assert!(matches!(
        check_pose(&mirror),
        Err(TsdfError::InvalidTransform)
    ));
}

#[cfg(target_arch = "aarch64")]
#[test]
#[allow(unsafe_code)]
fn tsdf_fp_environment_rejected() {
    std::thread::spawn(|| {
        let cam = pinhole(8, 6, 8.0, (0.1, 4.0));
        let d = DepthImage::from_size_val(cam.sensor.size(), 1.0f32, 1.0).unwrap();
        let mut vol = volume::<8>(64);
        let mut ws = TsdfWorkspace::default();
        let old: u64;
        // SAFETY: reads FPCR, then sets flush-to-zero on this thread only;
        // restored right after the call under test.
        unsafe {
            std::arch::asm!("mrs {}, fpcr", out(reg) old);
            std::arch::asm!("msr fpcr, {}", in(reg) old | (1 << 24));
        }
        let r = vol.integrate(&mut ws, &[obs(&cam, Pose3d::IDENTITY, &d)]);
        // SAFETY: restores this thread's original FPCR.
        unsafe {
            std::arch::asm!("msr fpcr, {}", in(reg) old);
        }
        assert!(matches!(r, Err(TsdfError::FpEnvironment)));
        assert_eq!(vol.num_blocks(), 0);
    })
    .join()
    .unwrap();
}
