//! `TsdfKernel::Simd` (NEON plus rayon over blocks on aarch64, parallel
//! scalar elsewhere) against the serial scalar reference, bit for bit: block
//! keys, every voxel's bits, counters. Scenes: analytic pinhole and LiDAR,
//! stress inputs where fused and unfused arithmetic differ, image widths of
//! every residue mod 4, multi-sensor steps, multi-agent batches and, with
//! `KORNIA_TSDF_REPLICA` set, 20 Replica room0 frames.

use super::*;

/// The first voxel whose bits differ, for the failure message.
fn first_diff<const B: usize>(a: &TsdfVolume<B>, b: &TsdfVolume<B>) -> Option<String>
where
    BlockDim<B>: SupportedBlockDim,
{
    if !a.block_keys().eq(b.block_keys()) {
        return Some("block keys differ".into());
    }
    for key in a.block_keys() {
        let (va, vb) = (a.block(key)?, b.block(key)?);
        for i in 0..va.tsdf.len() {
            let rgb = |v: &BlockView<'_>| v.rgbw.map(|c| c[i]);
            if va.tsdf[i].to_bits() != vb.tsdf[i].to_bits()
                || va.weight[i] != vb.weight[i]
                || rgb(&va) != rgb(&vb)
            {
                return Some(format!(
                    "block {:?} voxel {i}: scalar ({}, {}, {:?}) simd ({}, {}, {:?})",
                    key.coords(),
                    va.tsdf[i],
                    va.weight[i],
                    rgb(&va),
                    vb.tsdf[i],
                    vb.weight[i],
                    rgb(&vb)
                ));
            }
        }
    }
    None
}

/// Integrates `steps` with both kernels into fresh volumes and checks the
/// counters of every step and the final bits match; returns the voxel
/// updates (so callers can require a non-vacuous scene).
fn same<const B: usize>(
    label: &str,
    volume: impl Fn() -> TsdfVolume<B>,
    steps: &[Vec<SensorObservation<'_>>],
) -> Result<u64, Box<dyn std::error::Error>>
where
    BlockDim<B>: SupportedBlockDim,
{
    let (mut a, mut b) = (volume(), volume());
    let mut wa = TsdfWorkspace::new(TsdfKernel::Scalar);
    let mut wb = TsdfWorkspace::new(TsdfKernel::Simd);
    let mut updated = 0;
    for (k, s) in steps.iter().enumerate() {
        let sa = a.integrate(&mut wa, s)?;
        let sb = b.integrate(&mut wb, s)?;
        assert_eq!(sa, sb, "{label} B{B} step {k}: counters");
        updated += sa.updated_voxels;
    }
    if let Some(d) = first_diff(&a, &b) {
        panic!("{label} B{B}: {d}");
    }
    assert_eq!(a.content_hash(), b.content_hash(), "{label} B{B}");
    assert!(a.num_blocks() > 0, "{label} B{B}: no blocks");
    eprintln!(
        "{label} B{B}: {} blocks, {updated} voxel updates",
        a.num_blocks()
    );
    Ok(updated)
}

fn with_color<'a>(o: SensorObservation<'a>, c: &'a Image<u8, 3>) -> SensorObservation<'a> {
    SensorObservation {
        color: Some(c),
        ..o
    }
}

fn analytic<const B: usize>() -> R
where
    BlockDim<B>: SupportedBlockDim,
{
    // pinhole, two views of a plane and a sphere, colour on
    let cam = pinhole(160, 120, 150.0, (0.1, 4.0));
    let scene = [
        tilted_plane(),
        Surface::Sphere {
            centre: [0.1, 0.0, 1.0],
            radius: 0.2,
        },
    ];
    let poses: Vec<Pose3d> = (0..3)
        .map(|k| {
            let f = k as f64;
            pose(
                0.02 * f,
                -0.05 * f,
                0.01 * f,
                [0.05 * f, -0.02 * f, 0.03 * f],
            )
        })
        .collect();
    let depths: Vec<_> = poses.iter().map(|p| render(&cam, p, &scene)).collect();
    let colors: Vec<_> = (0..3).map(|k| color_image(cam.sensor.size(), k)).collect();
    let steps: Vec<Vec<_>> = (0..3)
        .map(|k| vec![with_color(obs(&cam, poses[k], &depths[k]), &colors[k])])
        .collect();
    let n = same("pinhole", || volume::<B>(4096), &steps)?;
    assert!(n > 100_000, "pinhole: {n} updates");

    // full-turn LiDAR, sphere and plane
    let l = lidar(512, 32, 0.3, -0.3, (0.2, 3.0));
    let ls = [
        Surface::Sphere {
            centre: [1.2, 0.2, 0.05],
            radius: 0.4,
        },
        Surface::Plane {
            n: [0.0, -1.0, 0.0],
            c: -1.0,
        },
    ];
    let d0 = render(&l, &Pose3d::IDENTITY, &ls);
    let p1 = pose(0.0, 0.0, 0.3, [0.1, 0.05, 0.0]);
    let d1 = render(&l, &p1, &ls);
    let steps = vec![vec![obs(&l, Pose3d::IDENTITY, &d0)], vec![obs(&l, p1, &d1)]];
    let n = same("lidar", || volume::<B>(16384), &steps)?;
    assert!(n > 50_000, "lidar: {n} updates");
    Ok(())
}

#[test]
fn tsdf_integrate_neon_matches_scalar_analytic_b8() -> R {
    analytic::<8>()
}

#[test]
fn tsdf_integrate_neon_matches_scalar_analytic_b16() -> R {
    analytic::<16>()
}

/// A partial-field LiDAR with a clockwise (negative) azimuth step: the
/// column gate rejects instead of wrapping.
fn lidar_partial() -> TestSensor {
    let (w, h, start, step, top, bottom) = (203usize, 17usize, 1.0, -0.011, 0.4, -0.35);
    let size = ImageSize {
        width: w,
        height: h,
    };
    let az = AzimuthGrid {
        start,
        step,
        wrap: false,
    };
    let s =
        SphericalLidar::uniform(size, az, top, bottom, DepthRange::new(0.2, 4.0).unwrap()).unwrap();
    let dirs = (0..w * h)
        .map(|i| {
            let el = top + (bottom - top) * ((i / w) as f64) / (h as f64 - 1.0);
            let a = start + step * (i % w) as f64;
            [el.cos() * a.cos(), el.cos() * a.sin(), el.sin()]
        })
        .collect();
    TestSensor {
        sensor: s.into(),
        dirs,
    }
}

/// `world_from_sensor` applied to a sensor-frame surface.
fn moved(s: Surface, p: &Pose3d) -> Surface {
    match s {
        Surface::Plane { n, c } => {
            let nw = p.rotation * Vec3F64::new(n[0], n[1], n[2]);
            let t = p.translation;
            let nw = [nw.x, nw.y, nw.z];
            Surface::Plane {
                n: nw,
                c: c + dot(nw, [t.x, t.y, t.z]),
            }
        }
        Surface::Sphere { centre, radius } => {
            let cw = p.transform_point(&Vec3F64::new(centre[0], centre[1], centre[2]));
            Surface::Sphere {
                centre: [cw.x, cw.y, cw.z],
                radius,
            }
        }
    }
}

/// Unfused `R * p + t` in the contract's nesting (each product rounded).
fn xform_unfused(r: &tc::Rigid, p: [f32; 3]) -> [f32; 3] {
    let row =
        |i: usize| r.r[3 * i] * p[0] + (r.r[3 * i + 1] * p[1] + (r.r[3 * i + 2] * p[2] + r.t[i]));
    [row(0), row(1), row(2)]
}

fn stress<const B: usize>() -> R
where
    BlockDim<B>: SupportedBlockDim,
{
    // non-dyadic voxel and truncation, irrational rotation, far offset
    let layout = TsdfLayout::new(0.0137, 3.7)?;
    let new = || TsdfVolume::<B>::new(layout, params(16384)).unwrap();
    let off = [123.456, -78.9, 3.21];
    let p = pose(0.3141, -0.2718, 1.4171, off);
    let cam = pinhole(97, 61, 83.3, (0.1, 4.0));
    let local = [
        Surface::Plane {
            n: [0.0, 0.0, -1.0],
            c: -1.0,
        },
        Surface::Sphere {
            centre: [0.0, 0.0, 0.9],
            radius: 0.3,
        },
    ];
    let scene: Vec<Surface> = local.iter().map(|s| moved(*s, &p)).collect();
    let d = render(&cam, &p, &scene);
    let c = color_image(cam.sensor.size(), 7);
    let l = lidar_partial();
    let pl = p.compose(&lidar_mount([0.01, 0.02, 0.03]));
    let dl = render(&l, &pl, &scene);
    let steps = vec![vec![with_color(obs(&cam, p, &d), &c), obs(&l, pl, &dl)]];
    let n = same("stress", new, &steps)?;
    assert!(n > 20_000, "stress: {n} updates");

    // the inputs separate fused from unfused arithmetic
    let mut v = new();
    v.integrate(&mut TsdfWorkspace::new(TsdfKernel::Scalar), &steps[0])?;
    let sfv = rigid_from_pose(&p.inverse())?;
    let (mut total, mut differ) = (0usize, 0usize);
    for key in v.block_keys() {
        let g0 = key.coords().map(|c| c * B as i32);
        for i in 0..B * B * B {
            let g = [
                g0[0] + (i % B) as i32,
                g0[1] + (i / B % B) as i32,
                g0[2] + (i / (B * B)) as i32,
            ];
            let c = v.layout().centre_from_voxel(g);
            let fused = tc::xform::<StdOps>(&sfv.r, &sfv.t, c);
            total += 1;
            differ += (fused != xform_unfused(&sfv, c)) as usize;
        }
    }
    assert!(
        differ * 4 > total,
        "only {differ} of {total} voxel transforms differ unfused"
    );
    Ok(())
}

#[test]
fn tsdf_integrate_neon_matches_scalar_stress_b8() -> R {
    stress::<8>()
}

#[test]
fn tsdf_integrate_neon_matches_scalar_stress_b16() -> R {
    stress::<16>()
}

/// Image widths of every residue mod 4 (and odd heights): blocks straddle
/// the image border, so registers mix in-image and gated-out lanes.
#[test]
fn tsdf_integrate_neon_tails_bit_exact() -> R {
    for w in 60..64 {
        let h = 2 * (w / 3) + 1;
        let cam = pinhole(w, h, 0.9 * w as f64, (0.1, 4.0));
        let scene = [
            tilted_plane(),
            Surface::Sphere {
                centre: [0.05, 0.02, 0.8],
                radius: 0.15,
            },
        ];
        let p = pose(0.01, 0.02, 0.03, [0.003, -0.002, 0.001]);
        let d = render(&cam, &p, &scene);
        let c = color_image(cam.sensor.size(), w);
        let steps = vec![vec![with_color(obs(&cam, p, &d), &c)]];
        let label = format!("width {w}");
        assert!(same(&label, || volume::<8>(4096), &steps)? > 5_000);
        assert!(same(&label, || volume::<16>(4096), &steps)? > 5_000);
    }
    Ok(())
}

fn multi_sensor<const B: usize>() -> R
where
    BlockDim<B>: SupportedBlockDim,
{
    let cam_a = pinhole(96, 72, 90.0, (0.1, 4.0));
    let cam_b = pinhole(81, 59, 70.0, (0.1, 4.0));
    let lid = lidar(512, 32, 0.3, -0.3, (0.2, 4.0));
    let scene = [
        tilted_plane(),
        Surface::Sphere {
            centre: [0.0, 0.1, 1.0],
            radius: 0.2,
        },
    ];
    let mut data = Vec::new();
    for k in 0..3 {
        let f = k as f64;
        let body = pose(0.0, 0.02 * f, 0.0, [0.03 * f, 0.0, 0.01 * f]);
        let pa = body;
        let pb = body.compose(&pose(0.0, 0.15, 0.0, [-0.1, 0.0, 0.05]));
        let pl = body.compose(&lidar_mount([0.0, 0.05, 0.0]));
        data.push((
            [pa, pb, pl],
            render(&cam_a, &pa, &scene),
            render(&cam_b, &pb, &scene),
            render(&lid, &pl, &scene),
            color_image(cam_a.sensor.size(), k),
            color_image(cam_b.sensor.size(), k + 5),
        ));
    }
    let steps: Vec<Vec<_>> = data
        .iter()
        .map(|(p, da, db, dl, ca, cb)| {
            vec![
                with_color(obs(&cam_a, p[0], da), ca),
                obs(&lid, p[2], dl),
                with_color(obs(&cam_b, p[1], db), cb),
            ]
        })
        .collect();
    let n = same("multi-sensor", || volume::<B>(8192), &steps)?;
    assert!(n > 50_000, "multi-sensor: {n} updates");
    Ok(())
}

#[test]
fn tsdf_integrate_neon_matches_scalar_multi_sensor_b8() -> R {
    multi_sensor::<8>()
}

#[test]
fn tsdf_integrate_neon_matches_scalar_multi_sensor_b16() -> R {
    multi_sensor::<16>()
}

/// Two agents' batches: every submap's bits and pose, per kernel.
#[test]
fn tsdf_integrate_neon_matches_scalar_multi_agent() -> R {
    let w = world();
    let data = steps(&w);
    let frames: Vec<Vec<SensorFrame<'_>>> = data.iter().map(frames).collect();
    let all: Vec<AgentStep<'_>> = (0..data.len())
        .map(|i| agent_step(&data[i], &frames[i]))
        .collect();
    let run = |kernel| -> Result<_, TsdfError> {
        let mut map = new_map(&w);
        let mut ws = TsdfWorkspace::new(kernel);
        let r = map.integrate_batch(&mut ws, &all)?;
        Ok((snapshot(&map), r))
    };
    let (a, ra) = run(TsdfKernel::Scalar)?;
    let (b, rb) = run(TsdfKernel::Simd)?;
    assert_eq!(a.len(), 6);
    assert!(ra.stats.updated_voxels > 10_000);
    assert_eq!(ra, rb);
    assert_eq!(a, b);
    Ok(())
}

/// Root of the NICE-SLAM Replica export (holding `room0/` and
/// `cam_params.json`); the test skips without it.
const REPLICA_ENV: &str = "KORNIA_TSDF_REPLICA";
/// Frames used: every fifth of the first 100.
const REPLICA_FRAMES: usize = 20;
const REPLICA_STRIDE: usize = 5;

/// Room0 frame `k`: depth (u16, `scale` units per metre), colour and
/// `world_from_camera` (traj.txt is row-major camera-to-world).
struct ReplicaFrame {
    depth: DepthImage<u16>,
    color: Image<u8, 3>,
    world_from_camera: Pose3d,
}

fn replica_frames(
    root: &std::path::Path,
    size: ImageSize,
    scale: f32,
) -> Result<Vec<ReplicaFrame>, Box<dyn std::error::Error>> {
    let room = root.join("room0");
    let traj = std::fs::read_to_string(room.join("traj.txt"))?;
    let rows: Vec<Vec<f64>> = traj
        .lines()
        .map(|l| l.split_whitespace().map(str::parse).collect())
        .collect::<Result<_, _>>()?;
    let mut out = Vec::new();
    for k in (0..REPLICA_FRAMES).map(|i| i * REPLICA_STRIDE) {
        let m = &rows[k];
        let rot =
            Mat3F64::from_cols_array(&[m[0], m[4], m[8], m[1], m[5], m[9], m[2], m[6], m[10]]);
        let world_from_camera = Pose3d::new(rot, Vec3F64::new(m[3], m[7], m[11]));
        let d =
            kornia_io::png::read_image_png_mono16(room.join(format!("results/depth{k:06}.png")))?;
        let c =
            kornia_io::jpeg::read_image_jpeg_rgb8(room.join(format!("results/frame{k:06}.jpg")))?;
        out.push(ReplicaFrame {
            depth: DepthImage::new(Image::new(size, d.as_slice().to_vec())?, scale)?,
            color: Image::new(size, c.as_slice().to_vec())?,
            world_from_camera,
        });
    }
    Ok(out)
}

fn replica<const B: usize>(root: &std::path::Path) -> R
where
    BlockDim<B>: SupportedBlockDim,
{
    // cam_params.json of the export: 1200 x 680, f = 600, depth 6553.5 / m
    let size = ImageSize {
        width: 1200,
        height: 680,
    };
    let camera = crate::camera::PinholeCamera {
        fx: 600.0,
        fy: 600.0,
        cx: 599.5,
        cy: 339.5,
        ..crate::camera::PinholeCamera::IDENTITY
    };
    let sensor = RangeSensor::from(PinholeSensor::new(
        &camera,
        size,
        DepthRange::new(0.1, 7.0)?,
    )?);
    let frames = replica_frames(root, size, 6553.5)?;
    let steps: Vec<Vec<_>> = frames
        .iter()
        .map(|f| {
            vec![SensorObservation {
                sensor: &sensor,
                volume_from_sensor: f.world_from_camera,
                depth: (&f.depth).into(),
                color: Some(&f.color),
            }]
        })
        .collect();
    let layout = TsdfLayout::new(0.02, 4.0)?;
    let n = same(
        "replica room0",
        || TsdfVolume::<B>::new(layout, params(1 << 16)).unwrap(),
        &steps,
    )?;
    assert!(n > 10_000_000, "replica: {n} updates");
    Ok(())
}

/// 20 Replica room0 frames (voxel 2 cm, colour on), B = 8 and 16. Set
/// `KORNIA_TSDF_REPLICA` to the Replica root; run in release.
#[test]
fn tsdf_integrate_neon_matches_scalar_replica() -> R {
    let Some(root) = std::env::var_os(REPLICA_ENV).map(std::path::PathBuf::from) else {
        eprintln!("skipped: {REPLICA_ENV} is not set");
        return Ok(());
    };
    if !root.join("room0/traj.txt").exists() {
        eprintln!("skipped: no room0 under {}", root.display());
        return Ok(());
    }
    replica::<8>(&root)?;
    replica::<16>(&root)
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod lanes {
    //! The core's lane functions with `NeonOps` against `StdOps`, lane by
    //! lane, on random and special inputs.

    use super::*;
    use crate::tsdf::integrate_neon::{NeonOps, Vf, Vu};

    /// xorshift64*: deterministic test inputs.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
        }

        /// Mostly uniform in `[-s, s]`, with special values mixed in.
        fn f(&mut self, s: f32) -> f32 {
            const SPECIAL: [f32; 8] = [0.0, -0.0, 1.0, -1.0, 1e-30, -1e-30, 0.5, 2.0];
            let u = self.next();
            if u.is_multiple_of(16) {
                SPECIAL[(u >> 8) as usize % SPECIAL.len()]
            } else {
                ((u >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * s
            }
        }

        fn quad(&mut self, s: f32) -> [f32; 4] {
            [self.f(s), self.f(s), self.f(s), self.f(s)]
        }
    }

    fn bits(v: [f32; 4]) -> [u32; 4] {
        v.map(f32::to_bits)
    }

    const N: usize = 1 << 16;

    #[test]
    fn tsdf_lanes_atan2_xform_neon_matches_scalar() {
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        let r = [0.3, -0.7, 0.648, 0.9, 0.2, -0.38, -0.31, 0.68, 0.66f32];
        let t = [1.25, -0.5, 3.0f32];
        for _ in 0..N {
            let (y, x) = (rng.quad(10.0), rng.quad(10.0));
            let lane = tc::atan2_det::<NeonOps>(Vf::from_array(y), Vf::from_array(x)).to_array();
            let scalar = [0, 1, 2, 3].map(|l| tc::atan2_det::<StdOps>(y[l], x[l]));
            assert_eq!(bits(lane), bits(scalar), "atan2_det({y:?}, {x:?})");
            let p = [rng.quad(50.0), rng.quad(50.0), rng.quad(50.0)];
            let lane = tc::xform::<NeonOps>(&r, &t, p.map(Vf::from_array)).map(Vf::to_array);
            for l in 0..4 {
                let s = tc::xform::<StdOps>(&r, &t, [p[0][l], p[1][l], p[2][l]]);
                for a in 0..3 {
                    assert_eq!(lane[a][l].to_bits(), s[a].to_bits(), "xform lane {l}");
                }
            }
        }
    }

    #[test]
    fn tsdf_lanes_update_neon_matches_scalar() {
        let mut rng = Rng(0x0123_4567_89ab_cdef);
        for _ in 0..N {
            let sdf = rng.quad(0.2);
            let (ok, tsd) = tc::truncate_lanes::<Vf>(Vf::from_array(sdf), 0.08, 12.5);
            let ok = ok.to_array();
            let tsd = tsd.to_array();
            for l in 0..4 {
                let s = tc::truncate(sdf[l], 0.08, 12.5);
                assert_eq!(ok[l] != 0, s.is_some(), "truncate gate {}", sdf[l]);
                if let Some(s) = s {
                    assert_eq!(tsd[l].to_bits(), s.to_bits(), "truncate {}", sdf[l]);
                }
            }
            let old = rng.quad(1.0);
            let w0 = [0, 1, 2, 3].map(|_| (rng.next() % 70_000).min(65_535) as u16);
            let tsd = rng.quad(1.0);
            let cap = (rng.next() % 300) as u16 + 1;
            let (t, w) = tc::update_tsdf_lanes::<NeonOps>(
                Vf::from_array(old),
                Vu::from_array(w0.map(u32::from)),
                Vf::from_array(tsd),
                cap,
            );
            let (t, w) = (t.to_array(), w.to_array());
            for l in 0..4 {
                let (st, sw) = tc::update_tsdf::<StdOps>(old[l], w0[l], tsd[l], cap);
                assert_eq!((t[l].to_bits(), w[l]), (st.to_bits(), u32::from(sw)));
            }
        }
    }
}
