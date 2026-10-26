use super::*;
use kornia_algebra::{Mat3F64, Vec3F64};
use kornia_image::ImageSize;

type TestResult = Result<(), Box<dyn std::error::Error>>;

const MM: f32 = 0.001;

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
    /// Uniform in `[0, 1)`.
    fn unit(&mut self) -> f32 {
        (self.next() >> 8) as f32 / (1u32 << 23) as f32
    }
}

/// Millimetre depth in `500..12000` with ~20% missing samples.
fn random_depth_u16(w: usize, h: usize, seed: u64) -> Vec<u16> {
    let mut rng = Lcg(seed);
    (0..w * h)
        .map(|_| {
            if rng.next().is_multiple_of(5) {
                0
            } else {
                500 + (rng.next() % 11_500) as u16
            }
        })
        .collect()
}

/// Metric depth in `0.3..8.0` with zeros, NaNs and a few negatives.
fn random_depth_f32(w: usize, h: usize, seed: u64) -> Vec<f32> {
    let mut rng = Lcg(seed);
    (0..w * h)
        .map(|_| match rng.next() % 10 {
            0 => 0.0,
            1 => f32::NAN,
            2 => -1.0,
            _ => 0.3 + 7.7 * rng.unit(),
        })
        .collect()
}

// ── unproject ──────────────────────────────────────────────────────────────

/// Values from the Open3D / pinhole formula `((x - cx) z / fx, (y - cy) z / fy, z)`,
/// worked by hand with exactly representable inputs.
#[test]
fn unproject_pins_reference_values() -> TestResult {
    let camera = pinhole(4.0, 2.0, 1.0, 0.5);
    let depth = Image::<f32, 1>::new(sz(3, 2), vec![2.0, 0.0, 4.0, 1.0, 8.0, f32::NAN])?;
    let cloud = unproject_depth(&depth, 1.0, &camera, None)?;
    assert_eq!(
        cloud.points(),
        &vec![
            [-0.5, -0.5, 2.0],  // (0, 0): (0 - 1) * 2 / 4, (0 - 0.5) * 2 / 2
            [1.0, -1.0, 4.0],   // (2, 0)
            [-0.25, 0.25, 1.0], // (0, 1)
            [0.0, 2.0, 8.0],    // (1, 1)
        ]
    );
    assert!(cloud.colors().is_none() && cloud.normals().is_none());
    Ok(())
}

#[test]
fn unproject_u16_matches_formula_and_index() -> TestResult {
    let (w, h) = (97, 61);
    let data = random_depth_u16(w, h, 7);
    let depth = Image::<u16, 1>::new(sz(w, h), data.clone())?;
    let camera = pinhole(91.5, 88.25, 47.3, 29.9);
    let mut points = Vec::new();
    let mut index = Vec::new();
    unproject_depth_into(&depth, MM, &camera, None, &mut points, Some(&mut index))?;

    let expected: Vec<usize> = (0..w * h).filter(|&i| data[i] != 0).collect();
    assert_eq!(points.len(), expected.len());
    assert_eq!(
        index.iter().map(|&i| i as usize).collect::<Vec<_>>(),
        expected
    );
    for (p, &i) in points.iter().zip(&index) {
        let (x, y) = ((i as usize % w) as f64, (i as usize / w) as f64);
        let z = f64::from(data[i as usize]) * 1e-3;
        let want = [
            (x - camera.cx) / camera.fx * z,
            (y - camera.cy) / camera.fy * z,
            z,
        ];
        for k in 0..3 {
            assert!(
                (p[k] - want[k]).abs() <= 1e-5 * z.max(1.0),
                "pixel {i} axis {k}: {} vs {}",
                p[k],
                want[k]
            );
        }
    }
    Ok(())
}

/// Unproject -> project returns each point to its source pixel and depth.
#[test]
fn unproject_project_round_trip() -> TestResult {
    let (w, h) = (64, 48);
    let data = random_depth_f32(w, h, 11);
    let depth = Image::<f32, 1>::new(sz(w, h), data.clone())?;
    let camera = pinhole(60.0, 62.5, 31.7, 23.2);
    let mut points = Vec::new();
    let mut index = Vec::new();
    unproject_depth_into(&depth, 1.0, &camera, None, &mut points, Some(&mut index))?;
    let n_valid = data.iter().filter(|&&d| d > 0.0 && d.is_finite()).count();
    assert_eq!(points.len(), n_valid);
    for (p, &i) in points.iter().zip(&index) {
        let uv = camera
            .project_to_pixel(&Vec3F64::new(p[0], p[1], p[2]), 0.0)
            .ok_or("point behind the camera")?;
        let (x, y) = ((i as usize % w) as f64, (i as usize / w) as f64);
        assert!((uv.x - x).abs() < 1e-3 && (uv.y - y).abs() < 1e-3);
        assert_eq!(p[2], f64::from(data[i as usize]));
    }
    Ok(())
}

#[test]
fn unproject_mask_skips_pixels() -> TestResult {
    let (w, h) = (40, 30);
    let data = random_depth_u16(w, h, 3);
    let mut rng = Lcg(5);
    let mask: Vec<u8> = (0..w * h).map(|_| (rng.next() % 2) as u8 * 255).collect();
    let depth = Image::<u16, 1>::new(sz(w, h), data.clone())?;
    let mask_img = Image::<u8, 1>::new(sz(w, h), mask.clone())?;
    let camera = pinhole(30.0, 30.0, 20.0, 15.0);
    let mut points = Vec::new();
    let mut index = Vec::new();
    unproject_depth_into(
        &depth,
        MM,
        &camera,
        Some(&mask_img),
        &mut points,
        Some(&mut index),
    )?;
    let expected: Vec<u32> = (0..w * h)
        .filter(|&i| data[i] != 0 && mask[i] != 0)
        .map(|i| i as u32)
        .collect();
    assert_eq!(index, expected);
    assert_eq!(points.len(), expected.len());
    Ok(())
}

/// The banded parallel writer equals the sequential reference, on heights
/// that do and do not divide into the band count.
#[test]
fn unproject_parallel_matches_sequential() -> TestResult {
    for &(w, h) in &[(1, 1), (5, 1), (1, 70), (33, 64), (17, 129), (211, 333)] {
        let data = random_depth_f32(w, h, (w * h) as u64);
        let camera = pinhole(50.0, 51.0, w as f64 / 2.0, h as f64 / 2.0);
        let p = kernels::unproject_params(&camera, 1.0, w, h)?;
        let (mut par_pts, mut par_ix) = (Vec::new(), Vec::new());
        let (mut seq_pts, mut seq_ix) = (Vec::new(), Vec::new());
        kernels::unproject_into(&data, None, &p, &mut par_pts, Some(&mut par_ix));
        kernels::unproject_sequential(&data, None, &p, &mut seq_pts, Some(&mut seq_ix));
        assert_eq!(par_pts, seq_pts, "{w}x{h}");
        assert_eq!(par_ix, seq_ix, "{w}x{h}");
    }
    Ok(())
}

#[test]
fn unproject_into_reuses_buffers() -> TestResult {
    let (w, h) = (50, 40);
    let depth = Image::<u16, 1>::new(sz(w, h), random_depth_u16(w, h, 9))?;
    let camera = pinhole(40.0, 40.0, 25.0, 20.0);
    let mut points = vec![[9.0; 3]; 7];
    let mut index = vec![9u32; 3];
    unproject_depth_into(&depth, MM, &camera, None, &mut points, Some(&mut index))?;
    let first = points.clone();
    let (cap, ptr) = (points.capacity(), points.as_ptr());
    unproject_depth_into(&depth, MM, &camera, None, &mut points, Some(&mut index))?;
    assert_eq!(points, first);
    assert_eq!(index.len(), points.len());
    assert_eq!((points.capacity(), points.as_ptr()), (cap, ptr));
    // Without an index buffer the points are the same.
    let mut plain = Vec::new();
    unproject_depth_into(&depth, MM, &camera, None, &mut plain, None)?;
    assert_eq!(plain, first);
    Ok(())
}

#[test]
fn unproject_empty_and_all_invalid() -> TestResult {
    let camera = pinhole(1.0, 1.0, 0.0, 0.0);
    let zeros = Image::<u16, 1>::from_size_val(sz(8, 4), 0)?;
    assert!(unproject_depth(&zeros, MM, &camera, None)?.is_empty());
    let empty = Image::<f32, 1>::new(sz(0, 3), vec![])?;
    assert!(unproject_depth(&empty, 1.0, &camera, None)?.is_empty());
    Ok(())
}

#[test]
fn unproject_rejects_bad_arguments() -> TestResult {
    let camera = pinhole(1.0, 1.0, 0.0, 0.0);
    let depth = Image::<u16, 1>::from_size_val(sz(4, 3), 1)?;
    // Same pixel count, different shape.
    let mask = Image::<u8, 1>::from_size_val(sz(3, 4), 1)?;
    assert!(matches!(
        unproject_depth(&depth, MM, &camera, Some(&mask)),
        Err(DepthError::Image(ImageError::InvalidImageSize(4, 3, 3, 4)))
    ));
    for scale in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        assert!(matches!(
            unproject_depth(&depth, scale, &camera, None),
            Err(DepthError::InvalidDepthScale(_))
        ));
    }
    for bad in [
        pinhole(0.0, 1.0, 0.0, 0.0),
        pinhole(1.0, f64::NAN, 0.0, 0.0),
        pinhole(1.0, 1.0, f64::INFINITY, 0.0),
    ] {
        assert!(matches!(
            unproject_depth(&depth, MM, &bad, None),
            Err(DepthError::InvalidIntrinsics { .. })
        ));
    }
    Ok(())
}

// ── register ───────────────────────────────────────────────────────────────

fn translation(x: f64, y: f64, z: f64) -> Pose3d {
    Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(x, y, z))
}

/// Identity transform, same camera, no splat: every valid sample lands on
/// its own pixel with its own value; invalid samples become `0`.
#[test]
fn register_identity_reproduces_input() -> TestResult {
    let (w, h) = (73, 41);
    let camera = pinhole(70.0, 69.0, 36.2, 20.7);

    let data = random_depth_u16(w, h, 21);
    let src = Image::<u16, 1>::new(sz(w, h), data.clone())?;
    let mut dst = Image::<u16, 1>::from_size_val(sz(w, h), 7)?;
    register_depth(&src, MM, &camera, &Pose3d::IDENTITY, &camera, 0, &mut dst)?;
    assert_eq!(dst.as_slice(), &data[..]);

    let data = random_depth_f32(w, h, 22);
    let src = Image::<f32, 1>::new(sz(w, h), data.clone())?;
    let mut dst = Image::<f32, 1>::from_size_val(sz(w, h), 7.0)?;
    register_depth(&src, 1.0, &camera, &Pose3d::IDENTITY, &camera, 0, &mut dst)?;
    for (i, (&o, &d)) in dst.as_slice().iter().zip(&data).enumerate() {
        let want = if d > 0.0 && d.is_finite() { d } else { 0.0 };
        assert_eq!(o.to_bits(), want.to_bits(), "pixel {i}");
    }
    Ok(())
}

/// A foreground strip moves further (parallax) than the background behind
/// it and lands on top of it: the nearer depth must win every shared pixel,
/// and the background it uncovers is a hole.
#[test]
fn register_occlusion_nearer_wins() -> TestResult {
    let (w, h) = (24, 3);
    // fx * tx / z: background (4 m) moves 1 px, foreground (1 m) 4 px.
    let camera = pinhole(100.0, 100.0, 12.0, 1.0);
    let mut data = vec![4000u16; w * h];
    for y in 0..h {
        for x in 10..=13 {
            data[y * w + x] = 1000;
        }
    }
    let src = Image::<u16, 1>::new(sz(w, h), data)?;
    let mut dst = Image::<u16, 1>::from_size_val(sz(w, h), 0)?;
    register_depth(
        &src,
        MM,
        &camera,
        &translation(0.04, 0.0, 0.0),
        &camera,
        0,
        &mut dst,
    )?;
    let mut row = vec![0u16; w];
    row[1..=10].fill(4000); // background src 0..=9
    row[14..=17].fill(1000); // foreground src 10..=13, over background src 14..=16
    row[18..].fill(4000); // background src 17..=22; src 23 leaves the image
    for y in 0..h {
        assert_eq!(&dst.as_slice()[y * w..(y + 1) * w], &row[..], "row {y}");
    }

    // The same scene in f32 metres.
    let src_f = Image::<f32, 1>::new(
        sz(w, h),
        src.as_slice().iter().map(|&d| f32::from(d) * MM).collect(),
    )?;
    let mut dst_f = Image::<f32, 1>::from_size_val(sz(w, h), 0.0)?;
    register_depth(
        &src_f,
        1.0,
        &camera,
        &translation(0.04, 0.0, 0.0),
        &camera,
        0,
        &mut dst_f,
    )?;
    for (i, (&o, &r)) in dst_f.as_slice().iter().zip(row.iter().cycle()).enumerate() {
        assert!((o - f32::from(r) * MM).abs() < 1e-6, "pixel {i}: {o}");
    }
    Ok(())
}

/// A 2x upsample leaves three holes per source pixel without a splat; a
/// radius-1 splat fills the whole destination.
#[test]
fn register_splat_fills_2x_upsample() -> TestResult {
    let (w, h) = (8, 6);
    let src_cam = pinhole(10.0, 10.0, 3.5, 2.5);
    // u_dst = 2 x + 0.25, so source pixel x lands on destination pixel 2x.
    let dst_cam = pinhole(20.0, 20.0, 7.25, 5.25);
    let src = Image::<u16, 1>::from_size_val(sz(w, h), 1500)?;
    let mut dst = Image::<u16, 1>::from_size_val(sz(2 * w, 2 * h), 0)?;

    register_depth(&src, MM, &src_cam, &Pose3d::IDENTITY, &dst_cam, 0, &mut dst)?;
    let d = dst.as_slice();
    assert_eq!(d.iter().filter(|&&v| v != 0).count(), w * h);
    for y in 0..2 * h {
        for x in 0..2 * w {
            let want = if x % 2 == 0 && y % 2 == 0 { 1500 } else { 0 };
            assert_eq!(d[y * 2 * w + x], want, "({x}, {y})");
        }
    }

    register_depth(&src, MM, &src_cam, &Pose3d::IDENTITY, &dst_cam, 1, &mut dst)?;
    assert!(dst.as_slice().iter().all(|&v| v == 1500));
    Ok(())
}

/// Points behind the destination camera, or whose depth does not fit the
/// `u16` output, are dropped.
#[test]
fn register_drops_unrepresentable_points() -> TestResult {
    let camera = pinhole(10.0, 10.0, 1.5, 1.5);
    let src = Image::<u16, 1>::from_size_val(sz(4, 4), 1000)?;
    let mut dst = Image::<u16, 1>::from_size_val(sz(4, 4), 9)?;
    // 1 m in front of the source, 1 m behind the destination.
    register_depth(
        &src,
        MM,
        &camera,
        &translation(0.0, 0.0, -2.0),
        &camera,
        0,
        &mut dst,
    )?;
    assert!(dst.as_slice().iter().all(|&v| v == 0));
    // 1000 mm + 65 m = 66000 mm > 65534.
    register_depth(
        &src,
        MM,
        &camera,
        &translation(0.0, 0.0, 65.0),
        &camera,
        0,
        &mut dst,
    )?;
    assert!(dst.as_slice().iter().all(|&v| v == 0));
    // 1000 mm + 64 m = 65000 mm still fits (the image shrinks onto the centre).
    register_depth(
        &src,
        MM,
        &camera,
        &translation(0.0, 0.0, 64.0),
        &camera,
        0,
        &mut dst,
    )?;
    assert!(dst.as_slice().contains(&65000));
    Ok(())
}

#[test]
fn register_rejects_bad_arguments() -> TestResult {
    let camera = pinhole(1.0, 1.0, 0.0, 0.0);
    let src = Image::<u16, 1>::from_size_val(sz(4, 3), 1)?;
    let mut dst = Image::<u16, 1>::from_size_val(sz(4, 3), 0)?;
    let id = Pose3d::IDENTITY;
    assert!(matches!(
        register_depth(
            &src,
            MM,
            &camera,
            &id,
            &camera,
            MAX_SPLAT_RADIUS + 1,
            &mut dst
        ),
        Err(DepthError::SplatRadiusTooLarge { .. })
    ));
    assert!(matches!(
        register_depth(&src, 0.0, &camera, &id, &camera, 0, &mut dst),
        Err(DepthError::InvalidDepthScale(_))
    ));
    assert!(matches!(
        register_depth(
            &src,
            MM,
            &camera,
            &id,
            &pinhole(0.0, 1.0, 0.0, 0.0),
            0,
            &mut dst
        ),
        Err(DepthError::InvalidIntrinsics { .. })
    ));
    assert!(matches!(
        register_depth(
            &src,
            MM,
            &camera,
            &translation(f64::NAN, 0.0, 0.0),
            &camera,
            0,
            &mut dst
        ),
        Err(DepthError::InvalidTransform)
    ));
    Ok(())
}

/// Straight-line oracle of `register_depth` for `u16`: one sequential pass,
/// one pixel at a time, writing the smallest depth into each splat cell.
fn register_u16_oracle(
    src: &[u16],
    (w, h): (usize, usize),
    src_cam: &PinholeCamera,
    pose: &Pose3d,
    dst_cam: &PinholeCamera,
    radius: usize,
    (dw, dh): (usize, usize),
) -> Vec<u16> {
    let (ifx, ify) = ((1.0 / src_cam.fx) as f32, (1.0 / src_cam.fy) as f32);
    let (scx, scy) = (src_cam.cx as f32, src_cam.cy as f32);
    let (fx, fy) = (dst_cam.fx as f32, dst_cam.fy as f32);
    let (dcx, dcy) = (dst_cam.cx as f32, dst_cam.cy as f32);
    let m = pose.rotation.to_cols_array();
    let r: [f32; 9] = std::array::from_fn(|k| m[3 * (k % 3) + k / 3] as f32);
    let t = [
        pose.translation.x as f32,
        pose.translation.y as f32,
        pose.translation.z as f32,
    ];
    let rad = radius as i64;
    let mut out = vec![u16::MAX; dw * dh];
    for y in 0..h {
        for x in 0..w {
            let z = f32::from(src[y * w + x]) * MM;
            if z <= 0.0 {
                continue;
            }
            let px = ((x as f32 - scx) * ifx) * z;
            let py = ((y as f32 - scy) * ify) * z;
            let qx = r[0] * px + r[1] * py + r[2] * z + t[0];
            let qy = r[3] * px + r[4] * py + r[5] * z + t[1];
            let qz = r[6] * px + r[7] * py + r[8] * z + t[2];
            if qz <= 0.0 {
                continue;
            }
            let iz = 1.0 / qz;
            let u = (fx * qx * iz + dcx + 0.5).floor() as i64;
            let v = (fy * qy * iz + dcy + 0.5).floor() as i64;
            let q = (qz * (1.0 / MM)).round();
            if !(1.0..=65534.0).contains(&q) {
                continue;
            }
            for yy in (v - rad).max(0)..=(v + rad).min(dh as i64 - 1) {
                for xx in (u - rad).max(0)..=(u + rad).min(dw as i64 - 1) {
                    let cell = &mut out[yy as usize * dw + xx as usize];
                    *cell = (*cell).min(q as u16);
                }
            }
        }
    }
    out.iter_mut()
        .filter(|c| **c == u16::MAX)
        .for_each(|c| *c = 0);
    out
}

/// The banded scatter equals the sequential oracle when splats stay in
/// their band's window and when strong rotations (roll, pitch) and
/// near-field parallax push them out of it into the deferred path.
#[test]
fn register_matches_sequential_oracle() -> TestResult {
    let rot = |roll: f64, pitch: f64| {
        let (c, s) = (roll.cos(), roll.sin());
        let rz = Mat3F64::from_cols_array(&[c, s, 0.0, -s, c, 0.0, 0.0, 0.0, 1.0]);
        let (c, s) = (pitch.cos(), pitch.sin());
        let rx = Mat3F64::from_cols_array(&[1.0, 0.0, 0.0, 0.0, c, s, 0.0, -s, c]);
        Mat3F64::from(rz.0 * rx.0)
    };
    let poses = [
        Pose3d::new(rot(0.0, 0.0), Vec3F64::new(0.05, 0.0, 0.0)),
        Pose3d::new(rot(0.5, 0.0), Vec3F64::new(0.0, 0.0, 0.0)),
        Pose3d::new(rot(0.1, 0.25), Vec3F64::new(0.02, 0.3, 0.1)),
        Pose3d::new(rot(-0.3, -0.2), Vec3F64::new(0.0, -0.4, 0.0)),
    ];
    for &((w, h), (dw, dh)) in &[
        ((160, 130), (160, 130)),
        ((97, 211), (194, 422)),
        ((200, 70), (100, 35)),
    ] {
        let src_cam = pinhole(
            0.8 * w as f64,
            0.8 * w as f64,
            w as f64 / 2.0,
            h as f64 / 2.0,
        );
        let dst_cam = pinhole(
            0.8 * dw as f64,
            0.8 * dw as f64,
            dw as f64 / 2.0,
            dh as f64 / 2.0,
        );
        let data = random_depth_u16(w, h, (w * h) as u64);
        let src = Image::<u16, 1>::new(sz(w, h), data.clone())?;
        for pose in &poses {
            for radius in [0, 1, 3] {
                let want =
                    register_u16_oracle(&data, (w, h), &src_cam, pose, &dst_cam, radius, (dw, dh));
                let mut dst = Image::<u16, 1>::from_size_val(sz(dw, dh), 1)?;
                register_depth(&src, MM, &src_cam, pose, &dst_cam, radius, &mut dst)?;
                assert_eq!(dst.as_slice(), &want[..], "{w}x{h}->{dw}x{dh} r={radius}");
                assert!(want.iter().any(|&v| v != 0));
            }
        }
    }
    Ok(())
}

/// The NEON projection equals the scalar one bit for bit, lane by lane:
/// keys and pixels for both dtypes' keys, on random poses and depths that
/// include missing, negative, infinite, behind-camera and out-of-image
/// points, and on batch lengths that are not multiples of 4.
#[test]
fn register_projection_simd_matches_scalar() -> TestResult {
    let mut rng = Lcg(77);
    let mut splats = 0usize;
    for case in 0..40 {
        let (w, h) = (
            64 + (rng.next() % 300) as usize,
            2 + (rng.next() % 300) as usize,
        );
        let cam = pinhole(
            20.0 + 900.0 * f64::from(rng.unit()),
            20.0 + 900.0 * f64::from(rng.unit()),
            w as f64 * f64::from(rng.unit()),
            h as f64 * f64::from(rng.unit()),
        );
        let (a, b) = (f64::from(rng.unit()) - 0.5, f64::from(rng.unit()) - 0.5);
        let rot = Mat3F64::from_cols_array(&[
            a.cos(),
            0.0,
            -a.sin(),
            a.sin() * b.sin(),
            b.cos(),
            a.cos() * b.sin(),
            a.sin() * b.cos(),
            -b.sin(),
            a.cos() * b.cos(),
        ]);
        let pose = Pose3d::new(
            rot,
            Vec3F64::new(
                f64::from(rng.unit()) - 0.5,
                f64::from(rng.unit()) - 0.5,
                2.0 * f64::from(rng.unit()) - 1.0,
            ),
        );
        let radius = (rng.next() % 4) as usize;
        let p = kernels::register_params(&cam, 1.0, &pose, &cam, radius, sz(w, h), sz(w, h))?;
        let row: Vec<f32> = (0..kernels::PROJECT_BATCH)
            .map(|_| match rng.next() % 12 {
                0 => 0.0,
                1 => f32::NAN,
                2 => -1.0,
                3 => f32::INFINITY,
                4 => 1e-30,
                5 => 70_000.0,
                _ => 0.05 + 20.0 * rng.unit(),
            })
            .collect();
        let len = 1 + (rng.next() as usize % kernels::PROJECT_BATCH);
        let y = rng.next() as usize % h;
        for u16_keys in [false, true] {
            let mut simd = kernels::Batch::new();
            let mut scalar = kernels::Batch::new();
            kernels::project_batch(&row[..len], 3, y, &p, u16_keys, true, &mut simd);
            kernels::project_batch(&row[..len], 3, y, &p, u16_keys, false, &mut scalar);
            for j in 0..len {
                assert_eq!(simd.key[j], scalar.key[j], "case {case} lane {j}");
                if scalar.key[j] != 0 {
                    assert_eq!(
                        (simd.u[j], simd.v[j]),
                        (scalar.u[j], scalar.v[j]),
                        "case {case} lane {j}"
                    );
                }
            }
            splats += scalar.key[..len].iter().filter(|&&k| k != 0).count();
        }
    }
    // Not vacuous: plenty of lanes are real splats.
    assert!(splats > 200, "{splats} splats");
    Ok(())
}
