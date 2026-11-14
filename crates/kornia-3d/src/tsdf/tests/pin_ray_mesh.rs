//! Pins of the raycast march and the marching-cubes rules (contract
//! sections 12 and 13): stop rule, step rule, hit interpolation, normal
//! validity, edge vertices, vertex and triangle order, validity gates,
//! tie-breaking over submaps and the world frame.

use super::*;
use crate::tsdf::mesh::mesh_block;

const ID: tc::Rigid = tc::Rigid {
    r: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    t: [0.0; 3],
};

fn fma(a: f32, b: f32, c: f32) -> f32 {
    libm::fmaf(a, b, c)
}

fn succ(x: f32) -> f32 {
    f32::from_bits(x.to_bits() + 1)
}

fn bits3(v: [f32; 3]) -> [u32; 3] {
    [v[0].to_bits(), v[1].to_bits(), v[2].to_bits()]
}

fn witness(name: &str, differing: usize) {
    assert!(differing > 0, "{name}: no input separates it from the spec");
}

// ------------------------------------------------------------ the march ----

/// Voxels `(0, 0, k)` of a column, `None` elsewhere.
struct Column(Vec<Option<(f32, u16)>>);

impl tc::VoxelSampler for Column {
    fn sample(&self, g: [i32; 3]) -> Option<(f32, u16)> {
        if g[0] == 0 && g[1] == 0 && g[2] >= 0 {
            self.0.get(g[2] as usize).copied().flatten()
        } else {
            None
        }
    }
}

fn col(f: &[f32]) -> Column {
    Column(f.iter().map(|&v| Some((v, 1))).collect())
}

/// Unit voxels, pixel (0, 0) looks along +z from z = 0.25: sample k is voxel k.
fn ray(trunc: f32, max_steps: u32, threshold: u32, depth_max: f32) -> tc::RaycastParams {
    tc::RaycastParams {
        volume_from_camera: ID,
        camera_from_volume_r: ID.r,
        cx: 0.0,
        cy: 0.0,
        inv_fx: 1.0,
        inv_fy: 1.0,
        depth_min: 0.25,
        depth_max,
        voxel_size: 1.0,
        inv_voxel: 1.0,
        trunc,
        block_step: 1.0,
        weight_threshold: threshold,
        max_steps,
    }
}

fn march(c: &Column, p: &tc::RaycastParams) -> Option<tc::RayHit> {
    tc::raycast_pixel::<StdOps, _>(p, c, 0, 0)
}

fn z_of(c: &Column, p: &tc::RaycastParams) -> Option<u32> {
    march(c, p).map(|h| h.z.to_bits())
}

#[test]
fn tsdf_raycast_stop_rule() {
    let p = ray(1.0, 64, 1, 100.0);
    let hit = |z: f32| Some(z.to_bits());
    // + , 0 , - : the crossing is at the zero voxel (f_prev >= 0 and f < 0)
    assert_eq!(z_of(&col(&[0.5, 0.0, -0.5]), &p), hit(1.25));
    assert_eq!(
        z_of(&col(&[0.5, -0.0, -0.5]), &p),
        hit(1.25),
        "-0 is not negative"
    );
    // + , 0 , + : never negative, no hit (a `<= 0` test would stop at the 0)
    assert_eq!(march(&col(&[0.5, 0.0, 0.5, 0.5]), &p), None);
    // 0 , - : a zero start counts as non-negative
    assert_eq!(z_of(&col(&[0.0, -0.5]), &p), hit(0.25));
    // the crossing is linear between the two samples: fma(a, z - z_prev, z_prev)
    assert_eq!(z_of(&col(&[0.5, -0.5]), &p), hit(0.75));
    let h = march(&col(&[0.5, -0.5]), &p).unwrap();
    assert_eq!(
        h.vertex,
        [0.0, 0.0, 0.75],
        "[xn z, yn z, z] on the optical axis"
    );
    assert_eq!(h.point, [0.0, 0.0, 0.75]);
    // first crossing wins
    assert_eq!(z_of(&col(&[0.5, -0.5, 0.5, -0.5]), &p), hit(0.75));
    // + then - only counts between two consecutive valid samples
    let gap = Column(vec![Some((0.5, 1)), None, Some((-0.5, 1))]);
    assert_eq!(
        march(&gap, &p),
        None,
        "an unallocated gap resets the previous sample"
    );
    // a low-weight voxel resets it too (and is stepped over by one voxel)
    let thr2 = ray(1.0, 64, 2, 100.0);
    let low = Column(vec![Some((0.5, 2)), Some((0.5, 1)), Some((-0.5, 2))]);
    assert_eq!(march(&low, &thr2), None);
    let low_hit = Column(vec![Some((0.5, 2)), Some((0.5, 2)), Some((-0.5, 2))]);
    assert!(march(&low_hit, &thr2).is_some());
    let low_neg = Column(vec![Some((0.5, 2)), Some((-0.5, 1)), Some((-0.5, 2))]);
    assert_eq!(
        march(&low_neg, &thr2),
        None,
        "a weak negative voxel is not a crossing"
    );
    // step cap: the sixth sample is the crossing
    let late = col(&[0.5, 0.5, 0.5, 0.5, 0.5, -0.5]);
    assert!(march(&late, &ray(1.0, 6, 1, 100.0)).is_some());
    assert_eq!(march(&late, &ray(1.0, 5, 1, 100.0)), None);
    // depth_max is exclusive
    let two = col(&[0.5, -0.5]);
    assert_eq!(march(&two, &ray(1.0, 64, 1, 1.25)), None);
    assert!(march(&two, &ray(1.0, 64, 1, succ(1.25))).is_some());
    // the step is max(f * trunc, voxel): with f = 0.75 and trunc 4 the ray
    // jumps 3 voxels, skipping the negative voxel 1 and landing on voxel 3
    let jump = col(&[0.75, -0.5, 0.5, -0.25]);
    let p4 = ray(4.0, 64, 1, 100.0);
    assert_eq!(z_of(&jump, &p4), hit(2.5), "fma(0.75, 3, 0.25)");
    // the free-space step never goes below one voxel
    let slow = col(&[0.001, 0.001, -0.5]);
    assert_eq!(z_of(&slow, &p4), z_of(&slow, &ray(1.0, 64, 1, 100.0)));
    // a miss
    assert_eq!(march(&col(&[0.5, 0.5, 0.5]), &p), None);
    assert_eq!(march(&Column(vec![]), &p), None);
}

/// A field defined for every voxel.
struct Field<F: Fn([i32; 3]) -> Option<(f32, u16)>>(F);

impl<F: Fn([i32; 3]) -> Option<(f32, u16)>> tc::VoxelSampler for Field<F> {
    fn sample(&self, g: [i32; 3]) -> Option<(f32, u16)> {
        (self.0)(g)
    }
}

#[test]
fn tsdf_raycast_normal_rules() {
    let p = ray(1.0, 64, 1, 100.0);
    // f = (3 - (gz + 0.5)) / 8: positive in front, the gradient points to -z
    let plane =
        |s: f32| Field(move |g: [i32; 3]| Some((s * (3.0 - (g[2] as f32 + 0.5)) * 0.125, 1)));
    let normal = |f: &dyn tc::VoxelSampler, q: [f32; 3]| {
        // the generic fn wants a sized sampler: wrap the reference
        struct Dyn<'a>(&'a dyn tc::VoxelSampler);
        impl tc::VoxelSampler for Dyn<'_> {
            fn sample(&self, g: [i32; 3]) -> Option<(f32, u16)> {
                self.0.sample(g)
            }
        }
        tc::raycast_normal::<StdOps, _>(&p, &Dyn(f), q, q)
    };
    let q = [0.2, 0.3, 2.5];
    for s in [1.0f32, -1.0] {
        // either gradient direction faces the camera
        let n = normal(&plane(s), q);
        assert_eq!(n, [0.0, 0.0, -1.0], "scale {s}");
    }
    // all 8 voxels around the point must be valid: allocated and weight >= threshold
    let hole = Field(|g: [i32; 3]| (g != [0, 0, 2]).then_some((0.25, 1)));
    assert_eq!(normal(&hole, q), [0.0; 3], "an unallocated corner");
    let weak = Field(|g: [i32; 3]| Some((g[2] as f32, if g == [0, 0, 2] { 0 } else { 1 })));
    assert_eq!(normal(&weak, q), [0.0; 3], "a low-weight corner");
    let flat = Field(|_: [i32; 3]| Some((0.25, 1)));
    assert_eq!(normal(&flat, q), [0.0; 3], "no gradient");
    // the domain gate on q = p / voxel - 0.5: [-2^22, 2^22 - 1)
    let big = plane(1.0);
    assert_eq!(normal(&big, [0.0, 0.0, 1.0e7]), [0.0; 3]);
    assert_eq!(
        normal(&big, [0.0, 0.0, 4_194_303.5]),
        [0.0; 3],
        "q == 2^22 - 1"
    );
    assert_ne!(normal(&big, [0.0, 0.0, 4_194_304.0 - 0.75]), [0.0; 3]);
    assert_ne!(
        normal(&big, [0.0, 0.0, -4_194_303.5]),
        [0.0; 3],
        "q == -2^22"
    );
    assert_eq!(normal(&big, [0.0, 0.0, -4_194_304.0]), [0.0; 3]);
    assert_eq!(normal(&big, [f32::NAN, 0.0, 0.0]), [0.0; 3]);
}

#[test]
fn tsdf_trilinear_gradient_op_order() {
    let mut s = 21u64;
    let (mut unfused, mut weighted) = (0, 0);
    let mut u = |lo: f32, hi: f32| lo + (hi - lo) * ((rng(&mut s) >> 40) as f32 / 16_777_216.0);
    for _ in 0..50_000 {
        let f: [f32; 8] = std::array::from_fn(|_| u(-1.0, 1.0));
        let al = [u(0.0, 1.0), u(0.0, 1.0), u(0.0, 1.0)];
        // section 13: lerp(a, b, t) = fma(t, b - a, a) on differences along each axis
        let lerp = |a: f32, b: f32, t: f32| fma(t, b - a, a);
        let (f000, f100, f010, f110, f001, f101, f011, f111) =
            (f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7]);
        let want = [
            lerp(
                lerp(f100 - f000, f110 - f010, al[1]),
                lerp(f101 - f001, f111 - f011, al[1]),
                al[2],
            ),
            lerp(
                lerp(f010 - f000, f110 - f100, al[0]),
                lerp(f011 - f001, f111 - f101, al[0]),
                al[2],
            ),
            lerp(
                lerp(f001 - f000, f101 - f100, al[0]),
                lerp(f011 - f010, f111 - f110, al[0]),
                al[1],
            ),
        ];
        assert_eq!(bits3(tc::trilinear_gradient::<StdOps>(&f, al)), bits3(want));
        let plain = |a: f32, b: f32, t: f32| a + t * (b - a);
        let w0 = plain(
            plain(f100 - f000, f110 - f010, al[1]),
            plain(f101 - f001, f111 - f011, al[1]),
            al[2],
        );
        unfused += (w0.to_bits() != want[0].to_bits()) as usize;
        let blend = |a: f32, b: f32, t: f32| (1.0 - t) * a + t * b;
        let w1 = blend(
            blend(f100 - f000, f110 - f010, al[1]),
            blend(f101 - f001, f111 - f011, al[1]),
            al[2],
        );
        weighted += (w1.to_bits() != want[0].to_bits()) as usize;
    }
    witness("unfused lerp", unfused);
    witness("(1 - t) a + t b", weighted);
}

#[test]
fn tsdf_raycast_params_and_outputs() -> R {
    // max_steps = ceil((max - min) / voxel) + 2, in f64 on the host
    let l = TsdfLayout::new(0.02, 4.0)?;
    let spec = (((4.0f32 as f64 - 0.1f32 as f64) / 0.02f32 as f64).ceil() + 2.0) as u32;
    let p = RaycastParams::new(&l, 0.1, 4.0)?;
    assert_eq!(
        (p.max_steps, p.weight_threshold, p.colors),
        (spec, 1, false)
    );
    assert_eq!(spec, 198);
    // the span is divided in f64: an f32 span gives another count for some ranges
    let mut differs = 0;
    for k in 0..2000 {
        let (lo, hi) = (0.05 + 0.001 * (k % 7) as f32, 1.0 + 0.0137 * k as f32);
        let want = (((hi as f64 - lo as f64) / 0.02f32 as f64).ceil() + 2.0) as u32;
        assert_eq!(RaycastParams::new(&l, lo, hi)?.max_steps, want);
        differs += ((((hi - lo) / 0.02f32).ceil() + 2.0) as u32 != want) as usize;
    }
    witness("f32 span", differs);
    assert!(RaycastParams::new(&l, 0.0, 1.0).is_ok());
    for (a, b) in [
        (-0.1, 1.0),
        (1.0, 1.0),
        (1.0, 0.5),
        (f32::NAN, 1.0),
        (0.1, f32::NAN),
        (0.1, f32::INFINITY),
    ] {
        assert!(
            matches!(
                RaycastParams::new(&l, a, b),
                Err(TsdfError::InvalidParameter { .. })
            ),
            "{a} {b}"
        );
    }
    let cam = pinhole(32, 24, 30.0, (0.1, 4.0));
    let RangeSensor::Pinhole(ps) = &cam.sensor else {
        unreachable!()
    };
    let vol = volume::<8>(256);
    let mut maps = RaycastMaps::new(cam.sensor.size(), false)?;
    let mut ws = TsdfWorkspace::default();
    for bad in [
        RaycastParams {
            weight_threshold: 0,
            ..p
        },
        RaycastParams { max_steps: 0, ..p },
        RaycastParams { colors: true, ..p },
    ] {
        let e = vol
            .raycast_into(&mut ws, ps.camera(), &Pose3d::IDENTITY, &bad, &mut maps)
            .unwrap_err();
        assert!(matches!(e, TsdfError::InvalidParameter { .. }), "{e}");
    }
    assert!(RaycastMaps::new(
        ImageSize {
            width: 0,
            height: 4
        },
        false
    )
    .is_err());
    // an invalid camera pose and a distorted camera
    let e = vol
        .raycast_into(&mut ws, ps.camera(), &mirror_pose(), &p, &mut maps)
        .unwrap_err();
    assert!(matches!(e, TsdfError::InvalidTransform));
    let distorted = PinholeCamera {
        k1: 0.1,
        ..ps.camera().clone()
    };
    let e = vol
        .raycast_into(&mut ws, &distorted, &Pose3d::IDENTITY, &p, &mut maps)
        .unwrap_err();
    assert!(matches!(e, TsdfError::DistortedCamera));
    Ok(())
}

fn mirror_pose() -> Pose3d {
    Pose3d::new(
        Mat3F64::from_cols_array(&[-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
        Vec3F64::ZERO,
    )
}

#[test]
fn tsdf_raycast_miss_and_colour_outputs() -> R {
    let cam = pinhole(32, 24, 30.0, (0.1, 4.0));
    let sz = cam.sensor.size();
    let RangeSensor::Pinhole(ps) = &cam.sensor else {
        unreachable!()
    };
    let d = DepthImage::from_size_val(sz, 1.0f32, 1.0)?;
    let colour = Image::<u8, 3>::new(sz, [10u8, 20, 30].repeat(sz.width * sz.height))?;
    let mut vol = volume::<8>(2048);
    let mut ws = TsdfWorkspace::default();
    vol.integrate(
        &mut ws,
        &[SensorObservation {
            color: Some(&colour),
            ..obs(&cam, Pose3d::IDENTITY, &d)
        }],
    )?;
    let p = RaycastParams {
        colors: true,
        ..RaycastParams::new(vol.layout(), 0.1, 4.0)?
    };
    let mut maps = RaycastMaps::new(sz, true)?;
    vol.raycast_into(&mut ws, ps.camera(), &Pose3d::IDENTITY, &p, &mut maps)?;
    let hits: Vec<usize> = (0..sz.width * sz.height)
        .filter(|&i| maps.depth().as_slice()[i] > 0.0)
        .collect();
    assert!(
        hits.len() * 10 >= sz.width * sz.height * 8,
        "{} hits",
        hits.len()
    );
    let c = maps.colors().unwrap().as_slice();
    // the colour of the voxel at the hit, or 0 where that voxel was never observed
    let coloured = hits
        .iter()
        .filter(|&&i| c[3 * i..3 * i + 3] == [10, 20, 30])
        .count();
    assert!(hits
        .iter()
        .all(|&i| c[3 * i..3 * i + 3] == [10, 20, 30] || c[3 * i..3 * i + 3] == [0, 0, 0]));
    assert!(
        coloured * 10 >= hits.len() * 9,
        "{coloured} of {}",
        hits.len()
    );
    // a miss writes zeros over everything the buffers held, colour included
    let away = Pose3d::new(mat(0.0, std::f64::consts::PI, 0.0), Vec3F64::ZERO);
    vol.raycast_into(&mut ws, ps.camera(), &away, &p, &mut maps)?;
    assert!(maps.depth().as_slice().iter().all(|&v| v == 0.0));
    assert!(maps.vertices().as_slice().iter().all(|&v| v == 0.0));
    assert!(maps.normals().as_slice().iter().all(|&v| v == 0.0));
    assert!(maps.colors().unwrap().as_slice().iter().all(|&v| v == 0));
    // colour buffers are cleared when the call does not ask for colour
    vol.raycast_into(&mut ws, ps.camera(), &Pose3d::IDENTITY, &p, &mut maps)?;
    assert!(maps.colors().unwrap().as_slice().iter().any(|&v| v != 0));
    let no_colour = RaycastParams { colors: false, ..p };
    vol.raycast_into(
        &mut ws,
        ps.camera(),
        &Pose3d::IDENTITY,
        &no_colour,
        &mut maps,
    )?;
    assert!(maps.colors().unwrap().as_slice().iter().all(|&v| v == 0));
    // asking for colour needs a colour map and colour storage
    let mut bare = RaycastMaps::new(sz, false)?;
    let e = vol
        .raycast_into(&mut ws, ps.camera(), &Pose3d::IDENTITY, &p, &mut bare)
        .unwrap_err();
    assert!(matches!(
        e,
        TsdfError::InvalidParameter {
            field: "colors",
            ..
        }
    ));
    let plain = TsdfVolume::<8>::new(
        TsdfLayout::new(VOXEL, 4.0)?,
        TsdfParams {
            color: false,
            ..params(64)
        },
    )?;
    let e = plain
        .raycast_into(&mut ws, ps.camera(), &Pose3d::IDENTITY, &p, &mut maps)
        .unwrap_err();
    assert!(matches!(
        e,
        TsdfError::InvalidParameter {
            field: "colors",
            ..
        }
    ));
    Ok(())
}

#[test]
fn tsdf_raycast_multi_submap_tie_is_lower_id() -> R {
    let size = ImageSize {
        width: 48,
        height: 36,
    };
    let cam = pinhole(48, 36, 45.0, (0.1, 4.0));
    let RangeSensor::Pinhole(ps) = &cam.sensor else {
        unreachable!()
    };
    let d = DepthImage::from_size_val(size, 1.25f32, 1.0)?;
    let colour_of =
        |c: [u8; 3]| Image::<u8, 3>::new(size, c.repeat(size.width * size.height)).unwrap();
    let (red, blue) = (colour_of([200, 10, 10]), colour_of([10, 10, 200]));
    let centre = (size.height / 2) * size.width + size.width / 2;
    let p = RaycastParams {
        colors: true,
        ..RaycastParams::new(&submap_params(true).layout, 0.1, 4.0)?
    };
    // identical geometry, different colours; integrate in either order
    for order in [[0u32, 1], [1, 0]] {
        let mut map = SubmapTsdf::<8>::new(submap_params(true))?;
        for a in [0, 1] {
            map.add_agent(
                AgentId(a),
                rig(&vec![(
                    SensorId(0),
                    pinhole(48, 36, 45.0, (0.1, 4.0)),
                    Pose3d::IDENTITY,
                )]),
                SpawnPolicy::default(),
            )?;
        }
        let mut ws = TsdfWorkspace::default();
        for (n, a) in order.iter().enumerate() {
            let f = [SensorFrame {
                sensor: SensorId(0),
                depth: (&d).into(),
                color: Some(if *a == 0 { &red } else { &blue }),
            }];
            map.integrate(
                &mut ws,
                &AgentStep {
                    agent: AgentId(*a),
                    stamp: n as u64 + 1,
                    world_from_body: Pose3d::IDENTITY,
                    frames: &f,
                },
            )?;
        }
        let mut cast = |filter: SubmapFilter<'_>| -> Result<(f32, [u8; 3]), TsdfError> {
            let mut maps = RaycastMaps::new(size, true)?;
            map.raycast_into(
                &mut ws,
                ps.camera(),
                &Pose3d::IDENTITY,
                filter,
                &p,
                &mut maps,
            )?;
            let c = maps.colors().unwrap().as_slice();
            Ok((
                maps.depth().as_slice()[centre],
                [c[3 * centre], c[3 * centre + 1], c[3 * centre + 2]],
            ))
        };
        let (z0, c0) = cast(SubmapFilter::Agents(&[AgentId(0)]))?;
        let (z1, c1) = cast(SubmapFilter::Agents(&[AgentId(1)]))?;
        assert!(
            z0 > 1.0 && z0.to_bits() == z1.to_bits(),
            "an exact tie: {z0} {z1}"
        );
        assert_eq!((c0, c1), ([200, 10, 10], [10, 10, 200]));
        let (z, c) = cast(SubmapFilter::All)?;
        assert_eq!(
            (z.to_bits(), c),
            (z0.to_bits(), [200, 10, 10]),
            "the lower id wins a tie ({order:?})"
        );
        let (_, c) = cast(SubmapFilter::Submaps(&[SubmapId {
            agent: AgentId(1),
            seq: 0,
        }]))?;
        assert_eq!(c, [10, 10, 200]);
    }
    Ok(())
}

// ------------------------------------------------------- marching cubes ----

/// One block (and optionally more) where every voxel is `+0.5` except `at`.
fn octa(keys: &[[i32; 3]], at: [i32; 3], inside: f32) -> Result<TsdfVolume<8>, TsdfError> {
    let mut vol = volume::<8>(64);
    let layout = *vol.layout();
    let mut k: Vec<u64> = keys
        .iter()
        .map(|c| BlockKey::new(c[0], c[1], c[2]).map(BlockKey::bits))
        .collect::<Result<_, _>>()?;
    vol.fill_for_test(&mut k, |c| {
        let g = layout.voxel_from_point(c)?;
        Some(if g == at { inside } else { 0.5 })
    })?;
    Ok(vol)
}

fn mesh_and_points(
    vol: &TsdfVolume<8>,
    mp: &MeshParams,
) -> Result<(TriangleMesh, Vec<[f32; 3]>), TsdfError> {
    let mut ws = TsdfWorkspace::default();
    let mut mesh = TriangleMesh::new();
    vol.extract_mesh_into(&mut ws, mp, &mut mesh)?;
    let mut pts = Vec::new();
    vol.extract_point_cloud_into(&mut ws, mp, &mut pts, None, None)?;
    Ok((mesh, pts))
}

#[test]
fn tsdf_mc_zero_corner_is_outside() -> R {
    let mp = MeshParams::default();
    let at = [3, 3, 3];
    // a corner exactly at 0 (either sign) is not inside: no surface at all
    for zero in [0.0f32, -0.0] {
        let (m, p) = mesh_and_points(&octa(&[[0, 0, 0]], at, zero)?, &mp)?;
        assert!(
            m.triangles.is_empty() && m.vertices.is_empty() && p.is_empty(),
            "{zero:?}"
        );
    }
    // the smallest negative value is: an octahedron, 6 edges and 8 cubes
    let (m, p) = mesh_and_points(&octa(&[[0, 0, 0]], at, -f32::MIN_POSITIVE)?, &mp)?;
    assert_eq!((m.vertices.len(), m.triangles.len(), p.len()), (6, 8, 6));
    Ok(())
}

#[test]
fn tsdf_mc_vertex_and_triangle_order() -> R {
    let at = [3, 3, 3];
    let vol = octa(&[[0, 0, 0]], at, -0.25)?;
    let (mesh, pts) = mesh_and_points(&vol, &MeshParams::default())?;
    assert_eq!(
        (
            mesh.vertices.len(),
            mesh.normals.len(),
            mesh.triangles.len()
        ),
        (6, 6, 8)
    );
    let l = *vol.layout();
    let f = |g: [i32; 3]| if g == at { -0.25f32 } else { 0.5 };
    // vertices ascend by e = 3 (ox + 9 (oy + 9 oz)) + axis, one per crossed edge
    let edges: [([i32; 3], usize); 6] = [
        ([3, 3, 2], 2),
        ([3, 2, 3], 1),
        ([2, 3, 3], 0),
        ([3, 3, 3], 0),
        ([3, 3, 3], 1),
        ([3, 3, 3], 2),
    ];
    let e_of = |o: [i32; 3], k: usize| 3 * (o[0] + 9 * (o[1] + 9 * o[2])) + k as i32;
    assert!(edges
        .windows(2)
        .all(|w| e_of(w[0].0, w[0].1) < e_of(w[1].0, w[1].1)));
    for (i, (o, k)) in edges.iter().enumerate() {
        let mut b = *o;
        b[*k] += 1;
        let (fa, fb) = (f(*o), f(b));
        let t = fa / (fa - fb);
        let ca = l.centre_from_voxel(*o);
        let mut want = ca;
        want[*k] = fma(t, l.voxel_size(), ca[*k]);
        assert_eq!(bits3(mesh.vertices[i]), bits3(want), "vertex {i}");
    }
    // triangles follow the cubes in ascending lx + 8 (ly + 8 lz), here one each
    let mut j = 0;
    let c_in = l.centre_from_voxel(at).map(f64::from);
    for lz in 2..=3 {
        for ly in 2..=3 {
            for lx in 2..=3 {
                let (lo, hi) = (
                    l.centre_from_voxel([lx, ly, lz]),
                    l.centre_from_voxel([lx + 1, ly + 1, lz + 1]),
                );
                let t = mesh.triangles[j];
                let v = |i: usize| mesh.vertices[t[i] as usize];
                for i in 0..3 {
                    for a in 0..3 {
                        assert!(
                            lo[a] <= v(i)[a] && v(i)[a] <= hi[a],
                            "triangle {j} is not in cube {:?}",
                            [lx, ly, lz]
                        );
                    }
                }
                // counter-clockwise seen from free space: away from the inside voxel
                let d = |a: [f32; 3], b: [f32; 3]| {
                    [
                        (b[0] - a[0]) as f64,
                        (b[1] - a[1]) as f64,
                        (b[2] - a[2]) as f64,
                    ]
                };
                let (e1, e2) = (d(v(0), v(1)), d(v(0), v(2)));
                let cr = [
                    e1[1] * e2[2] - e1[2] * e2[1],
                    e1[2] * e2[0] - e1[0] * e2[2],
                    e1[0] * e2[1] - e1[1] * e2[0],
                ];
                let out = [
                    v(0)[0] as f64 - c_in[0],
                    v(0)[1] as f64 - c_in[1],
                    v(0)[2] as f64 - c_in[2],
                ];
                assert!(
                    cr[0] * out[0] + cr[1] * out[1] + cr[2] * out[2] > 0.0,
                    "winding of triangle {j}"
                );
                j += 1;
            }
        }
    }
    // vertex normals are unit and point toward +tsdf (away from the inside voxel)
    for (v, n) in mesh.vertices.iter().zip(&mesh.normals) {
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        assert!((len - 1.0).abs() < 1e-6, "{n:?}");
        let out = [
            v[0] as f64 - c_in[0],
            v[1] as f64 - c_in[1],
            v[2] as f64 - c_in[2],
        ];
        assert!(n[0] as f64 * out[0] + n[1] as f64 * out[1] + n[2] as f64 * out[2] > 0.0);
    }
    // the point cloud holds the same 6 edges, in the same ascending order
    assert_eq!(
        pts.iter().map(|p| bits3(*p)).collect::<Vec<_>>(),
        mesh.vertices.iter().map(|p| bits3(*p)).collect::<Vec<_>>()
    );
    Ok(())
}

#[test]
fn tsdf_mc_validity_gates() -> R {
    let at = [3, 3, 3];
    let tri = |vol: &TsdfVolume<8>, thr: u16| -> Result<usize, TsdfError> {
        Ok(mesh_and_points(
            vol,
            &MeshParams {
                weight_threshold: thr,
                ..MeshParams::default()
            },
        )?
        .0
        .triangles
        .len())
    };
    // weight >= threshold: a voxel at the threshold is valid
    let mut vol = octa(&[[0, 0, 0]], at, -0.25)?;
    vol.pools.weight.fill(2);
    assert_eq!((tri(&vol, 1)?, tri(&vol, 2)?, tri(&vol, 3)?), (8, 8, 0));
    // one weak neighbour drops the 4 cubes it shares with the inside voxel
    let key = BlockKey::new(0, 0, 0)?.bits();
    let base = vol.view().block_base(key).unwrap();
    vol.pools.weight[base + tc::local_index([4, 3, 3], 3)] = 1;
    assert_eq!((tri(&vol, 1)?, tri(&vol, 2)?), (8, 4));
    // a cube needs its 8 corners in allocated blocks: at the +x face of the
    // block, the cubes reaching into the missing neighbour are dropped ...
    let edge = [7, 3, 3];
    assert_eq!(tri(&octa(&[[0, 0, 0]], edge, -0.25)?, 1)?, 4);
    // ... and come back once the neighbour exists (they belong to block 0)
    assert_eq!(tri(&octa(&[[0, 0, 0], [1, 0, 0]], edge, -0.25)?, 1)?, 8);
    // the same cubes, from a block on the other side of the face
    assert_eq!(
        tri(&octa(&[[0, 0, 0], [-1, 0, 0]], [0, 3, 3], -0.25)?, 1)?,
        8
    );
    assert_eq!(tri(&octa(&[[0, 0, 0]], [0, 3, 3], -0.25)?, 1)?, 4);
    Ok(())
}

#[test]
fn tsdf_mc_edge_interpolation_op_order() {
    let mut s = 17u64;
    let mut u = |lo: f32, hi: f32| lo + (hi - lo) * ((rng(&mut s) >> 40) as f32 / 16_777_216.0);
    let (mut vtx, mut nrm, mut colour_trunc) = (0, 0, 0);
    for i in 0..50_000 {
        let (fa, fb) = if i % 2 == 0 {
            (u(0.0, 1.0), -u(1e-3, 1.0))
        } else {
            (-u(1e-3, 1.0), u(0.0, 1.0))
        };
        let t = tc::edge_t(fa, fb);
        assert_eq!(t.to_bits(), (fa / (fa - fb)).to_bits());
        assert!((0.0..=1.0).contains(&t));
        // the vertex: the owner centre moved by t * voxel along one axis, one fma
        let (ca, vs, k) = (
            [u(-9.0, 9.0), u(-9.0, 9.0), u(-9.0, 9.0)],
            u(0.01, 0.3),
            i % 3,
        );
        let got = tc::edge_vertex::<StdOps>(ca, k, vs, t);
        let mut want = ca;
        want[k] = fma(t, vs, ca[k]);
        assert_eq!(bits3(got), bits3(want));
        let mut plain = ca;
        plain[k] = ca[k] + t * vs;
        vtx += (bits3(plain) != bits3(want)) as usize;
        // the normal: fma lerp of the end gradients, then 1 / sqrt, plain products
        let (ga, gb) = (
            [u(-1.0, 1.0), u(-1.0, 1.0), u(-1.0, 1.0)],
            [u(-1.0, 1.0), u(-1.0, 1.0), u(-1.0, 1.0)],
        );
        let n = [
            fma(t, gb[0] - ga[0], ga[0]),
            fma(t, gb[1] - ga[1], ga[1]),
            fma(t, gb[2] - ga[2], ga[2]),
        ];
        let l2 = fma(n[0], n[0], fma(n[1], n[1], n[2] * n[2]));
        let inv = 1.0 / libm::sqrtf(l2);
        let want_n = [n[0] * inv, n[1] * inv, n[2] * inv];
        assert_eq!(bits3(tc::edge_normal::<StdOps>(t, ga, gb)), bits3(want_n));
        let pl = [
            ga[0] + t * (gb[0] - ga[0]),
            ga[1] + t * (gb[1] - ga[1]),
            ga[2] + t * (gb[2] - ga[2]),
        ];
        let pl_l2 = pl[0] * pl[0] + pl[1] * pl[1] + pl[2] * pl[2];
        let pl_inv = 1.0 / libm::sqrtf(pl_l2);
        nrm += (bits3([pl[0] * pl_inv, pl[1] * pl_inv, pl[2] * pl_inv]) != bits3(want_n)) as usize;
        // the colour: fma lerp, rounded half up
        let (ra, rb) = (
            [(i % 251) as u8, (i % 7) as u8, 255],
            [(i % 13) as u8, (i % 241) as u8, (i % 5) as u8],
        );
        let c = tc::lerp_color::<StdOps>(t, ra, rb);
        for ch in 0..3 {
            let v = fma(t, rb[ch] as f32 - ra[ch] as f32, ra[ch] as f32) + 0.5;
            let want = if v >= 255.0 { 255 } else { v as u32 as u8 };
            assert_eq!(c[ch], want);
            colour_trunc += ((v - 0.5) as u32 as u8 != want) as usize;
        }
    }
    witness("unfused edge vertex", vtx);
    witness("unfused edge normal", nrm);
    witness("truncated colour", colour_trunc);
    // a zero gradient gives the zero normal, never NaN
    assert_eq!(tc::edge_normal::<StdOps>(0.5, [0.0; 3], [0.0; 3]), [0.0; 3]);
}

#[test]
fn tsdf_mc_gradient_fallbacks() {
    // central difference, else one-sided toward the valid neighbour, else 0
    assert_eq!(tc::grad_axis(Some(5.0), 2.0, Some(1.0)), 4.0);
    assert_eq!(tc::grad_axis(Some(5.0), 2.0, None), 3.0);
    assert_eq!(tc::grad_axis(None, 2.0, Some(1.0)), 1.0);
    assert_eq!(tc::grad_axis(None, 2.0, None).to_bits(), 0);
    // cube index: bit c is set for f_c < 0 (strict)
    let mut f = [0.5f32; 8];
    assert_eq!(tc::cube_index(&f), 0);
    f[0] = -0.0;
    f[3] = 0.0;
    assert_eq!(tc::cube_index(&f), 0, "zeros are outside");
    f[0] = -1e-30;
    f[7] = -0.5;
    assert_eq!(tc::cube_index(&f), 0b1000_0001);
    // edge ownership: the owner is the lower end along the axis
    assert_eq!(tc::edge_key([1, 2, 3], 1, 8), 3 * (1 + 9 * (2 + 9 * 3)) + 1);
}

#[test]
fn tsdf_mc_shared_edge_vertex() -> R {
    let (vol, _) = filled_sphere::<8>(0.3)?;
    let mut ws = TsdfWorkspace::default();
    let mut mesh = TriangleMesh::new();
    vol.extract_mesh_into(&mut ws, &MeshParams::default(), &mut mesh)?;
    // face vertices are duplicated in the neighbouring block, bit for bit
    // (position and normal), because both read the same voxels
    let mut groups: HashMap<[u32; 3], Vec<usize>> = HashMap::new();
    for (i, v) in mesh.vertices.iter().enumerate() {
        groups.entry(bits3(*v)).or_default().push(i);
    }
    let dups: Vec<&Vec<usize>> = groups.values().filter(|g| g.len() > 1).collect();
    assert!(dups.len() > 20, "{} shared vertices", dups.len());
    for g in dups {
        let n0 = bits3(mesh.normals[g[0]]);
        assert!(g.iter().all(|&i| bits3(mesh.normals[i]) == n0));
    }
    Ok(())
}

#[test]
fn tsdf_mc_winding_outward() -> R {
    let (vol, centre) = filled_sphere::<8>(0.3)?;
    let mut ws = TsdfWorkspace::default();
    let mut mesh = TriangleMesh::new();
    vol.extract_mesh_into(&mut ws, &MeshParams::default(), &mut mesh)?;
    assert!(mesh.triangles.len() > 500);
    // winding: the face normal agrees with the interpolated vertex normals
    // (toward +tsdf, i.e. out of the sphere) and with the radial direction
    for t in &mesh.triangles {
        let v = |i: usize| f64x3(mesh.vertices[t[i] as usize]);
        let (e1, e2) = (sub(v(1), v(0)), sub(v(2), v(0)));
        let cr = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ];
        let n: [f64; 3] =
            std::array::from_fn(|a| (0..3).map(|i| mesh.normals[t[i] as usize][a] as f64).sum());
        assert!(dot(cr, n) > 0.0);
        assert!(dot(cr, sub(v(0), centre)) > 0.0);
    }
    Ok(())
}

#[test]
fn tsdf_mc_block_mesh_depends_on_neighbourhood_only() -> R {
    // a block's mesh is a function of its 3 x 3 x 3 neighbourhood: changing a
    // voxel two blocks away leaves it unchanged
    let (mut vol, _) = filled_sphere::<8>(0.3)?;
    let key = BlockKey::new(1, 1, 0)?.bits();
    assert!(vol.view().block_base(key).is_some());
    let mut ws = TsdfWorkspace::default();
    let mp = MeshParams::default();
    let snap = |vol: &TsdfVolume<8>, ws: &mut TsdfWorkspace| {
        mesh_block(&vol.view(), key, &mp, ws);
        (
            ws.block_vertices
                .iter()
                .map(|v| bits3(*v))
                .collect::<Vec<_>>(),
            ws.block_normals
                .iter()
                .map(|v| bits3(*v))
                .collect::<Vec<_>>(),
            ws.block_triangles.clone(),
        )
    };
    let before = snap(&vol, &mut ws);
    assert!(!before.0.is_empty());
    let far = BlockKey::new(3, 1, 0)?.bits();
    if let Some(base) = vol.view().block_base(far) {
        vol.pools.tsdf[base] = -0.9;
    }
    assert_eq!(snap(&vol, &mut ws), before);
    // a voxel of the face-adjacent block moves a normal at the shared face
    let near = BlockKey::new(2, 1, 0)?.bits();
    let base = vol.view().block_base(near).unwrap();
    for i in 0..512 {
        vol.pools.tsdf[base + i] = -vol.pools.tsdf[base + i];
    }
    assert_ne!(snap(&vol, &mut ws), before);
    Ok(())
}

#[test]
fn tsdf_submap_mesh_world_frame_bits() -> R {
    let w = world();
    let data = steps(&w);
    let fr: Vec<Vec<SensorFrame<'_>>> = data.iter().map(frames).collect();
    let mut map = SubmapTsdf::<8>::new(submap_params(false))?;
    map.add_agent(AgentId(0), rig(&w.rig_a), SpawnPolicy::default())?;
    let mut ws = TsdfWorkspace::default();
    for i in 0..2 {
        map.integrate(&mut ws, &agent_step(&data[i], &fr[i]))?;
    }
    let sid = SubmapId {
        agent: AgentId(0),
        seq: 0,
    };
    let at = |map: &mut SubmapTsdf<8>, p: Pose3d| {
        map.set_submap_poses(&[SubmapPose {
            id: sid,
            world_from_submap: p,
        }])
    };
    let mut mesh_at = |map: &mut SubmapTsdf<8>, p: Pose3d| -> Result<TriangleMesh, TsdfError> {
        at(map, p)?;
        let mut m = TriangleMesh::new();
        map.extract_mesh_into(&mut ws, SubmapFilter::All, &MeshParams::default(), &mut m)?;
        Ok(m)
    };
    let local = mesh_at(&mut map, Pose3d::IDENTITY)?;
    assert!(local.vertices.len() > 100);
    let pz = pose(0.3, 0.2, 0.1, [1.0, -2.0, 0.5]);
    let world = mesh_at(&mut map, pz)?;
    // the world mesh is xform / rot of the local one, rounded once, no renormalisation
    let x = rigid_from_pose(&pz)?;
    assert_eq!(world.triangles, local.triangles);
    let mut renormalised = 0;
    for ((lv, ln), (wv, wn)) in local
        .vertices
        .iter()
        .zip(&local.normals)
        .zip(world.vertices.iter().zip(&world.normals))
    {
        assert_eq!(bits3(*wv), bits3(tc::xform::<StdOps>(&x.r, &x.t, *lv)));
        let rotated = tc::rot::<StdOps>(&x.r, *ln);
        assert_eq!(bits3(*wn), bits3(rotated));
        renormalised += (bits3(tc::normalize::<StdOps>(rotated)) != bits3(rotated)) as usize;
    }
    witness("renormalised world normals", renormalised);
    Ok(())
}

#[test]
fn tsdf_mesh_cache_dirty_set_covers_normal_neighbours() -> R {
    // the inside voxel (8, 3, 3) is on the -x face of block J = (1, 0, 0); its
    // vertex normals read voxel (7, 3, 3), which lives in block K = (0, 0, 0)
    let (k, j) = (
        BlockKey::new(0, 0, 0)?.bits(),
        BlockKey::new(1, 0, 0)?.bits(),
    );
    for normals in [true, false] {
        let mp = MeshParams {
            normals,
            ..MeshParams::default()
        };
        let mut vol = octa(&[[0, 0, 0], [1, 0, 0]], [8, 3, 3], -0.25)?;
        let (mut ws, mut cache) = (TsdfWorkspace::default(), BlockMeshCache::new());
        vol.update_mesh_cache(&mut ws, &mp, &mut cache)?;
        let mut before = TriangleMesh::new();
        cache.write_into(&mut before)?;
        // change K's voxel without changing any sign, as a step that touched only K
        let base = vol.view().block_base(k).unwrap();
        vol.pools.tsdf[base + tc::local_index([7, 3, 3], 3)] = 0.9;
        vol.index.epoch += 1;
        let pk = vol.index.lookup(k).unwrap() as usize;
        vol.index.modified[pk] = vol.index.epoch;
        let st = vol.update_mesh_cache(&mut ws, &mp, &mut cache)?;
        // a block's mesh reads J + {-1, 0, 1}^3 with normals, J + {0, 1}^3 without
        assert_eq!(
            st.remeshed_blocks,
            if normals { 2 } else { 1 },
            "normals {normals}"
        );
        let (mut inc, mut full) = (TriangleMesh::new(), TriangleMesh::new());
        cache.write_into(&mut inc)?;
        vol.extract_mesh_into(&mut ws, &mp, &mut full)?;
        assert_eq!(mesh_bits(&inc), mesh_bits(&full), "normals {normals}");
        assert_ne!(mesh_bits(&inc), mesh_bits(&before));
        // block J's own mesh changed only when normals are read
        let mesh_of = |vol: &TsdfVolume<8>, ws: &mut TsdfWorkspace, key: u64| {
            mesh_block(&vol.view(), key, &mp, ws);
            (
                ws.block_normals
                    .iter()
                    .map(|v| bits3(*v))
                    .collect::<Vec<_>>(),
                ws.block_vertices.len(),
            )
        };
        let (n_now, nv) = mesh_of(&vol, &mut ws, j);
        vol.pools.tsdf[base + tc::local_index([7, 3, 3], 3)] = 0.5;
        let (n_old, _) = mesh_of(&vol, &mut ws, j);
        assert!(nv > 0);
        assert_eq!(
            n_now != n_old,
            normals,
            "block J's normals depend on K only with normals on"
        );
    }
    Ok(())
}
