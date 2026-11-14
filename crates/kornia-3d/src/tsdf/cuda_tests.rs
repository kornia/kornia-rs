//! CUDA backend against the scalar CPU reference, bit for bit (TSDF
//! contract, section 14): per step the counters, then the whole state
//! (block keys, the key -> pool map, epochs, every voxel's bits), meshes
//! (full and incremental), raycast maps and errors. One test per kernel
//! entry, named `<entry>_bit_exact`. Needs a CUDA device and the real PTX
//! (`cargo run -p kornia-oxide-kernels --example build_ptx`).

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaStream};

use super::*;

fn stream() -> Arc<CudaStream> {
    CudaContext::new(0)
        .and_then(|c| c.new_stream())
        .expect("CUDA device 0")
}

fn layout() -> TsdfLayout {
    TsdfLayout::new(VOXEL, 4.0).unwrap()
}

/// A CPU (scalar) volume and its device twin with the same parameters.
struct Pair<const B: usize>
where
    BlockDim<B>: SupportedBlockDim,
{
    cpu: TsdfVolume<B>,
    gpu: CudaTsdfVolume<B>,
    ws: TsdfWorkspace,
}

impl<const B: usize> Pair<B>
where
    BlockDim<B>: SupportedBlockDim,
{
    fn new(p: TsdfParams) -> Self {
        Self {
            cpu: TsdfVolume::new(layout(), p.clone()).unwrap(),
            gpu: CudaTsdfVolume::new(layout(), p, stream()).unwrap(),
            ws: TsdfWorkspace::new(TsdfKernel::Scalar),
        }
    }

    /// One step on both; the outcome (counters or error variant) must match.
    fn step(
        &mut self,
        label: &str,
        obs: &[SensorObservation<'_>],
    ) -> Result<IntegrateStats, String> {
        let a = self.cpu.integrate(&mut self.ws, obs);
        let b = self.gpu.integrate(obs);
        match (a, b) {
            (Ok(a), Ok(b)) => {
                assert_eq!(a, b, "{label} B{B}: counters");
                Ok(a)
            }
            (Err(a), Err(b)) => {
                assert_eq!(format!("{a:?}"), format!("{b:?}"), "{label} B{B}: error");
                Err(format!("{a:?}"))
            }
            (a, b) => panic!("{label} B{B}: cpu {a:?} vs gpu {b:?}"),
        }
    }

    /// The whole device state equals the CPU's.
    fn same(&self, label: &str) {
        let mut d = TsdfVolume::<B>::new(layout(), self.cpu.params().clone()).unwrap();
        self.gpu.download_into(&mut d).unwrap();
        let (a, b) = (&self.cpu, &d);
        assert_eq!(a.num_blocks(), b.num_blocks(), "{label} B{B}: blocks");
        assert_eq!(
            a.num_blocks(),
            self.gpu.num_blocks(),
            "{label} B{B}: device blocks"
        );
        assert!(
            a.block_keys().eq(b.block_keys()),
            "{label} B{B}: sorted keys"
        );
        assert_eq!(
            a.index.pool_keys, b.index.pool_keys,
            "{label} B{B}: key -> pool"
        );
        assert_eq!(a.index.modified, b.index.modified, "{label} B{B}: epochs");
        assert_eq!(a.index.epoch, b.index.epoch, "{label} B{B}: step epoch");
        if let Some(diff) = first_voxel_diff(a, b) {
            panic!("{label} B{B}: {diff}");
        }
        assert_eq!(a.content_hash(), b.content_hash(), "{label} B{B}: hash");
        assert_eq!(
            a.content_hash(),
            self.gpu.content_hash().unwrap(),
            "{label} B{B}: device hash"
        );
        let mut keys = Vec::new();
        self.gpu.block_keys_into(&mut keys).unwrap();
        assert!(a.block_keys().eq(keys), "{label} B{B}: device keys");
    }
}

fn first_voxel_diff<const B: usize>(a: &TsdfVolume<B>, b: &TsdfVolume<B>) -> Option<String>
where
    BlockDim<B>: SupportedBlockDim,
{
    for key in a.block_keys() {
        let (va, vb) = (a.block(key)?, b.block(key)?);
        for i in 0..va.tsdf.len() {
            let rgb = |v: &BlockView<'_>| v.rgbw.map(|c| c[i]);
            if va.tsdf[i].to_bits() != vb.tsdf[i].to_bits()
                || va.weight[i] != vb.weight[i]
                || rgb(&va) != rgb(&vb)
            {
                return Some(format!(
                    "block {:?} voxel {i}: cpu ({}, {}, {:?}) gpu ({}, {}, {:?})",
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

fn u16_depth(d: &DepthImage<f32>, meter: f32) -> DepthImage<u16> {
    let data = d
        .image()
        .as_slice()
        .iter()
        .map(|&z| (z * meter).round().clamp(0.0, 65535.0) as u16)
        .collect();
    DepthImage::new(Image::new(d.size(), data).unwrap(), meter).unwrap()
}

fn with_color<'a>(o: SensorObservation<'a>, c: &'a Image<u8, 3>) -> SensorObservation<'a> {
    SensorObservation {
        color: Some(c),
        ..o
    }
}

/// Views around a sphere in front of a tilted plane.
fn views(n: usize) -> Vec<Pose3d> {
    (0..n)
        .map(|k| {
            let a = 0.15 * k as f64 - 0.2;
            pose(
                0.05 * k as f64,
                a,
                0.02 * k as f64,
                [0.3 * a, 0.05 * k as f64, -0.1 * k as f64],
            )
        })
        .collect()
}

fn scene() -> Vec<Surface> {
    vec![
        tilted_plane(),
        Surface::Sphere {
            centre: [0.05, -0.05, 0.9],
            radius: 0.2,
        },
    ]
}

/// Pinhole frames (f32 depth, u16 depth, colour) along `views`.
struct PinholeRun {
    sensor: TestSensor,
    depth: Vec<DepthImage<f32>>,
    depth_u16: Vec<DepthImage<u16>>,
    color: Vec<Image<u8, 3>>,
    poses: Vec<Pose3d>,
}

fn pinhole_run(w: usize, h: usize, n: usize) -> PinholeRun {
    let sensor = pinhole(w, h, 0.9 * w.max(h) as f64, (0.1, 4.0));
    let poses = views(n);
    let depth: Vec<_> = poses.iter().map(|p| render(&sensor, p, &scene())).collect();
    let depth_u16 = depth.iter().map(|d| u16_depth(d, 1000.0)).collect();
    let color = (0..n)
        .map(|k| color_image(sensor.sensor.size(), 7 * k))
        .collect();
    PinholeRun {
        sensor,
        depth,
        depth_u16,
        color,
        poses,
    }
}

fn pinhole_steps<const B: usize>(label: &str, run: &PinholeRun, colour: bool) -> Pair<B>
where
    BlockDim<B>: SupportedBlockDim,
{
    let mut pair = Pair::<B>::new(TsdfParams {
        color: colour,
        ..params(1 << 13)
    });
    for k in 0..run.poses.len() {
        let d: DepthRef<'_> = if k % 2 == 0 {
            (&run.depth[k]).into()
        } else {
            (&run.depth_u16[k]).into()
        };
        let o = obs_d(&run.sensor, run.poses[k], d);
        let o = if colour {
            with_color(o, &run.color[k])
        } else {
            o
        };
        pair.step(label, &[o]).unwrap();
        pair.same(&format!("{label} step {k}"));
    }
    pair
}

/// Pinhole integration (u16 and f32 depth, with and without colour), B =
/// 8 and 16, state equal after every step.
#[test]
fn tsdf_integrate_pinhole_bit_exact() {
    let run = pinhole_run(64, 48, 5);
    for colour in [true, false] {
        let p = pinhole_steps::<8>("pinhole", &run, colour);
        assert!(p.cpu.num_blocks() > 20, "vacuous scene");
        pinhole_steps::<16>("pinhole", &run, colour);
    }
}

/// Allocation of pinhole steps at edge sizes (1 x N, N x 1, odd) and with
/// samples whose band leaves the voxel domain (`out_of_range`).
#[test]
fn tsdf_touch_pinhole_bit_exact() {
    for (w, h) in [(1, 37), (41, 1), (33, 17), (7, 5)] {
        let run = pinhole_run(w, h, 3);
        pinhole_steps::<8>(&format!("touch {w}x{h}"), &run, true);
        pinhole_steps::<16>(&format!("touch {w}x{h}"), &run, false);
    }
    // a camera looking along +x at the voxel domain's edge: the left half of
    // the image sees 0.2 m (its band stays inside |g| < 2^22), the right
    // half 1 m (its band leaves it)
    let s = pinhole(16, 12, 14.0, (0.1, 4.0));
    let edge = (tc::G_F as f64 - 3.0) * VOXEL as f64;
    let p = pose(
        0.0,
        std::f64::consts::FRAC_PI_2,
        0.0,
        [edge - 0.5, 0.0, 0.0],
    );
    let size = s.sensor.size();
    let data = (0..size.width * size.height)
        .map(|i| {
            if i % size.width < size.width / 2 {
                0.2f32
            } else {
                1.0
            }
        })
        .collect();
    let d = DepthImage::new(Image::new(size, data).unwrap(), 1.0).unwrap();
    let mut pair = Pair::<8>::new(params(1 << 10));
    let st = pair.step("domain edge", &[obs(&s, p, &d)]).unwrap();
    assert!(
        st.out_of_range > 0 && st.out_of_range < st.valid_pixels && st.new_blocks > 0,
        "{st:?}"
    );
    pair.same("domain edge");
}

/// A miss set too small for the step: it overflows, regrows and the step
/// re-runs (growing and fixed pools); a fixed pool that is also too small
/// then fails with the exact block count.
#[test]
fn tsdf_touch_pinhole_miss_set_overflow_bit_exact() {
    let run = pinhole_run(48, 36, 2);
    for growth in [PoolGrowth::Geometric, PoolGrowth::Fixed] {
        let mut pair = Pair::<8>::new(TsdfParams {
            growth,
            ..params(1 << 13)
        });
        pair.gpu.shrink_miss_set(16).unwrap();
        for k in 0..2 {
            let o = obs(&run.sensor, run.poses[k], &run.depth[k]);
            let st = pair.step("miss overflow", &[o]).unwrap();
            assert!(k > 0 || st.new_blocks > 16, "{st:?}");
            pair.same(&format!("miss overflow {growth:?} step {k}"));
        }
        assert!(pair.gpu.miss_slots() > 16, "the miss set never regrew");
    }
    let mut small = Pair::<8>::new(TsdfParams {
        growth: PoolGrowth::Fixed,
        ..params(8)
    });
    small.gpu.shrink_miss_set(16).unwrap();
    let o = obs(&run.sensor, run.poses[0], &run.depth[0]);
    let e = small.step("miss overflow, pool full", &[o]).unwrap_err();
    assert!(e.contains("PoolFull"), "{e}");
    assert!(small.gpu.miss_slots() > 16, "the miss set never regrew");
    small.same("miss overflow, pool full");
}

/// LiDAR steps (f32 and u16 range) of one sensor.
fn lidar_steps<const B: usize>(label: &str, l: &TestSensor) -> Pair<B>
where
    BlockDim<B>: SupportedBlockDim,
{
    let mut pair = Pair::<B>::new(params(1 << 13));
    // the mount looks along volume +z: a wall ahead, a floor behind, a ball
    // to the side
    let scene = [
        Surface::Plane {
            n: [0.0, 0.0, -1.0],
            c: -1.5,
        },
        Surface::Plane {
            n: [0.0, 0.0, 1.0],
            c: -0.6,
        },
        Surface::Sphere {
            centre: [1.0, 0.3, 0.1],
            radius: 0.35,
        },
    ];
    for k in 0..3 {
        let m = lidar_mount([0.05 * k as f64, 0.0, 0.0]);
        let d = render(l, &m, &scene);
        let du = u16_depth(&d, 256.0);
        let o = if k == 1 {
            obs_d(l, m, (&du).into())
        } else {
            obs(l, m, &d)
        };
        pair.step(label, &[o]).unwrap();
        pair.same(&format!("{label} step {k}"));
    }
    assert!(pair.cpu.num_blocks() > 10, "{label}: vacuous scene");
    pair
}

/// A LiDAR covering 1.4 rad of azimuth (no wrap).
fn partial_lidar() -> TestSensor {
    let (w, h) = (90, 9);
    let step = 1.4 / w as f64;
    let az = AzimuthGrid {
        start: -0.7,
        step,
        wrap: false,
    };
    let size = ImageSize {
        width: w,
        height: h,
    };
    let r = DepthRange::new(0.2, 3.0).unwrap();
    let s = SphericalLidar::uniform(size, az, 0.3, -0.4, r).unwrap();
    let dirs = (0..w * h)
        .map(|i| {
            let el = 0.3 - 0.7 * ((i / w) as f64) / (h as f64 - 1.0);
            let a = -0.7 + (i % w) as f64 * step;
            [el.cos() * a.cos(), el.cos() * a.sin(), el.sin()]
        })
        .collect();
    TestSensor {
        sensor: s.into(),
        dirs,
    }
}

#[test]
fn tsdf_integrate_spherical_bit_exact() {
    let full = lidar(180, 16, 0.25, -0.4, (0.2, 3.0));
    lidar_steps::<8>("lidar", &full);
    lidar_steps::<16>("lidar", &full);
    lidar_steps::<8>("lidar partial", &partial_lidar());
}

#[test]
fn tsdf_touch_spherical_bit_exact() {
    // the smallest sensor (2 x 2) and an odd one
    for (w, h) in [(2, 2), (37, 3)] {
        let l = lidar(w, h, 0.2, -0.3, (0.2, 3.0));
        let m = lidar_mount([0.0; 3]);
        let d = render(&l, &m, &[tilted_plane()]);
        let mut pair = Pair::<8>::new(params(1 << 10));
        pair.step("lidar small", &[obs(&l, m, &d)]).unwrap();
        pair.same(&format!("lidar {w}x{h}"));
    }
    lidar_steps::<16>("lidar partial", &partial_lidar());
}

/// Steps with two cameras (one coloured) and a LiDAR in two orders, and an
/// empty step: fold order is slice order on both backends.
#[test]
fn tsdf_integrate_multi_sensor_bit_exact() {
    let run = pinhole_run(40, 30, 1);
    let cam2 = pinhole(24, 20, 20.0, (0.1, 4.0));
    let l = lidar(120, 12, 0.3, -0.3, (0.2, 3.0));
    let p2 = pose(0.1, -0.1, 0.0, [0.1, 0.0, -0.05]);
    let d2 = render(&cam2, &p2, &scene());
    let m = lidar_mount([0.0, 0.0, -0.1]);
    let dl = render(&l, &m, &scene());
    for colour in [true, false] {
        let mut pair = Pair::<8>::new(TsdfParams {
            color: colour,
            ..params(1 << 13)
        });
        let a = with_color(obs(&run.sensor, run.poses[0], &run.depth[0]), &run.color[0]);
        let b = obs(&cam2, p2, &d2);
        let c = obs(&l, m, &dl);
        pair.step("multi", &[a, b, c]).unwrap();
        pair.same("multi step 0");
        pair.step("multi", &[c, a, b]).unwrap();
        pair.same("multi step 1");
        pair.step("multi empty", &[]).unwrap();
        pair.same("multi empty step");
    }
}

/// Steps that each allocate (a camera sweeping sideways): the ranked pool
/// order and the merged sorted keys after every step.
fn sweep<const B: usize>(p: TsdfParams, steps: usize) -> Pair<B>
where
    BlockDim<B>: SupportedBlockDim,
{
    let s = pinhole(48, 36, 40.0, (0.1, 4.0));
    let mut pair = Pair::<B>::new(p);
    let mut allocating = 0;
    for k in 0..steps {
        let p = pose(0.0, 0.1 * k as f64, 0.0, [0.25 * k as f64, 0.0, 0.0]);
        let d = render(&s, &p, &scene());
        let st = pair.step("sweep", &[obs(&s, p, &d)]).unwrap();
        allocating += (k > 0 && st.new_blocks > 0) as usize;
        pair.same(&format!("sweep step {k}"));
    }
    assert!(
        allocating >= steps / 2,
        "sweep: {allocating} later steps allocate"
    );
    pair
}

#[test]
fn tsdf_rank_misses_bit_exact() {
    sweep::<8>(params(1 << 13), 6);
    sweep::<16>(params(1 << 12), 6);
}

/// A fixed pool that overflows fails like the CPU's and leaves both
/// volumes unchanged; a fixed pool that fits allocates like the CPU's.
#[test]
fn tsdf_assign_misses_bit_exact() {
    let fixed = |capacity: usize| TsdfParams {
        growth: PoolGrowth::Fixed,
        ..params(capacity)
    };
    sweep::<8>(fixed(1 << 13), 3);
    let s = pinhole(48, 36, 40.0, (0.1, 4.0));
    let mut small = Pair::<8>::new(fixed(8));
    let p = Pose3d::IDENTITY;
    let d = render(&s, &p, &scene());
    let e = small.step("pool full", &[obs(&s, p, &d)]).unwrap_err();
    assert!(e.contains("PoolFull"), "{e}");
    small.same("pool full");
}

#[test]
fn tsdf_merge_keys_bit_exact() {
    // repeated views add no block (the list is kept), a new view merges
    let run = pinhole_run(48, 36, 2);
    let mut pair = Pair::<8>::new(params(1 << 13));
    let o = obs(&run.sensor, run.poses[0], &run.depth[0]);
    for k in 0..3 {
        let st = pair.step("merge", &[o]).unwrap();
        assert_eq!(st.new_blocks == 0, k > 0, "step {k}: {st:?}");
        pair.same(&format!("merge step {k}"));
    }
    let o = obs(&run.sensor, run.poses[1], &run.depth[1]);
    pair.step("merge", &[o]).unwrap();
    pair.same("merge last");
}

/// Geometric growth from a tiny pool: the device buffers are re-laid out
/// and the hash rebuilt while the logical capacity follows the CPU.
#[test]
fn tsdf_rehash_bit_exact() {
    sweep::<8>(params(4), 5);
    sweep::<16>(params(2), 4);
    let mut pair = sweep::<8>(params(16), 2);
    pair.cpu.reserve(5000).unwrap();
    pair.gpu.reserve(5000).unwrap();
    pair.same("reserve");
    let s = pinhole(48, 36, 40.0, (0.1, 4.0));
    let p = pose(0.0, -0.3, 0.0, [-0.4, 0.0, 0.0]);
    let d = render(&s, &p, &scene());
    pair.step("after reserve", &[obs(&s, p, &d)]).unwrap();
    pair.same("after reserve");
}

fn sentinel_maps(size: ImageSize, colors: bool) -> RaycastMaps {
    let mut m = RaycastMaps::new(size, colors).unwrap();
    m.depth.as_slice_mut().fill(f32::from_bits(0x7fc0_dead));
    m.vertices.as_slice_mut().fill(f32::from_bits(0x7fc0_beef));
    m.normals.as_slice_mut().fill(-1.0);
    if let Some(c) = m.colors.as_mut() {
        c.as_slice_mut().fill(0xa5);
    }
    m
}

type MapBits = (Vec<u32>, Vec<u32>, Vec<u32>, Option<Vec<u8>>);

fn maps_bits(m: &RaycastMaps) -> MapBits {
    let f = |s: &[f32]| s.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    (
        f(m.depth().as_slice()),
        f(m.vertices().as_slice()),
        f(m.normals().as_slice()),
        m.colors().map(|c| c.as_slice().to_vec()),
    )
}

fn bits(s: &[f32]) -> Vec<u32> {
    s.iter().map(|x| x.to_bits()).collect()
}

fn raycast_same<const B: usize>(pair: &mut Pair<B>, label: &str)
where
    BlockDim<B>: SupportedBlockDim,
{
    let colors = pair.cpu.params().color;
    for (k, (w, h)) in [(64, 48), (1, 9), (9, 1), (31, 17)].into_iter().enumerate() {
        let f = 0.9 * w.max(h) as f64;
        let cam = PinholeCamera {
            fx: f,
            fy: f,
            cx: (w as f64 - 1.0) / 2.0,
            cy: (h as f64 - 1.0) / 2.0,
            ..PinholeCamera::IDENTITY
        };
        let size = ImageSize {
            width: w,
            height: h,
        };
        let v = pose(0.03 * k as f64, 0.1, 0.0, [0.02 * k as f64, 0.0, -0.05]);
        for threshold in [1, 2] {
            let mut rp = RaycastParams::new(pair.cpu.layout(), 0.1, 3.0).unwrap();
            rp.weight_threshold = threshold;
            rp.colors = colors;
            let (mut a, mut b) = (sentinel_maps(size, colors), sentinel_maps(size, colors));
            pair.cpu
                .raycast_into(&mut pair.ws, &cam, &v, &rp, &mut a)
                .unwrap();
            pair.gpu.raycast_into(&cam, &v, &rp, &mut b).unwrap();
            assert_eq!(
                maps_bits(&a),
                maps_bits(&b),
                "{label} B{B} {w}x{h} threshold {threshold}"
            );
            let hits = a.depth().as_slice().iter().filter(|&&z| z > 0.0).count();
            assert!(
                w * h < 64 || threshold > 1 || hits > 0,
                "{label}: vacuous raycast"
            );
        }
    }
    // ICP pyramid: vertices and normals at each level's camera
    let cam = PinholeCamera {
        fx: 50.0,
        fy: 50.0,
        cx: 31.5,
        cy: 23.5,
        ..PinholeCamera::IDENTITY
    };
    let size = ImageSize {
        width: 64,
        height: 48,
    };
    let rp = RaycastParams::new(pair.cpu.layout(), 0.1, 3.0).unwrap();
    let mut pa = RgbdPyramid::zeroed(&cam, size, 3).unwrap();
    let mut pb = RgbdPyramid::zeroed(&cam, size, 3).unwrap();
    let v = pose(0.0, 0.05, 0.0, [0.0; 3]);
    pair.cpu
        .raycast_pyramid_into(&mut pair.ws, &v, &rp, &mut pa)
        .unwrap();
    pair.gpu.raycast_pyramid_into(&v, &rp, &mut pb).unwrap();
    for (la, lb) in pa.levels.iter().zip(pb.levels.iter()) {
        assert_eq!(
            bits(la.vertices.as_slice()),
            bits(lb.vertices.as_slice()),
            "{label}: level vertices"
        );
        assert_eq!(
            bits(la.normals.as_slice()),
            bits(lb.normals.as_slice()),
            "{label}: level normals"
        );
    }
}

#[test]
fn tsdf_raycast_bit_exact() {
    let run = pinhole_run(64, 48, 4);
    for colour in [true, false] {
        let mut p = pinhole_steps::<8>("raycast", &run, colour);
        raycast_same(&mut p, "raycast");
        let mut p = pinhole_steps::<16>("raycast", &run, colour);
        raycast_same(&mut p, "raycast");
    }
    // the block-skip back-off of contract v1.1: a fronto-parallel plane at
    // 1.2 m seen from 0.1 m
    let s = pinhole(32, 24, 30.0, (0.1, 4.0));
    let plane = Surface::Plane {
        n: [0.0, 0.0, -1.0],
        c: -1.2,
    };
    let d = render(&s, &Pose3d::IDENTITY, &[plane]);
    let mut pair = Pair::<8>::new(params(1 << 12));
    pair.step("back-off", &[obs(&s, Pose3d::IDENTITY, &d)])
        .unwrap();
    raycast_same(&mut pair, "back-off");
    // the reproducer's own view: every ray of the integrated camera hits
    let RangeSensor::Pinhole(ph) = &s.sensor else {
        unreachable!()
    };
    let rp = RaycastParams::new(pair.cpu.layout(), 0.1, 3.0).unwrap();
    let size = ph.size();
    let (mut a, mut b) = (sentinel_maps(size, false), sentinel_maps(size, false));
    let cam = ph.camera().clone();
    pair.cpu
        .raycast_into(&mut pair.ws, &cam, &Pose3d::IDENTITY, &rp, &mut a)
        .unwrap();
    pair.gpu
        .raycast_into(&cam, &Pose3d::IDENTITY, &rp, &mut b)
        .unwrap();
    assert_eq!(maps_bits(&a), maps_bits(&b), "back-off view");
    let centre = a.depth().as_slice()[12 * 32 + 16];
    assert!((centre - 1.2).abs() < 0.02, "back-off centre ray: {centre}");
}

/// Raycast errors match the CPU's.
#[test]
fn tsdf_raycast_rejects_bad_input_like_cpu() {
    let run = pinhole_run(32, 24, 1);
    let mut pair = pinhole_steps::<8>("bad raycast", &run, false);
    let cam = PinholeCamera {
        fx: 30.0,
        fy: 30.0,
        cx: 15.5,
        cy: 11.5,
        ..PinholeCamera::IDENTITY
    };
    let size = ImageSize {
        width: 32,
        height: 24,
    };
    let good = RaycastParams::new(pair.cpu.layout(), 0.1, 3.0).unwrap();
    let scaled = Pose3d::new(
        kornia_algebra::Mat3F64::from_diagonal(Vec3F64::new(1.0, 1.0, 2.0)),
        Vec3F64::ZERO,
    );
    let distorted = PinholeCamera {
        k1: 0.1,
        ..cam.clone()
    };
    let cases = [
        ("pose", &cam, scaled, good),
        ("distortion", &distorted, Pose3d::IDENTITY, good),
        (
            "colours without storage",
            &cam,
            Pose3d::IDENTITY,
            RaycastParams {
                colors: true,
                ..good
            },
        ),
        (
            "weight threshold",
            &cam,
            Pose3d::IDENTITY,
            RaycastParams {
                weight_threshold: 0,
                ..good
            },
        ),
    ];
    for (label, c, p, rp) in cases {
        let mut a = RaycastMaps::new(size, rp.colors).unwrap();
        let mut b = RaycastMaps::new(size, rp.colors).unwrap();
        let ea = pair.cpu.raycast_into(&mut pair.ws, c, &p, &rp, &mut a);
        let eb = pair.gpu.raycast_into(c, &p, &rp, &mut b);
        assert!(ea.is_err(), "{label}");
        assert_eq!(format!("{ea:?}"), format!("{eb:?}"), "{label}");
    }
}

fn junk_mesh() -> TriangleMesh {
    TriangleMesh {
        vertices: vec![[f32::NAN; 3]; 17],
        normals: vec![[7.0; 3]; 5],
        colors: vec![[1; 3]; 3],
        triangles: vec![[9; 3]; 11],
    }
}

fn mesh_same<const B: usize>(pair: &mut Pair<B>, label: &str)
where
    BlockDim<B>: SupportedBlockDim,
{
    let colour = pair.cpu.params().color;
    for normals in [true, false] {
        for colors in [false, colour] {
            for weight_threshold in [1, 2] {
                let mp = MeshParams {
                    weight_threshold,
                    normals,
                    colors,
                };
                let (mut a, mut b) = (junk_mesh(), junk_mesh());
                pair.cpu
                    .extract_mesh_into(&mut pair.ws, &mp, &mut a)
                    .unwrap();
                pair.gpu.extract_mesh_into(&mp, &mut b).unwrap();
                assert!(!a.triangles.is_empty(), "{label}: vacuous mesh");
                assert_eq!(mesh_bits(&a), mesh_bits(&b), "{label} B{B} {mp:?}");
            }
        }
    }
    let bad = MeshParams {
        weight_threshold: 0,
        normals: true,
        colors: false,
    };
    let mut m = TriangleMesh::new();
    let ea = pair.cpu.extract_mesh_into(&mut pair.ws, &bad, &mut m);
    let eb = pair.gpu.extract_mesh_into(&bad, &mut m);
    assert_eq!(format!("{ea:?}"), format!("{eb:?}"), "{label}: bad params");
}

#[test]
fn tsdf_mc_count_bit_exact() {
    let run = pinhole_run(64, 48, 4);
    let mut p = pinhole_steps::<8>("mesh", &run, true);
    mesh_same(&mut p, "mesh");
    let mut p = pinhole_steps::<16>("mesh", &run, false);
    mesh_same(&mut p, "mesh");
}

#[test]
fn tsdf_mc_emit_bit_exact() {
    let full = lidar(180, 16, 0.25, -0.4, (0.2, 3.0));
    let mut p = lidar_steps::<8>("mesh lidar", &full);
    mesh_same(&mut p, "mesh lidar");
    let mut p = lidar_steps::<16>("mesh lidar", &full);
    mesh_same(&mut p, "mesh lidar");
    let run = pinhole_run(48, 36, 3);
    let mut p = pinhole_steps::<16>("mesh colour", &run, true);
    mesh_same(&mut p, "mesh colour");
}

/// Incremental meshing on the device equals the CPU cache and the full
/// mesh after every step (normals on: 27 neighbours; off: 8).
#[test]
fn tsdf_mc_dirty_bit_exact() {
    let run = pinhole_run(48, 36, 5);
    for normals in [true, false] {
        let mp = MeshParams {
            weight_threshold: 1,
            normals,
            colors: true,
        };
        let mut pair = Pair::<8>::new(params(1 << 13));
        let (mut ca, mut cb) = (BlockMeshCache::new(), BlockMeshCache::new());
        for k in 0..run.poses.len() {
            let o = with_color(obs(&run.sensor, run.poses[k], &run.depth[k]), &run.color[k]);
            pair.step("dirty", &[o]).unwrap();
            let sa = pair
                .cpu
                .update_mesh_cache(&mut pair.ws, &mp, &mut ca)
                .unwrap();
            let sb = pair.gpu.update_mesh_cache(&mp, &mut cb).unwrap();
            assert_eq!(sa, sb, "dirty step {k}: stats");
            let (mut ma, mut mb, mut full) = (junk_mesh(), junk_mesh(), TriangleMesh::new());
            ca.write_into(&mut ma).unwrap();
            cb.write_into(&mut mb).unwrap();
            pair.gpu.extract_mesh_into(&mp, &mut full).unwrap();
            assert_eq!(mesh_bits(&ma), mesh_bits(&mb), "dirty step {k}: cache");
            assert_eq!(mesh_bits(&mb), mesh_bits(&full), "dirty step {k}: full");
            // no new step: nothing to re-mesh
            let again = pair.gpu.update_mesh_cache(&mp, &mut cb).unwrap();
            assert_eq!(again.remeshed_blocks, 0, "dirty step {k}: idle update");
        }
    }
}

/// Invalid steps fail with the CPU's error and change neither volume.
#[test]
fn tsdf_integrate_rejects_bad_input_like_cpu() {
    let run = pinhole_run(32, 24, 2);
    let mut pair = pinhole_steps::<8>("bad input", &run, true);
    let l = lidar(16, 4, 0.2, -0.2, (0.2, 3.0));
    let m = lidar_mount([0.0; 3]);
    let dl = render(&l, &m, &[tilted_plane()]);
    let five = ImageSize {
        width: 5,
        height: 5,
    };
    let wrong = DepthImage::new(Image::from_size_val(five, 1.0f32).unwrap(), 1.0).unwrap();
    let nan_pose = Pose3d::new(
        kornia_algebra::Mat3F64::IDENTITY,
        Vec3F64::new(f64::NAN, 0.0, 0.0),
    );
    let tiny_color = color_image(five, 0);
    let good = obs(&run.sensor, run.poses[0], &run.depth[0]);
    let cases = [
        ("size", obs(&run.sensor, run.poses[0], &wrong)),
        ("pose", obs(&run.sensor, nan_pose, &run.depth[0])),
        (
            "colour on lidar",
            with_color(obs(&l, m, &dl), &run.color[0]),
        ),
        ("colour size", with_color(good, &tiny_color)),
    ];
    for (label, bad) in cases {
        let e = pair.step(label, &[good, bad]).unwrap_err();
        assert!(!e.is_empty(), "{label}");
        pair.same(label);
    }
}

/// 20 Replica room0 frames (2 cm, colour on), B = 8 and 16: state, mesh
/// and raycast. Set `KORNIA_TSDF_REPLICA` to the Replica root; release.
#[test]
fn tsdf_integrate_pinhole_bit_exact_replica() {
    let env = super::neon::REPLICA_ENV;
    let Some(root) = std::env::var_os(env).map(std::path::PathBuf::from) else {
        eprintln!("skipped: {env} is not set");
        return;
    };
    let size = ImageSize {
        width: 1200,
        height: 680,
    };
    let camera = PinholeCamera {
        fx: 600.0,
        fy: 600.0,
        cx: 599.5,
        cy: 339.5,
        ..PinholeCamera::IDENTITY
    };
    let range = DepthRange::new(0.1, 7.0).unwrap();
    let sensor = RangeSensor::from(PinholeSensor::new(&camera, size, range).unwrap());
    let frames = super::neon::replica_frames(&root, size, 6553.5).unwrap();
    replica_run::<8>(&sensor, &camera, &frames);
    replica_run::<16>(&sensor, &camera, &frames);
}

fn replica_run<const B: usize>(
    sensor: &RangeSensor,
    camera: &PinholeCamera,
    frames: &[super::neon::ReplicaFrame],
) where
    BlockDim<B>: SupportedBlockDim,
{
    let mut pair = Pair::<B>::new(params(1 << 16));
    let mut updated = 0;
    for (k, f) in frames.iter().enumerate() {
        let o = SensorObservation {
            sensor,
            volume_from_sensor: f.world_from_camera,
            depth: (&f.depth).into(),
            color: Some(&f.color),
        };
        updated += pair.step("replica", &[o]).unwrap().updated_voxels;
        if k % 5 == 4 {
            pair.same(&format!("replica frame {k}"));
        }
    }
    assert!(updated > 10_000_000, "replica: {updated} updates");
    let mp = MeshParams {
        weight_threshold: 1,
        normals: true,
        colors: true,
    };
    let (mut a, mut b) = (TriangleMesh::new(), TriangleMesh::new());
    pair.cpu
        .extract_mesh_into(&mut pair.ws, &mp, &mut a)
        .unwrap();
    pair.gpu.extract_mesh_into(&mp, &mut b).unwrap();
    assert_eq!(mesh_bits(&a), mesh_bits(&b), "replica B{B}: mesh");
    let mut rp = RaycastParams::new(pair.cpu.layout(), 0.1, 7.0).unwrap();
    rp.colors = true;
    let size = ImageSize {
        width: 1200,
        height: 680,
    };
    let (mut ma, mut mb) = (sentinel_maps(size, true), sentinel_maps(size, true));
    let v = frames[frames.len() / 2].world_from_camera;
    pair.cpu
        .raycast_into(&mut pair.ws, camera, &v, &rp, &mut ma)
        .unwrap();
    pair.gpu.raycast_into(camera, &v, &rp, &mut mb).unwrap();
    assert_eq!(maps_bits(&ma), maps_bits(&mb), "replica B{B}: raycast");
    eprintln!(
        "replica B{B}: {} blocks, {updated} updates, {} triangles",
        pair.cpu.num_blocks(),
        a.triangles.len()
    );
}

// ------------------------------------------------------------ submaps ----

fn spawn_every_two() -> SpawnPolicy {
    SpawnPolicy {
        max_steps: 2,
        ..SpawnPolicy::default()
    }
}

fn cpu_map<const B: usize>(w: &World, color: bool) -> SubmapTsdf<B>
where
    BlockDim<B>: SupportedBlockDim,
{
    let mut map = SubmapTsdf::<B>::new(submap_params(color)).unwrap();
    map.add_agent(AgentId(0), rig(&w.rig_a), spawn_every_two())
        .unwrap();
    map.add_agent(AgentId(1), rig(&w.rig_b), spawn_every_two())
        .unwrap();
    map
}

fn gpu_map<const B: usize>(w: &World, params: SubmapParams) -> CudaSubmapTsdf<B>
where
    BlockDim<B>: SupportedBlockDim,
{
    let mut map = CudaSubmapTsdf::<B>::new(params, stream()).unwrap();
    map.add_agent(AgentId(0), rig(&w.rig_a), spawn_every_two())
        .unwrap();
    map.add_agent(AgentId(1), rig(&w.rig_b), spawn_every_two())
        .unwrap();
    map
}

/// Colour images of every pinhole sensor of the steps (LiDARs get none).
fn step_colors(w: &World, data: &[StepData]) -> Vec<Vec<Option<Image<u8, 3>>>> {
    data.iter()
        .enumerate()
        .map(|(k, d)| {
            let rig = if d.agent == AgentId(0) {
                &w.rig_a
            } else {
                &w.rig_b
            };
            d.depth
                .iter()
                .map(|(id, _)| {
                    let (_, s, _) = rig.iter().find(|(i, _, _)| i == id).unwrap();
                    match &s.sensor {
                        RangeSensor::Pinhole(p) => Some(color_image(p.size(), 11 * k)),
                        RangeSensor::Spherical(_) => None,
                    }
                })
                .collect()
        })
        .collect()
}

/// The frames of a step, with colour when `color` holds images.
fn step_frames<'a>(
    s: &'a StepData,
    c: &'a [Option<Image<u8, 3>>],
    color: bool,
) -> Vec<SensorFrame<'a>> {
    if !color {
        return frames(s);
    }
    s.depth
        .iter()
        .zip(c)
        .rev()
        .map(|((id, d), c)| SensorFrame {
            sensor: *id,
            depth: d.into(),
            color: c.as_ref(),
        })
        .collect()
}

/// Every submap of the device map equals the CPU map's: ids, poses,
/// blocks, the key -> pool map, epochs and voxel bits.
fn same_maps<const B: usize>(cpu: &SubmapTsdf<B>, gpu: &CudaSubmapTsdf<B>, label: &str)
where
    BlockDim<B>: SupportedBlockDim,
{
    let ids: Vec<_> = cpu.submap_ids().collect();
    assert_eq!(ids, gpu.submap_ids().collect::<Vec<_>>(), "{label}: ids");
    assert!(ids.len() > 1, "{label}: one submap only");
    for id in ids {
        assert_eq!(
            cpu.submap_pose(id),
            gpu.submap_pose(id),
            "{label} {id:?}: pose"
        );
        assert_eq!(
            cpu.submap_blocks(id),
            gpu.submap_blocks(id),
            "{label} {id:?}: blocks"
        );
        let a = cpu.submap_index(id).unwrap();
        let (b, pools) = gpu.download_submap(id).unwrap();
        assert_eq!(a.pool_keys, b.pool_keys, "{label} {id:?}: key -> pool");
        assert_eq!(a.sorted, b.sorted, "{label} {id:?}: sorted keys");
        assert_eq!(a.modified, b.modified, "{label} {id:?}: epochs");
        assert_eq!(a.epoch, b.epoch, "{label} {id:?}: step epoch");
        assert_eq!(
            cpu.content_hash(id),
            Some(content_hash_of(&b, &pools)),
            "{label} {id:?}: voxels"
        );
        assert_eq!(
            cpu.content_hash(id),
            gpu.content_hash(id).unwrap(),
            "{label} {id:?}: device hash"
        );
    }
    for agent in [AgentId(0), AgentId(1)] {
        assert_eq!(cpu.active_submap(agent), gpu.active_submap(agent));
        assert_eq!(
            cpu.memory_bytes(Some(agent)).blocks,
            gpu.memory_bytes(Some(agent)).blocks
        );
    }
}

fn content_hash_of(
    index: &crate::tsdf::volume::BlockIndex,
    pools: &crate::tsdf::volume::Pools,
) -> u64 {
    crate::tsdf::volume::content_hash(index, pools)
}

/// Shuffled arrival as one device batch, and another interleaving over
/// several batches, equal the CPU's serial replay in `(stamp, agent)`
/// order, step reports included.
fn multi_agent<const B: usize>(color: bool)
where
    BlockDim<B>: SupportedBlockDim,
{
    let w = world();
    let data = steps(&w);
    let colors = step_colors(&w, &data);
    let frames: Vec<Vec<SensorFrame<'_>>> = data
        .iter()
        .zip(&colors)
        .map(|(d, c)| step_frames(d, c, color))
        .collect();
    let all: Vec<AgentStep<'_>> = data
        .iter()
        .zip(&frames)
        .map(|(d, f)| agent_step(d, f))
        .collect();
    let mut sorted: Vec<usize> = (0..all.len()).collect();
    sorted.sort_by_key(|&i| (all[i].stamp, all[i].agent));
    let label = format!("multi-agent B{B} colour {color}");

    let mut cpu = cpu_map::<B>(&w, color);
    let mut ws = TsdfWorkspace::new(TsdfKernel::Scalar);
    let mut serial = gpu_map::<B>(&w, submap_params(color));
    let mut cpu_stats = IntegrateStats::default();
    for &i in &sorted {
        let a = cpu.integrate(&mut ws, &all[i]).unwrap();
        let b = serial.integrate(&all[i]).unwrap();
        assert_eq!(a, b, "{label} step {i}: report");
        cpu_stats.accumulate(&a.stats);
    }
    same_maps(&cpu, &serial, &format!("{label} serial"));

    let shuffled: Vec<AgentStep<'_>> = [7, 2, 9, 0, 5, 3, 8, 1, 6, 4]
        .iter()
        .map(|&i| all[i])
        .collect();
    let mut whole = gpu_map::<B>(&w, submap_params(color));
    let rep = whole.integrate_batch(&shuffled).unwrap();
    assert_eq!(rep.steps, all.len());
    assert_eq!(rep.stats, cpu_stats, "{label}: batch counters");
    same_maps(&cpu, &whole, &format!("{label} batch"));

    // agent 0: 0..5, agent 1: 5..10, each in its own stamp order
    let mut streamed = gpu_map::<B>(&w, submap_params(color));
    let mut stats = IntegrateStats::default();
    for b in [&[5, 6, 0][..], &[1, 7], &[2, 3, 8, 9], &[4]] {
        let batch: Vec<AgentStep<'_>> = b.iter().map(|&i| all[i]).collect();
        stats.accumulate(&streamed.integrate_batch(&batch).unwrap().stats);
    }
    assert_eq!(stats, cpu_stats, "{label}: streamed counters");
    same_maps(&cpu, &streamed, &format!("{label} streamed"));
}

#[test]
fn tsdf_integrate_multi_agent_bit_exact() {
    multi_agent::<8>(false);
    multi_agent::<16>(true);
}

/// Loop closure moves submaps; the union mesh (all and per agent, colours
/// with colour storage), the incremental caches and the nearest-hit
/// raycast equal the CPU's.
fn submap_queries<const B: usize>(color: bool)
where
    BlockDim<B>: SupportedBlockDim,
{
    let w = world();
    let data = steps(&w);
    let colors = step_colors(&w, &data);
    let frames: Vec<Vec<SensorFrame<'_>>> = data
        .iter()
        .zip(&colors)
        .map(|(d, c)| step_frames(d, c, color))
        .collect();
    let all: Vec<AgentStep<'_>> = data
        .iter()
        .zip(&frames)
        .map(|(d, f)| agent_step(d, f))
        .collect();
    let label = format!("queries B{B} colour {color}");
    let mut cpu = cpu_map::<B>(&w, color);
    let mut gpu = gpu_map::<B>(&w, submap_params(color));
    let mut ws = TsdfWorkspace::new(TsdfKernel::Scalar);
    cpu.integrate_batch(&mut ws, &all).unwrap();
    gpu.integrate_batch(&all).unwrap();
    same_maps(&cpu, &gpu, &label);

    let mp = MeshParams {
        weight_threshold: 1,
        normals: true,
        colors: color,
    };
    let (mut ca, mut cb) = (SubmapMeshCache::new(), SubmapMeshCache::new());
    let moved: Vec<SubmapPose> = cpu
        .submap_ids()
        .filter(|id| id.agent == AgentId(1))
        .map(|id| SubmapPose {
            id,
            world_from_submap: pose(0.01, 0.02, -0.03, [0.04, -0.02, 0.01])
                .compose(&cpu.submap_pose(id).unwrap()),
        })
        .collect();
    let cam = PinholeCamera {
        fx: 50.0,
        fy: 50.0,
        cx: 31.5,
        cy: 23.5,
        ..PinholeCamera::IDENTITY
    };
    let size = ImageSize {
        width: 64,
        height: 48,
    };
    let mut rp = RaycastParams::new(&layout(), 0.1, 3.0).unwrap();
    rp.colors = color;
    let v = pose(0.0, 0.05, 0.0, [0.05, 0.0, -0.02]);
    let agents = [AgentId(1)];
    for round in 0..2 {
        if round == 1 {
            cpu.set_submap_poses(&moved).unwrap();
            gpu.set_submap_poses(&moved).unwrap();
        }
        let label = format!("{label} round {round}");
        for filter in [SubmapFilter::All, SubmapFilter::Agents(&agents)] {
            let (mut a, mut b) = (junk_mesh(), junk_mesh());
            cpu.extract_mesh_into(&mut ws, filter, &mp, &mut a).unwrap();
            gpu.extract_mesh_into(filter, &mp, &mut b).unwrap();
            assert!(!a.triangles.is_empty(), "{label}: vacuous mesh");
            let coloured = a.colors.iter().any(|c| *c != [0; 3]);
            assert!(!color || coloured, "{label}: grey mesh");
            assert_eq!(mesh_bits(&a), mesh_bits(&b), "{label}: union mesh");
        }
        let sa = cpu.update_mesh_cache(&mut ws, &mp, &mut ca).unwrap();
        let sb = gpu.update_mesh_cache(&mp, &mut cb).unwrap();
        assert_eq!(sa, sb, "{label}: cache stats");
        let (mut a, mut b) = (junk_mesh(), junk_mesh());
        ca.write_into(&cpu, SubmapFilter::All, &mut a).unwrap();
        gpu.write_mesh_cache_into(&cb, SubmapFilter::All, &mut b)
            .unwrap();
        assert_eq!(mesh_bits(&a), mesh_bits(&b), "{label}: cached union");

        let (mut a, mut b) = (sentinel_maps(size, color), sentinel_maps(size, color));
        cpu.raycast_into(&mut ws, &cam, &v, SubmapFilter::All, &rp, &mut a)
            .unwrap();
        gpu.raycast_into(&cam, &v, SubmapFilter::All, &rp, &mut b)
            .unwrap();
        assert_eq!(maps_bits(&a), maps_bits(&b), "{label}: raycast");
        let hits = a.depth().as_slice().iter().any(|&z| z > 0.0);
        assert!(hits, "{label}: vacuous raycast");
        let mut pa = RgbdPyramid::zeroed(&cam, size, 2).unwrap();
        let mut pb = RgbdPyramid::zeroed(&cam, size, 2).unwrap();
        let only_b = SubmapFilter::Agents(&agents);
        cpu.raycast_pyramid_into(&mut ws, &v, only_b, &rp, &mut pa)
            .unwrap();
        gpu.raycast_pyramid_into(&v, only_b, &rp, &mut pb).unwrap();
        for (la, lb) in pa.levels.iter().zip(pb.levels.iter()) {
            assert_eq!(bits(la.vertices.as_slice()), bits(lb.vertices.as_slice()));
            assert_eq!(bits(la.normals.as_slice()), bits(lb.normals.as_slice()));
        }
    }
}

#[test]
fn tsdf_submap_queries_bit_exact() {
    submap_queries::<8>(false);
    submap_queries::<16>(true);
}

/// Stale, duplicate and unknown steps, a full slot table and a slot table
/// that grows: the same outcome on both backends.
#[test]
fn tsdf_submap_errors_and_growth_like_cpu() {
    let w = world();
    let data = steps(&w);
    let frames: Vec<Vec<SensorFrame<'_>>> = data.iter().map(frames).collect();
    let all: Vec<AgentStep<'_>> = data
        .iter()
        .zip(&frames)
        .map(|(d, f)| agent_step(d, f))
        .collect();
    let mut ws = TsdfWorkspace::new(TsdfKernel::Scalar);
    let dbg = |r: Result<StepReport, TsdfError>| format!("{r:?}");
    let dbgb = |r: Result<BatchReport, TsdfError>| format!("{r:?}");

    let mut cpu = cpu_map::<8>(&w, false);
    let mut gpu = gpu_map::<8>(&w, submap_params(false));
    assert_eq!(
        dbg(cpu.integrate(&mut ws, &all[1])),
        dbg(gpu.integrate(&all[1]))
    );
    // stale stamp, duplicate step in a batch, unknown agent and sensor
    assert_eq!(
        dbg(cpu.integrate(&mut ws, &all[0])),
        dbg(gpu.integrate(&all[0]))
    );
    let dup = [all[2], all[2]];
    assert_eq!(
        dbgb(cpu.integrate_batch(&mut ws, &dup)),
        dbgb(gpu.integrate_batch(&dup))
    );
    let ghost = AgentStep {
        agent: AgentId(9),
        ..all[3]
    };
    assert_eq!(
        dbg(cpu.integrate(&mut ws, &ghost)),
        dbg(gpu.integrate(&ghost))
    );
    let wrong = [SensorFrame {
        sensor: SensorId(42),
        ..frames[3][0]
    }];
    let unknown = AgentStep {
        frames: &wrong,
        ..all[3]
    };
    assert_eq!(
        dbg(cpu.integrate(&mut ws, &unknown)),
        dbg(gpu.integrate(&unknown))
    );
    assert_eq!(snapshot(&cpu).len(), gpu.submap_ids().count());

    // two slots, every step spawning: fixed fails like the CPU, geometric
    // grows the slot table
    for growth in [PoolGrowth::Fixed, PoolGrowth::Geometric] {
        let p = SubmapParams {
            max_submaps: 2,
            volume: TsdfParams {
                growth,
                ..submap_params(false).volume
            },
            ..submap_params(false)
        };
        let mut cpu = SubmapTsdf::<8>::new(p.clone()).unwrap();
        let mut gpu = CudaSubmapTsdf::<8>::new(p, stream()).unwrap();
        let spawn = SpawnPolicy {
            max_steps: 1,
            ..SpawnPolicy::default()
        };
        cpu.add_agent(AgentId(0), rig(&w.rig_a), spawn).unwrap();
        gpu.add_agent(AgentId(0), rig(&w.rig_a), spawn).unwrap();
        for s in &all[..5] {
            assert_eq!(
                dbg(cpu.integrate(&mut ws, s)),
                dbg(gpu.integrate(s)),
                "{growth:?}"
            );
        }
        let ids: Vec<_> = cpu.submap_ids().collect();
        assert_eq!(ids, gpu.submap_ids().collect::<Vec<_>>());
        for id in ids {
            assert_eq!(
                cpu.content_hash(id),
                gpu.content_hash(id).unwrap(),
                "{growth:?} {id:?}"
            );
        }
    }
    // a residency budget is refused on the device
    let budget = SubmapParams {
        max_resident_bytes: Some(1 << 20),
        ..submap_params(false)
    };
    assert!(matches!(
        CudaSubmapTsdf::<8>::new(budget, stream()),
        Err(TsdfError::InvalidParameter {
            field: "max_resident_bytes",
            ..
        })
    ));
}
