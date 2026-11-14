//! A warmed TSDF frame must not allocate: integration (one sensor, several sensors, several
//! agents), raycast, meshing into reused buffers, a submap spawn within its reserved budget,
//! and the LiDAR range image, for both CPU kernels and both block sizes.
//!
//! Its own test binary with ONE test function (run it with `--test-threads=1`), because it
//! installs a counting global allocator. The count is process-wide, so a rayon worker that
//! allocates is seen; the calling thread's own count is checked as well. Every measured call
//! follows warm-up calls of the same shape, and the measured integration steps add new blocks
//! (within the reserved capacity), so a buffer sized by the block count rather than the
//! capacity regrows inside the window.
//!
//! The `Simd` kernel runs its voxel passes on rayon, so its frames are measured from a worker
//! of a pool (warm-up included). Entering a pool from outside allocates inside rayon at a fixed
//! cadence; one row pins that cadence from the main thread.

use kornia_3d::camera::PinholeCamera;
use kornia_3d::depth::DepthImage;
use kornia_3d::mesh::TriangleMesh;
use kornia_3d::pose::Pose3d;
use kornia_3d::registration::RgbdPyramid;
use kornia_3d::tsdf::{
    AgentId, AgentStep, AzimuthGrid, BlockDim, BlockMeshCache, DepthRange, MeshParams,
    PinholeSensor, PoolGrowth, RangeSensor, RaycastMaps, RaycastParams, RigSensor, SensorFrame,
    SensorId, SensorObservation, SensorRig, SpawnPolicy, SphericalLidar, SubmapFilter,
    SubmapMeshCache, SubmapParams, SubmapTsdf, SupportedBlockDim, TsdfError, TsdfKernel,
    TsdfLayout, TsdfParams, TsdfVolume, TsdfWorkspace, WorkspaceHint,
};
use kornia_algebra::{Mat3F64, Vec3F64};
use kornia_image::{Image, ImageSize};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};

struct Counting;

static TOTAL: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    static LOCAL: Cell<usize> = const { Cell::new(0) };
}

fn count() {
    TOTAL.fetch_add(1, Ordering::Relaxed);
    LOCAL.with(|c| c.set(c.get() + 1));
}

// SAFETY: forwards every call to the system allocator unchanged; the counters are an atomic and
// a const-initialised thread-local `Cell`, so touching them never allocates or re-enters.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count();
        // SAFETY: same contract as the caller's.
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count();
        // SAFETY: same contract as the caller's.
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` came from `System` above with this layout.
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        count();
        // SAFETY: same contract as the caller's.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// Runs `f` and returns its value with the allocations it made: (process-wide, this thread).
/// Nothing is printed or pushed inside the window.
fn measure<T>(f: impl FnOnce() -> T) -> (T, usize, usize) {
    let (t0, l0) = (TOTAL.load(Ordering::Relaxed), LOCAL.with(|c| c.get()));
    let out = f();
    let (t1, l1) = (TOTAL.load(Ordering::Relaxed), LOCAL.with(|c| c.get()));
    (out, t1 - t0, l1 - l0)
}

type R<T = ()> = Result<T, Box<dyn std::error::Error>>;

/// One measured call per row.
#[derive(Default)]
struct Audit {
    rows: Vec<(String, usize, usize)>,
}

impl Audit {
    fn row(&mut self, label: String, total: usize, local: usize) {
        self.rows.push((label, total, local));
    }

    /// Measures `f`, which must return `Ok`.
    fn call<T>(&mut self, label: String, f: impl FnOnce() -> Result<T, TsdfError>) -> R<T> {
        let (out, total, local) = measure(f);
        self.row(label, total, local);
        Ok(out?)
    }
}

const VOXEL: f32 = 0.02;
/// Sensor travel per step (m): more than a block (0.16 m), so a step allocates blocks.
const STRIDE: f64 = 0.25;
const SIZE: ImageSize = ImageSize {
    width: 64,
    height: 48,
};
const LIDAR_SIZE: ImageSize = ImageSize {
    width: 128,
    height: 4,
};

fn camera() -> PinholeCamera {
    PinholeCamera {
        fx: 60.0,
        fy: 60.0,
        cx: 31.5,
        cy: 23.5,
        ..PinholeCamera::IDENTITY
    }
}

fn pinhole() -> R<RangeSensor> {
    let range = DepthRange::new(0.1, 4.0)?;
    Ok(PinholeSensor::new(&camera(), SIZE, range)?.into())
}

fn lidar() -> R<SphericalLidar> {
    let az = AzimuthGrid {
        start: 0.0,
        step: std::f64::consts::TAU / LIDAR_SIZE.width as f64,
        wrap: true,
    };
    Ok(SphericalLidar::uniform(
        LIDAR_SIZE,
        az,
        0.1,
        -0.1,
        DepthRange::new(0.3, 4.0)?,
    )?)
}

fn at(x: f64) -> Pose3d {
    Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(x, 0.0, 0.0))
}

/// Blocks per volume (or submap): room for the warm-up and the measured steps, so that
/// `PoolGrowth::Fixed` never fails and a regrowth is a bug, not a policy.
fn capacity<const B: usize>() -> usize
where
    BlockDim<B>: SupportedBlockDim,
{
    2_000_000 / <BlockDim<B> as SupportedBlockDim>::VOXELS
}

fn params<const B: usize>() -> TsdfParams
where
    BlockDim<B>: SupportedBlockDim,
{
    TsdfParams {
        block_capacity: capacity::<B>(),
        growth: PoolGrowth::Fixed,
        ..TsdfParams::default()
    }
}

fn hint(blocks: usize) -> WorkspaceHint {
    WorkspaceHint {
        pixels_per_step: 3 * SIZE.width * SIZE.height + LIDAR_SIZE.width * LIDAR_SIZE.height,
        observations_per_step: 4,
        blocks,
        steps_per_batch: 4,
    }
}

/// A workspace sized for volumes of up to `capacity::<B>()` blocks.
fn workspace<const B: usize>(kernel: TsdfKernel) -> TsdfWorkspace
where
    BlockDim<B>: SupportedBlockDim,
{
    let mut ws = TsdfWorkspace::new(kernel);
    ws.reserve(&hint(capacity::<B>()));
    ws
}

/// Constant-depth images: a plane 1 m from the camera, a ring 1 m from the LiDAR.
struct Frames {
    depth_u16: DepthImage<u16>,
    depth_f32: DepthImage<f32>,
    range: DepthImage<f32>,
    color: Image<u8, 3>,
}

fn frames() -> R<Frames> {
    Ok(Frames {
        depth_u16: DepthImage::from_size_val(SIZE, 1000u16, 1000.0)?,
        depth_f32: DepthImage::from_size_val(SIZE, 1.0f32, 1.0)?,
        range: DepthImage::from_size_val(LIDAR_SIZE, 1.0f32, 1.0)?,
        color: Image::from_size_val(SIZE, 128u8)?,
    })
}

fn name<const B: usize>(kernel: TsdfKernel, what: &str) -> String {
    format!("{kernel:?} B={B} {what}")
}

/// `TsdfVolume`: one sensor (with colour), then two cameras and a LiDAR, then raycast and meshing.
fn volume_cases<const B: usize>(kernel: TsdfKernel, audit: &mut Audit) -> R
where
    BlockDim<B>: SupportedBlockDim,
{
    let f = frames()?;
    let (cam, ring) = (pinhole()?, RangeSensor::from(lidar()?));
    let layout = TsdfLayout::new(VOXEL, 4.0)?;
    let mut ws = workspace::<B>(kernel);

    // single sensor; each step moves one stride so the measured ones allocate blocks
    let mut vol = TsdfVolume::<B>::new(layout, params::<B>())?;
    let step = |x: f64| SensorObservation {
        sensor: &cam,
        volume_from_sensor: at(x),
        depth: (&f.depth_u16).into(),
        color: Some(&f.color),
    };
    for k in 0..3 {
        vol.integrate(&mut ws, &[step(STRIDE * k as f64)])?;
    }
    for k in 3..5 {
        let s = audit.call(name::<B>(kernel, "integrate, one sensor"), || {
            vol.integrate(&mut ws, &[step(STRIDE * k as f64)])
        })?;
        assert!(s.new_blocks > 0, "the measured step allocates no block");
    }

    // two cameras and a LiDAR in one step
    let mut vol = TsdfVolume::<B>::new(layout, params::<B>())?;
    let rig = |x: f64| {
        [
            SensorObservation {
                sensor: &cam,
                volume_from_sensor: at(x),
                depth: (&f.depth_f32).into(),
                color: None,
            },
            SensorObservation {
                sensor: &cam,
                volume_from_sensor: at(x + 0.2),
                depth: (&f.depth_f32).into(),
                color: None,
            },
            SensorObservation {
                sensor: &ring,
                volume_from_sensor: at(x),
                depth: (&f.range).into(),
                color: None,
            },
        ]
    };
    for k in 0..3 {
        vol.integrate(&mut ws, &rig(STRIDE * k as f64))?;
    }
    for k in 3..5 {
        let obs = rig(STRIDE * k as f64);
        let s = audit.call(name::<B>(kernel, "integrate, 2 cams + lidar"), || {
            vol.integrate(&mut ws, &obs)
        })?;
        assert!(s.new_blocks > 0, "the measured step allocates no block");
    }

    // raycast: the volume, an ICP level and a whole pyramid, from moving poses
    let rp = RaycastParams::new(&layout, 0.1, 3.0)?;
    let mut maps = RaycastMaps::new(SIZE, false)?;
    let mut pyramid = RgbdPyramid::zeroed(&camera(), SIZE, 3)?;
    vol.raycast_into(&mut ws, &camera(), &at(0.0), &rp, &mut maps)?;
    vol.raycast_pyramid_into(&mut ws, &at(0.0), &rp, &mut pyramid)?;
    for x in [0.05, 0.15] {
        audit.call(name::<B>(kernel, "raycast_into"), || {
            vol.raycast_into(&mut ws, &camera(), &at(x), &rp, &mut maps)
        })?;
        audit.call(name::<B>(kernel, "raycast_pyramid_into"), || {
            vol.raycast_pyramid_into(&mut ws, &at(x), &rp, &mut pyramid)
        })?;
    }
    assert!(
        maps.depth().as_slice().iter().any(|&d| d > 0.0),
        "the raycast hit nothing"
    );

    // meshing into reused buffers
    let mp = MeshParams::default();
    let mut mesh = TriangleMesh::new();
    vol.extract_mesh_into(&mut ws, &mp, &mut mesh)?;
    audit.call(name::<B>(kernel, "extract_mesh_into"), || {
        vol.extract_mesh_into(&mut ws, &mp, &mut mesh)
    })?;
    assert!(!mesh.triangles.is_empty(), "the mesh is empty");

    // incremental mesh: warm, integrate a shifted step (new blocks), then update and write
    let mut cache = BlockMeshCache::with_capacity(1 << 20, 1 << 20, capacity::<B>());
    let mut out = TriangleMesh::new();
    out.vertices.reserve(1 << 20);
    out.normals.reserve(1 << 20);
    out.triangles.reserve(1 << 20);
    vol.update_mesh_cache(&mut ws, &mp, &mut cache)?;
    cache.write_into(&mut out)?;
    for k in 5..7 {
        let obs = rig(STRIDE * k as f64);
        let s = vol.integrate(&mut ws, &obs)?;
        assert!(s.new_blocks > 0, "the step allocates no block");
        let st = audit.call(name::<B>(kernel, "update_mesh_cache"), || {
            vol.update_mesh_cache(&mut ws, &mp, &mut cache)
        })?;
        assert!(st.remeshed_blocks > 0, "nothing was re-meshed");
        audit.call(name::<B>(kernel, "BlockMeshCache::write_into"), || {
            cache.write_into(&mut out)
        })?;
    }
    Ok(())
}

/// The LiDAR range image from points.
fn lidar_cases(audit: &mut Audit) -> R {
    let ring = lidar()?;
    let points: Vec<[f32; 3]> = (0..2000)
        .map(|i| {
            let a = i as f32 * 0.00314;
            [a.cos(), a.sin(), 0.02 * (i % 5) as f32 - 0.04]
        })
        .collect();
    let mut dst = DepthImage::from_size_val(LIDAR_SIZE, 0.0f32, 1.0)?;
    ring.range_image_from_points_into(&points, &mut dst)?;
    audit.call("range_image_from_points_into".to_string(), || {
        ring.range_image_from_points_into(&points, &mut dst)
    })?;
    assert!(
        dst.image().as_slice().iter().any(|&r| r > 0.0),
        "no point projected"
    );
    Ok(())
}

fn submap_params<const B: usize>() -> SubmapParams
where
    BlockDim<B>: SupportedBlockDim,
{
    SubmapParams {
        layout: TsdfLayout::new(VOXEL, 4.0).unwrap_or_else(|e| panic!("{e}")),
        volume: params::<B>(),
        arena_blocks: 3 * capacity::<B>(),
        max_submaps: 8,
        max_resident_bytes: None,
    }
}

fn rig_a() -> R<SensorRig> {
    Ok(SensorRig::new(vec![
        RigSensor {
            id: SensorId(2),
            sensor: lidar()?.into(),
            body_from_sensor: Pose3d::IDENTITY,
        },
        RigSensor {
            id: SensorId(1),
            sensor: pinhole()?,
            body_from_sensor: Pose3d::IDENTITY,
        },
    ])?)
}

fn rig_b() -> R<SensorRig> {
    Ok(SensorRig::new(vec![RigSensor {
        id: SensorId(0),
        sensor: pinhole()?,
        body_from_sensor: at(0.1),
    }])?)
}

/// `SubmapTsdf`: `integrate` and `integrate_batch` with two agents, queries, and spawns.
fn submap_cases<const B: usize>(kernel: TsdfKernel, audit: &mut Audit) -> R
where
    BlockDim<B>: SupportedBlockDim,
{
    let f = frames()?;
    let mut ws = workspace::<B>(kernel);
    let frames_a = [
        SensorFrame {
            sensor: SensorId(1),
            depth: (&f.depth_f32).into(),
            color: None,
        },
        SensorFrame {
            sensor: SensorId(2),
            depth: (&f.range).into(),
            color: None,
        },
    ];
    let frames_b = [SensorFrame {
        sensor: SensorId(0),
        depth: (&f.depth_f32).into(),
        color: None,
    }];
    let step_a = |stamp: u64| AgentStep {
        agent: AgentId(0),
        stamp,
        world_from_body: at(STRIDE * stamp as f64),
        frames: &frames_a,
    };
    let step_b = |stamp: u64| AgentStep {
        agent: AgentId(1),
        stamp,
        world_from_body: at(2.0 + STRIDE * stamp as f64),
        frames: &frames_b,
    };

    let mut map = SubmapTsdf::<B>::new(submap_params::<B>())?;
    map.add_agent(AgentId(0), rig_a()?, SpawnPolicy::default())?;
    map.add_agent(AgentId(1), rig_b()?, SpawnPolicy::default())?;
    // warm-up: the first steps spawn the submaps
    map.integrate_batch(&mut ws, &[step_b(1), step_a(1)])?;
    map.integrate(&mut ws, &step_a(2))?;
    for stamp in 3..5 {
        let s = audit.call(name::<B>(kernel, "SubmapTsdf::integrate"), || {
            map.integrate(&mut ws, &step_a(stamp))
        })?;
        assert!(s.stats.new_blocks > 0 && !s.spawned);
    }
    for stamp in 5..7 {
        let steps = [step_a(stamp), step_b(stamp)];
        let r = audit.call(name::<B>(kernel, "integrate_batch, 2 agents"), || {
            map.integrate_batch(&mut ws, &steps)
        })?;
        assert!(r.stats.new_blocks > 0 && r.spawned == 0);
    }

    // queries over the union of the submaps
    let layout = TsdfLayout::new(VOXEL, 4.0)?;
    let rp = RaycastParams::new(&layout, 0.1, 3.0)?;
    let mut maps = RaycastMaps::new(SIZE, false)?;
    let mut pyramid = RgbdPyramid::zeroed(&camera(), SIZE, 3)?;
    let all = SubmapFilter::All;
    map.raycast_into(&mut ws, &camera(), &at(0.1), all, &rp, &mut maps)?;
    map.raycast_pyramid_into(&mut ws, &at(0.1), all, &rp, &mut pyramid)?;
    for x in [0.3, 0.4] {
        audit.call(name::<B>(kernel, "SubmapTsdf::raycast_into"), || {
            map.raycast_into(&mut ws, &camera(), &at(x), all, &rp, &mut maps)
        })?;
        audit.call(
            name::<B>(kernel, "SubmapTsdf::raycast_pyramid_into"),
            || map.raycast_pyramid_into(&mut ws, &at(x), all, &rp, &mut pyramid),
        )?;
    }
    let mp = MeshParams::default();
    let mut mesh = TriangleMesh::new();
    mesh.vertices.reserve(1 << 20);
    mesh.normals.reserve(1 << 20);
    mesh.triangles.reserve(1 << 20);
    map.extract_mesh_into(&mut ws, all, &mp, &mut mesh)?;
    audit.call(name::<B>(kernel, "SubmapTsdf::extract_mesh_into"), || {
        map.extract_mesh_into(&mut ws, all, &mp, &mut mesh)
    })?;
    assert!(!mesh.triangles.is_empty(), "the mesh is empty");

    // incremental mesh of the map, each submap slab reserved up front
    let mut cache = SubmapMeshCache::with_capacity(2, 1 << 19, 1 << 19, capacity::<B>());
    map.update_mesh_cache(&mut ws, &mp, &mut cache)?;
    cache.write_into(&map, all, &mut mesh)?;
    for stamp in 7..9 {
        map.integrate(&mut ws, &step_a(stamp))?;
        let st = audit.call(name::<B>(kernel, "SubmapTsdf::update_mesh_cache"), || {
            map.update_mesh_cache(&mut ws, &mp, &mut cache)
        })?;
        assert!(st.remeshed_blocks > 0, "nothing was re-meshed");
        audit.call(name::<B>(kernel, "SubmapMeshCache::write_into"), || {
            cache.write_into(&map, all, &mut mesh)
        })?;
    }

    spawn_cases::<B>(kernel, audit)
}

/// A new submap every step (`max_steps = 1`): after two spawns, the next ones stay in the budget.
fn spawn_cases<const B: usize>(kernel: TsdfKernel, audit: &mut Audit) -> R
where
    BlockDim<B>: SupportedBlockDim,
{
    let f = frames()?;
    let mut ws = workspace::<B>(kernel);
    let frames_b = [SensorFrame {
        sensor: SensorId(0),
        depth: (&f.depth_f32).into(),
        color: None,
    }];
    let step = |stamp: u64| AgentStep {
        agent: AgentId(1),
        stamp,
        world_from_body: at(0.5 * stamp as f64),
        frames: &frames_b,
    };
    let spawn = SpawnPolicy {
        max_steps: 1,
        ..SpawnPolicy::default()
    };
    let mut map = SubmapTsdf::<B>::new(submap_params::<B>())?;
    map.add_agent(AgentId(1), rig_b()?, spawn)?;
    map.integrate(&mut ws, &step(1))?;
    map.integrate(&mut ws, &step(2))?;
    for stamp in 3..6 {
        let s = audit.call(name::<B>(kernel, "submap spawn"), || {
            map.integrate(&mut ws, &step(stamp))
        })?;
        assert!(s.spawned, "the step did not spawn");
    }
    assert_eq!(map.submap_ids().count(), 5);
    Ok(())
}

/// Workers of the pool the `Simd` rows run in.
const SIMD_THREADS: usize = 4;
/// Pushes per block of crossbeam-deque's `Injector` (`BLOCK_CAP`): rayon-core's
/// `Registry::inject`, reached from `in_worker_cold` on every entry from outside the pool,
/// allocates the next block every 63 pushes.
const INJECTOR_BLOCK: usize = 63;
/// Warmed single-sensor `Simd` steps measured from outside any pool.
const OUTSIDE_STEPS: usize = 2 * INJECTOR_BLOCK;

/// Allocations of `OUTSIDE_STEPS` warmed one-observation `Simd` steps called from this
/// (non-rayon) thread: one pool entry each.
fn outside_pool_steps() -> R<(usize, usize)> {
    let sensor = pinhole()?;
    let f = frames()?;
    let mut vol = TsdfVolume::<8>::new(TsdfLayout::new(VOXEL, 4.0)?, params::<8>())?;
    let mut ws = workspace::<8>(TsdfKernel::Simd);
    let obs = [SensorObservation {
        sensor: &sensor,
        volume_from_sensor: at(0.0),
        depth: (&f.depth_u16).into(),
        color: None,
    }];
    for _ in 0..4 {
        vol.integrate(&mut ws, &obs)?;
    }
    let (out, total, local) = measure(|| {
        for _ in 0..OUTSIDE_STEPS {
            vol.integrate(&mut ws, &obs)?;
        }
        Ok::<(), TsdfError>(())
    });
    out?;
    Ok((total, local))
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "slow in a debug build; runs in the release test job"
)]
fn warmed_tsdf_frame_does_not_allocate() -> R {
    let mut audit = Audit::default();

    // the instrument can fail: a deliberate allocation is counted
    let ((), total, local) = measure(|| {
        std::hint::black_box(Vec::<u8>::with_capacity(1));
    });
    assert!(total >= 1 && local >= 1, "the counter does not count");

    let cases = |kernel: TsdfKernel, audit: &mut Audit| -> R {
        volume_cases::<8>(kernel, audit)?;
        volume_cases::<16>(kernel, audit)?;
        submap_cases::<8>(kernel, audit)?;
        submap_cases::<16>(kernel, audit)
    };
    cases(TsdfKernel::Scalar, &mut audit)?;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(SIMD_THREADS)
        .build()?;
    pool.install(|| cases(TsdfKernel::Simd, &mut audit).map_err(|e| e.to_string()))?;
    lidar_cases(&mut audit)?;

    // from outside a pool only rayon's injector allocates: at most one block per 63 entries,
    // where an allocation of ours would cost at least one per step
    let (total, local) = outside_pool_steps()?;
    eprintln!("Simd B=8 integrate x{OUTSIDE_STEPS} outside a pool: {total} allocations");
    assert!(
        total <= OUTSIDE_STEPS / INJECTOR_BLOCK + 1 && local == total,
        "{OUTSIDE_STEPS} steps outside a pool: {total} allocations ({local} on this thread)"
    );

    let mut bad = Vec::new();
    for (label, total, local) in &audit.rows {
        eprintln!("{label}: {total} allocations ({local} on the calling thread)");
        if *total != 0 || *local != 0 {
            bad.push(format!("{label}: {total} / {local}"));
        }
    }
    assert!(
        bad.is_empty(),
        "allocating warmed calls:\n{}",
        bad.join("\n")
    );
    Ok(())
}
