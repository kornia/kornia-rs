//! A warmed `CudaTsdfVolume` frame makes no host heap allocation: integration (one sensor,
//! two cameras and a LiDAR), raycast, full and incremental meshing, for both block sizes;
//! and a warmed `CudaSubmapTsdf` multi-agent batch, spawns within `max_submaps` included
//! (TSDF contract, section 17: raw launches, preallocated device scratch and host staging).
//!
//! Its own test binary with ONE test function (run it with `--test-threads=1`), because it
//! installs a counting global allocator. The volumes use `PoolGrowth::Fixed` without any
//! `reserve`, so this also pins that a fixed pool is allocated whole at construction; the
//! measured integration steps add new blocks, so a buffer sized by the block count rather
//! than the capacity regrows inside the window. Needs a CUDA device and the real PTX.
#![cfg(feature = "cuda")]

use kornia_3d::camera::PinholeCamera;
use kornia_3d::depth::DepthImage;
use kornia_3d::mesh::TriangleMesh;
use kornia_3d::pose::Pose3d;
use kornia_3d::registration::RgbdPyramid;
use kornia_3d::tsdf::{
    AgentId, AgentStep, AzimuthGrid, BlockDim, BlockMeshCache, CudaSubmapTsdf, CudaTsdfVolume,
    DepthRange, MeshParams, PinholeSensor, PoolGrowth, RangeSensor, RaycastMaps, RaycastParams,
    RigSensor, SensorFrame, SensorId, SensorObservation, SensorRig, SpawnPolicy, SphericalLidar,
    SubmapFilter, SubmapParams, SupportedBlockDim, TsdfError, TsdfLayout, TsdfParams,
};
use kornia_algebra::{Mat3F64, Vec3F64};
use kornia_image::{Image, ImageSize};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

struct Counting;

static TOTAL: AtomicUsize = AtomicUsize::new(0);

// SAFETY: forwards every call to the system allocator unchanged; the counter is an atomic,
// so touching it never allocates or re-enters.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        TOTAL.fetch_add(1, Ordering::Relaxed);
        // SAFETY: same contract as the caller's.
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        TOTAL.fetch_add(1, Ordering::Relaxed);
        // SAFETY: same contract as the caller's.
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` came from `System` above with this layout.
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        TOTAL.fetch_add(1, Ordering::Relaxed);
        // SAFETY: same contract as the caller's.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

type R<T = ()> = Result<T, Box<dyn std::error::Error>>;

/// Measured calls: (label, allocations). Nothing is printed inside a window.
#[derive(Default)]
struct Audit {
    rows: Vec<(String, usize)>,
}

impl Audit {
    fn call<T>(&mut self, label: String, f: impl FnOnce() -> Result<T, TsdfError>) -> R<T> {
        let t0 = TOTAL.load(Ordering::Relaxed);
        let out = f();
        let n = TOTAL.load(Ordering::Relaxed) - t0;
        self.rows.push((label, n));
        Ok(out?)
    }
}

const VOXEL: f32 = 0.02;
/// Sensor travel per step (m): more than a block, so a step allocates blocks.
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

fn at(x: f64) -> Pose3d {
    Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(x, 0.0, 0.0))
}

/// Room for the warm-up and the measured steps (a regrowth is a bug).
fn capacity<const B: usize>() -> usize
where
    BlockDim<B>: SupportedBlockDim,
{
    2_000_000 / <BlockDim<B> as SupportedBlockDim>::VOXELS
}

fn cases<const B: usize>(audit: &mut Audit) -> R
where
    BlockDim<B>: SupportedBlockDim,
{
    let stream = cudarc::driver::CudaContext::new(0)?.new_stream()?;
    let cam = RangeSensor::from(PinholeSensor::new(
        &camera(),
        SIZE,
        DepthRange::new(0.1, 4.0)?,
    )?);
    let az = AzimuthGrid {
        start: 0.0,
        step: std::f64::consts::TAU / LIDAR_SIZE.width as f64,
        wrap: true,
    };
    let ring = RangeSensor::from(SphericalLidar::uniform(
        LIDAR_SIZE,
        az,
        0.1,
        -0.1,
        DepthRange::new(0.3, 4.0)?,
    )?);
    let depth_u16 = DepthImage::from_size_val(SIZE, 1000u16, 1000.0)?;
    let depth_f32 = DepthImage::from_size_val(SIZE, 1.0f32, 1.0)?;
    let range = DepthImage::from_size_val(LIDAR_SIZE, 1.0f32, 1.0)?;
    let color = Image::from_size_val(SIZE, 128u8)?;
    let layout = TsdfLayout::new(VOXEL, 4.0)?;
    let params = TsdfParams {
        block_capacity: capacity::<B>(),
        growth: PoolGrowth::Fixed,
        ..TsdfParams::default()
    };
    let name = |what: &str| format!("cuda B={B} {what}");

    // one sensor with colour
    let mut vol = CudaTsdfVolume::<B>::new(layout, params.clone(), stream.clone())?;
    let step = |x: f64| SensorObservation {
        sensor: &cam,
        volume_from_sensor: at(x),
        depth: (&depth_u16).into(),
        color: Some(&color),
    };
    for k in 0..3 {
        vol.integrate(&[step(STRIDE * k as f64)])?;
    }
    for k in 3..5 {
        let s = audit.call(name("integrate, one sensor"), || {
            vol.integrate(&[step(STRIDE * k as f64)])
        })?;
        assert!(s.new_blocks > 0, "the measured step allocates no block");
    }

    // two cameras and a LiDAR in one step
    let mut vol = CudaTsdfVolume::<B>::new(layout, params, stream)?;
    let rig = |x: f64| {
        [
            SensorObservation {
                sensor: &cam,
                volume_from_sensor: at(x),
                depth: (&depth_f32).into(),
                color: None,
            },
            SensorObservation {
                sensor: &cam,
                volume_from_sensor: at(x + 0.2),
                depth: (&depth_f32).into(),
                color: None,
            },
            SensorObservation {
                sensor: &ring,
                volume_from_sensor: at(x),
                depth: (&range).into(),
                color: None,
            },
        ]
    };
    for k in 0..3 {
        vol.integrate(&rig(STRIDE * k as f64))?;
    }
    for k in 3..5 {
        let obs = rig(STRIDE * k as f64);
        let s = audit.call(name("integrate, 2 cams + lidar"), || vol.integrate(&obs))?;
        assert!(s.new_blocks > 0, "the measured step allocates no block");
    }

    // raycast: the volume and a pyramid, from moving poses
    let rp = RaycastParams::new(&layout, 0.1, 3.0)?;
    let mut maps = RaycastMaps::new(SIZE, false)?;
    let mut pyramid = RgbdPyramid::zeroed(&camera(), SIZE, 3)?;
    vol.raycast_into(&camera(), &at(0.0), &rp, &mut maps)?;
    vol.raycast_pyramid_into(&at(0.0), &rp, &mut pyramid)?;
    for x in [0.05, 0.15] {
        audit.call(name("raycast_into"), || {
            vol.raycast_into(&camera(), &at(x), &rp, &mut maps)
        })?;
        audit.call(name("raycast_pyramid_into"), || {
            vol.raycast_pyramid_into(&at(x), &rp, &mut pyramid)
        })?;
    }
    assert!(
        maps.depth().as_slice().iter().any(|&d| d > 0.0),
        "the raycast hit nothing"
    );

    // meshing into reused buffers
    let mp = MeshParams::default();
    let mut mesh = TriangleMesh::new();
    vol.extract_mesh_into(&mp, &mut mesh)?;
    audit.call(name("extract_mesh_into"), || {
        vol.extract_mesh_into(&mp, &mut mesh)
    })?;
    assert!(!mesh.triangles.is_empty(), "the mesh is empty");

    // incremental: warm, then a shifted step (new blocks), update and write
    let mut cache = BlockMeshCache::with_capacity(1 << 20, 1 << 20, capacity::<B>());
    let mut out = TriangleMesh::new();
    out.vertices.reserve(1 << 20);
    out.normals.reserve(1 << 20);
    out.triangles.reserve(1 << 20);
    vol.update_mesh_cache(&mp, &mut cache)?;
    cache.write_into(&mut out)?;
    for k in 5..7 {
        let obs = rig(STRIDE * k as f64);
        let s = vol.integrate(&obs)?;
        assert!(s.new_blocks > 0, "the step allocates no block");
        let st = audit.call(name("update_mesh_cache"), || {
            vol.update_mesh_cache(&mp, &mut cache)
        })?;
        assert!(st.remeshed_blocks > 0, "nothing was re-meshed");
    }
    Ok(())
}

/// Two agents (a camera + LiDAR rig and a camera), a spawn every second step, one batch
/// per stamp pair.
fn submap_cases<const B: usize>(audit: &mut Audit) -> R
where
    BlockDim<B>: SupportedBlockDim,
{
    let stream = cudarc::driver::CudaContext::new(0)?.new_stream()?;
    let cam = RangeSensor::from(PinholeSensor::new(
        &camera(),
        SIZE,
        DepthRange::new(0.1, 4.0)?,
    )?);
    let az = AzimuthGrid {
        start: 0.0,
        step: std::f64::consts::TAU / LIDAR_SIZE.width as f64,
        wrap: true,
    };
    let ring = RangeSensor::from(SphericalLidar::uniform(
        LIDAR_SIZE,
        az,
        0.1,
        -0.1,
        DepthRange::new(0.3, 4.0)?,
    )?);
    let depth = DepthImage::from_size_val(SIZE, 1.0f32, 1.0)?;
    let range = DepthImage::from_size_val(LIDAR_SIZE, 1.0f32, 1.0)?;
    let params = SubmapParams {
        layout: TsdfLayout::new(VOXEL, 4.0)?,
        volume: TsdfParams {
            block_capacity: capacity::<B>(),
            growth: PoolGrowth::Fixed,
            color: false,
            ..TsdfParams::default()
        },
        arena_blocks: 8 * capacity::<B>(),
        max_submaps: 32,
        max_resident_bytes: None,
    };
    let mut map = CudaSubmapTsdf::<B>::new(params, stream)?;
    let rig_a = SensorRig::new(vec![
        RigSensor {
            id: SensorId(0),
            sensor: cam.clone(),
            body_from_sensor: Pose3d::IDENTITY,
        },
        RigSensor {
            id: SensorId(1),
            sensor: ring,
            body_from_sensor: Pose3d::IDENTITY,
        },
    ])?;
    let rig_b = SensorRig::new(vec![RigSensor {
        id: SensorId(0),
        sensor: cam,
        body_from_sensor: Pose3d::IDENTITY,
    }])?;
    let spawn = SpawnPolicy {
        max_steps: 2,
        ..SpawnPolicy::default()
    };
    map.add_agent(AgentId(0), rig_a, spawn)?;
    map.add_agent(AgentId(1), rig_b, spawn)?;
    let fa = [
        SensorFrame {
            sensor: SensorId(1),
            depth: (&range).into(),
            color: None,
        },
        SensorFrame {
            sensor: SensorId(0),
            depth: (&depth).into(),
            color: None,
        },
    ];
    let fb = [SensorFrame {
        sensor: SensorId(0),
        depth: (&depth).into(),
        color: None,
    }];
    let batch = |k: u64| {
        [
            AgentStep {
                agent: AgentId(1),
                stamp: 2 * k + 1,
                world_from_body: at(0.1 * k as f64 + 0.05),
                frames: &fb,
            },
            AgentStep {
                agent: AgentId(0),
                stamp: 2 * k,
                world_from_body: at(0.1 * k as f64),
                frames: &fa,
            },
        ]
    };
    for k in 0..4 {
        map.integrate_batch(&batch(k))?;
    }
    for k in 4..8 {
        let steps = batch(k);
        let r = audit.call(format!("cuda B={B} submaps integrate_batch"), || {
            map.integrate_batch(&steps)
        })?;
        assert!(r.stats.updated_voxels > 0, "the batch updated nothing");
        if k % 2 == 0 {
            assert_eq!(r.spawned, 2, "the measured batch spawns no submap");
        }
    }
    let rp = RaycastParams::new(&map.params().layout, 0.1, 3.0)?;
    let mut maps = RaycastMaps::new(SIZE, false)?;
    map.raycast_into(&camera(), &at(0.0), SubmapFilter::All, &rp, &mut maps)?;
    audit.call(format!("cuda B={B} submaps raycast_into"), || {
        map.raycast_into(&camera(), &at(0.3), SubmapFilter::All, &rp, &mut maps)
    })?;
    Ok(())
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "slow in a debug build; runs in the release test job"
)]
fn warmed_cuda_tsdf_frame_does_not_allocate() -> R {
    // the instrument can fail: a deliberate allocation is counted
    let t0 = TOTAL.load(Ordering::Relaxed);
    std::hint::black_box(Vec::<u8>::with_capacity(1));
    assert!(
        TOTAL.load(Ordering::Relaxed) > t0,
        "the counter does not count"
    );

    let mut audit = Audit::default();
    cases::<8>(&mut audit)?;
    cases::<16>(&mut audit)?;
    submap_cases::<8>(&mut audit)?;
    submap_cases::<16>(&mut audit)?;
    let mut bad = Vec::new();
    for (label, n) in &audit.rows {
        eprintln!("{label}: {n} allocations");
        if *n != 0 {
            bad.push(format!("{label}: {n}"));
        }
    }
    assert!(
        bad.is_empty(),
        "allocating warmed calls:\n{}",
        bad.join("\n")
    );
    Ok(())
}
