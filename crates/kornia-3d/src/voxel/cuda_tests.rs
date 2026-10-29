//! Host vs device parity: the CUDA voxel ops must equal the CPU exactly,
//! in the same order.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaStream};

use super::tests::random_points;
use super::*;

type TestResult = Result<(), Box<dyn std::error::Error>>;

/// Below one sort tile, exactly one tile, one past it (padding), more than
/// one scan tile, and enough for several global merge stages.
const SIZES: [usize; 7] = [1, 7, 2048, 2049, 5000, 70_001, 300_000];

const MODES: [DownsampleMode; 3] = [
    DownsampleMode::Centroid,
    DownsampleMode::First,
    DownsampleMode::NearestToCentre,
];

const CONNECTIVITIES: [VoxelConnectivity; 3] = [
    VoxelConnectivity::Six,
    VoxelConnectivity::Eighteen,
    VoxelConnectivity::TwentySix,
];

fn stream() -> Result<Arc<CudaStream>, Box<dyn std::error::Error>> {
    Ok(CudaContext::new(0)?.default_stream())
}

/// Random points straddling zero, with every fifth point duplicated so
/// that voxels hold several points with equal coordinates.
fn cloud(n: usize, seed: u64) -> Vec<[f64; 3]> {
    let mut p = random_points(n, -5.0, 5.0, seed);
    for i in (0..n).step_by(5) {
        let j = (i * 7 + 3) % n;
        p[j] = p[i];
    }
    p
}

fn assert_downsample_parity(
    filter: &mut CudaVoxelFilter,
    points: &[[f64; 3]],
    layout: &VoxelLayout,
) -> TestResult {
    let st = filter.stream().clone();
    let dev = st.clone_htod(points.as_flattened())?;
    let mut out = filter.alloc_voxel_cloud(points.len())?;
    let mut ds = VoxelDownsampler::new();
    let (mut host, mut host_index) = (Vec::new(), Vec::new());
    for mode in MODES {
        ds.downsample_into(points, layout, mode, &mut host, Some(&mut host_index))?;
        filter.downsample(&dev, points.len(), layout, mode, &mut out)?;
        let device = out.to_host()?;
        let device_index = out.index_to_host()?;
        assert_eq!(device.len(), host.len(), "{mode:?} n={}", points.len());
        // Bit-exact: compare the bit patterns, not with a tolerance.
        let bits =
            |v: &[[f64; 3]]| -> Vec<[u64; 3]> { v.iter().map(|p| p.map(f64::to_bits)).collect() };
        assert_eq!(
            bits(device.points()),
            bits(&host),
            "{mode:?} n={}",
            points.len()
        );
        assert_eq!(device_index, host_index, "{mode:?} n={}", points.len());
    }
    Ok(())
}

fn assert_counts_parity(
    filter: &mut CudaVoxelFilter,
    points: &[[f64; 3]],
    layout: &VoxelLayout,
) -> TestResult {
    let st = filter.stream().clone();
    let dev = st.clone_htod(points.as_flattened())?;
    let mut out = filter.alloc_neighbour_counts(points.len())?;
    let mut ds = VoxelDownsampler::new();
    let mut host = Vec::new();
    for conn in CONNECTIVITIES {
        ds.neighbour_counts_into(points, layout, conn, &mut host)?;
        filter.neighbour_counts(&dev, points.len(), layout, conn, &mut out)?;
        assert_eq!(out.len(), points.len());
        assert_eq!(out.to_host()?, host, "{conn:?} n={}", points.len());
    }
    Ok(())
}

#[test]
fn downsample_matches_host_exactly() -> TestResult {
    let mut filter = CudaVoxelFilter::new(&stream()?)?;
    let layout = VoxelLayout::new(0.3, [0.05, -0.1, 0.0])?;
    for (i, &n) in SIZES.iter().enumerate() {
        assert_downsample_parity(&mut filter, &cloud(n, 100 + i as u64), &layout)?;
    }
    Ok(())
}

#[test]
fn downsample_matches_host_with_one_point_per_voxel_and_one_voxel() -> TestResult {
    let mut filter = CudaVoxelFilter::new(&stream()?)?;
    let p = cloud(40_000, 3);
    // Tiny voxels: (almost) every point alone; huge voxel: one run of all.
    assert_downsample_parity(&mut filter, &p, &VoxelLayout::new(1e-6, [0.0; 3])?)?;
    assert_downsample_parity(&mut filter, &p, &VoxelLayout::new(100.0, [-50.0; 3])?)?;
    Ok(())
}

/// Clouds whose key range makes one, two or three radix digits vary, so
/// the sorted words end in either ping-pong buffer.
#[test]
fn downsample_matches_host_for_each_radix_pass_parity() -> TestResult {
    let mut filter = CudaVoxelFilter::new(&stream()?)?;
    let layout = VoxelLayout::new(1.0, [0.0; 3])?;
    let line = |n: usize, axis: usize, span: f64| -> Vec<[f64; 3]> {
        random_points(n, 0.0, 1.0, 29 + axis as u64)
            .into_iter()
            .map(|mut p| {
                p[axis] *= span;
                p
            })
            .collect()
    };
    // z within 200 voxels: one varying digit (odd).
    let z_only = line(9000, 2, 200.0);
    // x across 300 voxels, past a digit boundary: two digits (even).
    let x_only = line(9000, 0, 300.0);
    // y within 100 voxels plus x across 300: three digits (odd).
    let mut xy = line(9000, 0, 300.0);
    for (p, q) in xy.iter_mut().zip(line(9000, 1, 100.0)) {
        p[1] = q[1];
    }
    for p in [z_only, x_only, xy] {
        assert_downsample_parity(&mut filter, &p, &layout)?;
        assert_counts_parity(&mut filter, &p, &layout)?;
    }
    Ok(())
}

/// Points on voxel boundaries and one or two doubles either side, where
/// the multiply-screened key falls back to the division.
#[test]
fn downsample_matches_host_on_voxel_boundaries() -> TestResult {
    let mut filter = CudaVoxelFilter::new(&stream()?)?;
    for (s, o) in [
        (0.1, 0.0),
        (1.0 / 3.0, -0.05),
        (0.07, 1.25),
        (1e-3, 1234.5678),
        (1e-41, 0.0),
        (1e39, -1e40),
        (2.5e-8, -3e-7),
    ] {
        let layout = VoxelLayout::new(s, [o; 3])?;
        let mut p = Vec::new();
        for k in -40i32..40 {
            let b = o + f64::from(k) * s;
            let near = [
                b.next_down().next_down(),
                b.next_down(),
                b,
                b.next_up(),
                b.next_up().next_up(),
            ];
            for (j, &x) in near.iter().enumerate() {
                p.push([x, near[(j + 2) % 5], o + f64::from(k % 7) * s]);
            }
        }
        assert_downsample_parity(&mut filter, &p, &layout)?;
        assert_counts_parity(&mut filter, &p, &layout)?;
    }
    Ok(())
}

#[test]
fn downsample_matches_host_near_the_key_limits() -> TestResult {
    let mut filter = CudaVoxelFilter::new(&stream()?)?;
    let layout = VoxelLayout::new(1.0, [0.0; 3])?;
    let (max, min) = (i32::MAX as f64, i32::MIN as f64);
    let mut p = vec![
        [max + 0.5, min, 0.5],
        [max - 0.5, min + 0.25, -0.5],
        [min, max + 0.75, -1e-9],
        [0.0, -0.0, 0.0],
        [-0.5, 0.5, -0.5],
    ];
    p.extend(random_points(3000, -1.5, 1.5, 17));
    assert_downsample_parity(&mut filter, &p, &layout)?;
    assert_counts_parity(&mut filter, &p, &layout)?;
    Ok(())
}

#[test]
fn neighbour_counts_match_host_exactly() -> TestResult {
    let mut filter = CudaVoxelFilter::new(&stream()?)?;
    let layout = VoxelLayout::new(0.45, [0.0; 3])?;
    for (i, &n) in SIZES.iter().enumerate() {
        assert_counts_parity(&mut filter, &cloud(n, 200 + i as u64), &layout)?;
    }
    Ok(())
}

#[test]
fn device_runs_are_deterministic() -> TestResult {
    let mut filter = CudaVoxelFilter::new(&stream()?)?;
    let st = filter.stream().clone();
    let p = cloud(123_457, 9);
    let layout = VoxelLayout::new(0.2, [0.0; 3])?;
    let dev = st.clone_htod(p.as_flattened())?;
    let mut out = filter.alloc_voxel_cloud(p.len())?;
    filter.downsample(&dev, p.len(), &layout, DownsampleMode::Centroid, &mut out)?;
    let a = out.to_host()?;
    filter.downsample(&dev, p.len(), &layout, DownsampleMode::Centroid, &mut out)?;
    let b = out.to_host()?;
    assert_eq!(a.points(), b.points());
    Ok(())
}

#[test]
fn device_empty_cloud() -> TestResult {
    let mut filter = CudaVoxelFilter::new(&stream()?)?;
    let st = filter.stream().clone();
    let dev = st.alloc_zeros::<f64>(3)?;
    let layout = VoxelLayout::new(1.0, [0.0; 3])?;
    let mut out = filter.alloc_voxel_cloud(4)?;
    filter.downsample(&dev, 0, &layout, DownsampleMode::First, &mut out)?;
    assert_eq!(out.len_to_host()?, 0);
    assert!(out.to_host()?.is_empty());
    let mut counts = filter.alloc_neighbour_counts(4)?;
    filter.neighbour_counts(&dev, 0, &layout, VoxelConnectivity::Six, &mut counts)?;
    assert!(counts.to_host()?.is_empty());
    Ok(())
}

#[test]
fn device_reports_the_same_invalid_point_as_host() -> TestResult {
    let mut filter = CudaVoxelFilter::new(&stream()?)?;
    let st = filter.stream().clone();
    let layout = VoxelLayout::new(1.0, [0.0; 3])?;
    let mut p = cloud(5000, 4);
    p[4321] = [f64::NAN, 0.0, 0.0];
    p[1234] = [0.0, 1e12, 0.0];
    let dev = st.clone_htod(p.as_flattened())?;
    let mut out = filter.alloc_voxel_cloud(p.len())?;
    filter.downsample(&dev, p.len(), &layout, DownsampleMode::Centroid, &mut out)?;
    assert!(matches!(
        out.to_host(),
        Err(VoxelError::PointOutOfRange { index: 1234 })
    ));
    let mut counts = filter.alloc_neighbour_counts(p.len())?;
    filter.neighbour_counts(&dev, p.len(), &layout, VoxelConnectivity::Six, &mut counts)?;
    assert!(matches!(
        counts.to_host(),
        Err(VoxelError::PointOutOfRange { index: 1234 })
    ));
    assert!(matches!(
        voxel_neighbour_counts(&p, &layout, VoxelConnectivity::Six),
        Err(VoxelError::PointOutOfRange { index: 1234 })
    ));

    // The error does not stick: a valid call afterwards succeeds.
    let ok = cloud(100, 5);
    let dev = st.clone_htod(ok.as_flattened())?;
    filter.downsample(&dev, ok.len(), &layout, DownsampleMode::Centroid, &mut out)?;
    assert!(out.to_host().is_ok());
    Ok(())
}

#[test]
fn device_rejects_bad_buffers() -> TestResult {
    let st = stream()?;
    let mut filter = CudaVoxelFilter::new(&st)?;
    let layout = VoxelLayout::new(1.0, [0.0; 3])?;
    let dev = st.alloc_zeros::<f64>(30)?;
    let mut small = filter.alloc_voxel_cloud(5)?;
    assert!(matches!(
        filter.downsample(&dev, 10, &layout, DownsampleMode::First, &mut small),
        Err(VoxelError::CapacityTooSmall {
            capacity: 5,
            needed: 10
        })
    ));
    let mut out = filter.alloc_voxel_cloud(20)?;
    assert!(matches!(
        filter.downsample(&dev, 11, &layout, DownsampleMode::First, &mut out),
        Err(VoxelError::PointBufferTooSmall {
            len: 30,
            needed: 33
        })
    ));
    let other = CudaVoxelFilter::new(&st.context().new_stream()?)?;
    let mut foreign = other.alloc_voxel_cloud(20)?;
    assert!(matches!(
        filter.downsample(&dev, 10, &layout, DownsampleMode::First, &mut foreign),
        Err(VoxelError::StreamMismatch)
    ));
    Ok(())
}
