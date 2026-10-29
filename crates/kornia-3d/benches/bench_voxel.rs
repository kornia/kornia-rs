//! Voxel downsampling, voxel neighbour counts and the radius / statistical
//! outlier filters on 100k and 1M uniform random points in a 4 m cube
//! (0.1 m voxels), on the CPU.
//!
//! The outlier filters also run on the same cloud with a few NaN points
//! (`*_nan_cpu`: non-finite points are dropped before the grid search, which
//! must stay in use) and on a clustered cloud (`*_clustered_cpu`: dense
//! clusters on a sparse background, where the statistical filter shrinks
//! its cells and sends the background queries to the k-d tree).
//!
//! `cargo bench -p kornia-3d --bench bench_voxel`
//! Single-thread numbers: `RAYON_NUM_THREADS=1`.
//! Baselines on the same inputs: `benches/bench_voxel_baselines.py`.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use kornia_3d::pointcloud::filter::{radius_inlier_indices, statistical_inlier_indices};
use kornia_3d::voxel::{DownsampleMode, VoxelConnectivity, VoxelDownsampler, VoxelLayout};

const SIZES: [usize; 2] = [100_000, 1_000_000];
const EXTENT_M: f64 = 4.0;
const VOXEL_M: f64 = 0.1;
const RADIUS_M: f64 = 0.05;
const RADIUS_MIN_NEIGHBOURS: usize = 2;
const STAT_K: usize = 16;
const STAT_STD_RATIO: f64 = 1.0;
/// NaN points spliced into the `*_nan_cpu` clouds.
const NAN_POINTS: usize = 3;
/// Clusters of the clustered cloud, and the edge of each cluster's cube.
const CLUSTERS: usize = 20;
const CLUSTER_EDGE_M: f64 = 0.05;
/// One point in this many of the clustered cloud is uniform background.
const BACKGROUND_EVERY: usize = 100;

/// xorshift64 points in `[0, EXTENT_M)³`; the Python baselines use the same
/// generator.
fn points(n: usize) -> Vec<[f64; 3]> {
    let mut s = 0x2545_f491_4f6c_dd1du64;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64 * EXTENT_M
    };
    (0..n).map(|_| [next(), next(), next()]).collect()
}

/// `points(n)` with [`NAN_POINTS`] of them, evenly spaced, set to NaN.
fn points_with_nan(n: usize) -> Vec<[f64; 3]> {
    let mut pts = points(n);
    for i in 0..NAN_POINTS {
        pts[i * n / NAN_POINTS] = [f64::NAN; 3];
    }
    pts
}

/// [`CLUSTERS`] cubes of edge [`CLUSTER_EDGE_M`] at random centres, plus
/// one uniform background point in [`BACKGROUND_EVERY`].
fn clustered_points(n: usize) -> Vec<[f64; 3]> {
    let base = points(n + CLUSTERS);
    let (centres, offsets) = base.split_at(CLUSTERS);
    let scale = CLUSTER_EDGE_M / EXTENT_M;
    offsets
        .iter()
        .enumerate()
        .map(|(i, p)| {
            if i % BACKGROUND_EVERY == 0 {
                return *p;
            }
            let c = centres[i % CLUSTERS];
            [
                c[0] + p[0] * scale,
                c[1] + p[1] * scale,
                c[2] + p[2] * scale,
            ]
        })
        .collect()
}

fn bench_voxel(c: &mut Criterion) {
    let mut group = c.benchmark_group("Voxel");
    group.sample_size(10);
    let layout = VoxelLayout::new(VOXEL_M, [0.0; 3]).unwrap();
    for n in SIZES {
        let id = format!("{}k", n / 1000);
        let pts = points(n);
        let mut ds = VoxelDownsampler::new();
        let mut out = Vec::with_capacity(n);
        let mut index = Vec::with_capacity(n);
        let mut counts = Vec::with_capacity(n);
        let mut kept = Vec::with_capacity(n);

        for (name, mode) in [
            ("downsample_centroid_cpu", DownsampleMode::Centroid),
            ("downsample_first_cpu", DownsampleMode::First),
            ("downsample_nearest_cpu", DownsampleMode::NearestToCentre),
        ] {
            group.bench_function(BenchmarkId::new(name, &id), |b| {
                b.iter(|| {
                    ds.downsample_into(&pts, &layout, mode, &mut out, Some(&mut index))
                        .unwrap()
                })
            });
        }
        group.bench_function(BenchmarkId::new("neighbour_counts26_cpu", &id), |b| {
            b.iter(|| {
                ds.neighbour_counts_into(&pts, &layout, VoxelConnectivity::TwentySix, &mut counts)
                    .unwrap()
            })
        });
        group.bench_function(BenchmarkId::new("radius_outliers_cpu", &id), |b| {
            b.iter(|| {
                radius_inlier_indices(&pts, RADIUS_M, RADIUS_MIN_NEIGHBOURS, &mut kept).unwrap()
            })
        });
        group.bench_function(BenchmarkId::new("statistical_outliers_cpu", &id), |b| {
            b.iter(|| statistical_inlier_indices(&pts, STAT_K, STAT_STD_RATIO, &mut kept).unwrap())
        });
        for (suffix, cloud) in [
            ("nan", points_with_nan(n)),
            ("clustered", clustered_points(n)),
        ] {
            group.bench_function(
                BenchmarkId::new(format!("radius_outliers_{suffix}_cpu"), &id),
                |b| {
                    b.iter(|| {
                        radius_inlier_indices(&cloud, RADIUS_M, RADIUS_MIN_NEIGHBOURS, &mut kept)
                            .unwrap()
                    })
                },
            );
            group.bench_function(
                BenchmarkId::new(format!("statistical_outliers_{suffix}_cpu"), &id),
                |b| {
                    b.iter(|| {
                        statistical_inlier_indices(&cloud, STAT_K, STAT_STD_RATIO, &mut kept)
                            .unwrap()
                    })
                },
            );
        }
    }
    group.finish();
}

criterion_group!(benches, bench_voxel);
criterion_main!(benches);
