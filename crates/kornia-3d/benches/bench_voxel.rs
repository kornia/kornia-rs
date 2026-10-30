//! Voxel downsampling and voxel neighbour counts on 100k and 1M uniform
//! random points in a 4 m cube (0.1 m voxels), on the CPU.
//! `voxel_downsample_*_attrs_cpu` times `voxel_downsample` on the same
//! cloud with colors and normals, so it includes the attribute reduction
//! (or copy). `view_*_cpu` times the zero-copy [`VoxelView`] reductions on a
//! reused scratch (`VoxelDownsampler::view`), `view_first_to_point_cloud_cpu`
//! adds the copy into an owned cloud, and `remove_sparse_voxels26_*` times
//! the density filter returning its view, and with the copy.
//!
//! `cargo bench -p kornia-3d --bench bench_voxel`
//! Single-thread numbers: `RAYON_NUM_THREADS=1`.
//! Baselines on the same inputs: `benches/bench_voxel_baselines.py`.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use kornia_3d::pointcloud::PointCloud;
use kornia_3d::voxel::{
    remove_sparse_voxels, voxel_downsample, DownsampleMode, VoxelConnectivity, VoxelDownsampler,
    VoxelLayout, VoxelView,
};

const SIZES: [usize; 2] = [100_000, 1_000_000];
const EXTENT_M: f64 = 4.0;
const VOXEL_M: f64 = 0.1;

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
        let colors = pts
            .iter()
            .map(|p| p.map(|v| (v / EXTENT_M * 255.0) as u8))
            .collect();
        let normals = pts
            .iter()
            .map(|p| {
                let len = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt().max(1e-12);
                p.map(|v| v / len)
            })
            .collect();
        let cloud = PointCloud::new(pts.clone(), Some(colors), Some(normals));
        for (name, mode) in [
            (
                "voxel_downsample_centroid_attrs_cpu",
                DownsampleMode::Centroid,
            ),
            ("voxel_downsample_first_attrs_cpu", DownsampleMode::First),
            (
                "voxel_downsample_nearest_attrs_cpu",
                DownsampleMode::NearestToCentre,
            ),
        ] {
            group.bench_function(BenchmarkId::new(name, &id), |b| {
                b.iter(|| voxel_downsample(&cloud, &layout, mode).unwrap())
            });
        }
        group.bench_function(BenchmarkId::new("view_first_cpu", &id), |b| {
            b.iter(|| ds.view(&cloud, &layout).unwrap().first().len())
        });
        group.bench_function(BenchmarkId::new("view_nearest_cpu", &id), |b| {
            b.iter(|| ds.view(&cloud, &layout).unwrap().nearest().len())
        });
        group.bench_function(BenchmarkId::new("view_centroids_attrs_cpu", &id), |b| {
            b.iter(|| ds.view(&cloud, &layout).unwrap().centroids())
        });
        group.bench_function(
            BenchmarkId::new("view_first_to_point_cloud_cpu", &id),
            |b| b.iter(|| ds.view(&cloud, &layout).unwrap().first().to_point_cloud()),
        );
        group.bench_function(BenchmarkId::new("voxel_view_new_cpu", &id), |b| {
            b.iter(|| VoxelView::new(&cloud, &layout).unwrap().len())
        });
        let conn = VoxelConnectivity::TwentySix;
        group.bench_function(
            BenchmarkId::new("remove_sparse_voxels26_view_cpu", &id),
            |b| {
                b.iter(|| {
                    remove_sparse_voxels(&cloud, &layout, conn, 3)
                        .unwrap()
                        .len()
                })
            },
        );
        group.bench_function(
            BenchmarkId::new("remove_sparse_voxels26_attrs_cpu", &id),
            |b| {
                b.iter(|| {
                    remove_sparse_voxels(&cloud, &layout, conn, 3)
                        .unwrap()
                        .to_point_cloud()
                })
            },
        );
        group.bench_function(BenchmarkId::new("neighbour_counts26_cpu", &id), |b| {
            b.iter(|| {
                ds.neighbour_counts_into(&pts, &layout, VoxelConnectivity::TwentySix, &mut counts)
                    .unwrap()
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_voxel);
criterion_main!(benches);
