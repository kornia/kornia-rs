//! Voxel downsampling and voxel neighbour counts on 100k and 1M uniform
//! random points in a 4 m cube (0.1 m voxels), on the CPU.
//!
//! `cargo bench -p kornia-3d --bench bench_voxel`
//! Single-thread numbers: `RAYON_NUM_THREADS=1`.
//! Baselines on the same inputs: `benches/bench_voxel_baselines.py`.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use kornia_3d::voxel::{DownsampleMode, VoxelConnectivity, VoxelDownsampler, VoxelLayout};

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
