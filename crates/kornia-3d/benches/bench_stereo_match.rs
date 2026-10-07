//! Sparse stereo matching on a 752x480 rectified pair (EuRoC's resolution): CPU
//! reference vs the CUDA twin, binary (ORB-like, 32 B) and float (XFeat-like, 64-D).
//!
//! `cargo bench -p kornia-3d --bench bench_stereo_match [--features cuda]`

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use kornia_3d::stereo::{
    StereoDescriptors, StereoKeypoints, StereoMatchConfig, StereoMatcher, StereoMatches,
};
use kornia_image::{Image, ImageSize};
use rand::{rngs::StdRng, RngExt, SeedableRng};

const W: usize = 752;
const H: usize = 480;

struct Scene {
    left: Image<u8, 1>,
    right: Image<u8, 1>,
    lxy: Vec<[f32; 2]>,
    rxy: Vec<[f32; 2]>,
    lbin: Vec<u8>,
    rbin: Vec<u8>,
    lf: Vec<f32>,
    rf: Vec<f32>,
}

/// Random texture with a constant 23 px disparity; `n` left keypoints, each with a
/// partner on the right plus 30% distractors.
fn scene(n: usize) -> Scene {
    let mut rng = StdRng::seed_from_u64(42);
    let d = 23usize;
    let base: Vec<u8> = (0..(W + d) * H).map(|_| rng.random::<u8>()).collect();
    let mut l = vec![0u8; W * H];
    let mut r = vec![0u8; W * H];
    for y in 0..H {
        for x in 0..W {
            l[y * W + x] = base[y * (W + d) + x];
            r[y * W + x] = base[y * (W + d) + x + d];
        }
    }
    let size = ImageSize {
        width: W,
        height: H,
    };
    let unit = |rng: &mut StdRng| -> Vec<f32> {
        let v: Vec<f32> = (0..64).map(|_| rng.random::<f32>() - 0.5).collect();
        let s = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.into_iter().map(|x| x / s).collect()
    };
    let mut s = Scene {
        left: Image::new(size, l).unwrap(),
        right: Image::new(size, r).unwrap(),
        lxy: Vec::new(),
        rxy: Vec::new(),
        lbin: Vec::new(),
        rbin: Vec::new(),
        lf: Vec::new(),
        rf: Vec::new(),
    };
    for _ in 0..n {
        let (x, y) = (
            rng.random_range(40..W - 20) as f32,
            rng.random_range(20..H - 20) as f32,
        );
        let b: Vec<u8> = (0..32).map(|_| rng.random::<u8>()).collect();
        let f = unit(&mut rng);
        s.lxy.push([x, y]);
        s.rxy.push([x - d as f32, y]);
        s.lbin.extend(&b);
        s.rbin.extend(&b);
        s.lf.extend(&f);
        s.rf.extend(&f);
    }
    for _ in 0..n * 3 / 10 {
        s.rxy
            .push([rng.random_range(0..W) as f32, rng.random_range(0..H) as f32]);
        s.rbin.extend((0..32).map(|_| rng.random::<u8>()));
        s.rf.extend(unit(&mut rng));
    }
    s
}

/// Borrows single-scale keypoints with either 32-byte binary or 64-float descriptors.
fn keypoints<'a>(
    xy: &'a [[f32; 2]],
    bin: &'a [u8],
    f: &'a [f32],
    binary: bool,
) -> StereoKeypoints<'a> {
    StereoKeypoints {
        xy,
        octaves: None,
        descriptors: if binary {
            StereoDescriptors::Binary {
                data: bin,
                bytes: 32,
            }
        } else {
            StereoDescriptors::Float { data: f, dim: 64 }
        },
    }
}

/// Benchmarks both descriptor kinds at 1000 and 2048 keypoints on CPU and, if enabled, CUDA.
fn bench_stereo_match(c: &mut Criterion) {
    let mut group = c.benchmark_group("StereoMatch");
    let m = StereoMatcher::new(StereoMatchConfig::new(435.0, 0.11)).unwrap();
    for n in [1000usize, 2048] {
        let s = scene(n);
        for binary in [true, false] {
            let kind = if binary { "bin32" } else { "f32x64" };
            let (l, r) = (
                keypoints(&s.lxy, &s.lbin, &s.lf, binary),
                keypoints(&s.rxy, &s.rbin, &s.rf, binary),
            );
            let lp = [s.left.clone()];
            let rp = [s.right.clone()];
            let mut out = StereoMatches::default();
            group.bench_with_input(BenchmarkId::new(format!("cpu_{kind}"), n), &n, |b, _| {
                b.iter(|| m.match_into(&lp, &rp, &l, &r, &mut out).unwrap())
            });
            #[cfg(feature = "cuda")]
            cuda::bench(&mut group, &m, &s, binary, kind, n);
        }
    }
    group.finish();
}

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    use criterion::measurement::WallTime;
    use criterion::BenchmarkGroup;
    use cudarc::driver::CudaContext;
    use kornia_3d::stereo::{CudaStereoDescriptors, CudaStereoKeypoints, KeypointCount};

    /// Benchmarks device-resident stereo matching, synchronizing after each iteration.
    /// Uploads and output allocation occur outside the timed loop.
    ///
    /// # Arguments
    ///
    /// * `group` - Criterion group receiving the CUDA benchmark.
    /// * `m` - CPU matcher whose configuration is copied to the CUDA backend.
    /// * `s` - Synthetic stereo images, keypoints, and descriptors to upload.
    /// * `binary` - Selects binary descriptors when true, float descriptors otherwise.
    /// * `kind` - Descriptor label used in the benchmark identifier.
    /// * `n` - Keypoint count used in the benchmark identifier and input.
    ///
    /// # Returns
    ///
    /// Registers and runs the benchmark without returning a value.
    ///
    /// # Errors
    ///
    /// Errors are not returned; setup and matching failures panic.
    ///
    /// # Panics
    ///
    /// Panics if CUDA initialization, uploads, allocation, matching, or synchronization fails.
    pub fn bench(
        group: &mut BenchmarkGroup<'_, WallTime>,
        m: &StereoMatcher,
        s: &Scene,
        binary: bool,
        kind: &str,
        n: usize,
    ) {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let mut dm = m.to_cuda(&stream).unwrap();
        let flat = |xy: &[[f32; 2]]| xy.iter().flatten().copied().collect::<Vec<f32>>();
        let (lxy, rxy) = (
            stream.clone_htod(&flat(&s.lxy)).unwrap(),
            stream.clone_htod(&flat(&s.rxy)).unwrap(),
        );
        let (lb, rb) = (
            stream.clone_htod(&s.lbin).unwrap(),
            stream.clone_htod(&s.rbin).unwrap(),
        );
        let (lf, rf) = (
            stream.clone_htod(&s.lf).unwrap(),
            stream.clone_htod(&s.rf).unwrap(),
        );
        let side = |xy, b, f, count| CudaStereoKeypoints {
            xy,
            octaves: None,
            descriptors: if binary {
                CudaStereoDescriptors::Binary { data: b, bytes: 32 }
            } else {
                CudaStereoDescriptors::Float { data: f, dim: 64 }
            },
            count: KeypointCount::Host(count),
        };
        let (l, r) = (
            side(&lxy, &lb, &lf, s.lxy.len()),
            side(&rxy, &rb, &rf, s.rxy.len()),
        );
        let lp = [s.left.to_cuda(&stream).unwrap()];
        let rp = [s.right.to_cuda(&stream).unwrap()];
        let mut out = dm.alloc_matches(s.lxy.len()).unwrap();
        group.bench_with_input(BenchmarkId::new(format!("cuda_{kind}"), n), &n, |b, _| {
            b.iter(|| {
                dm.match_device(&lp, &rp, &l, &r, &mut out).unwrap();
                stream.synchronize().unwrap();
            })
        });
    }
}

criterion_group!(benches, bench_stereo_match);
criterion_main!(benches);
