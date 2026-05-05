//! Head-to-head bench: LSL (kornia, this branch) vs Suzuki/Abe (kornia, main)
//! emitting CSV in the same shape as bench_opencv_contours.py so the rows can
//! be merged directly.
//!
//! Compare against External + CHAIN_APPROX_NONE rows from the python bench
//! (LSL currently always emits every boundary pixel — same semantics as NONE).

use kornia_image::{allocator::CpuAllocator, Image, ImageSize};
use kornia_imgproc::contours::{find_contours, ContourApproximationMode, RetrievalMode};
use kornia_imgproc::contours_lsl::LslExecutor;
use std::time::Instant;

const REPS: usize = 20;
const WARMUP: usize = 5;

fn make_filled_square(w: usize, h: usize) -> Vec<u8> {
    let mw = w / 8;
    let mh = h / 8;
    let mut d = vec![0u8; w * h];
    for r in mh..(h - mh) {
        for c in mw..(w - mw) {
            d[r * w + c] = 1;
        }
    }
    d
}

fn make_hollow_square(w: usize, h: usize) -> Vec<u8> {
    let ow = w / 8;
    let oh = h / 8;
    let iw = w / 4;
    let ih = h / 4;
    let mut d = vec![0u8; w * h];
    for r in oh..(h - oh) {
        for c in ow..(w - ow) {
            d[r * w + c] = 1;
        }
    }
    for r in ih..(h - ih) {
        for c in iw..(w - iw) {
            d[r * w + c] = 0;
        }
    }
    d
}

fn make_noise(w: usize, h: usize, seed: u64) -> Vec<u8> {
    let mut state = seed;
    (0..w * h)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) & 1) as u8
        })
        .collect()
}

fn median(mut s: Vec<f64>) -> f64 {
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s[s.len() / 2]
}

fn min_(s: &[f64]) -> f64 {
    s.iter().cloned().fold(f64::INFINITY, f64::min)
}

fn run_lsl(label: &str, w: usize, h: usize, data: &[u8]) {
    let mut exec = LslExecutor::new();
    for _ in 0..WARMUP {
        let _ = exec.find_external_contours(data, w, h);
    }
    let mut samples = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        let t = Instant::now();
        let _ = exec.find_external_contours(data, w, h);
        samples.push(t.elapsed().as_secs_f64());
    }
    let mn = min_(&samples) * 1e6;
    let md = median(samples.clone()) * 1e6;
    let mu = samples.iter().sum::<f64>() / samples.len() as f64 * 1e6;
    let mpix = (w * h) as f64 / md;
    println!("kornia_lsl,{label},{w}x{h},{mn:.1},{md:.1},{mu:.1},{mpix:.1}");
}

fn run_suzuki(label: &str, w: usize, h: usize, data: Vec<u8>) {
    let img = Image::<u8, 1, _>::new(
        ImageSize { width: w, height: h },
        data,
        CpuAllocator,
    )
    .unwrap();
    for _ in 0..WARMUP {
        let _ = find_contours(&img, RetrievalMode::External, ContourApproximationMode::None);
    }
    let mut samples = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        let t = Instant::now();
        let _ = find_contours(&img, RetrievalMode::External, ContourApproximationMode::None);
        samples.push(t.elapsed().as_secs_f64());
    }
    let mn = min_(&samples) * 1e6;
    let md = median(samples.clone()) * 1e6;
    let mu = samples.iter().sum::<f64>() / samples.len() as f64 * 1e6;
    let mpix = (w * h) as f64 / md;
    println!("kornia_suzuki,{label},{w}x{h},{mn:.1},{md:.1},{mu:.1},{mpix:.1}");
}

fn main() {
    println!("# CSV: impl,fixture,size,min_us,med_us,mean_us,Mpix_per_s_median");
    println!("impl,fixture,size,min_us,med_us,mean_us,Mpix_s");

    let sizes = [(128, 128), (256, 256), (512, 512), (1024, 1024), (2048, 2048)];

    for (w, h) in sizes {
        let d = make_filled_square(w, h);
        run_lsl("filled_square_external_none", w, h, &d);
        run_suzuki("filled_square_external_none", w, h, d);
    }
    for (w, h) in sizes {
        let d = make_hollow_square(w, h);
        run_lsl("hollow_square_external_none", w, h, &d);
        run_suzuki("hollow_square_external_none", w, h, d);
    }
    for (w, h) in sizes.iter().take(4) {
        let (w, h) = (*w, *h);
        let d = make_noise(w, h, 0xC0FFEE);
        run_lsl("sparse_noise_external_none", w, h, &d);
        run_suzuki("sparse_noise_external_none", w, h, d);
    }
}
