//! Cold-start bench: NEW LinkRunsExecutor per iteration.
//! Measures the worst case for one-shot callers (no executor reuse).

use kornia_imgproc::contours_linkruns::LinkRunsExecutor;
use std::time::Instant;

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

fn make_filled(w: usize, h: usize) -> Vec<u8> {
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

fn median(mut s: Vec<f64>) -> f64 {
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s[s.len() / 2]
}

fn run(label: &str, w: usize, h: usize, data: &[u8], reps: usize) {
    // No warmup — measure cold start
    let mut samples = Vec::with_capacity(reps);
    for _ in 0..reps {
        let mut exec = LinkRunsExecutor::new();
        let t = Instant::now();
        let _ = exec.find_external_contours(data, w, h);
        samples.push(t.elapsed().as_secs_f64());
    }
    let med_us = median(samples) * 1e6;
    println!("cold,{label},{w}x{h},{med_us:.1}");
}

fn main() {
    println!("# CSV: mode,fixture,size,med_us");
    for s in [128, 256, 512, 1024] {
        run("sparse_noise", s, s, &make_noise(s, s, 0xC0FFEE), 10);
    }
    for s in [128, 512, 1024, 2048] {
        run("filled_square", s, s, &make_filled(s, s), 10);
    }
}
