//! Quick LINKRUNS_PROF=1 helper: run sparse_noise 1024^2 8x with executor reuse
//! and dump per-call phase breakdown.

use kornia_imgproc::contours_linkruns::LinkRunsExecutor;

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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let s: usize = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(1024);
    let data = make_noise(s, s, 0xC0FFEE);
    let mut exec = LinkRunsExecutor::new();
    eprintln!("--- sparse_noise {s}x{s}, 8 reuse calls ---");
    for i in 0..8 {
        eprintln!("call {i}:");
        let t0 = std::time::Instant::now();
        exec.find_external_contours(&data, s, s);
        eprintln!("[lr-prof] TOTAL: {}us\n", t0.elapsed().as_micros());
    }
}
