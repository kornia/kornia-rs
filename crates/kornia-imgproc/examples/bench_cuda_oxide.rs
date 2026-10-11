//! Interleaved A/B: cuda-oxide kernels vs their NVRTC twins.
//!
//! Protocol (see docs/cuda-oxide.md): owned stream, both arms verified equal
//! first, then `CYCLES` cycles that alternate which arm runs first. Each arm
//! runs `ROUNDS` x `ITERS` back-to-back launches bracketed by events, synced
//! per round; a cycle keeps its fastest round. Reported: min over cycles and
//! the spread of the cycle minima, plus the GPU clock seen after each round.
//!
//! ```text
//! cargo run --release -p kornia-imgproc --example bench_cuda_oxide --features cuda-oxide
//! ```

use std::sync::Arc;

use cudarc::driver::{sys::CUevent_flags, CudaContext, CudaSlice, CudaStream};
use kornia_imgproc::cuda::{
    color::{gray::launch_gray_from_rgb_u8_nvrtc, CudaColorError},
    oxide,
};

const WARMUP: u32 = 50;
const ROUNDS: u32 = 5;
const ITERS: u32 = 100;
const CYCLES: usize = 5;
const SIZES: &[(usize, usize)] = &[(320, 240), (1920, 1080), (3840, 2160)];
const GPU_FREQ: &str = "/sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu/cur_freq";

type Launch =
    fn(&Arc<CudaStream>, &CudaSlice<u8>, &mut CudaSlice<u8>, usize) -> Result<(), CudaColorError>;

const ARMS: [(&str, Launch); 2] = [
    ("NVRTC", launch_gray_from_rgb_u8_nvrtc),
    ("oxide", oxide::launch_gray_from_rgb_u8),
];

fn gpu_mhz() -> u32 {
    std::fs::read_to_string(GPU_FREQ)
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .map_or(0, |hz| (hz / 1_000_000) as u32)
}

/// Fastest per-launch time (ms) over `ROUNDS`, and the clock range seen.
fn time_arm(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    launch: Launch,
    src: &CudaSlice<u8>,
    dst: &mut CudaSlice<u8>,
    n: usize,
) -> (f64, u32, u32) {
    let ev = || {
        ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
            .expect("event")
    };
    let (start, end) = (ev(), ev());
    let (mut best, mut lo, mut hi) = (f64::INFINITY, u32::MAX, 0);
    for _ in 0..ROUNDS {
        start.record(stream).expect("record");
        for _ in 0..ITERS {
            launch(stream, src, dst, n).expect("launch");
        }
        end.record(stream).expect("record");
        stream.synchronize().expect("sync");
        best = best.min(start.elapsed_ms(&end).expect("elapsed") as f64 / ITERS as f64);
        let mhz = gpu_mhz();
        (lo, hi) = (lo.min(mhz), hi.max(mhz));
    }
    (best, lo, hi)
}

fn main() {
    let ctx = CudaContext::new(0).expect("CUDA device 0");
    let stream = ctx.new_stream().expect("owned stream");
    oxide::ensure_loaded(&stream).expect("oxide PTX must load, or both arms are NVRTC");
    let (major, minor) = ctx.compute_capability().expect("cc");
    println!(
        "<!-- bench_cuda_oxide  GPU: {} (sm_{major}{minor})  warmup={WARMUP} rounds={ROUNDS} iters={ITERS} cycles={CYCLES} -->",
        ctx.name().unwrap_or_default()
    );
    println!();
    println!("| Op | Size | Arm | Min (ms) | Median of cycle mins (ms) | Spread | GB/s @ min | GPU MHz seen | oxide / NVRTC |");
    println!("|---|---|---|---:|---:|---:|---:|---:|---:|");

    for &(w, h) in SIZES {
        let n = w * h;
        let host: Vec<u8> = (0..n * 3)
            .map(|i| (i as u32).wrapping_mul(2_654_435_761).rotate_left(7) as u8)
            .collect();
        let src = stream.clone_htod(&host).expect("htod");
        let mut dst = stream.alloc_zeros::<u8>(n).expect("alloc");

        let mut outs = Vec::new();
        for (_, launch) in ARMS {
            launch(&stream, &src, &mut dst, n).expect("launch");
            outs.push(stream.clone_dtoh(&dst).expect("dtoh"));
        }
        assert_eq!(outs[0], outs[1], "arms disagree at {w}x{h}");

        for (_, launch) in ARMS {
            for _ in 0..WARMUP {
                launch(&stream, &src, &mut dst, n).expect("warmup");
            }
        }
        stream.synchronize().expect("sync");

        let mut cycle_mins: [Vec<f64>; 2] = [Vec::new(), Vec::new()];
        let mut mhz = [(u32::MAX, 0u32); 2];
        for c in 0..CYCLES {
            let order = if c % 2 == 0 { [0, 1] } else { [1, 0] };
            for a in order {
                let (t, lo, hi) = time_arm(&ctx, &stream, ARMS[a].1, &src, &mut dst, n);
                cycle_mins[a].push(t);
                mhz[a] = (mhz[a].0.min(lo), mhz[a].1.max(hi));
            }
        }

        let mut nvrtc_min = f64::NAN;
        for (((name, _), v), (lo, hi)) in ARMS.iter().zip(&mut cycle_mins).zip(mhz) {
            v.sort_by(f64::total_cmp);
            let (min, med, max) = (v[0], v[CYCLES / 2], v[CYCLES - 1]);
            let ratio = if nvrtc_min.is_nan() {
                nvrtc_min = min;
                String::new()
            } else {
                format!("{:.3}", min / nvrtc_min)
            };
            let gbps = (n * 4) as f64 / (min * 1e-3) / 1e9;
            let spread = (max - min) / min * 100.0;
            println!(
                "| gray_from_rgb_u8 | {w}x{h} | {name} | {min:.4} | {med:.4} | {spread:.1}% | {gbps:.1} | {lo}-{hi} | {ratio} |"
            );
        }
    }
}
