//! cuda-oxide kernels, through their public launchers, vs the CPU path they are
//! byte-exact with. Adding a kernel = adding rows to `CASES`.
//!
//! Protocol (docs/cuda-oxide.md): owned stream, GPU output verified equal to
//! the CPU output first, then `CYCLES` cycles alternating which arm runs first.
//! The GPU arm runs `ROUNDS` x `ITERS` back-to-back launches bracketed by events
//! (kernel time, no transfers); the CPU arm times `ITERS` calls on the wall
//! clock. Each cycle keeps its fastest round. Reported: min and median of the
//! cycle minima, their spread, and the GPU clock seen after each round.
//! End-to-end numbers (H2D + kernel + D2H) stay in `bench_cuda_imgproc`.
//!
//! ```text
//! cargo bench -p kornia-imgproc --bench bench_cuda_oxide --features cuda
//! ```
//!
//! `harness = false` like the other CUDA benches: criterion's sampling cannot
//! express the alternated, event-timed min-of-rounds protocol. Arguments
//! (cargo passes `--bench`) are ignored.

use std::{sync::Arc, time::Instant};

use cudarc::driver::{sys::CUevent_flags, CudaContext, CudaSlice, CudaStream};
use kornia_image::{Image, ImageError, ImageSize};
use kornia_imgproc::{
    color::gray_from_rgb_u8,
    cuda::color::{gray::launch_gray_from_rgb_u8, CudaColorError},
};

const WARMUP: u32 = 50;
const ROUNDS: u32 = 5;
const ITERS: u32 = 100;
const CYCLES: usize = 5;
const GPU_FREQ: &str = "/sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu/cur_freq";

type GpuFn = fn(
    &Arc<CudaStream>,
    &CudaSlice<u8>,
    &mut CudaSlice<u8>,
    ImageSize,
) -> Result<(), CudaColorError>;

/// One row: an op at one size, its CPU path and its public GPU launcher.
struct Case {
    op: &'static str,
    size: (usize, usize),
    /// Input channels (the input is `w * h * in_ch` bytes).
    in_ch: usize,
    /// Bytes read + written per pixel, for GB/s.
    bytes_per_px: usize,
    /// Builds the CPU runner around the input bytes.
    cpu: fn(ImageSize, Vec<u8>) -> Box<dyn CpuRun>,
    gpu: GpuFn,
}

fn gray_cpu(size: ImageSize, input: Vec<u8>) -> Box<dyn CpuRun> {
    cpu_op::<3, 1>(size, input, gray_from_rgb_u8)
}

fn gray_gpu(
    stream: &Arc<CudaStream>,
    src: &CudaSlice<u8>,
    dst: &mut CudaSlice<u8>,
    size: ImageSize,
) -> Result<(), CudaColorError> {
    launch_gray_from_rgb_u8(stream, src, dst, size.width * size.height)
}

const fn gray(size: (usize, usize)) -> Case {
    Case {
        op: "gray_from_rgb_u8",
        size,
        in_ch: 3,
        bytes_per_px: 4,
        cpu: gray_cpu,
        gpu: gray_gpu,
    }
}

const CASES: &[Case] = &[gray((320, 240)), gray((1920, 1080)), gray((3840, 2160))];

/// A CPU op with its images preallocated.
trait CpuRun {
    fn run(&mut self);
    fn output(&self) -> &[u8];
}

struct CpuOp<const C: usize, const D: usize> {
    src: Image<u8, C>,
    dst: Image<u8, D>,
    f: fn(&Image<u8, C>, &mut Image<u8, D>) -> Result<(), ImageError>,
}

impl<const C: usize, const D: usize> CpuRun for CpuOp<C, D> {
    fn run(&mut self) {
        (self.f)(&self.src, &mut self.dst).expect("cpu op");
    }
    fn output(&self) -> &[u8] {
        self.dst.as_slice()
    }
}

fn cpu_op<const C: usize, const D: usize>(
    size: ImageSize,
    input: Vec<u8>,
    f: fn(&Image<u8, C>, &mut Image<u8, D>) -> Result<(), ImageError>,
) -> Box<dyn CpuRun> {
    Box::new(CpuOp {
        src: Image::new(size, input).expect("src image"),
        dst: Image::from_size_val(size, 0).expect("dst image"),
        f,
    })
}

fn gpu_mhz() -> u32 {
    std::fs::read_to_string(GPU_FREQ)
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .map_or(0, |hz| (hz / 1_000_000) as u32)
}

/// Fastest per-launch time (ms) over `ROUNDS`, and the clock range seen.
fn time_gpu(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    case: &Case,
    src: &CudaSlice<u8>,
    dst: &mut CudaSlice<u8>,
    size: ImageSize,
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
            (case.gpu)(stream, src, dst, size).expect("launch");
        }
        end.record(stream).expect("record");
        stream.synchronize().expect("sync");
        best = best.min(start.elapsed_ms(&end).expect("elapsed") as f64 / ITERS as f64);
        let mhz = gpu_mhz();
        (lo, hi) = (lo.min(mhz), hi.max(mhz));
    }
    (best, lo, hi)
}

/// Fastest per-call wall time (ms) over `ROUNDS`.
fn time_cpu(cpu: &mut dyn CpuRun) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..ROUNDS {
        let t = Instant::now();
        for _ in 0..ITERS {
            cpu.run();
        }
        best = best.min(t.elapsed().as_secs_f64() * 1e3 / ITERS as f64);
    }
    best
}

/// (min, median, spread %) of the cycle minima.
fn stats(v: &mut [f64]) -> (f64, f64, f64) {
    v.sort_by(f64::total_cmp);
    let (min, med, max) = (v[0], v[v.len() / 2], v[v.len() - 1]);
    (min, med, (max - min) / min * 100.0)
}

fn main() {
    let ctx = CudaContext::new(0).expect("CUDA device 0");
    let stream = ctx.new_stream().expect("owned stream");
    let (major, minor) = ctx.compute_capability().expect("cc");
    println!(
        "<!-- bench_cuda_oxide  GPU: {} (sm_{major}{minor})  warmup={WARMUP} rounds={ROUNDS} iters={ITERS} cycles={CYCLES}  CPU threads: {} -->",
        ctx.name().unwrap_or_default(),
        std::thread::available_parallelism().map_or(0, |n| n.get()),
    );
    println!();
    println!("| Op | Size | GPU min (ms) | GPU median | GPU spread | GB/s @ min | GPU MHz seen | CPU min (ms) | CPU median | CPU spread | CPU / GPU |");
    println!("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|");

    for case in CASES {
        let (w, h) = case.size;
        let size = ImageSize {
            width: w,
            height: h,
        };
        let n = w * h;
        let input: Vec<u8> = (0..n * case.in_ch)
            .map(|i| (i as u32).wrapping_mul(2_654_435_761).rotate_left(7) as u8)
            .collect();
        let src = stream.clone_htod(&input).expect("htod");
        let mut cpu = (case.cpu)(size, input);
        cpu.run();
        let mut dst = stream.alloc_zeros::<u8>(cpu.output().len()).expect("alloc");
        (case.gpu)(&stream, &src, &mut dst, size).expect("launch");
        let out = stream.clone_dtoh(&dst).expect("dtoh");
        assert_eq!(out, cpu.output(), "{} GPU != CPU at {w}x{h}", case.op);
        for _ in 0..WARMUP {
            (case.gpu)(&stream, &src, &mut dst, size).expect("warmup");
        }
        stream.synchronize().expect("sync");

        let (mut g, mut c) = (Vec::new(), Vec::new());
        let (mut lo, mut hi) = (u32::MAX, 0);
        for cycle in 0..CYCLES {
            let cpu_first = cycle % 2 == 1;
            if cpu_first {
                c.push(time_cpu(cpu.as_mut()));
            }
            let (t, l, hh) = time_gpu(&ctx, &stream, case, &src, &mut dst, size);
            g.push(t);
            (lo, hi) = (lo.min(l), hi.max(hh));
            if !cpu_first {
                c.push(time_cpu(cpu.as_mut()));
            }
        }
        let (gmin, gmed, gspread) = stats(&mut g);
        let (cmin, cmed, cspread) = stats(&mut c);
        let gbps = (n * case.bytes_per_px) as f64 / (gmin * 1e-3) / 1e9;
        println!(
            "| {} | {w}x{h} | {gmin:.4} | {gmed:.4} | {gspread:.1}% | {gbps:.1} | {lo}-{hi} | {cmin:.4} | {cmed:.4} | {cspread:.1}% | {:.1}x |",
            case.op,
            cmin / gmin
        );
    }
}
