//! Time `XFeat::extract()` directly — avoids criterion overhead, tests the real path.
//!
//! Loads the embedded weights and a real fixture image, runs 20 warm-up iterations,
//! then 50 timed iterations at 480×640.  Prints median, min, max latency in ms.
//!
//! Thread count is controlled by the `RAYON_NUM_THREADS` environment variable
//! (Rayon default: all logical cores) or the `--threads N` flag (which wins).
//! `--pin A-B` pins Rayon worker i to core A+i (and the main thread to the
//! same range) via raw `sched_setaffinity` — used to test whether avoiding
//! the IRQ-heavy core 0 tightens the latency distribution.
//!
//! Run examples:
//!   cargo run --release -p kornia-xfeat --example bench_extract
//!   RAYON_NUM_THREADS=1 cargo run --release -p kornia-xfeat --example bench_extract
//!   cargo run --release -p kornia-xfeat --example bench_extract -- --threads 5 --pin 1-5

use std::path::Path;
use std::time::Instant;

use kornia_xfeat::{
    preproc::{align_to_32, bilinear_resample_gray, rgb_u8_to_gray_f32},
    weights::PackedWeights,
    XFeat, XFeatConfig,
};

// ─── PNG loading (identical to detect_and_match.rs) ───────────────────────────

fn load_png(path: &Path) -> (Vec<u8>, usize, usize, usize) {
    let file =
        std::fs::File::open(path).unwrap_or_else(|e| panic!("cannot open {}: {e}", path.display()));
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info().expect("png read_info");
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).expect("png decode frame");
    let channels = info.color_type.samples();
    buf.truncate(info.buffer_size());
    (buf, info.height as usize, info.width as usize, channels)
}

fn preprocess_image(path: &Path, target_h: usize, target_w: usize) -> Vec<f32> {
    let (raw, h, w, channels) = load_png(path);

    let mut gray_f32 = vec![0.0f32; h * w];
    match channels {
        1 => kornia_xfeat::preproc::gray_u8_to_gray_f32(&raw, &mut gray_f32),
        3 => rgb_u8_to_gray_f32(&raw, &mut gray_f32, h, w),
        4 => {
            let rgb: Vec<u8> = raw
                .chunks_exact(4)
                .flat_map(|px| [px[0], px[1], px[2]])
                .collect();
            rgb_u8_to_gray_f32(&rgb, &mut gray_f32, h, w);
        }
        c => panic!("unsupported channel count {c}"),
    }

    let mut resized = vec![0.0f32; target_h * target_w];
    bilinear_resample_gray(&gray_f32, &mut resized, h, w, target_h, target_w);
    resized
}

/// Parse "A-B" into an inclusive core range.
fn parse_core_range(s: &str) -> Option<(u32, u32)> {
    let (a, b) = s.split_once('-')?;
    let a: u32 = a.parse().ok()?;
    let b: u32 = b.parse().ok()?;
    (a <= b && b < 64).then_some((a, b))
}

// ─── Percentile helper ────────────────────────────────────────────────────────

fn median_f64(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    }
}

// ─── Main ─────────────────────────────────────────────────────────────────────

fn main() {
    // ── CLI: --threads N  --pin A-B ───────────────────────────────────────────
    let args: Vec<String> = std::env::args().collect();
    let flag = |name: &str| -> Option<String> {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1).cloned())
    };
    let cli_threads: Option<usize> = flag("--threads").and_then(|v| v.parse().ok());
    let pin_range: Option<(u32, u32)> = flag("--pin").as_deref().and_then(parse_core_range);
    // `--timing` = convenience alias for XFEAT_TIMING=1 (per-layer t_lap! output).
    if args.iter().any(|a| a == "--timing") {
        std::env::set_var("XFEAT_TIMING", "1");
    }

    let threads = cli_threads.unwrap_or_else(|| {
        std::env::var("RAYON_NUM_THREADS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0) // 0 → Rayon default (all cores)
    });

    // Build the global pool explicitly when pinning (worker i → core A+i) or
    // when --threads was given. Must happen before any Rayon use.
    if let Some((a, b)) = pin_range {
        // The crate's production pinning API: one worker pinned per core of A-B.
        // (--threads is implied by the range size in this mode.)
        kornia_xfeat::affinity::init_pinned_threadpool(a, b)
            .expect("global Rayon pool already initialised?");
    } else if threads > 0 && cli_threads.is_some() {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .expect("global Rayon pool already initialised?");
    }

    let thread_label = match (threads, pin_range) {
        (_, Some((a, b))) => format!("{}, pinned {a}-{b}", b - a + 1),
        (0, None) => "all cores (Rayon default)".to_string(),
        (n, None) => format!("{n}"),
    };

    // Fix the target resolution to 480×640 (must be multiples of 32).
    let (target_h, target_w) = {
        // align_to_32 of 480×640 is 480×640 — but call it to be correct.
        align_to_32(480, 640)
    };
    assert!(
        target_h > 0 && target_w > 0,
        "align_to_32 returned zero for 480x640"
    );

    // Load the fixture image once; reuse the same pixel buffer every iteration.
    let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let img_path = crate_dir.join("tests/fixtures/v1/ref/input.png");
    eprintln!("Loading image: {}", img_path.display());
    let gray = preprocess_image(&img_path, target_h, target_w);
    eprintln!("  -> ({target_h} x {target_w}) f32 gray");

    // Load weights once; reuse across all iterations via cloning the bytes.
    let weights_bytes = kornia_xfeat::weights::embedded_bytes();

    let cfg = XFeatConfig {
        height: target_h,
        width: target_w,
        ..XFeatConfig::default()
    };

    // Construct the model once — this is the one-time cost (Winograd cache, etc.).
    let t_construct_start = Instant::now();
    let weights =
        PackedWeights::from_safetensors_bytes(weights_bytes).expect("embedded weights must parse");
    let mut model = XFeat::new(cfg.clone(), weights).expect("construct model");
    let t_construct = t_construct_start.elapsed();
    eprintln!(
        "Model construction (one-time): {:.1} ms",
        t_construct.as_secs_f64() * 1000.0
    );
    eprintln!("RAYON_NUM_THREADS={thread_label}");
    eprintln!();

    const WARMUP: usize = 20;
    const ITERS: usize = 50;

    // ── Warm-up ───────────────────────────────────────────────────────────────
    eprint!("Warm-up ({WARMUP} iters) ...");
    for _ in 0..WARMUP {
        let _ = model.extract(&gray).expect("extract (warmup)");
    }
    eprintln!(" done");

    // ── Timed iters ───────────────────────────────────────────────────────────
    eprint!("Timing ({ITERS} iters) ...");
    let mut samples_ms: Vec<f64> = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let t0 = Instant::now();
        let out = model.extract(&gray).expect("extract (timed)");
        let elapsed = t0.elapsed().as_secs_f64() * 1000.0;
        // Force the result to not be optimised away.
        std::hint::black_box(out.keypoints.len());
        samples_ms.push(elapsed);
    }
    eprintln!(" done");
    eprintln!();

    samples_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = median_f64(&samples_ms);
    let min = samples_ms[0];
    let max = samples_ms[samples_ms.len() - 1];
    let p95 = samples_ms[(ITERS as f64 * 0.95) as usize];

    // Report on stdout (parseable by scripts).
    println!("extract() latency @ {target_h}x{target_w}  threads={thread_label}");
    println!("  median : {med:.1} ms");
    println!("  min    : {min:.1} ms");
    println!("  max    : {max:.1} ms");
    println!("  p95    : {p95:.1} ms");

    // Speedup vs prior real extract() baseline (fp16+rayon wired 2026-06-08).
    // MT baseline = real extract() MT before fp16 GEMM + parallel postproc (267ms).
    // ST baseline = real extract() ST before fp16 GEMM + parallel postproc (1193ms).
    let baseline_mt = 267.0_f64;
    let baseline_st = 1193.0_f64;
    if threads == 1 {
        println!(
            "  speedup vs ST baseline ({baseline_st} ms): {:.2}×",
            baseline_st / med
        );
    } else {
        println!(
            "  speedup vs MT baseline ({baseline_mt} ms): {:.2}×",
            baseline_mt / med
        );
    }
}
