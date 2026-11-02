//! CPU adaptive-mean benchmark worker for benches/bench_adaptive_dataset.py.
//! Input is an uncompressed grayscale u8 buffer, loaded outside the timed loop.

use std::{
    env,
    error::Error,
    fs,
    hint::black_box,
    io::{self, BufRead, Write},
    path::Path,
    time::{Duration, Instant},
};

use kornia_image::{Image, ImageSize};
use kornia_imgproc::threshold::{adaptive_threshold_mean, AdaptiveThresholdMeanWorkspace};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().collect();
    if args.len() != 5 && !(args.len() == 6 && args[5] == "--server") {
        return Err("usage: bench_adaptive_dataset INPUT.raw WIDTH HEIGHT MASK_DIRECTORY".into());
    }
    let width = args[2].parse()?;
    let height = args[3].parse()?;
    let src = Image::<u8, 1>::new(ImageSize { width, height }, fs::read(&args[1])?)?;
    let mut dst = Image::from_size_val(src.size(), 0)?;
    let mut workspace = AdaptiveThresholdMeanWorkspace::new();
    let output = Path::new(&args[4]);
    fs::create_dir_all(output)?;
    const WARMUP: usize = 10;
    const ROUNDS: usize = 5;
    const ITERATIONS: usize = 20;
    const OFFSET: i16 = 10;
    if args.len() == 6 {
        for line in io::stdin().lock().lines() {
            let line = line?;
            let mut fields = line.split_whitespace();
            let command = fields.next().ok_or("missing command")?;
            let window: usize = fields.next().ok_or("missing window")?.parse()?;
            match command {
                "W" => {
                    let start = Instant::now();
                    let mut calls = 0;
                    while calls < WARMUP || start.elapsed() < Duration::from_millis(250) {
                        adaptive_threshold_mean(
                            black_box(&src),
                            &mut dst,
                            &mut workspace,
                            window,
                            OFFSET,
                            255,
                        )?;
                        black_box(dst.as_slice());
                        calls += 1;
                    }
                    println!("ok");
                }
                "R" => {
                    let iterations: usize = fields.next().ok_or("missing iterations")?.parse()?;
                    if iterations == 0 {
                        return Err("iterations must be positive".into());
                    }
                    let start = Instant::now();
                    for _ in 0..iterations {
                        adaptive_threshold_mean(
                            black_box(&src),
                            &mut dst,
                            &mut workspace,
                            window,
                            OFFSET,
                            255,
                        )?;
                        black_box(dst.as_slice());
                    }
                    println!(
                        "{:.9}",
                        start.elapsed().as_secs_f64() * 1000.0 / iterations as f64
                    );
                }
                "M" => {
                    adaptive_threshold_mean(&src, &mut dst, &mut workspace, window, OFFSET, 255)?;
                    fs::write(output.join(format!("{window}.raw")), dst.as_slice())?;
                    println!("ok");
                }
                _ => return Err("unknown command".into()),
            }
            io::stdout().flush()?;
        }
        return Ok(());
    }
    for window in [3, 5, 15, 31, 51, 67, 101] {
        let warmup_start = Instant::now();
        for _ in 0..WARMUP {
            adaptive_threshold_mean(
                black_box(&src),
                &mut dst,
                &mut workspace,
                window,
                OFFSET,
                255,
            )?;
            black_box(dst.as_slice());
        }
        while warmup_start.elapsed() < Duration::from_millis(250) {
            adaptive_threshold_mean(
                black_box(&src),
                &mut dst,
                &mut workspace,
                window,
                OFFSET,
                255,
            )?;
            black_box(dst.as_slice());
        }
        let mut samples = [0.0; ROUNDS];
        for sample in &mut samples {
            let start = Instant::now();
            for _ in 0..ITERATIONS {
                adaptive_threshold_mean(
                    black_box(&src),
                    &mut dst,
                    &mut workspace,
                    window,
                    OFFSET,
                    255,
                )?;
                black_box(dst.as_slice());
            }
            *sample = start.elapsed().as_secs_f64() * 1000.0 / ITERATIONS as f64;
        }
        fs::write(output.join(format!("{window}.raw")), dst.as_slice())?;
        print!("{window}");
        for sample in samples {
            print!(",{sample:.9}");
        }
        println!();
    }
    Ok(())
}
