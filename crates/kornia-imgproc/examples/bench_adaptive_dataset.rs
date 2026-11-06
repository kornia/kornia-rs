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
use kornia_imgproc::threshold::AdaptiveThresholdMean;

/// Serve warmup, timed-round and output-mask requests over standard input.
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().collect();
    if args.len() != 6 || args[5] != "--server" {
        return Err(
            "usage: bench_adaptive_dataset INPUT.raw WIDTH HEIGHT MASK_DIRECTORY --server".into(),
        );
    }
    let width = args[2].parse()?;
    let height = args[3].parse()?;
    let src = Image::<u8, 1>::new(ImageSize { width, height }, fs::read(&args[1])?)?;
    let mut dst = Image::from_size_val(src.size(), 0)?;
    let mut current_window = 3;
    let mut threshold = AdaptiveThresholdMean::new(current_window, 10, 255)?;
    let output = Path::new(&args[4]);
    fs::create_dir_all(output)?;
    for line in io::stdin().lock().lines() {
        let line = line?;
        let mut fields = line.split_whitespace();
        let command = fields.next().ok_or("missing command")?;
        let window: usize = fields.next().ok_or("missing window")?.parse()?;
        if window != current_window {
            threshold = AdaptiveThresholdMean::new(window, 10, 255)?;
            current_window = window;
        }
        match command {
            "W" => {
                let start = Instant::now();
                let mut calls = 0;
                while calls < 10 || start.elapsed() < Duration::from_millis(250) {
                    threshold.apply(black_box(&src), &mut dst)?;
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
                    threshold.apply(black_box(&src), &mut dst)?;
                    black_box(dst.as_slice());
                }
                println!(
                    "{:.9}",
                    start.elapsed().as_secs_f64() * 1000.0 / iterations as f64
                );
            }
            "M" => {
                threshold.apply(&src, &mut dst)?;
                fs::write(output.join(format!("{window}.raw")), dst.as_slice())?;
                println!("ok");
            }
            _ => return Err("unknown command".into()),
        }
        io::stdout().flush()?;
    }
    Ok(())
}
