use kornia_apriltag::{family::TagFamilyKind, AprilTagDecoder, DecodeTagsConfig};
use kornia_image::{allocator::CpuAllocator, Image};
use kornia_imgproc::color::gray_from_rgb_u8;
use kornia_io::jpeg::read_image_jpeg_rgb8;
use std::path::PathBuf;
use std::time::Instant;

fn main() {
    let img_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/data/apriltags_tag36h11.jpg");

    let img = read_image_jpeg_rgb8(img_path).unwrap();
    let mut src = Image::<u8, 1, CpuAllocator>::from_size_val(img.size(), 0, CpuAllocator).unwrap();
    gray_from_rgb_u8(&img, &mut src).unwrap();

    eprintln!("image: {}x{}", src.width(), src.height());

    let config = DecodeTagsConfig::new(vec![TagFamilyKind::Tag36H11]).unwrap();
    let mut det = AprilTagDecoder::new(config, src.size()).unwrap();

    // Warmup
    for _ in 0..20 {
        let _ = det.decode(&src).unwrap();
        det.clear();
    }

    // Run multiple batches and keep the minimum (min = least interference from scheduler/thermals).
    const BATCHES: usize = 5;
    const N: usize = 200;

    let labels = ["decimate", "threshold", "conn_comp", "gradient_clust", "fit_quads", "decode_tags"];
    let mut best_us = [u64::MAX; 6];
    let mut best_wall = f64::MAX;

    for _b in 0..BATCHES {
        let mut acc = [0u64; 6];
        let t_wall = Instant::now();
        for _ in 0..N {
            let (_, us) = det.decode_timed(&src).unwrap();
            det.clear();
            for (a, u) in acc.iter_mut().zip(us.iter()) {
                *a += u;
            }
        }
        let wall_us = t_wall.elapsed().as_micros() as f64 / N as f64;

        // Keep per-stage minimum across batches
        for (b, a) in best_us.iter_mut().zip(acc.iter()) {
            let per = *a / N as u64;
            if per < *b { *b = per; }
        }
        if wall_us < best_wall { best_wall = wall_us; }
    }

    eprintln!("\n--- per-stage (best of {} × {} iterations) ---", BATCHES, N);
    let mut subtotal = 0f64;
    for (label, &b) in labels.iter().zip(best_us.iter()) {
        let us = b as f64;
        subtotal += us;
        eprintln!("  {:<18} {:7.1} µs  ({:.1}%)", label, us, us / best_wall * 100.0);
    }
    eprintln!("  {:<18} {:7.1} µs  (sum of above)", "subtotal", subtotal);
    eprintln!("  {:<18} {:7.1} µs  (best wall clock)", "wall total", best_wall);
}
