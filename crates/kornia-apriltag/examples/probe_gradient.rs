// Multi-run gradient sub-phase profiler with warmup.
// Prints sub-phase averages after the AtomicBool resets on each run via a counter approach.
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

    let config = DecodeTagsConfig::new(vec![TagFamilyKind::Tag36H11]).unwrap();
    let mut det = AprilTagDecoder::new(config, src.size()).unwrap();

    // Warmup
    eprintln!("warming up...");
    for _ in 0..30 {
        let _ = det.decode(&src).unwrap();
        det.clear();
    }
    eprintln!("warmup done — now timing 200 iterations...");

    // Time 200 iterations, measuring total gradient_clust wall time
    let mut total_gradient_us = 0u64;
    for _ in 0..200 {
        let t = Instant::now();
        let (_, us) = det.decode_timed(&src).unwrap();
        total_gradient_us += us[3];
        det.clear();
        let _ = t.elapsed();
    }
    eprintln!("gradient_clust avg: {} µs", total_gradient_us / 200);
    eprintln!("(check the ONE sub-phase line that was printed during warmup above)");
}
