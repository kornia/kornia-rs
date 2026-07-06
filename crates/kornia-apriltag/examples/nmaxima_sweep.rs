//! Sweep `FitQuadConfig::max_nmaxima` and report speed vs detection parity.
//!
//! For each candidate value: run the detector over the test images, compare
//! the detection set (family, id, center within 0.5 px) against the
//! nmaxima=10 baseline, and time the fit_quads stage on the bench image.

use kornia_apriltag::{
    decoder::Detection, family::TagFamilyKind, AprilTagDecoder, DecodeTagsConfig,
};
use kornia_image::Image;
use kornia_imgproc::color::gray_from_rgb_u8;
use kornia_io::functional::read_image_any_rgb8;
use std::path::PathBuf;

fn detections_for(nmaxima: usize, gray: &Image<u8, 1>) -> Vec<Detection> {
    let mut config = DecodeTagsConfig::new(vec![
        TagFamilyKind::Tag36H11,
        TagFamilyKind::Tag16H5,
        TagFamilyKind::TagStandard41H12,
    ])
    .unwrap();
    config.fit_quad_config.max_nmaxima = nmaxima;
    let mut det = AprilTagDecoder::new(config, gray.size()).unwrap();
    let mut out = det.decode(gray).unwrap();
    out.sort_by_key(|d| (d._family_idx, d.id));
    out
}

fn parity(base: &[Detection], cand: &[Detection]) -> Option<String> {
    if base.len() != cand.len() {
        return Some(format!("count {} vs {}", base.len(), cand.len()));
    }
    for (b, c) in base.iter().zip(cand) {
        if b.id != c.id || b._family_idx != c._family_idx {
            return Some(format!("id mismatch {} vs {}", b.id, c.id));
        }
        let dx = (b.center.x - c.center.x).abs();
        let dy = (b.center.y - c.center.y).abs();
        if dx > 0.5 || dy > 0.5 {
            return Some(format!("id {} center off by ({dx:.3},{dy:.3})", b.id));
        }
    }
    None
}

/// `decode_timed` stage order: [decimate, threshold, conn_comp, gradient,
/// fit_quads, decode_tags].
const FIT_QUADS_STAGE: usize = 4;

fn main() {
    let img_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/data/apriltags_tag36h11.jpg");
    let img = read_image_any_rgb8(img_path).unwrap();
    let mut gray = Image::<u8, 1>::from_size_val(img.size(), 0).unwrap();
    gray_from_rgb_u8(&img, &mut gray).unwrap();

    // Baseline detections at the C-default nmaxima=10.
    let base = detections_for(10, &gray);
    println!("apriltags_tag36h11.jpg: {} baseline detections", base.len());

    for nmaxima in [10usize, 8, 6, 5, 4] {
        let why = parity(&base, &detections_for(nmaxima, &gray));

        // fit_quads timing: min over batches (least scheduler noise).
        let mut config = DecodeTagsConfig::new(vec![TagFamilyKind::Tag36H11]).unwrap();
        config.fit_quad_config.max_nmaxima = nmaxima;
        let mut det = AprilTagDecoder::new(config, gray.size()).unwrap();
        for _ in 0..20 {
            let _ = det.decode_timed(&gray).unwrap();
            det.clear();
        }
        let mut best_fit = u64::MAX;
        let mut best_total = u64::MAX;
        for _ in 0..5 {
            let mut fit_us = 0u64;
            let mut total_us = 0u64;
            const N: usize = 50;
            for _ in 0..N {
                let (_, us) = det.decode_timed(&gray).unwrap();
                det.clear();
                fit_us += us[FIT_QUADS_STAGE];
                total_us += us.iter().sum::<u64>();
            }
            best_fit = best_fit.min(fit_us / N as u64);
            best_total = best_total.min(total_us / N as u64);
        }
        println!(
            "nmaxima={nmaxima:>2}: fit_quads {best_fit:>4} µs  total {best_total:>5} µs  parity {}",
            match &why {
                None => "OK".to_string(),
                Some(w) => format!("FAIL ({w})"),
            }
        );
    }
}
