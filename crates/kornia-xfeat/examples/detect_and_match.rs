//! XFeat detection and matching example.
//!
//! Loads two fixture images, runs XFeat extraction on each, and then matches
//! descriptors via cosine similarity.  The example is self-contained: weights
//! are embedded via the `xfeat-embed` Cargo feature (the default).
//!
//! Run with:
//!   cargo run --example detect_and_match -p kornia-xfeat --release

use std::path::Path;

use kornia_xfeat::{
    preproc::{align_to_32, bilinear_resample_gray, rgb_u8_to_gray_f32},
    weights::PackedWeights,
    XFeat, XFeatConfig,
};

// ─── PNG loading ─────────────────────────────────────────────────────────────

/// Decode a PNG from `path`, returning raw bytes and `(h, w, channels)`.
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

// ─── Preprocessing ────────────────────────────────────────────────────────────

/// Load a PNG image, convert to f32 gray, and resize to the nearest dimensions
/// that are multiples of 32.  Returns the pixel buffer and `(h_out, w_out)`.
fn preprocess_image(path: &Path) -> (Vec<f32>, usize, usize) {
    let (raw, h, w, channels) = load_png(path);

    // Convert to f32 gray via channel mean (matching upstream XFeat convention).
    let mut gray_f32 = vec![0.0f32; h * w];
    match channels {
        1 => kornia_xfeat::preproc::gray_u8_to_gray_f32(&raw, &mut gray_f32),
        3 => rgb_u8_to_gray_f32(&raw, &mut gray_f32, h, w),
        4 => {
            // Drop alpha, treat as RGB.
            let rgb: Vec<u8> = raw
                .chunks_exact(4)
                .flat_map(|px| [px[0], px[1], px[2]])
                .collect();
            rgb_u8_to_gray_f32(&rgb, &mut gray_f32, h, w);
        }
        c => panic!("unsupported channel count {c}"),
    }

    // Align to multiples of 32.
    let (h_out, w_out) = align_to_32(h, w);
    if h_out == 0 || w_out == 0 {
        panic!("image too small to align to 32: {h}x{w}");
    }

    let mut resized = vec![0.0f32; h_out * w_out];
    bilinear_resample_gray(&gray_f32, &mut resized, h, w, h_out, w_out);

    (resized, h_out, w_out)
}

// ─── Simple cosine matcher ────────────────────────────────────────────────────

/// Brute-force cosine similarity match between two descriptor sets.
///
/// Each descriptor is `dim` consecutive f32 values in a flat slice.
/// Returns `(i, j)` pairs where `j` is the nearest neighbour of `i` with
/// cosine similarity >= `min_cos`.  Assumes descriptors are already L2-normalised
/// (as guaranteed by `XFeat::extract`).
fn match_cosine(
    descs1: &[f32],
    n1: usize,
    descs2: &[f32],
    n2: usize,
    dim: usize,
    min_cos: f32,
) -> Vec<(usize, usize)> {
    assert_eq!(descs1.len(), n1 * dim);
    assert_eq!(descs2.len(), n2 * dim);

    let dot = |a: &[f32], b: &[f32]| -> f32 { a.iter().zip(b).map(|(x, y)| x * y).sum() };

    let mut matches = Vec::new();
    for i in 0..n1 {
        let d1 = &descs1[i * dim..(i + 1) * dim];
        let mut best_j = 0usize;
        let mut best_cos = f32::NEG_INFINITY;
        for j in 0..n2 {
            let d2 = &descs2[j * dim..(j + 1) * dim];
            let cos = dot(d1, d2);
            if cos > best_cos {
                best_cos = cos;
                best_j = j;
            }
        }
        if best_cos >= min_cos {
            matches.push((i, best_j));
        }
    }
    matches
}

// ─── Main ─────────────────────────────────────────────────────────────────────

fn main() {
    // Locate fixture images relative to the crate root.
    let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let ref_path = crate_dir.join("tests/fixtures/v1/ref/input.png");
    let tgt_path = crate_dir.join("tests/fixtures/v1/tgt/input.png");

    // ── Preprocess ────────────────────────────────────────────────────────
    eprintln!("Loading ref  : {}", ref_path.display());
    let (ref_gray, h_ref, w_ref) = preprocess_image(&ref_path);
    eprintln!("  -> ({h_ref} x {w_ref}) f32 gray");

    eprintln!("Loading tgt  : {}", tgt_path.display());
    let (tgt_gray, h_tgt, w_tgt) = preprocess_image(&tgt_path);
    eprintln!("  -> ({h_tgt} x {w_tgt}) f32 gray");

    // ── Load weights ─────────────────────────────────────────────────────
    let weights_bytes = kornia_xfeat::weights::embedded_bytes();
    let weights = PackedWeights::from_safetensors_bytes(weights_bytes)
        .expect("embedded weights must parse; rebuild with the xfeat-embed feature");

    // ── Extract ref ──────────────────────────────────────────────────────
    let cfg_ref = XFeatConfig {
        height: h_ref,
        width: w_ref,
        ..XFeatConfig::default()
    };
    let mut model_ref = XFeat::new(cfg_ref, weights).expect("construct model_ref");
    let out_ref = model_ref.extract(&ref_gray).expect("extract ref");

    let n_ref = out_ref.keypoints.len();
    let desc_ref: Vec<f32> = out_ref.descriptors.to_vec();

    // ── Extract tgt ──────────────────────────────────────────────────────
    // Re-load weights from embedded bytes for the second model instance.
    let weights2 = PackedWeights::from_safetensors_bytes(kornia_xfeat::weights::embedded_bytes())
        .expect("embedded weights");
    let cfg_tgt = XFeatConfig {
        height: h_tgt,
        width: w_tgt,
        ..XFeatConfig::default()
    };
    let mut model_tgt = XFeat::new(cfg_tgt, weights2).expect("construct model_tgt");
    let out_tgt = model_tgt.extract(&tgt_gray).expect("extract tgt");

    let n_tgt = out_tgt.keypoints.len();
    let desc_tgt: Vec<f32> = out_tgt.descriptors.to_vec();

    // ── Match descriptors ─────────────────────────────────────────────────
    // XFeat descriptors are 64-dim L2-normalised float vectors.
    let matches = match_cosine(&desc_ref, n_ref, &desc_tgt, n_tgt, 64, 0.82);
    let n_matches = matches.len();

    // ── Report ─────────────────────────────────────────────────────────────
    println!("ref: {n_ref} keypoints, tgt: {n_tgt} keypoints, matches: {n_matches}");
}
