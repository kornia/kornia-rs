//! End-to-end NEON-backend vs scalar-oracle parity at the FIXTURE resolution.
//!
//! The fixture parity tests (`parity.rs`) force `with_scalar_backend()`, and
//! the benchmark runs 480×640 — so before this test, no CI path ever validated
//! the full NEON pipeline's *output* at 576×800. That blind spot hid a
//! col-split underflow panic (fixed) and would hide numerical divergence.
//!
//! Bar: the two backends run the same image; keypoint sets must overlap
//! strongly (IoU on rounded integer coordinates) and descriptors at shared
//! keypoints must be near-identical (cosine distance ≤ 1e-2 — fp16 activation
//! storage in the NEON path vs f32 scalar gives small but nonzero drift).

use std::collections::HashMap;
use std::path::Path;

use kornia_xfeat::{weights::PackedWeights, XFeat, XFeatConfig};

fn load_gray(path: &Path, h: usize, w: usize) -> Vec<f32> {
    let file = std::fs::File::open(path).expect("open fixture");
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info().expect("png read_info");
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).expect("png decode");
    let (ih, iw) = (info.height as usize, info.width as usize);
    let channels = info.color_type.samples();
    let mut gray = vec![0.0f32; ih * iw];
    match channels {
        1 => kornia_xfeat::preproc::gray_u8_to_gray_f32(&buf[..ih * iw], &mut gray),
        3 => kornia_xfeat::preproc::rgb_u8_to_gray_f32(&buf[..ih * iw * 3], &mut gray, ih, iw),
        c => panic!("unsupported channels {c}"),
    }
    let mut resized = vec![0.0f32; h * w];
    kornia_xfeat::preproc::bilinear_resample_gray(&gray, &mut resized, ih, iw, h, w);
    resized
}

#[test]
fn e2e_neon_vs_scalar_480x640() {
    run_e2e(480, 640);
}

#[test]
fn e2e_neon_vs_scalar_576x800() {
    run_e2e(576, 800);
}

fn run_e2e(h: usize, w: usize) {
    let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let img = load_gray(&crate_dir.join("tests/fixtures/v1/ref/input.png"), h, w);

    let cfg = XFeatConfig {
        height: h,
        width: w,
        ..XFeatConfig::default()
    };
    let weights_bytes = kornia_xfeat::weights::embedded_bytes();

    let mk = || PackedWeights::from_safetensors_bytes(weights_bytes).expect("weights");
    let mut neon_model = XFeat::new(cfg.clone(), mk()).expect("neon model");
    let mut scalar_model = XFeat::new(cfg.clone(), mk())
        .expect("scalar model")
        .with_scalar_backend();

    struct Owned {
        keypoints: Vec<kornia_xfeat::KeyPoint>,
        descriptors: Vec<f32>,
    }
    let mut run = |model: &mut XFeat| -> Owned {
        let out = model.extract(&img).expect("extract");
        Owned {
            keypoints: out.keypoints.to_vec(),
            descriptors: out.descriptors.to_vec(),
        }
    };
    let neon_out = run(&mut neon_model);
    let scalar_out = run(&mut scalar_model);

    // NaN guard FIRST: f32::max and `>` comparisons silently ignore NaN, so
    // every tolerance check below is blind to it. Count explicitly.
    let nan_count = |v: &[f32]| v.iter().filter(|x| x.is_nan()).count();
    let neon_desc_nan = nan_count(&neon_out.descriptors);
    let scalar_desc_nan = nan_count(&scalar_out.descriptors);
    eprintln!(
        "[e2e {h}x{w} nan] neon_desc={}/{} scalar_desc={}/{}",
        neon_desc_nan,
        neon_out.descriptors.len(),
        scalar_desc_nan,
        scalar_out.descriptors.len()
    );
    assert_eq!(scalar_desc_nan, 0, "scalar descriptors contain NaN");
    assert_eq!(neon_desc_nan, 0, "NEON descriptors contain NaN");

    // Dense-tensor probes: localize any divergence before the sparse checks.
    let max_abs = |a: &[f32], b: &[f32]| -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    };
    let k1h_err = max_abs(neon_model.k1h_slice(), scalar_model.k1h_slice());
    let rel_err = max_abs(neon_model.h1_rel_slice(), scalar_model.h1_rel_slice());
    eprintln!("[e2e {h}x{w} dense] k1h max err={k1h_err:.3e}  h1_rel max err={rel_err:.3e}");
    // The NEON path stores activations in f16 and accumulates conv sums in
    // f16 (FMLA.8H); after ~10 layers the dense heads drift ~1-2e-2 from the
    // all-f32 scalar oracle. That is inherent to the fp16 design, not a bug —
    // the bound below catches catastrophic divergence (NaN, garbage, wrong
    // layout), while the sparse IoU + descriptor checks below are the real
    // functional bar.
    assert!(k1h_err <= 5e-2, "k1h diverges: {k1h_err:.3e}");
    assert!(rel_err <= 5e-2, "h1_rel diverges: {rel_err:.3e}");

    assert!(
        !neon_out.keypoints.is_empty() && !scalar_out.keypoints.is_empty(),
        "both backends must produce keypoints (neon={}, scalar={})",
        neon_out.keypoints.len(),
        scalar_out.keypoints.len()
    );

    // Keypoint-set IoU on integer-rounded coordinates.
    let key = |x: f32, y: f32| -> (i64, i64) { (x.round() as i64, y.round() as i64) };
    let scalar_idx: HashMap<(i64, i64), usize> = scalar_out
        .keypoints
        .iter()
        .enumerate()
        .map(|(i, kp)| (key(kp.x, kp.y), i))
        .collect();
    let mut shared = 0usize;
    let mut worst_cos_dist = 0.0f32;
    let c = 64usize;
    for (i, kp) in neon_out.keypoints.iter().enumerate() {
        if let Some(&j) = scalar_idx.get(&key(kp.x, kp.y)) {
            shared += 1;
            let dn = &neon_out.descriptors[i * c..(i + 1) * c];
            let ds = &scalar_out.descriptors[j * c..(j + 1) * c];
            let dot: f32 = dn.iter().zip(ds.iter()).map(|(a, b)| a * b).sum();
            worst_cos_dist = worst_cos_dist.max(1.0 - dot);
        }
    }
    let union = neon_out.keypoints.len() + scalar_out.keypoints.len() - shared;
    let iou = shared as f32 / union as f32;
    eprintln!(
        "[e2e {h}x{w}] neon={} scalar={} shared={} IoU={:.3} worst_cos_dist={:.3e}",
        neon_out.keypoints.len(),
        scalar_out.keypoints.len(),
        shared,
        iou,
        worst_cos_dist
    );
    assert!(
        iou >= 0.90,
        "NEON/scalar keypoint IoU too low at {h}x{w}: {iou:.3}"
    );
    assert!(
        worst_cos_dist <= 1e-2,
        "NEON/scalar descriptor divergence at shared keypoints: {worst_cos_dist:.3e}"
    );
}
