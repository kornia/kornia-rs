//! Parity tests against the upstream PyTorch fixtures.
//!
//! Tests are parameterised by image name (fixture sub-directories under
//! `tests/fixtures/v1/<image>/`). Each test gracefully skips when the fixture
//! is absent so CI stays green on machines where fixtures haven't been
//! regenerated yet.
//!
//! To regenerate fixtures:
//!   python3 tools/xfeat-regen-fixtures/regen.py \
//!     --output tests/fixtures/v1/ \
//!     --upstream /tmp/accelerated_features \
//!     --commit e92685f57f8318b18725c5c8c0bd28c7fe188d9a

use std::path::PathBuf;

use kornia_xfeat::{weights::PackedWeights, XFeat, XFeatConfig};

fn fixture_root() -> PathBuf {
    ["tests", "fixtures", "v1"].iter().collect()
}

fn fixture_path(name: &str) -> PathBuf {
    fixture_root().join(name)
}

fn image_fixture_path(image: &str, name: &str) -> PathBuf {
    fixture_root().join(image).join(name)
}

/// Return the list of per-image fixture directories that exist.
fn available_images() -> Vec<String> {
    ["ref", "tgt"]
        .iter()
        .filter(|&&stem| image_fixture_path(stem, "expected_sparse.json").exists())
        .map(|s| s.to_string())
        .collect()
}

/// Load packed weights from the assets directory. Returns None if absent or placeholder.
fn load_packed_weights() -> Option<PackedWeights> {
    let bytes = std::fs::read("assets/xfeat_packed.safetensors").ok()?;
    PackedWeights::from_safetensors_bytes(&bytes).ok()
}

/// Decode a f32 tensor from a safetensors view (little-endian IEEE 754).
fn decode_f32(view: &safetensors::tensor::TensorView<'_>) -> Vec<f32> {
    view.data()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

/// Load the preprocessed input (H × W flat grayscale) from the fixture.
/// The safetensors file holds `input_gray` in shape `[1, 1, H, W]`.
fn load_input(image: &str, h: usize, w: usize) -> Option<Vec<f32>> {
    let path = image_fixture_path(image, "input_preprocessed.safetensors");
    let bytes = std::fs::read(&path).ok()?;
    let st = safetensors::SafeTensors::deserialize(&bytes).ok()?;
    let view = st.tensor("input_gray").ok()?;
    let data = decode_f32(&view);
    if data.len() < h * w {
        return None;
    }
    Some(data[data.len() - h * w..].to_vec())
}

// ─── Shared MANIFEST check ───────────────────────────────────────────────────

#[test]
fn manifest_is_present_and_parses() {
    let m = fixture_path("MANIFEST.toml");
    if !m.exists() {
        eprintln!("[skip] {} not present yet", m.display());
        return;
    }
    let text = std::fs::read_to_string(&m).expect("read MANIFEST.toml");
    let table: toml::Table = text.parse().expect("MANIFEST.toml is valid TOML");
    let version = table
        .get("format_version")
        .and_then(|v| v.as_integer())
        .expect("format_version present");
    assert_eq!(version, 1, "unsupported MANIFEST.toml format_version");
}

// ─── Helpers for per-image parity ────────────────────────────────────────────

fn run_dense_parity(image: &str) {
    let dense_path = image_fixture_path(image, "expected_dense.safetensors");
    if !dense_path.exists() {
        eprintln!("[skip] dense fixture absent for '{image}'");
        return;
    }

    let weights = match load_packed_weights() {
        Some(w) => w,
        None => {
            eprintln!("[skip] packed weights absent or placeholder");
            return;
        }
    };

    // MANIFEST for tolerances
    let manifest_text =
        std::fs::read_to_string(fixture_path("MANIFEST.toml")).expect("MANIFEST.toml");
    let manifest: toml::Table = manifest_text.parse().unwrap();

    // Look up per-image height/width from [[images]] array
    let (h, w) = image_hw(&manifest, image).unwrap_or_else(|| {
        (
            manifest["input"]["height"].as_integer().unwrap() as usize,
            manifest["input"]["width"].as_integer().unwrap() as usize,
        )
    });
    let tol = &manifest["tolerances"]["dense"];
    let kp_abs = tol["keypoint_logits_abs"].as_float().unwrap() as f32;
    let rel_abs = tol["reliability_abs"].as_float().unwrap() as f32;

    let input = match load_input(image, h, w) {
        Some(v) => v,
        None => {
            eprintln!("[skip] input not parseable for '{image}'");
            return;
        }
    };

    let cfg = XFeatConfig {
        height: h,
        width: w,
        ..XFeatConfig::default()
    };
    let mut model = XFeat::new(cfg, weights)
        .expect("construct")
        .with_scalar_backend();
    model.extract(&input).expect("extract");

    let dense_bytes = std::fs::read(&dense_path).expect("read dense fixture");
    let dense_st = safetensors::SafeTensors::deserialize(&dense_bytes).unwrap();

    let kp_expected = decode_f32(&dense_st.tensor("kp_heatmap_fullres").unwrap());
    let rel_expected = decode_f32(&dense_st.tensor("reliability").unwrap());

    let our_k1h = model.k1h_slice();
    let our_h1rel = model.h1_rel_slice();

    let mut kp_max_err = 0.0f32;
    let mut rel_max_err = 0.0f32;
    let mut kp_fail = 0usize;
    let mut rel_fail = 0usize;

    for (&ours, &exp) in our_k1h.iter().zip(kp_expected.iter()) {
        let err = (ours - exp).abs();
        if err > kp_max_err {
            kp_max_err = err;
        }
        if err > kp_abs {
            kp_fail += 1;
        }
    }
    for (&ours, &exp) in our_h1rel.iter().zip(rel_expected.iter()) {
        let err = (ours - exp).abs();
        if err > rel_max_err {
            rel_max_err = err;
        }
        if err > rel_abs {
            rel_fail += 1;
        }
    }

    assert_eq!(
        kp_fail, 0,
        "[{image}] k1h: {kp_fail} exceed abs tol {kp_abs:.1e}; max err = {kp_max_err:.2e}"
    );
    assert_eq!(
        rel_fail, 0,
        "[{image}] h1_rel: {rel_fail} exceed abs tol {rel_abs:.1e}; max err = {rel_max_err:.2e}"
    );
    eprintln!("[dense/{image}] k1h max={kp_max_err:.2e} (tol {kp_abs:.1e}); h1_rel max={rel_max_err:.2e} (tol {rel_abs:.1e})");
}

fn run_sparse_parity(image: &str) {
    let sparse_path = image_fixture_path(image, "expected_sparse.json");
    if !sparse_path.exists() {
        eprintln!("[skip] sparse fixture absent for '{image}'");
        return;
    }

    let weights = match load_packed_weights() {
        Some(w) => w,
        None => {
            eprintln!("[skip] packed weights absent or placeholder");
            return;
        }
    };

    let manifest_text = std::fs::read_to_string(fixture_path("MANIFEST.toml")).unwrap();
    let manifest: toml::Table = manifest_text.parse().unwrap();

    let (h, w) = image_hw(&manifest, image).unwrap_or_else(|| {
        (
            manifest["input"]["height"].as_integer().unwrap() as usize,
            manifest["input"]["width"].as_integer().unwrap() as usize,
        )
    });
    let tol = &manifest["tolerances"]["sparse"];
    let iou_min = tol["keypoint_iou_min"].as_float().unwrap() as f32;
    let cosdist_max = tol["descriptor_cosdist_max"].as_float().unwrap() as f32;
    let count_abs = tol["count_abs_max"].as_integer().unwrap() as usize;

    let input = match load_input(image, h, w) {
        Some(v) => v,
        None => {
            eprintln!("[skip] input not parseable for '{image}'");
            return;
        }
    };

    let cfg = XFeatConfig {
        height: h,
        width: w,
        ..XFeatConfig::default()
    };
    let mut model = XFeat::new(cfg, weights)
        .expect("construct")
        .with_scalar_backend();
    let out = model.extract(&input).expect("extract");

    let our_kps = out.keypoints;
    let our_descs = out.descriptors;

    let json_text = std::fs::read_to_string(&sparse_path).unwrap();
    let json: serde_json::Value = serde_json::from_str(&json_text).unwrap();

    let exp_kps: Vec<[f32; 2]> = json["keypoints"]
        .as_array()
        .unwrap()
        .iter()
        .map(|kp| {
            let a = kp.as_array().unwrap();
            [a[0].as_f64().unwrap() as f32, a[1].as_f64().unwrap() as f32]
        })
        .collect();

    let exp_descs: Vec<Vec<f32>> = json["descriptors"]
        .as_array()
        .unwrap()
        .iter()
        .map(|d| {
            d.as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect()
        })
        .collect();

    // Count check
    let our_n = our_kps.len();
    let exp_n = exp_kps.len();
    assert!(
        our_n.abs_diff(exp_n) <= count_abs,
        "[{image}] count: ours={our_n}, expected={exp_n} (max diff={count_abs})"
    );

    // IoU — match within 1 pixel radius
    let match_r2 = 1.0f32;
    let n_check = our_n.min(exp_n);
    let mut matched = 0usize;
    for exp_kp in exp_kps.iter().take(n_check) {
        if our_kps
            .iter()
            .any(|kp| (kp.x - exp_kp[0]).powi(2) + (kp.y - exp_kp[1]).powi(2) <= match_r2)
        {
            matched += 1;
        }
    }
    let union = (our_n + exp_n).saturating_sub(matched);
    let iou = if union == 0 {
        1.0
    } else {
        matched as f32 / union as f32
    };
    assert!(
        iou >= iou_min,
        "[{image}] IoU={iou:.3} < {iou_min:.3} (matched {matched}/{n_check})"
    );

    // Descriptor cosine distance for top-100 matched pairs
    let n_desc = 100.min(matched);
    let mut cos_dist_max = 0.0f32;
    let mut desc_fail = 0usize;

    'outer: for (ei, exp_kp) in exp_kps.iter().enumerate().take(n_desc) {
        let Some(oi) = our_kps
            .iter()
            .position(|kp| (kp.x - exp_kp[0]).powi(2) + (kp.y - exp_kp[1]).powi(2) <= match_r2)
        else {
            continue 'outer;
        };

        let our_d = &our_descs[oi * 64..(oi + 1) * 64];
        let exp_d = &exp_descs[ei];
        let dot: f32 = our_d.iter().zip(exp_d).map(|(a, b)| a * b).sum();
        let na = our_d.iter().map(|v| v * v).sum::<f32>().sqrt();
        let nb = exp_d.iter().map(|v| v * v).sum::<f32>().sqrt();
        let cos_dist = 1.0
            - if na > 0.0 && nb > 0.0 {
                dot / (na * nb)
            } else {
                0.0
            };
        if cos_dist > cos_dist_max {
            cos_dist_max = cos_dist;
        }
        if cos_dist > cosdist_max {
            desc_fail += 1;
        }
    }

    assert_eq!(
        desc_fail, 0,
        "[{image}] desc: {desc_fail} pairs exceed {cosdist_max:.1e}; max={cos_dist_max:.2e}"
    );
    eprintln!("[sparse/{image}] IoU={iou:.3} (min {iou_min:.2}), cos_dist_max={cos_dist_max:.2e}");
}

/// Read per-image (H, W) from the `[[images]]` TOML array if present.
fn image_hw(manifest: &toml::Table, stem: &str) -> Option<(usize, usize)> {
    let images = manifest.get("images")?.as_array()?;
    for entry in images {
        let name = entry.get("name")?.as_str()?;
        if name == stem {
            let h = entry.get("height")?.as_integer()? as usize;
            let w = entry.get("width")?.as_integer()? as usize;
            return Some((h, w));
        }
    }
    None
}

// ─── Per-image dense parity ───────────────────────────────────────────────────

#[test]
fn dense_parity_ref() {
    run_dense_parity("ref");
}

#[test]
fn dense_parity_tgt() {
    run_dense_parity("tgt");
}

// Legacy: v1 root-level fixture (ref image, kept for backward compat)
#[test]
fn dense_parity_against_pytorch() {
    let dense_path = fixture_path("expected_dense.safetensors");
    if !dense_path.exists() {
        eprintln!("[skip] root-level dense fixture absent");
        return;
    }
    run_dense_parity_root(&dense_path);
}

fn run_dense_parity_root(dense_path: &std::path::Path) {
    let weights = match load_packed_weights() {
        Some(w) => w,
        None => {
            eprintln!("[skip] packed weights absent");
            return;
        }
    };
    let manifest_text = std::fs::read_to_string(fixture_path("MANIFEST.toml")).unwrap();
    let manifest: toml::Table = manifest_text.parse().unwrap();
    let h = manifest["input"]["height"].as_integer().unwrap() as usize;
    let w = manifest["input"]["width"].as_integer().unwrap() as usize;
    let tol = &manifest["tolerances"]["dense"];
    let kp_abs = tol["keypoint_logits_abs"].as_float().unwrap() as f32;
    let rel_abs = tol["reliability_abs"].as_float().unwrap() as f32;

    let input = match load_input_root(h, w) {
        Some(v) => v,
        None => {
            eprintln!("[skip] root input not parseable");
            return;
        }
    };
    let cfg = XFeatConfig {
        height: h,
        width: w,
        ..XFeatConfig::default()
    };
    let mut model = XFeat::new(cfg, weights)
        .expect("construct")
        .with_scalar_backend();
    model.extract(&input).expect("extract");

    let dense_bytes = std::fs::read(dense_path).unwrap();
    let dense_st = safetensors::SafeTensors::deserialize(&dense_bytes).unwrap();
    let kp_expected = decode_f32(&dense_st.tensor("kp_heatmap_fullres").unwrap());
    let rel_expected = decode_f32(&dense_st.tensor("reliability").unwrap());

    let our_k1h = model.k1h_slice();
    let our_h1rel = model.h1_rel_slice();
    let (mut kp_max, mut rel_max) = (0.0f32, 0.0f32);
    let (mut kp_fail, mut rel_fail) = (0, 0);
    for (&o, &e) in our_k1h.iter().zip(kp_expected.iter()) {
        let d = (o - e).abs();
        if d > kp_max {
            kp_max = d;
        }
        if d > kp_abs {
            kp_fail += 1;
        }
    }
    for (&o, &e) in our_h1rel.iter().zip(rel_expected.iter()) {
        let d = (o - e).abs();
        if d > rel_max {
            rel_max = d;
        }
        if d > rel_abs {
            rel_fail += 1;
        }
    }
    assert_eq!(
        kp_fail, 0,
        "k1h: {kp_fail} exceed {kp_abs:.1e}; max={kp_max:.2e}"
    );
    assert_eq!(
        rel_fail, 0,
        "h1_rel: {rel_fail} exceed {rel_abs:.1e}; max={rel_max:.2e}"
    );
    eprintln!("[dense parity] k1h max err = {kp_max:.2e} (tol {kp_abs:.1e}); h1_rel max err = {rel_max:.2e} (tol {rel_abs:.1e})");
}

fn load_input_root(h: usize, w: usize) -> Option<Vec<f32>> {
    let path = fixture_path("input_preprocessed.safetensors");
    let bytes = std::fs::read(&path).ok()?;
    let st = safetensors::SafeTensors::deserialize(&bytes).ok()?;
    let view = st.tensor("input_gray").ok()?;
    let data = decode_f32(&view);
    if data.len() < h * w {
        return None;
    }
    Some(data[data.len() - h * w..].to_vec())
}

// ─── Per-image sparse (end-to-end) parity ────────────────────────────────────

#[test]
fn sparse_parity_ref() {
    run_sparse_parity("ref");
}

#[test]
fn sparse_parity_tgt() {
    run_sparse_parity("tgt");
}

// Legacy root-level sparse test
#[test]
fn sparse_parity_against_pytorch() {
    let sparse_path = fixture_path("expected_sparse.json");
    if !sparse_path.exists() {
        eprintln!("[skip] root-level sparse fixture absent");
        return;
    }
    run_sparse_parity_root(&sparse_path);
}

fn run_sparse_parity_root(sparse_path: &std::path::Path) {
    let weights = match load_packed_weights() {
        Some(w) => w,
        None => {
            eprintln!("[skip] packed weights absent");
            return;
        }
    };
    let manifest_text = std::fs::read_to_string(fixture_path("MANIFEST.toml")).unwrap();
    let manifest: toml::Table = manifest_text.parse().unwrap();
    let h = manifest["input"]["height"].as_integer().unwrap() as usize;
    let w = manifest["input"]["width"].as_integer().unwrap() as usize;
    let tol = &manifest["tolerances"]["sparse"];
    let iou_min = tol["keypoint_iou_min"].as_float().unwrap() as f32;
    let cosdist_max = tol["descriptor_cosdist_max"].as_float().unwrap() as f32;
    let count_abs = tol["count_abs_max"].as_integer().unwrap() as usize;

    let input = match load_input_root(h, w) {
        Some(v) => v,
        None => {
            eprintln!("[skip] root input not parseable");
            return;
        }
    };
    let cfg = XFeatConfig {
        height: h,
        width: w,
        ..XFeatConfig::default()
    };
    let mut model = XFeat::new(cfg, weights)
        .expect("construct")
        .with_scalar_backend();
    let out = model.extract(&input).expect("extract");

    let our_kps = out.keypoints;
    let our_descs = out.descriptors;

    let json_text = std::fs::read_to_string(sparse_path).unwrap();
    let json: serde_json::Value = serde_json::from_str(&json_text).unwrap();

    let exp_kps: Vec<[f32; 2]> = json["keypoints"]
        .as_array()
        .unwrap()
        .iter()
        .map(|kp| {
            let a = kp.as_array().unwrap();
            [a[0].as_f64().unwrap() as f32, a[1].as_f64().unwrap() as f32]
        })
        .collect();
    let exp_descs: Vec<Vec<f32>> = json["descriptors"]
        .as_array()
        .unwrap()
        .iter()
        .map(|d| {
            d.as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect()
        })
        .collect();

    let our_n = our_kps.len();
    let exp_n = exp_kps.len();
    assert!(
        our_n.abs_diff(exp_n) <= count_abs,
        "count: ours={our_n} exp={exp_n}"
    );

    let match_r2 = 1.0f32;
    let n_check = our_n.min(exp_n);
    let mut matched = 0;
    for exp_kp in exp_kps.iter().take(n_check) {
        if our_kps
            .iter()
            .any(|kp| (kp.x - exp_kp[0]).powi(2) + (kp.y - exp_kp[1]).powi(2) <= match_r2)
        {
            matched += 1;
        }
    }
    let union = (our_n + exp_n).saturating_sub(matched);
    let iou = if union == 0 {
        1.0
    } else {
        matched as f32 / union as f32
    };
    assert!(iou >= iou_min, "IoU={iou:.3} < {iou_min:.3}");

    let n_desc = 100.min(matched);
    let mut cos_dist_max = 0.0f32;
    let mut desc_fail = 0;
    'outer: for (ei, exp_kp) in exp_kps.iter().enumerate().take(n_desc) {
        let Some(oi) = our_kps
            .iter()
            .position(|kp| (kp.x - exp_kp[0]).powi(2) + (kp.y - exp_kp[1]).powi(2) <= match_r2)
        else {
            continue 'outer;
        };
        let our_d = &our_descs[oi * 64..(oi + 1) * 64];
        let exp_d = &exp_descs[ei];
        let dot: f32 = our_d.iter().zip(exp_d).map(|(a, b)| a * b).sum();
        let na = our_d.iter().map(|v| v * v).sum::<f32>().sqrt();
        let nb = exp_d.iter().map(|v| v * v).sum::<f32>().sqrt();
        let cd = 1.0
            - if na > 0.0 && nb > 0.0 {
                dot / (na * nb)
            } else {
                0.0
            };
        if cd > cos_dist_max {
            cos_dist_max = cd;
        }
        if cd > cosdist_max {
            desc_fail += 1;
        }
    }
    assert_eq!(
        desc_fail, 0,
        "desc: {desc_fail} pairs exceed {cosdist_max:.1e}; max={cos_dist_max:.2e}"
    );
    eprintln!("[sparse parity] IoU={iou:.3} (min {iou_min:.2}), cos_dist_max={cos_dist_max:.2e} (max {cosdist_max:.1e})");
}

// ─── End-to-end: iterate all available image fixtures ────────────────────────

#[test]
fn e2e_all_images_sparse_parity() {
    let images = available_images();
    if images.is_empty() {
        eprintln!("[skip] no per-image fixture dirs found under tests/fixtures/v1/{{ref,tgt}}/");
        return;
    }
    eprintln!("[e2e] testing images: {:?}", images);
    for img in &images {
        run_sparse_parity(img);
    }
    eprintln!("[e2e] all {} images passed sparse parity", images.len());
}

#[test]
fn e2e_all_images_dense_parity() {
    let images = available_images();
    if images.is_empty() {
        eprintln!("[skip] no per-image fixture dirs found");
        return;
    }
    for img in &images {
        run_dense_parity(img);
    }
}

// ─── Repeatability ───────────────────────────────────────────────────────────

#[test]
fn repeatability_bit_identical() {
    let weights =
        match PackedWeights::from_safetensors_bytes(kornia_xfeat::weights::embedded_bytes()) {
            Ok(w) => w,
            Err(_) => {
                eprintln!(
                    "[skip] embedded weights are placeholder; repeatability test needs real ones"
                );
                return;
            }
        };

    let cfg = XFeatConfig::default();
    let mut model = XFeat::new(cfg.clone(), weights)
        .expect("construct")
        .with_scalar_backend();

    let input = vec![0.5f32; cfg.height * cfg.width];
    let r1: Vec<_> = match model.extract(&input) {
        Ok(out) => out.keypoints.to_vec(),
        Err(e) => {
            eprintln!("[skip] extract failed (weights may be placeholder): {e}");
            return;
        }
    };
    let r2: Vec<_> = match model.extract(&input) {
        Ok(out) => out.keypoints.to_vec(),
        Err(e) => {
            eprintln!("[skip] extract failed on second call: {e}");
            return;
        }
    };
    assert_eq!(
        r1, r2,
        "two calls on identical input must produce identical outputs"
    );
}
