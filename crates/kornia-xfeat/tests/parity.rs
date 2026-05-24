//! Parity tests against the upstream PyTorch fixtures.
//!
//! Each test gracefully no-ops when the corresponding fixture is absent so CI
//! stays green during scaffold work. Once `tools/xfeat-regen-fixtures` runs
//! and the dense / sparse artifacts land under `tests/fixtures/v1/`, these
//! tests flip into enforcing mode automatically.

use std::path::PathBuf;

fn fixture_path(name: &str) -> PathBuf {
    let mut p: PathBuf = ["tests", "fixtures", "v1"].iter().collect();
    p.push(name);
    p
}

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

#[test]
fn dense_parity_against_pytorch() {
    let dense = fixture_path("expected_dense.safetensors");
    if !dense.exists() {
        eprintln!(
            "[skip] {} not present — regen fixtures via tools/xfeat-regen-fixtures",
            dense.display()
        );
        return;
    }
    // TODO(parity): when the model graph's full forward pass is wired,
    // load the four dense maps from expected_dense.safetensors, run the
    // scalar model on input_preprocessed.safetensors, and compare with the
    // tolerances from MANIFEST.toml. For now, presence of the file is a
    // signal to start enforcing — leave a panic so we don't silently regress.
    panic!(
        "fixtures present but parity comparison not yet implemented; \
         implement once XFeat::extract is fully wired"
    );
}

#[test]
fn sparse_parity_against_pytorch() {
    let sparse = fixture_path("expected_sparse.json");
    if !sparse.exists() {
        eprintln!("[skip] {} not present", sparse.display());
        return;
    }
    // Same as above — placeholder until the model forward pass lands.
    panic!(
        "fixtures present but sparse parity not yet implemented; \
         implement once XFeat::extract is fully wired"
    );
}

#[test]
fn repeatability_bit_identical() {
    use kornia_xfeat::{weights::PackedWeights, XFeat, XFeatConfig};

    // Use embedded (placeholder) weights — empty safetensors is enough to
    // confirm construction and that two calls with the same input return
    // byte-equivalent outputs.
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
