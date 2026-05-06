//! Snapshot regression tests for `find_contours`.
//!
//! For each (fixture, mode, method) combo we compute a FNV-1a 64-bit hash of
//! the canonical JSON output and compare it against the hash recorded in
//! `tests/snapshots/digests.txt`. Any unintentional change to find_contours'
//! output flips a hash and fails the test.
//!
//! The full JSON files themselves live in `tests/snapshots/*.json` (gitignored,
//! ~1.6 MB total). Regenerate with:
//!
//! ```bash
//! bash crates/kornia-imgproc/examples/dump_snapshots.sh
//! ```
//!
//! To intentionally update the recorded hashes after an algorithm change:
//!
//! ```bash
//! UPDATE_DIGESTS=1 cargo test --release -p kornia-imgproc --test contours_snapshots -- --nocapture
//! ```
//!
//! Tests are skipped (not failed) when fixture PNGs aren't downloaded.

use kornia_image::{allocator::CpuAllocator, Image, ImageSize};
use kornia_imgproc::contours::{
    find_contours, ContourApproximationMode, RetrievalMode,
};
use std::fmt::Write as _;
use std::path::PathBuf;

const COMBOS: &[(&str, RetrievalMode, ContourApproximationMode)] = &[
    ("pic1_external_simple", RetrievalMode::External, ContourApproximationMode::Simple),
    ("pic1_external_none",   RetrievalMode::External, ContourApproximationMode::None),
    ("pic1_list_simple",     RetrievalMode::List,     ContourApproximationMode::Simple),
    ("pic1_list_none",       RetrievalMode::List,     ContourApproximationMode::None),
    ("pic2_external_simple", RetrievalMode::External, ContourApproximationMode::Simple),
    ("pic2_external_none",   RetrievalMode::External, ContourApproximationMode::None),
    ("pic3_external_simple", RetrievalMode::External, ContourApproximationMode::Simple),
    ("pic3_external_none",   RetrievalMode::External, ContourApproximationMode::None),
    ("pic3_list_simple",     RetrievalMode::List,     ContourApproximationMode::Simple),
    ("pic3_list_none",       RetrievalMode::List,     ContourApproximationMode::None),
    ("pic4_external_simple", RetrievalMode::External, ContourApproximationMode::Simple),
    ("pic4_external_none",   RetrievalMode::External, ContourApproximationMode::None),
    ("pic4_list_simple",     RetrievalMode::List,     ContourApproximationMode::Simple),
    ("pic4_list_none",       RetrievalMode::List,     ContourApproximationMode::None),
];

/// FNV-1a 64-bit. Stable across Rust versions, no deps, plenty of bits to detect changes.
fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Canonical JSON emission — must byte-match `dump_contours.rs` output so the
/// hash computed here matches the hash of the on-disk snapshot file.
fn emit_json(contours: &[Vec<[i32; 2]>]) -> String {
    let mut s = String::with_capacity(contours.len() * 64);
    s.push_str("{\"contours\": [");
    for (i, c) in contours.iter().enumerate() {
        if i > 0 { s.push(','); }
        s.push('[');
        for (j, p) in c.iter().enumerate() {
            if j > 0 { s.push(','); }
            // dump_contours uses println! (trailing \n) for the outer wrapper
            // and print! for each [x,y] — match exactly.
            write!(&mut s, "[{},{}]", p[0], p[1]).unwrap();
        }
        s.push(']');
    }
    s.push_str("]}\n");
    s
}

fn load_binary(name: &str) -> Option<Image<u8, 1, CpuAllocator>> {
    let path: PathBuf = [env!("CARGO_MANIFEST_DIR"), "examples", "data", name]
        .iter().collect();
    if !path.exists() {
        return None;
    }
    let rgb = kornia_io::png::read_image_png_rgb8(&path).ok()?;
    let (w, h) = (rgb.width(), rgb.height());
    let mut gray = Image::<u8, 1, _>::from_size_val(
        ImageSize { width: w, height: h }, 0, CpuAllocator,
    ).ok()?;
    kornia_imgproc::color::gray_from_rgb_u8(&rgb, &mut gray).ok()?;
    let mut bw = Image::<u8, 1, _>::from_size_val(
        ImageSize { width: w, height: h }, 0, CpuAllocator,
    ).ok()?;
    kornia_imgproc::threshold::threshold_binary(&gray, &mut bw, 127, 1).ok()?;
    Some(bw)
}

fn fixture_name(combo_name: &str) -> &'static str {
    match &combo_name[..4] {
        "pic1" => "pic1.png",
        "pic2" => "pic2.png",
        "pic3" => "pic3.png",
        "pic4" => "pic4.png",
        _ => unreachable!(),
    }
}

fn read_digests() -> std::collections::HashMap<String, String> {
    let path: PathBuf = [env!("CARGO_MANIFEST_DIR"), "tests", "snapshots", "digests.txt"]
        .iter().collect();
    // In UPDATE mode (or first run after deletion) the file may be absent — that
    // is not a test failure; we'll write it next.
    let Ok(text) = std::fs::read_to_string(&path) else {
        return std::collections::HashMap::new();
    };
    text.lines()
        .filter(|l| !l.trim().is_empty() && !l.starts_with('#'))
        .map(|l| {
            let mut it = l.split_whitespace();
            let hash = it.next().unwrap().to_string();
            let name = it.next().unwrap().to_string();
            (name, hash)
        })
        .collect()
}

#[test]
fn snapshot_digests_match() {
    // Digests are anchored to the default trace path. The cv2-parity feature
    // intentionally produces different (cv2-matching) output — its correctness
    // is verified separately by `examples/diff_snapshots.py`.
    if cfg!(feature = "contours_cv2_parity") {
        eprintln!("  snapshot_digests_match skipped (contours_cv2_parity feature is on)");
        return;
    }
    let expected = read_digests();
    let update = std::env::var("UPDATE_DIGESTS").is_ok();

    let mut new_digests: Vec<(String, String)> = Vec::new();
    let mut mismatches: Vec<String> = Vec::new();
    let mut skipped: Vec<&str> = Vec::new();

    for &(combo, mode, method) in COMBOS {
        let Some(img) = load_binary(fixture_name(combo)) else {
            skipped.push(combo);
            continue;
        };
        let r = find_contours(&img, mode, method)
            .unwrap_or_else(|e| panic!("{combo}: find_contours failed: {e:?}"));
        let json = emit_json(&r.contours);
        let got = format!("{:016x}", fnv1a64(json.as_bytes()));
        let key = format!("kornia_{combo}.json");

        if update {
            new_digests.push((key.clone(), got));
            continue;
        }
        match expected.get(&key) {
            None => mismatches.push(format!("  {key}: missing from digests.txt (got {got})")),
            Some(exp) if exp != &got => mismatches.push(format!(
                "  {key}: expected {exp}, got {got}",
            )),
            Some(_) => {}
        }
    }

    if update {
        let mut out = String::from("# FNV-1a64 hashes of canonical JSON output of find_contours.\n");
        out.push_str("# Format: <hex64>  <kornia_<fixture>_<mode>_<method>.json>\n");
        out.push_str("# Regenerate: UPDATE_DIGESTS=1 cargo test ... --test contours_snapshots -- --nocapture\n");
        new_digests.sort();
        for (name, hash) in &new_digests {
            out.push_str(&format!("{hash}  {name}\n"));
        }
        let path: PathBuf = [env!("CARGO_MANIFEST_DIR"), "tests", "snapshots", "digests.txt"]
            .iter().collect();
        std::fs::write(&path, out).expect("write digests.txt");
        eprintln!("UPDATED {path:?} with {} entries", new_digests.len());
        return;
    }

    if !skipped.is_empty() {
        eprintln!("  snapshot test skipped {} combo(s) (fixtures missing): {:?}",
                  skipped.len(), skipped);
    }
    assert!(
        mismatches.is_empty(),
        "snapshot digest mismatches:\n{}\n\nIf the change is intentional, regenerate with:\n  bash crates/kornia-imgproc/examples/dump_snapshots.sh\n  UPDATE_DIGESTS=1 cargo test --release -p kornia-imgproc --test contours_snapshots -- --nocapture",
        mismatches.join("\n"),
    );
}
