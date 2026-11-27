//! Source rules of the TSDF module that clippy cannot state: per-element code spells fused
//! multiply-add and square root only through `F32Ops` (kornia-3d TSDF contract, section 1).
//! The typed rules (`f32::min`/`max`/`clamp`, inherent `mul_add`/`sqrt`/`floor`) are the
//! `disallowed-methods` of `kornia-oxide-kernels/clippy.toml`, which covers the shared core.

use std::fs;
use std::path::Path;

/// Tokens no TSDF source may contain: they fuse, approximate or reassociate differently per target.
const FORBIDDEN: &[&str] = &[
    "F32x4",
    "F32x8",
    "vmlaq_f32",
    "vmlsq_f32",
    "vrecpe",
    "vrecps",
    "vrsqrte",
    "vrsqrts",
    "_fast(",
    "_algebraic(",
    "fmuladdf32",
];

/// The module's sources (not its tests), `//` comments removed: (file name, code).
fn sources() -> Vec<(String, String)> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/tsdf");
    let mut out = Vec::new();
    for entry in fs::read_dir(&dir).expect("src/tsdf") {
        let path = entry.expect("dir entry").path();
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if name.ends_with(".rs") && name != "tests.rs" {
            let code = fs::read_to_string(&path)
                .expect("read source")
                .lines()
                .map(|l| l.split("//").next().unwrap_or(""))
                .collect::<Vec<_>>()
                .join("\n");
            out.push((name.to_string(), code));
        }
    }
    out
}

#[test]
fn tsdf_sources_use_no_target_dependent_float_op() {
    let mut hits = Vec::new();
    for (name, code) in sources() {
        for tok in FORBIDDEN {
            if code.contains(tok) {
                hits.push(format!("src/tsdf/{name}: `{tok}`"));
            }
        }
    }
    assert!(hits.is_empty(), "forbidden float ops:\n{}", hits.join("\n"));
}

#[test]
fn tsdf_fma_is_only_the_std_ops_impl() {
    let sites: Vec<String> = sources()
        .into_iter()
        .filter(|(_, code)| code.contains("mul_add("))
        .map(|(name, _)| name)
        .collect();
    assert_eq!(sites, ["mod.rs"], "mul_add( outside StdOps");
}
