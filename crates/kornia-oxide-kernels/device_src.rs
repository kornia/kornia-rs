//! What `device/` builds: its `#[kernel]` names and a hash of every input of
//! the PTX. Shared by `build.rs` and `examples/build_ptx.rs`, so the PTX the
//! builder stamps and the stamp the build checks cannot disagree.

use std::{
    fs,
    path::{Path, PathBuf},
};

use crate::ptx_gate::ARCH;

/// Shipped as `.in` so `cargo package` keeps `device/` instead of skipping it
/// as a nested package; staged as `Cargo.toml` by the builder.
pub const MANIFEST_IN: &str = "Cargo.toml.in";
/// `device/` files besides `src/` that decide the PTX.
pub const INPUTS: &[&str] = &[MANIFEST_IN, "Cargo.lock", "rust-toolchain.toml"];
/// Flags of the nested build (the ones `cargo oxide build` passes, plus
/// `-Zalways-encode-mir`), after `-Z codegen-backend=<path>`.
pub const RUSTFLAGS: &str = "-C opt-level=3 -C debug-assertions=off \
     -Z mir-enable-passes=-JumpThreading -Zalways-encode-mir -Csymbol-mangling-version=v0";
/// First line of a stamped `kernels.ptx`, followed by the source hash in hex.
pub const STAMP: &str = "// kornia-oxide-kernels source fnv1a64 ";

/// `device/src/**/*.rs`, sorted.
pub fn sources(device: &Path) -> Result<Vec<PathBuf>, String> {
    let mut files = Vec::new();
    let mut dirs = vec![device.join("src")];
    while let Some(d) = dirs.pop() {
        for e in fs::read_dir(&d).map_err(|e| format!("reading {}: {e}", d.display()))? {
            let path = e.map_err(|e| e.to_string())?.path();
            if path.is_dir() {
                dirs.push(path);
            } else if path.extension().is_some_and(|x| x == "rs") {
                files.push(path);
            }
        }
    }
    files.sort();
    Ok(files)
}

/// `STAMP` + hash of the sources, `INPUTS`, `ARCH` and `RUSTFLAGS`.
pub fn stamp(device: &Path) -> Result<String, String> {
    let mut files: Vec<PathBuf> = INPUTS.iter().map(|f| device.join(f)).collect();
    files.extend(sources(device)?);
    let mut bytes = format!("{ARCH}\0{RUSTFLAGS}\0").into_bytes();
    for f in &files {
        let rel = f.strip_prefix(device).unwrap_or(f);
        bytes.extend(rel.to_string_lossy().replace('\\', "/").as_bytes());
        bytes.push(0);
        let body = fs::read(f).map_err(|e| format!("reading {}: {e}", f.display()))?;
        bytes.extend((body.len() as u64).to_le_bytes());
        bytes.extend(body);
    }
    Ok(format!("{STAMP}{:016x}", fnv1a(&bytes)))
}

/// 64-bit FNV-1a: stable across Rust releases, unlike std's hasher. It
/// catches a stale PTX; it is not a defence against a forged one.
fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
    })
}

/// Names of the `#[kernel] pub unsafe fn` items in `device/src`, sorted;
/// duplicates are an error (PTX entry names are flat). Textual: `#[kernel]`
/// on its own line or before `pub unsafe fn` on the same line; `//` comments
/// are skipped. The PTX gate checks the result against the real entries.
pub fn entries(device: &Path) -> Result<Vec<String>, String> {
    let mut names = Vec::new();
    for f in sources(device)? {
        let text = fs::read_to_string(&f).map_err(|e| format!("reading {}: {e}", f.display()))?;
        let mut pending = false;
        for (i, raw) in text.lines().enumerate() {
            let line = raw.split("//").next().unwrap_or("").trim();
            let rest = match line.strip_prefix("#[kernel]") {
                Some(rest) => rest.trim(),
                None if pending && !line.is_empty() => line,
                None => continue,
            };
            pending = rest.is_empty();
            if !pending {
                let name = rest
                    .strip_prefix("pub unsafe fn ")
                    .and_then(|r| r.split(['(', '<']).next())
                    .map(str::trim)
                    .filter(|n| !n.is_empty())
                    .ok_or_else(|| {
                        format!(
                            "{}:{}: `#[kernel]` must be followed by `pub unsafe fn <name>(`",
                            f.display(),
                            i + 1
                        )
                    })?;
                names.push(name.to_string());
            }
        }
    }
    names.sort();
    if let Some(w) = names.windows(2).find(|w| w[0] == w[1]) {
        return Err(format!("kernel `{}` is defined twice", w[0]));
    }
    if names.is_empty() {
        return Err("no `#[kernel]` in device/src".into());
    }
    Ok(names)
}
