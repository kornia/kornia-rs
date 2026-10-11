//! No-op unless the `cuda-oxide` feature is on. With it: compile
//! `kernels-oxide/` to PTX through the cuda-oxide codegen backend, gate the
//! PTX, and write `$OUT_DIR/kernels.ptx` for `src/cuda/oxide/mod.rs`.
//! Toolchain setup: docs/cuda-oxide.md.

use std::{
    env, fs,
    path::{Path, PathBuf},
    process::Command,
};

/// PTX target: JITs on any Turing-or-newer GPU.
const ARCH: &str = "sm_75";
const PTX_NAME: &str = "kornia_kernels_oxide.ptx";

#[derive(Debug, thiserror::Error)]
enum OxideBuildError {
    #[error(
        "{0}\n\nThe `cuda-oxide` feature compiles Rust kernels to PTX at build time and needs \
         that toolchain; see docs/cuda-oxide.md and run `cargo oxide doctor`. Build without the \
         feature to use the NVRTC kernels."
    )]
    Toolchain(String),
    #[error("nested kernels-oxide build failed:\n{0}")]
    Build(String),
    #[error("PTX gate failed for {path}: {reason}")]
    Gate { path: String, reason: String },
    #[error("{0}")]
    Io(String),
}
use OxideBuildError::{Build, Gate, Io, Toolchain};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    if env::var_os("CARGO_FEATURE_CUDA_OXIDE").is_none() {
        return;
    }
    if let Err(e) = build_oxide_ptx() {
        panic!("\n\ncuda-oxide: {e}\n");
    }
}

fn env_path(key: &str) -> Result<PathBuf, OxideBuildError> {
    env::var_os(key)
        .map(PathBuf::from)
        .ok_or_else(|| Io(format!("cargo did not set {key}")))
}

fn read(path: &Path) -> Result<String, OxideBuildError> {
    fs::read_to_string(path).map_err(|e| Io(format!("reading {}: {e}", path.display())))
}

/// `Some(stdout)` if `cmd` ran and exited 0.
fn probe(cmd: &mut Command) -> Option<String> {
    let out = cmd.output().ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// First `key = "value"` in a TOML file (enough for the two pins we read).
fn toml_string(text: &str, key: &str) -> Option<String> {
    text.lines().find_map(|l| {
        let rest = l.split(key).nth(1)?.trim_start().strip_prefix('=')?;
        Some(rest.trim().split('"').nth(1)?.to_string())
    })
}

fn build_oxide_ptx() -> Result<(), OxideBuildError> {
    let manifest_dir = env_path("CARGO_MANIFEST_DIR")?;
    let out_dir = env_path("OUT_DIR")?;
    let kernels = manifest_dir.join("../../kernels-oxide");
    if !kernels.join("Cargo.toml").is_file() {
        return Err(Toolchain(format!(
            "{} not found: the feature builds only from a kornia-rs git checkout",
            kernels.display()
        )));
    }
    for f in ["src", "Cargo.toml", "Cargo.lock", "rust-toolchain.toml"] {
        println!("cargo:rerun-if-changed={}", kernels.join(f).display());
    }
    for v in [
        "CUDA_OXIDE_LLC",
        "CUDA_OXIDE_BACKEND",
        "CARGO_HOME",
        "CUDA_HOME",
        "CUDA_TOOLKIT_PATH",
    ] {
        println!("cargo:rerun-if-env-changed={v}");
    }

    // -- toolchain preflight: one actionable error, never an empty PTX ------
    let toolchain = toml_string(&read(&kernels.join("rust-toolchain.toml"))?, "channel")
        .ok_or_else(|| Io("no channel in kernels-oxide/rust-toolchain.toml".into()))?;
    let sysroot =
        probe(Command::new("rustup").args(["run", &toolchain, "rustc", "--print", "sysroot"]))
            .ok_or_else(|| {
                Toolchain(format!(
                    "toolchain {toolchain} not available via rustup; install it with \
                 `rustup toolchain install {toolchain} -c rust-src,rustc-dev`"
                ))
            })?;

    let backend = match env::var_os("CUDA_OXIDE_BACKEND") {
        Some(p) => PathBuf::from(p),
        None => env::var_os("CARGO_HOME")
            .map(PathBuf::from)
            .or_else(|| env::var_os("HOME").map(|h| PathBuf::from(h).join(".cargo")))
            .ok_or_else(|| Io("neither CARGO_HOME nor HOME is set".into()))?
            .join("cuda-oxide/librustc_codegen_cuda.so"),
    };
    if !backend.is_file() {
        return Err(Toolchain(format!(
            "codegen backend not found at {} (set CUDA_OXIDE_BACKEND to override)",
            backend.display()
        )));
    }
    // A backend built from another rev may mis-compile the pinned cuda-device.
    let pin = toml_string(&read(&kernels.join("Cargo.toml"))?, "rev")
        .ok_or_else(|| Io("no cuda-oxide rev in kernels-oxide/Cargo.toml".into()))?;
    if let Some(src) = backend
        .parent()
        .map(|d| d.join("src"))
        .filter(|s| s.join(".git").exists())
    {
        let head = probe(
            Command::new("git")
                .arg("-C")
                .arg(&src)
                .args(["rev-parse", "HEAD"]),
        );
        if head.as_deref().is_some_and(|h| h != pin) {
            return Err(Toolchain(format!(
                "backend source {} is at {}, kernels-oxide pins {pin}; rebuild the backend at the pin",
                src.display(),
                head.unwrap_or_default()
            )));
        }
    }

    let llc_ok = env::var_os("CUDA_OXIDE_LLC").is_some()
        || ["llc-22", "llc-21"]
            .iter()
            .any(|llc| probe(Command::new(llc).arg("--version")).is_some());
    if !llc_ok {
        return Err(Toolchain(
            "llc-21 or llc-22 (LLVM 21+) not on PATH; install LLVM 21 or set CUDA_OXIDE_LLC".into(),
        ));
    }

    let cuda_root = env::var_os("CUDA_TOOLKIT_PATH")
        .or_else(|| env::var_os("CUDA_HOME"))
        .map_or_else(|| PathBuf::from("/usr/local/cuda"), PathBuf::from);
    if !cuda_root.join("include/cuda.h").is_file() {
        return Err(Toolchain(format!(
            "{}/include/cuda.h not found (needed by cuda-oxide's bindgen); set CUDA_HOME",
            cuda_root.display()
        )));
    }

    // -- nested build ---------------------------------------------------------
    let target_dir = out_dir.join("kernels-oxide-target");
    let ptx_dir = out_dir.join("kernels-oxide-ptx");
    let ptx = ptx_dir.join(PTX_NAME);
    let jobs = env::var("CARGO_BUILD_JOBS")
        .or_else(|_| env::var("NUM_JOBS"))
        .unwrap_or_else(|_| "2".into());
    let ld_path = match env::var("LD_LIBRARY_PATH") {
        Ok(p) if !p.is_empty() => format!("{sysroot}/lib:{p}"),
        _ => format!("{sysroot}/lib"),
    };
    // Same flags `cargo oxide build` passes; driven directly because cargo-oxide
    // rewrites src/main.rs on every run, which would re-trigger this script forever.
    let rustflags = format!(
        "-Z codegen-backend={} -C opt-level=3 -C debug-assertions=off \
         -Z mir-enable-passes=-JumpThreading -Csymbol-mangling-version=v0",
        backend.display()
    );

    let run_nested = |args: &[&str]| -> Result<(), OxideBuildError> {
        let mut cmd = Command::new("rustup");
        cmd.args(["run", &toolchain, "cargo"])
            .args(args)
            .current_dir(&kernels);
        scrub_cargo_env(&mut cmd);
        let out = cmd
            .env("CARGO_TARGET_DIR", &target_dir)
            .env("RUSTFLAGS", &rustflags)
            .env("CUDA_OXIDE_TARGET", ARCH)
            .env("CUDA_OXIDE_PTX_DIR", &ptx_dir)
            .env("LD_LIBRARY_PATH", &ld_path)
            .output()
            .map_err(|e| Toolchain(format!("cannot spawn rustup: {e}")))?;
        if out.status.success() {
            Ok(())
        } else {
            Err(Build(String::from_utf8_lossy(&out.stderr).into_owned()))
        }
    };
    // The backend does not create its output directory.
    fs::create_dir_all(&ptx_dir).map_err(|e| Io(format!("creating {}: {e}", ptx_dir.display())))?;
    let build_args = ["build", "--release", "--locked", "--jobs", &jobs];
    run_nested(&build_args)?;
    if !ptx.is_file() {
        // The PTX is emitted only when the kernel crate is re-codegened; force
        // that if a fresh nested target dir outlived its PTX.
        run_nested(&["clean", "--release", "-p", "kornia_kernels_oxide"])?;
        run_nested(&build_args)?;
    }

    let text = read(&ptx)?;
    gate(&text).map_err(|reason| Gate {
        path: ptx.display().to_string(),
        reason,
    })?;
    let dest = out_dir.join("kernels.ptx");
    if fs::read_to_string(&dest).ok().as_deref() != Some(text.as_str()) {
        fs::write(&dest, &text).map_err(|e| Io(format!("writing {}: {e}", dest.display())))?;
    }
    Ok(())
}

/// Strip what the outer cargo exports to build scripts, so the nested cargo
/// resolves its own toolchain, flags, target dir and jobserver.
fn scrub_cargo_env(cmd: &mut Command) {
    const KEEP: &[&str] = &["CARGO_HOME", "CARGO_BUILD_JOBS"];
    const KEEP_PREFIX: &[&str] = &["CARGO_NET_", "CARGO_HTTP_", "CARGO_REGISTRIES_"];
    const DROP: &[&str] = &[
        "RUSTC",
        "RUSTC_WRAPPER",
        "RUSTC_WORKSPACE_WRAPPER",
        "RUSTC_LINKER",
        "RUSTDOC",
        "RUSTFLAGS",
        "RUSTDOCFLAGS",
        "RUSTUP_TOOLCHAIN",
        "MAKEFLAGS",
        "MFLAGS",
        "OUT_DIR",
        "TARGET",
        "HOST",
        "NUM_JOBS",
        "OPT_LEVEL",
        "PROFILE",
        "DEBUG",
    ];
    for (key, _) in env::vars_os() {
        let Some(k) = key.to_str() else { continue };
        let cargo_owned = k.starts_with("CARGO_")
            && !KEEP.contains(&k)
            && !KEEP_PREFIX.iter().any(|p| k.starts_with(p));
        if cargo_owned || DROP.contains(&k) {
            cmd.env_remove(k);
        }
    }
}

/// `.target` is `ARCH`; no `fma` (CPU byte parity); every `ld`/`st` names a
/// state space (generic access means address-space inference was lost).
fn gate(ptx: &str) -> Result<(), String> {
    if !ptx.contains(".entry ") {
        return Err("no .entry in PTX".into());
    }
    if !ptx.lines().any(|l| l.trim() == format!(".target {ARCH}")) {
        return Err(format!(".target is not {ARCH}"));
    }
    if ptx.contains("fma.") {
        return Err("contains fma".into());
    }
    const SPACES: &[&str] = &["global", "param", "shared", "local", "const"];
    for line in ptx.lines() {
        let mut toks = line.split("//").next().unwrap_or("").split_whitespace();
        let op = match toks.next() {
            Some(t) if t.starts_with('@') => toks.next(),
            t => t,
        };
        if let Some(op) = op.filter(|o| o.starts_with("ld.") || o.starts_with("st.")) {
            if !SPACES.contains(&op.split('.').nth(1).unwrap_or("")) {
                return Err(format!("generic memory op `{op}`"));
            }
        }
    }
    Ok(())
}
