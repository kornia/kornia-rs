//! Build `device/` to PTX with cuda-oxide, gate it and write the stamped
//! `kernels.ptx` that the `ptx` feature ships (docs/cuda-oxide.md).
//!
//! ```text
//! scripts/cuda-oxide/install-toolchain.sh && . ~/.cargo/cuda-oxide/env.sh
//! cargo run -p kornia-oxide-kernels --example build_ptx [-- OUT]
//! ```
//!
//! Needs the nightly pinned in `device/rust-toolchain.toml` and the codegen
//! backend at the rev pinned in `device/Cargo.toml.in` (`CUDA_OXIDE_BACKEND`,
//! default `~/.cargo/cuda-oxide/librustc_codegen_cuda.so`); no CUDA toolkit.

use std::{
    env, fs,
    path::{Path, PathBuf},
    process::{Command, ExitCode},
};

#[path = "../device_src.rs"]
mod device_src;
#[path = "../ptx_gate.rs"]
mod ptx_gate;

use device_src::MANIFEST_IN;
use ptx_gate::ARCH;

fn main() -> ExitCode {
    match run() {
        Ok(out) => {
            println!("build_ptx: wrote {}", out.display());
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("build_ptx: {e}");
            ExitCode::FAILURE
        }
    }
}

/// `Some(stdout)` if `cmd` ran and exited 0.
fn probe(cmd: &mut Command) -> Option<String> {
    let out = cmd.output().ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// The one value of `key = "value"` in a TOML file (enough for the pins we
/// read); differing values are an error, comment lines are skipped.
fn toml_string(path: &Path, key: &str) -> Result<String, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("reading {}: {e}", path.display()))?;
    let mut vals = text
        .lines()
        .filter(|l| !l.trim_start().starts_with('#'))
        .filter_map(|l| {
            let rest = l.split(key).nth(1)?.trim_start().strip_prefix('=')?;
            Some(rest.trim().split('"').nth(1)?.to_string())
        });
    let first = vals
        .next()
        .ok_or_else(|| format!("no `{key}` in {}", path.display()))?;
    match vals.find(|v| *v != first) {
        Some(other) => Err(format!(
            "`{key}` differs in {}: {first} vs {other}",
            path.display()
        )),
        None => Ok(first),
    }
}

fn run() -> Result<PathBuf, String> {
    let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let device = crate_dir.join("device");
    let out = env::args_os()
        .nth(1)
        .map_or_else(|| crate_dir.join("kernels.ptx"), PathBuf::from);
    let entries = device_src::entries(&device)?;
    // Hash the inputs before staging them: an edit during the build must not
    // stamp the old PTX as current.
    let stamp_line = device_src::stamp(&device)?;

    // -- toolchain: one actionable error per missing piece
    let toolchain = toml_string(&device.join("rust-toolchain.toml"), "channel")?;
    let sysroot = probe(Command::new("rustup").args(["run", &toolchain, "rustc", "--print", "sysroot"]))
        .ok_or_else(|| {
            format!(
                "toolchain {toolchain} not available via rustup; run scripts/cuda-oxide/install-toolchain.sh"
            )
        })?;
    let backend = match env::var_os("CUDA_OXIDE_BACKEND") {
        Some(p) => PathBuf::from(p),
        None => env::var_os("CARGO_HOME")
            .map(PathBuf::from)
            .or_else(|| env::var_os("HOME").map(|h| PathBuf::from(h).join(".cargo")))
            .ok_or("neither CARGO_HOME nor HOME is set")?
            .join("cuda-oxide/librustc_codegen_cuda.so"),
    };
    if !backend.is_file() {
        return Err(format!(
            "codegen backend not found at {} (run scripts/cuda-oxide/install-toolchain.sh, \
             or set CUDA_OXIDE_BACKEND)",
            backend.display()
        ));
    }
    // A backend from another rev or nightly may mis-compile the pinned
    // cuda-device; install-toolchain.sh stamps it "<rev> <toolchain>".
    let pin = toml_string(&device.join(MANIFEST_IN), "rev")?;
    let stamp_path = backend.with_file_name(".backend");
    let stamp = fs::read_to_string(&stamp_path).unwrap_or_default();
    if stamp
        .split_whitespace()
        .ne([pin.as_str(), toolchain.as_str()])
    {
        return Err(format!(
            "{} says `{}`, device/ pins `{pin} {toolchain}`; re-run \
             scripts/cuda-oxide/install-toolchain.sh",
            stamp_path.display(),
            stamp.trim()
        ));
    }

    // -- nested build on a staged copy: device/ must never hold a Cargo.toml,
    // or `cargo package` would skip it as a nested package.
    let target = env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| crate_dir.join("../../target"))
        .join("kornia-oxide-kernels");
    let staged = target.join("device-src");
    let _ = fs::remove_dir_all(&staged);
    copy_dir(&device.join("src"), &staged.join("src"))?;
    for f in device_src::INPUTS {
        let to = if *f == MANIFEST_IN { "Cargo.toml" } else { f };
        fs::copy(device.join(f), staged.join(to)).map_err(|e| format!("staging {f}: {e}"))?;
    }
    let ptx_dir = target.join("ptx");
    let _ = fs::remove_dir_all(&ptx_dir);
    fs::create_dir_all(&ptx_dir).map_err(|e| e.to_string())?;
    let rustflags = format!(
        "-Z codegen-backend={} {}",
        backend.display(),
        device_src::RUSTFLAGS
    );
    let manifest = staged.join("Cargo.toml");
    let nested = |args: &[&str]| -> Result<(), String> {
        let mut cmd = Command::new("rustup");
        // cwd outside the repo: cargo reads `.cargo/config.toml` from the cwd
        // up, and the repo's (host linker, target rustflags) is not for this.
        cmd.args(["run", &toolchain, "cargo"])
            .args(args)
            .arg("--manifest-path")
            .arg(&manifest)
            .current_dir(env::temp_dir());
        // Whatever the caller's cargo exported (sccache wrapper, RUSTFLAGS,
        // the outer target dir, cargo-run's CARGO_*) must not reach it.
        for (key, _) in env::vars_os() {
            let Some(k) = key.to_str() else { continue };
            let keep = ["CARGO_HOME", "CARGO_BUILD_JOBS"].contains(&k)
                || ["CARGO_NET_", "CARGO_HTTP_", "CARGO_REGISTRIES_"]
                    .iter()
                    .any(|p| k.starts_with(p));
            if (k.starts_with("CARGO") && !keep)
                || k.starts_with("RUSTC")
                || k.starts_with("RUSTDOC")
                || ["RUSTFLAGS", "RUSTUP_TOOLCHAIN"].contains(&k)
            {
                cmd.env_remove(k);
            }
        }
        let ld_path = match env::var("LD_LIBRARY_PATH") {
            Ok(p) if !p.is_empty() => format!("{sysroot}/lib:{p}"),
            _ => format!("{sysroot}/lib"),
        };
        let status = cmd
            .env("CARGO_TARGET_DIR", target.join("device-target"))
            .env("RUSTFLAGS", &rustflags)
            .env("CUDA_OXIDE_TARGET", ARCH)
            // cuda-oxide contracts fmul+fadd by default (nvcc --fmad=true);
            // CPU byte parity needs it off.
            .env("CUDA_OXIDE_NO_FMA", "1")
            .env("CUDA_OXIDE_PTX_DIR", &ptx_dir)
            .env("LD_LIBRARY_PATH", ld_path)
            // Nor a `build.rustc-wrapper` from a cargo config: sccache does
            // not know the codegen backend.
            .env("CARGO_BUILD_RUSTC_WRAPPER", "")
            .status()
            .map_err(|e| format!("cannot spawn rustup: {e}"))?;
        status
            .success()
            .then_some(())
            .ok_or_else(|| format!("nested `cargo {}` failed", args.join(" ")))
    };
    // The backend emits PTX only while it compiles the kernel crate, and a
    // reinstalled backend at the same path is not in cargo's fingerprint:
    // always rebuild the crate, and the deps when the backend changed.
    let key_path = target.join("device-target.backend");
    let key = format!(
        "{stamp} {:?}",
        fs::metadata(&backend).and_then(|m| m.modified()).ok()
    );
    if fs::read_to_string(&key_path).ok().as_deref() != Some(key.as_str()) {
        let _ = fs::remove_dir_all(target.join("device-target"));
    }
    nested(&["clean", "--release", "-p", "kornia_kernels_oxide"])?;
    nested(&["build", "--release", "--locked"])?;
    fs::write(&key_path, &key).map_err(|e| e.to_string())?;

    let ptx = ptx_dir.join("kornia_kernels_oxide.ptx");
    let text = fs::read_to_string(&ptx).map_err(|e| format!("reading {}: {e}", ptx.display()))?;
    ptx_gate::gate(&text, &entries).map_err(|r| format!("PTX gate failed:\n{}", r.join("\n")))?;
    if device_src::stamp(&device)? != stamp_line {
        return Err("device/ changed during the build; re-run".into());
    }
    let stamped = format!("{stamp_line}\n{text}");
    fs::write(&out, stamped).map_err(|e| format!("writing {}: {e}", out.display()))?;
    Ok(out)
}

fn copy_dir(from: &Path, to: &Path) -> Result<(), String> {
    fs::create_dir_all(to).map_err(|e| e.to_string())?;
    for e in fs::read_dir(from).map_err(|e| format!("reading {}: {e}", from.display()))? {
        let path = e.map_err(|e| e.to_string())?.path();
        let dest = to.join(path.file_name().unwrap_or_default());
        if path.is_dir() {
            copy_dir(&path, &dest)?;
        } else {
            fs::copy(&path, &dest).map_err(|e| format!("staging {}: {e}", path.display()))?;
        }
    }
    Ok(())
}
