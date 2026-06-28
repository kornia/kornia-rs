// Build script for kornia-xfeat.
//
// Registers the `kornia_nightly_fp16` cfg key so that
// `#[cfg(kornia_nightly_fp16)]` gates in the source compile without a
// "unexpected cfg" warning.  The cfg is activated by setting
//   RUSTFLAGS="--cfg kornia_nightly_fp16"
// or, in .cargo/config.toml:
//   [build]
//   rustflags = ["--cfg", "kornia_nightly_fp16"]
//
// When enabled, the crate MUST be built with a nightly toolchain that has
// the `stdarch_neon_f16` feature (Rust ≥ nightly-2026-04-03 on aarch64).

fn main() {
    println!("cargo::rustc-check-cfg=cfg(kornia_nightly_fp16)");
}
