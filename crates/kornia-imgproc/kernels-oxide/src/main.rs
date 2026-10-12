//! Rust-authored CUDA kernels for kornia-imgproc.
//!
//! Every `#[kernel]` in this crate lands in ONE PTX module, which
//! kornia-imgproc's `build.rs` compiles into its `OUT_DIR`. PTX entry names are
//! the plain Rust function names.

mod color_gray;

/// PTX-only crate: `main` exists so the codegen backend has a binary to compile.
fn main() {
    println!("kornia_kernels_oxide: PTX-only crate, see docs/cuda-oxide.md");
}
