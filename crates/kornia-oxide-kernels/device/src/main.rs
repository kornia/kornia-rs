//! Rust-authored CUDA kernels for kornia.
//!
//! One module per kernel family (`color`, `resize`, ...). Every `#[kernel]`
//! lands in ONE PTX module, built by kornia-oxide-kernels'
//! `examples/build_ptx.rs`; each entry is named after the Rust function, which
//! keeps the name of the NVRTC kernel it replaces.

mod color;
mod features;

/// PTX-only crate: `main` exists so the codegen backend has a binary to compile.
fn main() {
    println!("kornia_kernels_oxide: PTX-only crate, see docs/cuda-oxide.md");
}
