//! Rust-authored CUDA kernels for kornia.
//!
//! One module per kernel family (`color`, `tsdf`, ...). Every `#[kernel]`
//! lands in ONE PTX module, built by kornia-oxide-kernels'
//! `examples/build_ptx.rs`; each entry is named after the Rust function, which
//! keeps the name of the NVRTC kernel it replaces.

mod color;
mod tsdf;
// Shared TSDF math (also the host lib's `tsdf_core`); not all of it runs on
// the device.
#[allow(dead_code)]
mod tsdf_core;
mod tsdf_ops;

/// PTX-only crate: `main` exists so the codegen backend has a binary to compile.
fn main() {
    println!("kornia_kernels_oxide: PTX-only crate, see docs/cuda-oxide.md");
}
