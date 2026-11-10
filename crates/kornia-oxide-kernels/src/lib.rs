//! kornia's CUDA kernels written in Rust (`device/`), compiled to one PTX
//! module by [cuda-oxide](https://github.com/NVIDIA/cuda-rust) ahead of time
//! and shipped in this crate (`ptx` feature). Without `cuda` this is
//! host-only, `no_std` data; the `cuda` feature adds [`OxideError`] and the
//! loader, which only kornia's own launchers call (hidden `__private` path:
//! a loaded kernel's launch takes unchecked arguments). See `docs/cuda-oxide.md`.
#![cfg_attr(not(feature = "cuda"), no_std)]
#![deny(missing_docs)]

#[cfg(feature = "cuda")]
mod loader;
#[cfg(feature = "cuda")]
pub use loader::OxideError;

/// Not public API: the loader, for kornia's launchers only. A kernel from it
/// launches with whatever arguments the caller passes, unchecked.
#[cfg(feature = "cuda")]
#[doc(hidden)]
pub mod __private {
    pub use crate::loader::{kernel, load_ptx_many};
}

/// The PTX module (target `sm_75`) holding every entry in [`ENTRIES`].
/// Empty when [`IS_STUB`].
pub const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx"));

/// True when built without the `ptx` feature.
pub const IS_STUB: bool = PTX.is_empty();

// `ENTRIES` and `entry::<NAME>`, generated from device/src.
include!(concat!(env!("OUT_DIR"), "/entries.rs"));

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entries() {
        assert!(!ENTRIES.is_empty());
        assert!(ENTRIES.windows(2).all(|w| w[0] < w[1]));
        assert!(ENTRIES.contains(&entry::GRAY_FROM_RGB_U8));
        assert!(IS_STUB || ENTRIES.iter().all(|e| PTX.contains(e)));
    }
}
