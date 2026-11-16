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

/// Not public API: the TSDF per-voxel math, the same file the device crate
/// compiles, so kornia-3d's CPU path and the kernels share one source.
#[doc(hidden)]
#[path = "../device/src/tsdf_core.rs"]
pub mod tsdf_core;

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

    /// Every by-value kernel parameter (`.param .align A .b8 p[N]`) is one of
    /// the `tsdf_core` structs, with the host's size and alignment: the
    /// launcher passes `&T` and the driver copies `N` bytes from it.
    #[test]
    fn byval_params_match_the_host_layout() {
        use core::mem::{align_of, size_of};
        use tsdf_core as tc;
        let layout = |s: usize, a: usize| (s, a);
        let step = layout(size_of::<tc::StepParams>(), align_of::<tc::StepParams>());
        let pin = layout(
            size_of::<tc::PinholeParams>(),
            align_of::<tc::PinholeParams>(),
        );
        let sph = layout(
            size_of::<tc::SphericalParams>(),
            align_of::<tc::SphericalParams>(),
        );
        let want = [
            ("tsdf_assign_misses", 0, step),
            ("tsdf_integrate_pinhole", 0, step),
            ("tsdf_integrate_pinhole", 1, pin),
            ("tsdf_integrate_spherical", 0, step),
            ("tsdf_integrate_spherical", 1, sph),
            ("tsdf_merge_keys", 0, step),
            ("tsdf_rank_misses", 0, step),
            ("tsdf_touch_pinhole", 0, step),
            ("tsdf_touch_pinhole", 1, pin),
            ("tsdf_touch_spherical", 0, step),
            ("tsdf_touch_spherical", 1, sph),
        ];
        let mut got: Vec<(&str, usize, (usize, usize))> = PTX
            .lines()
            .filter_map(|l| {
                // `.param .align 8 .b8 tsdf_rank_misses_param_0[64],`
                let rest = l.trim().strip_prefix(".param .align ")?;
                let (align, rest) = rest.split_once(' ')?;
                let (name, n) = rest.strip_prefix(".b8 ")?.split_once('[')?;
                let (entry, idx) = name.rsplit_once("_param_")?;
                let n = n.split(']').next()?;
                Some((
                    entry,
                    idx.parse().ok()?,
                    (n.parse().ok()?, align.parse().ok()?),
                ))
            })
            .collect();
        got.sort();
        if IS_STUB {
            assert!(got.is_empty());
        } else {
            assert_eq!(got, want);
        }
    }
}
