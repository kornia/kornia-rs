//! SIMD-accelerated image processing (proof-of-concept backend).
//!
//! This module is the scaffolding for a portable-SIMD backend for
//! `kornia-imgproc`. It currently provides vectorized variants of a handful of
//! representative ops so that the dispatch pattern, feature gating, and
//! benchmark wiring can be validated before the rest of the crate is ported.
//!
//! # Design
//!
//! * **Portability**: built on the `wide` crate, which compiles to SSE2/AVX2
//!   on x86_64 and NEON on aarch64 without requiring nightly `std::simd` or
//!   explicit `target_feature` attributes. Baseline targets are covered; use
//!   `RUSTFLAGS="-C target-cpu=native"` to unlock AVX2/FMA codegen.
//! * **Coexistence**: these functions are exposed *alongside* the scalar
//!   implementations in `color::` and `threshold::` rather than replacing
//!   them. This keeps the scalar path as the reference implementation and
//!   makes A/B benchmarking trivial.
//! * **Row parallelism**: the outer loop still uses `rayon::par_chunks_exact`
//!   over rows (mirroring `crate::parallel`); only the per-row inner loop is
//!   vectorized.
//!
//! # Available ops
//!
//! * [`gray_from_rgb_u8`] — u8 RGB→grayscale, 3→1 deinterleave + int MAC.
//! * [`threshold_binary`] — u8 elementwise compare + select (4× unrolled AVX2).

mod color;
#[cfg(target_arch = "x86_64")]
mod color_x86;
mod threshold;
#[cfg(target_arch = "x86_64")]
mod threshold_x86;

pub use color::gray_from_rgb_u8;
pub use threshold::threshold_binary;
