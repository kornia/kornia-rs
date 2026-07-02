//! Native CUDA (cudarc + NVRTC) kernels for color-space conversion.
//!
//! Enabled by the `gpu-cuda` feature. Each conversion follows the
//! [`super::resize_cuda`] pattern: an inline CUDA-C source string compiled once
//! per process via NVRTC (cached in a `OnceLock`), launched through
//! [`kornia_tensor::CudaKernel`] on device slices.
//!
//! # Layers
//!
//! - **Low-level launchers** (`launch_*` in the per-family submodules) take a
//!   stream plus raw `CudaSlice` operands — for callers that manage device
//!   memory themselves.
//! - **High-level dispatch** lives in `crate::color::cuda_dispatch`: the
//!   `ConvertColor` trait detects device-resident operands and routes to these
//!   launchers automatically.
//!
//! # Kernel conventions
//!
//! - One `extern "C"` entry per (op, dtype) variant — variants are selected on
//!   the Rust side, never by a per-pixel branch inside a kernel (a single
//!   branching kernel measurably regressed the preprocess path).
//! - Grouped variants that always ship together (e.g. the four Bayer patterns)
//!   share one source compiled once via [`CudaKernel::compile_many`].
//! - `u8` kernels replicate the CPU integer fixed-point math **bit-for-bit**
//!   so tests can `assert_eq!` device output against the CPU path.
//! - Elementwise ops launch 1-D (256-thread blocks); neighborhood ops (Bayer,
//!   subsampled-chroma video) launch 2-D 32×8 blocks.

use std::sync::{Arc, OnceLock};

use cudarc::driver::CudaStream;
use kornia_tensor::CudaKernel;

pub mod gray;
pub mod hsv_hls;
pub mod misc;
pub mod swizzle;
pub mod yuv;

/// Shared device helpers prepended to every color kernel source before NVRTC
/// compilation. Constants must stay in lock-step with the CPU kernels they
/// mirror (paths noted inline) — the u8 tests assert bit-exact equality.
pub(crate) const CUDA_COLOR_COMMON: &str = r#"
// Saturate an int to the u8 range.
__device__ __forceinline__ unsigned char sat_u8(int v) {
    return (unsigned char)min(max(v, 0), 255);
}

// Clamp a float to [0, 1].
__device__ __forceinline__ float clamp01(float v) {
    return fminf(fmaxf(v, 0.0f), 1.0f);
}

// BT.601 luma weights.
// u8 Q8 fixed point — matches color/gray/kernels.rs (77*R + 150*G + 29*B) >> 8.
#define GRAY_WR_Q8 77
#define GRAY_WG_Q8 150
#define GRAY_WB_Q8 29
// f32 — matches color/gray/kernels.rs RW_F32/GW_F32/BW_F32.
#define GRAY_WR_F 0.299f
#define GRAY_WG_F 0.587f
#define GRAY_WB_F 0.114f

// YCbCr/YUV Family A — full-range OpenCV Q14 fixed point.
// MUST match color/yuv/kernels.rs:23-37. CUDA `int >>` is an arithmetic shift,
// same as Rust `i32 >>`, so the negative chroma terms shift identically.
#define Q14_SHIFT 14
#define Q14_HALF  (1 << 13)
#define C_YR   4899
#define C_YG   9617
#define C_YB   1868
#define C_YCRI 11682
#define C_YCBI 9241
#define C_CR2R 22987
#define C_CR2G (-11698)
#define C_CB2G (-5636)
#define C_CB2B 29049
// f32 full-range coefficients (inputs in [0,1]) — color/yuv/kernels.rs:40-44.
#define F_YR 0.299f
#define F_YG 0.587f
#define F_YB 0.114f
#define F_CR 0.713f
#define F_CB 0.564f
"#;

// ── Error type ────────────────────────────────────────────────────────────────

/// Error type returned by the CUDA color-conversion launchers.
#[derive(Debug, thiserror::Error)]
pub enum CudaColorError {
    /// NVRTC compilation failure (deterministic; cached and returned on every call).
    #[error("CUDA color kernel compile error: {0}")]
    Compile(String),
    /// CUDA driver / launch failure.
    #[error("CUDA color kernel launch error: {0}")]
    Cuda(String),
    /// A device slice is smaller than the required element count.
    #[error("device slice length {got} < required {need} ({what})")]
    SliceTooSmall {
        /// Which operand was too small (e.g. "src", "dst").
        what: &'static str,
        /// Actual slice length (in elements).
        got: usize,
        /// Minimum required length.
        need: usize,
    },
}

impl From<kornia_tensor::CudaError> for CudaColorError {
    fn from(e: kornia_tensor::CudaError) -> Self {
        CudaColorError::Cuda(e.to_string())
    }
}

// ── Kernel cache helpers ─────────────────────────────────────────────────────

/// A once-compiled kernel cache cell. Compilation failure is deterministic, so
/// the error string is cached too and returned on every subsequent call
/// (instead of panicking like the resize path — with 40+ color kernels a
/// library-level panic on first use is not acceptable).
pub(crate) type KernelCell = OnceLock<Result<CudaKernel, String>>;

/// A once-compiled multi-entry kernel cache cell (one source, several
/// `extern "C"` entries loaded via [`CudaKernel::compile_many`]).
pub(crate) type KernelSuiteCell = OnceLock<Result<Vec<CudaKernel>, String>>;

/// Get (compiling on first use) the kernel for `fn_name` from `cell`.
///
/// The full source is `CUDA_COLOR_COMMON` + `src`; NVRTC targets the compute
/// capability of `stream`'s device. L1-preferred cache config is applied
/// best-effort (the kernels use no shared memory).
pub(crate) fn get_kernel<'a>(
    cell: &'a KernelCell,
    stream: &Arc<CudaStream>,
    src: &str,
    fn_name: &str,
) -> Result<&'a CudaKernel, CudaColorError> {
    cell.get_or_init(|| {
        let full = format!("{CUDA_COLOR_COMMON}\n{src}");
        CudaKernel::compile(stream.context(), &full, fn_name)
            .inspect(|k| {
                let _ = k.prefer_l1_cache();
            })
            .map_err(|e| e.to_string())
    })
    .as_ref()
    .map_err(|e| CudaColorError::Compile(e.clone()))
}

/// Get (compiling on first use) the kernel suite for `fn_names` from `cell`.
/// Returns the kernel at `index` (position in `fn_names`).
pub(crate) fn get_kernel_suite<'a>(
    cell: &'a KernelSuiteCell,
    stream: &Arc<CudaStream>,
    src: &str,
    fn_names: &[&str],
    index: usize,
) -> Result<&'a CudaKernel, CudaColorError> {
    let suite = cell
        .get_or_init(|| {
            let full = format!("{CUDA_COLOR_COMMON}\n{src}");
            CudaKernel::compile_many(stream.context(), &full, fn_names)
                .inspect(|ks| {
                    for k in ks {
                        let _ = k.prefer_l1_cache();
                    }
                })
                .map_err(|e| e.to_string())
        })
        .as_ref()
        .map_err(|e| CudaColorError::Compile(e.clone()))?;
    Ok(&suite[index])
}

/// Validate that a device slice holds at least `need` elements.
pub(crate) fn check_len(what: &'static str, got: usize, need: usize) -> Result<(), CudaColorError> {
    if got < need {
        return Err(CudaColorError::SliceTooSmall { what, got, need });
    }
    Ok(())
}

// ── Test helpers ─────────────────────────────────────────────────────────────

#[cfg(test)]
pub(crate) mod test_utils {
    use std::sync::Arc;

    use cudarc::driver::{CudaContext, CudaStream};

    /// Deterministic LCG byte pattern seeded with 0/255 extremes and gray runs
    /// so fixed-point edge cases are always covered.
    pub fn pattern_u8(len: usize) -> Vec<u8> {
        let mut v = Vec::with_capacity(len);
        // Deterministic edge-case prefix: extremes and a gray ramp.
        let prefix: &[u8] = &[
            0, 255, 255, 0, 0, 0, 255, 255, 255, 1, 254, 128, 128, 128, 64,
        ];
        v.extend_from_slice(&prefix[..prefix.len().min(len)]);
        let mut state = 0x1234_5678u32;
        while v.len() < len {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            v.push((state >> 24) as u8);
        }
        v
    }

    /// Deterministic f32 pattern in [0, 1] including exact 0.0 / 1.0.
    pub fn pattern_f32(len: usize) -> Vec<f32> {
        pattern_u8(len).iter().map(|&b| b as f32 / 255.0).collect()
    }

    /// Default-stream handle for device tests (Jetson: single GPU, ordinal 0).
    pub fn default_stream() -> Arc<CudaStream> {
        CudaContext::new(0)
            .expect("CUDA device 0 must be available for gpu-cuda tests")
            .default_stream()
    }
}
