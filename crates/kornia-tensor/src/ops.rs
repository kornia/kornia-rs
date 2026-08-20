//! Element-wise and reduction operations on [`Tensor<f32, N>`] with automatic
//! CPU / GPU dispatch based on the tensor's [`MemoryDomain`].
//!
//! # Operations
//!
//! | Group | Ops |
//! |-------|-----|
//! | Unary | [`UnaryOp::Abs`], [`UnaryOp::Relu`], [`UnaryOp::Neg`], [`UnaryOp::Clamp`] |
//! | Binary | [`BinaryOp::Add`], [`BinaryOp::Sub`], [`BinaryOp::Mul`], [`BinaryOp::Div`], [`BinaryOp::Min`], [`BinaryOp::Max`] |
//! | Reduce | [`ReduceOp::Sum`], [`ReduceOp::Mean`] |
//!
//! # Dispatch
//!
//! [`apply_unary`], [`apply_binary`], and [`reduce`] inspect the tensor's
//! [`MemoryDomain`] at runtime:
//!
//! - [`MemoryDomain::Host`] / [`MemoryDomain::Unified`] → CPU loops (unified
//!   memory is host-accessible so the CPU path is always valid).
//! - [`MemoryDomain::Device`] → NVRTC GPU kernels (requires `cuda` feature;
//!   returns [`OpsError::CudaNotEnabled`] otherwise).

use crate::{
    resource::MemoryDomain,
    tensor::{Tensor, TensorError},
};

// ── Op enums ──────────────────────────────────────────────────────────────────

/// Element-wise unary operation.
#[derive(Clone, Copy, Debug)]
pub enum UnaryOp {
    /// Absolute value: `|x|`
    Abs,
    /// Rectified linear unit: `max(0, x)`
    Relu,
    /// Negation: `-x`
    Neg,
    /// Clamp to `[min, max]`
    Clamp {
        /// Lower bound (inclusive).
        min: f32,
        /// Upper bound (inclusive).
        max: f32,
    },
}

/// Element-wise binary operation.
#[derive(Clone, Copy, Debug)]
pub enum BinaryOp {
    /// `a + b`
    Add,
    /// `a - b`
    Sub,
    /// `a * b`
    Mul,
    /// `a / b`
    Div,
    /// `min(a, b)`
    Min,
    /// `max(a, b)`
    Max,
}

/// Reduction operation (collapses all elements to a scalar).
#[derive(Clone, Copy, Debug)]
pub enum ReduceOp {
    /// Sum of all elements.
    Sum,
    /// Arithmetic mean.
    Mean,
}

// ── Error type ────────────────────────────────────────────────────────────────

/// Error type for tensor element-wise and reduction operations.
#[derive(Debug, thiserror::Error)]
pub enum OpsError {
    /// Shape or element-count mismatch between operands.
    #[error("shape mismatch: {0}")]
    ShapeMismatch(String),

    /// Operation requires device-accessible memory.
    #[error("tensor is not device-accessible (domain={0:?})")]
    NotDeviceAccessible(MemoryDomain),

    /// Tried to run a GPU op but the `cuda` feature is not enabled.
    #[error("CUDA feature not enabled — rebuild with --features cuda")]
    CudaNotEnabled,

    /// Underlying CUDA driver error.
    #[cfg(feature = "cuda")]
    #[error("CUDA error: {0}")]
    Cuda(#[from] crate::cuda::CudaError),

    /// Wrapped tensor error.
    #[error("{0}")]
    Tensor(#[from] TensorError),
}

// ── CPU implementations ───────────────────────────────────────────────────────

fn cpu_apply_unary<const N: usize>(
    input: &Tensor<f32, N>,
    output: &mut Tensor<f32, N>,
    op: UnaryOp,
) -> Result<(), OpsError> {
    if input.shape != output.shape {
        return Err(OpsError::ShapeMismatch(format!(
            "{:?} vs {:?}",
            input.shape, output.shape
        )));
    }
    let src = input.as_slice();
    let dst = output.as_slice_mut();
    match op {
        UnaryOp::Abs => {
            for (d, s) in dst.iter_mut().zip(src) {
                *d = s.abs();
            }
        }
        UnaryOp::Relu => {
            for (d, s) in dst.iter_mut().zip(src) {
                *d = s.max(0.0);
            }
        }
        UnaryOp::Neg => {
            for (d, s) in dst.iter_mut().zip(src) {
                *d = -*s;
            }
        }
        UnaryOp::Clamp { min, max } => {
            for (d, s) in dst.iter_mut().zip(src) {
                *d = s.clamp(min, max);
            }
        }
    }
    Ok(())
}

fn cpu_apply_binary<const N: usize>(
    a: &Tensor<f32, N>,
    b: &Tensor<f32, N>,
    out: &mut Tensor<f32, N>,
    op: BinaryOp,
) -> Result<(), OpsError> {
    if a.shape != b.shape || a.shape != out.shape {
        return Err(OpsError::ShapeMismatch(format!(
            "{:?} vs {:?} vs {:?}",
            a.shape, b.shape, out.shape
        )));
    }
    let as_ = a.as_slice();
    let bs = b.as_slice();
    let ds = out.as_slice_mut();
    match op {
        BinaryOp::Add => {
            for ((d, a), b) in ds.iter_mut().zip(as_).zip(bs) {
                *d = a + b;
            }
        }
        BinaryOp::Sub => {
            for ((d, a), b) in ds.iter_mut().zip(as_).zip(bs) {
                *d = a - b;
            }
        }
        BinaryOp::Mul => {
            for ((d, a), b) in ds.iter_mut().zip(as_).zip(bs) {
                *d = a * b;
            }
        }
        BinaryOp::Div => {
            for ((d, a), b) in ds.iter_mut().zip(as_).zip(bs) {
                *d = a / b;
            }
        }
        BinaryOp::Min => {
            for ((d, a), b) in ds.iter_mut().zip(as_).zip(bs) {
                *d = a.min(*b);
            }
        }
        BinaryOp::Max => {
            for ((d, a), b) in ds.iter_mut().zip(as_).zip(bs) {
                *d = a.max(*b);
            }
        }
    }
    Ok(())
}

fn cpu_reduce<const N: usize>(input: &Tensor<f32, N>, op: ReduceOp) -> Result<f32, OpsError> {
    let src = input.as_slice();
    let n = src.len();
    if n == 0 {
        return Ok(0.0);
    }
    let sum: f32 = src.iter().copied().sum();
    Ok(match op {
        ReduceOp::Sum => sum,
        ReduceOp::Mean => sum / n as f32,
    })
}

// ── CUDA implementations ──────────────────────────────────────────────────────

#[cfg(feature = "cuda")]
mod cuda_ops {
    use std::{mem::ManuallyDrop, sync::Arc};

    use cudarc::driver::{CudaSlice, CudaStream, DeviceRepr, LaunchConfig};

    use crate::{
        cuda::{CudaError, CudaKernel},
        tensor::Tensor,
    };

    use super::{BinaryOp, OpsError, ReduceOp, UnaryOp};

    // ── kernel sources ────────────────────────────────────────────────────────

    const UNARY_SRC: &str = r#"
extern "C" __global__ void kernel_unary(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int n, int op, float lo, float hi)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = src[i];
    switch (op) {
        case 0: dst[i] = fabsf(v); break;
        case 1: dst[i] = fmaxf(0.0f, v); break;
        case 2: dst[i] = -v; break;
        case 3: dst[i] = fmaxf(lo, fminf(hi, v)); break;
    }
}
"#;

    const BINARY_SRC: &str = r#"
extern "C" __global__ void kernel_binary(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ dst,
    int n, int op)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float va = a[i], vb = b[i];
    switch (op) {
        case 0: dst[i] = va + vb; break;
        case 1: dst[i] = va - vb; break;
        case 2: dst[i] = va * vb; break;
        case 3: dst[i] = va / vb; break;
        case 4: dst[i] = fminf(va, vb); break;
        case 5: dst[i] = fmaxf(va, vb); break;
    }
}
"#;

    // Block-level reduction: shared memory + atomicAdd into a single output slot.
    const REDUCE_SRC: &str = r#"
extern "C" __global__ void kernel_reduce_sum(
    const float* __restrict__ src, float* out, int n)
{
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < (unsigned int)n) ? src[i] : 0.0f;
    __syncthreads();
    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, sdata[0]);
}
"#;

    // ── lazy-compiled kernel singletons ───────────────────────────────────────

    use std::sync::OnceLock;
    static UNARY_KERNEL: OnceLock<Result<CudaKernel, String>> = OnceLock::new();
    static BINARY_KERNEL: OnceLock<Result<CudaKernel, String>> = OnceLock::new();
    static REDUCE_KERNEL: OnceLock<Result<CudaKernel, String>> = OnceLock::new();

    fn get_unary(ctx: &Arc<cudarc::driver::CudaContext>) -> Result<&'static CudaKernel, CudaError> {
        UNARY_KERNEL
            .get_or_init(|| {
                CudaKernel::compile(ctx, UNARY_SRC, "kernel_unary").map_err(|e| e.to_string())
            })
            .as_ref()
            .map_err(|e| CudaError::Driver(e.clone()))
    }

    fn get_binary(
        ctx: &Arc<cudarc::driver::CudaContext>,
    ) -> Result<&'static CudaKernel, CudaError> {
        BINARY_KERNEL
            .get_or_init(|| {
                CudaKernel::compile(ctx, BINARY_SRC, "kernel_binary").map_err(|e| e.to_string())
            })
            .as_ref()
            .map_err(|e| CudaError::Driver(e.clone()))
    }

    fn get_reduce(
        ctx: &Arc<cudarc::driver::CudaContext>,
    ) -> Result<&'static CudaKernel, CudaError> {
        REDUCE_KERNEL
            .get_or_init(|| {
                CudaKernel::compile(ctx, REDUCE_SRC, "kernel_reduce_sum").map_err(|e| e.to_string())
            })
            .as_ref()
            .map_err(|e| CudaError::Driver(e.clone()))
    }

    // ── non-owning CudaSlice alias ────────────────────────────────────────────

    /// Wrap a tensor's raw device pointer as a non-owning `CudaSlice<T>`.
    ///
    /// The returned slice is wrapped in `ManuallyDrop` to prevent cudarc from
    /// calling `cuMemFreeAsync` — the tensor owns the memory and will free it on drop.
    ///
    /// # Safety
    ///
    /// The alias is valid only while `tensor` is alive and its domain is Device.
    unsafe fn alias<T: DeviceRepr, const N: usize>(
        tensor: &Tensor<T, N>,
        stream: &Arc<CudaStream>,
    ) -> ManuallyDrop<CudaSlice<T>> {
        let numel = tensor.shape.iter().product::<usize>();
        let slice = stream.upgrade_device_ptr::<T>(tensor.as_ptr() as u64, numel);
        ManuallyDrop::new(slice)
    }

    // ── public CUDA ops ───────────────────────────────────────────────────────

    pub fn apply_unary<const N: usize>(
        input: &Tensor<f32, N>,
        output: &mut Tensor<f32, N>,
        op: UnaryOp,
    ) -> Result<(), OpsError> {
        if input.shape != output.shape {
            return Err(OpsError::ShapeMismatch(format!(
                "{:?} vs {:?}",
                input.shape, output.shape
            )));
        }
        let stream = input
            .cuda_stream()
            .ok_or_else(|| OpsError::NotDeviceAccessible(input.storage.domain()))?;
        let ctx = stream.context().clone();
        let kernel = get_unary(&ctx)?;

        let numel = input.shape.iter().product::<usize>() as i32;
        let (op_code, lo, hi): (i32, f32, f32) = match op {
            UnaryOp::Abs => (0, 0.0, 0.0),
            UnaryOp::Relu => (1, 0.0, 0.0),
            UnaryOp::Neg => (2, 0.0, 0.0),
            UnaryOp::Clamp { min, max } => (3, min, max),
        };

        // SAFETY: aliases are ManuallyDrop — cuMemFreeAsync never runs on them.
        let src_alias = unsafe { alias(input, stream) };
        let mut dst_alias = unsafe { alias(output, stream) };

        kernel
            .launch_builder(stream)
            .arg(&*src_alias)
            .arg(&mut *dst_alias)
            .arg(&numel)
            .arg(&op_code)
            .arg(&lo)
            .arg(&hi)
            .launch_1d(numel as u32)?;

        Ok(())
    }

    pub fn apply_binary<const N: usize>(
        a: &Tensor<f32, N>,
        b: &Tensor<f32, N>,
        out: &mut Tensor<f32, N>,
        op: BinaryOp,
    ) -> Result<(), OpsError> {
        if a.shape != b.shape || a.shape != out.shape {
            return Err(OpsError::ShapeMismatch(format!(
                "{:?} / {:?} / {:?}",
                a.shape, b.shape, out.shape
            )));
        }
        let stream = a
            .cuda_stream()
            .ok_or_else(|| OpsError::NotDeviceAccessible(a.storage.domain()))?;
        let ctx = stream.context().clone();
        let kernel = get_binary(&ctx)?;

        let numel = a.shape.iter().product::<usize>() as i32;
        let op_code: i32 = match op {
            BinaryOp::Add => 0,
            BinaryOp::Sub => 1,
            BinaryOp::Mul => 2,
            BinaryOp::Div => 3,
            BinaryOp::Min => 4,
            BinaryOp::Max => 5,
        };

        let a_alias = unsafe { alias(a, stream) };
        let b_alias = unsafe { alias(b, stream) };
        let mut dst_alias = unsafe { alias(out, stream) };

        kernel
            .launch_builder(stream)
            .arg(&*a_alias)
            .arg(&*b_alias)
            .arg(&mut *dst_alias)
            .arg(&numel)
            .arg(&op_code)
            .launch_1d(numel as u32)?;

        Ok(())
    }

    pub fn reduce<const N: usize>(input: &Tensor<f32, N>, op: ReduceOp) -> Result<f32, OpsError> {
        let stream = input
            .cuda_stream()
            .ok_or_else(|| OpsError::NotDeviceAccessible(input.storage.domain()))?;
        let ctx = stream.context().clone();
        let kernel = get_reduce(&ctx)?;

        let numel = input.shape.iter().product::<usize>();
        if numel == 0 {
            return Ok(0.0);
        }

        // Single-element output buffer, zero-initialised.
        let mut out_dev: CudaSlice<f32> = stream
            .alloc_zeros::<f32>(1)
            .map_err(|e| CudaError::Driver(e.to_string()))?;

        let src_alias = unsafe { alias(input, stream) };

        const BLOCK: u32 = 256;
        let grid = (numel as u32).div_ceil(BLOCK);
        let n_i32 = numel as i32;

        kernel
            .launch_builder(stream)
            .arg(&*src_alias)
            .arg(&mut out_dev)
            .arg(&n_i32)
            .launch_cfg(LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: BLOCK * 4, // BLOCK floats of shared memory
            })?;

        let result = stream
            .clone_dtoh(&out_dev)
            .map_err(|e| CudaError::Driver(e.to_string()))?;
        stream
            .synchronize()
            .map_err(|e| CudaError::Driver(e.to_string()))?;

        let sum = result[0];
        Ok(match op {
            ReduceOp::Sum => sum,
            ReduceOp::Mean => sum / numel as f32,
        })
    }
}

// ── Public dispatch functions ─────────────────────────────────────────────────

/// Apply an element-wise unary operation to `input`, writing results into
/// `output`.
///
/// Dispatches to the GPU kernel when `input` lives in `Device` memory.
/// `Host` and `Unified` tensors are processed on the CPU (unified memory is
/// host-accessible so the CPU path is always valid).
///
/// # Errors
///
/// - [`OpsError::ShapeMismatch`] — shapes differ.
/// - [`OpsError::CudaNotEnabled`] — `Device` tensor but `cuda` feature absent.
pub fn apply_unary<const N: usize>(
    input: &Tensor<f32, N>,
    output: &mut Tensor<f32, N>,
    op: UnaryOp,
) -> Result<(), OpsError> {
    match input.storage.domain() {
        MemoryDomain::Host | MemoryDomain::Unified { .. } => cpu_apply_unary(input, output, op),
        MemoryDomain::Device { .. } => {
            #[cfg(feature = "cuda")]
            return cuda_ops::apply_unary(input, output, op);
            #[cfg(not(feature = "cuda"))]
            Err(OpsError::CudaNotEnabled)
        }
    }
}

/// Apply an element-wise binary operation across `a` and `b`, writing into
/// `out`.
///
/// Dispatches to GPU when `a` is `Device`-backed; `Host` / `Unified` use CPU.
pub fn apply_binary<const N: usize>(
    a: &Tensor<f32, N>,
    b: &Tensor<f32, N>,
    out: &mut Tensor<f32, N>,
    op: BinaryOp,
) -> Result<(), OpsError> {
    match a.storage.domain() {
        MemoryDomain::Host | MemoryDomain::Unified { .. } => cpu_apply_binary(a, b, out, op),
        MemoryDomain::Device { .. } => {
            #[cfg(feature = "cuda")]
            return cuda_ops::apply_binary(a, b, out, op);
            #[cfg(not(feature = "cuda"))]
            Err(OpsError::CudaNotEnabled)
        }
    }
}

/// Reduce all elements of `input` to a scalar.
///
/// Dispatches to GPU when `input` is `Device`-backed; `Host` / `Unified` use
/// CPU.
pub fn reduce<const N: usize>(input: &Tensor<f32, N>, op: ReduceOp) -> Result<f32, OpsError> {
    match input.storage.domain() {
        MemoryDomain::Host | MemoryDomain::Unified { .. } => cpu_reduce(input, op),
        MemoryDomain::Device { .. } => {
            #[cfg(feature = "cuda")]
            return cuda_ops::reduce(input, op);
            #[cfg(not(feature = "cuda"))]
            Err(OpsError::CudaNotEnabled)
        }
    }
}

// ── Tensor convenience methods ────────────────────────────────────────────────

impl<const N: usize> Tensor<f32, N> {
    /// Apply an element-wise unary operation, writing results into `output`.
    /// Dispatches to GPU automatically when the tensor is `Device`-backed.
    pub fn apply_unary(&self, output: &mut Self, op: UnaryOp) -> Result<(), OpsError> {
        apply_unary(self, output, op)
    }

    /// Apply an element-wise binary operation with `other`, writing into `out`.
    pub fn apply_binary(&self, other: &Self, out: &mut Self, op: BinaryOp) -> Result<(), OpsError> {
        apply_binary(self, other, out, op)
    }

    /// Reduce all elements to a scalar.
    pub fn reduce(&self, op: ReduceOp) -> Result<f32, OpsError> {
        reduce(self, op)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    // ── CPU tests ─────────────────────────────────────────────────────────────

    #[test]
    fn cpu_unary_abs() {
        let t = Tensor::<f32, 1>::from_shape_vec([4], vec![-1.0, 2.0, -3.0, 0.0]).unwrap();
        let mut out = Tensor::<f32, 1>::zeros([4]);
        t.apply_unary(&mut out, UnaryOp::Abs).unwrap();
        assert_eq!(out.as_slice(), &[1.0, 2.0, 3.0, 0.0]);
    }

    #[test]
    fn cpu_unary_relu() {
        let t = Tensor::<f32, 1>::from_shape_vec([4], vec![-1.0, 0.5, -0.1, 2.0]).unwrap();
        let mut out = Tensor::<f32, 1>::zeros([4]);
        t.apply_unary(&mut out, UnaryOp::Relu).unwrap();
        assert_eq!(out.as_slice(), &[0.0, 0.5, 0.0, 2.0]);
    }

    #[test]
    fn cpu_unary_neg() {
        let t = Tensor::<f32, 1>::from_shape_vec([3], vec![1.0, -2.0, 0.0]).unwrap();
        let mut out = Tensor::<f32, 1>::zeros([3]);
        t.apply_unary(&mut out, UnaryOp::Neg).unwrap();
        assert_eq!(out.as_slice(), &[-1.0, 2.0, 0.0]);
    }

    #[test]
    fn cpu_unary_clamp() {
        let t = Tensor::<f32, 1>::from_shape_vec([4], vec![-2.0, 0.5, 1.5, 0.0]).unwrap();
        let mut out = Tensor::<f32, 1>::zeros([4]);
        t.apply_unary(&mut out, UnaryOp::Clamp { min: 0.0, max: 1.0 })
            .unwrap();
        assert_eq!(out.as_slice(), &[0.0, 0.5, 1.0, 0.0]);
    }

    #[test]
    fn cpu_binary_add() {
        let a = Tensor::<f32, 1>::from_shape_vec([3], vec![1.0, 2.0, 3.0]).unwrap();
        let b = Tensor::<f32, 1>::from_shape_vec([3], vec![4.0, 5.0, 6.0]).unwrap();
        let mut out = Tensor::<f32, 1>::zeros([3]);
        a.apply_binary(&b, &mut out, BinaryOp::Add).unwrap();
        assert_eq!(out.as_slice(), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn cpu_binary_mul() {
        let a = Tensor::<f32, 1>::from_shape_vec([3], vec![2.0, 3.0, 4.0]).unwrap();
        let b = Tensor::<f32, 1>::from_shape_vec([3], vec![1.0, 2.0, 3.0]).unwrap();
        let mut out = Tensor::<f32, 1>::zeros([3]);
        a.apply_binary(&b, &mut out, BinaryOp::Mul).unwrap();
        assert_eq!(out.as_slice(), &[2.0, 6.0, 12.0]);
    }

    #[test]
    fn cpu_reduce_sum() {
        let t = Tensor::<f32, 1>::from_shape_vec([4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let s = t.reduce(ReduceOp::Sum).unwrap();
        assert!((s - 10.0).abs() < 1e-6);
    }

    #[test]
    fn cpu_reduce_mean() {
        let t = Tensor::<f32, 1>::from_shape_vec([4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let m = t.reduce(ReduceOp::Mean).unwrap();
        assert!((m - 2.5).abs() < 1e-6);
    }

    #[test]
    fn cpu_shape_mismatch_is_error() {
        let a = Tensor::<f32, 1>::zeros([4]);
        let mut b = Tensor::<f32, 1>::zeros([5]);
        assert!(apply_unary(&a, &mut b, UnaryOp::Abs).is_err());
    }

    // ── CUDA parity tests (require the cuda feature) ──────────────────────────

    #[cfg(feature = "cuda")]
    mod cuda_parity {
        use std::sync::Arc;

        use cudarc::driver::CudaContext;

        use crate::cuda::zeros_cuda;

        use super::super::*;

        fn max_err(cpu: &[f32], gpu: &[f32]) -> f32 {
            cpu.iter()
                .zip(gpu)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max)
        }

        fn zeros_dev(
            n: usize,
            stream: &std::sync::Arc<cudarc::driver::CudaStream>,
        ) -> Tensor<f32, 1> {
            zeros_cuda::<f32, 1>([n], stream).unwrap()
        }

        #[test]
        fn parity_unary_all_ops() {
            let ctx = Arc::new(CudaContext::new(0).unwrap());
            let stream = ctx.default_stream();

            for n in [1024, 1_000_000] {
                let data: Vec<f32> = (0..n)
                    .map(|i| (i as f32 / (n - 1).max(1) as f32) * 2.0 - 1.0)
                    .collect();
                let cpu_in = Tensor::<f32, 1>::from_shape_vec([n], data).unwrap();
                let gpu_in = cpu_in.to_cuda(&stream).unwrap();

                for op in [
                    UnaryOp::Abs,
                    UnaryOp::Relu,
                    UnaryOp::Neg,
                    UnaryOp::Clamp {
                        min: -0.5,
                        max: 0.5,
                    },
                ] {
                    let mut cpu_out = Tensor::<f32, 1>::zeros([n]);
                    apply_unary(&cpu_in, &mut cpu_out, op).unwrap();

                    let mut gpu_out = zeros_dev(n, &stream);
                    apply_unary(&gpu_in, &mut gpu_out, op).unwrap();
                    stream.synchronize().unwrap();
                    let host_out = gpu_out.to_host(&stream).unwrap();

                    let err = max_err(cpu_out.as_slice(), host_out.as_slice());
                    assert!(err < 1e-6, "unary {op:?}  n={n}  max_err={err:.2e} > 1e-6");
                }
            }
        }

        #[test]
        fn parity_binary_all_ops() {
            let ctx = Arc::new(CudaContext::new(0).unwrap());
            let stream = ctx.default_stream();

            for n in [1024, 1_000_000] {
                let a_data: Vec<f32> = (0..n)
                    .map(|i| (i as f32 / (n - 1).max(1) as f32) * 2.0 - 0.5)
                    .collect();
                let b_data: Vec<f32> = (0..n).map(|i| 0.5 + (i as f32 / n as f32)).collect();

                let cpu_a = Tensor::<f32, 1>::from_shape_vec([n], a_data).unwrap();
                let cpu_b = Tensor::<f32, 1>::from_shape_vec([n], b_data).unwrap();
                let gpu_a = cpu_a.to_cuda(&stream).unwrap();
                let gpu_b = cpu_b.to_cuda(&stream).unwrap();

                for op in [
                    BinaryOp::Add,
                    BinaryOp::Sub,
                    BinaryOp::Mul,
                    BinaryOp::Div,
                    BinaryOp::Min,
                    BinaryOp::Max,
                ] {
                    let mut cpu_out = Tensor::<f32, 1>::zeros([n]);
                    apply_binary(&cpu_a, &cpu_b, &mut cpu_out, op).unwrap();

                    let mut gpu_out = zeros_dev(n, &stream);
                    apply_binary(&gpu_a, &gpu_b, &mut gpu_out, op).unwrap();
                    stream.synchronize().unwrap();
                    let host_out = gpu_out.to_host(&stream).unwrap();

                    let err = max_err(cpu_out.as_slice(), host_out.as_slice());
                    assert!(err < 1e-5, "binary {op:?}  n={n}  max_err={err:.2e} > 1e-5");
                }
            }
        }

        #[test]
        fn parity_reduce_sum_and_mean() {
            let ctx = Arc::new(CudaContext::new(0).unwrap());
            let stream = ctx.default_stream();

            for n in [1024, 1_000_000] {
                // All-positive data in [1, 2) avoids catastrophic cancellation.
                // GPU and CPU reductions accumulate in different orders, so the
                // achievable relative error on f32 for 1M elements is ~1e-3.
                let data: Vec<f32> = (0..n).map(|i| 1.0 + i as f32 / n as f32).collect();
                let cpu_t = Tensor::<f32, 1>::from_shape_vec([n], data).unwrap();
                let gpu_t = cpu_t.to_cuda(&stream).unwrap();

                for op in [ReduceOp::Sum, ReduceOp::Mean] {
                    let cpu_val = reduce(&cpu_t, op).unwrap();
                    let gpu_val = reduce(&gpu_t, op).unwrap();
                    let rel_err = (cpu_val - gpu_val).abs() / (cpu_val.abs() + 1e-8);
                    assert!(
                        rel_err < 1e-3,
                        "reduce {op:?}  n={n}  cpu={cpu_val:.6}  gpu={gpu_val:.6}  rel_err={rel_err:.2e}"
                    );
                }
            }
        }
    }
}
