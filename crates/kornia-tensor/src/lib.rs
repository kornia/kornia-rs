#![deny(missing_docs)]
#![doc = env!("CARGO_PKG_DESCRIPTION")]

/// Memory allocation traits and implementations.
///
/// Defines the [`TensorAllocator`] trait and provides [`CpuAllocator`] for
/// heap-allocated tensors. Custom allocators can be implemented for GPU or
/// shared memory backends.
pub mod allocator;

/// Bincode serialization support (feature-gated).
///
/// Serialize and deserialize tensors using the bincode format.
/// Requires the `bincode` feature flag.
#[cfg(feature = "bincode")]
pub mod bincode;

/// Serde serialization support (feature-gated).
///
/// Serialize and deserialize tensors using serde-compatible formats.
/// Requires the `serde` feature flag.
#[cfg(feature = "serde")]
pub mod serde;

/// Tensor storage implementations.
///
/// Internal storage layer wrapping Arrow buffers for efficient, thread-safe
/// data management with zero-copy sharing capabilities.
pub mod storage;

/// Core tensor types and operations.
///
/// Defines the [`Tensor`] struct and fundamental tensor operations.
/// Tensors are multi-dimensional arrays with compile-time dimension checking.
pub mod tensor;

/// Tensor views and slicing.
///
/// Provides non-owning views into tensor data for efficient sub-tensor access
/// without copying.
pub mod view;

pub use crate::allocator::{CpuAllocator, TensorAllocator};
pub(crate) use crate::tensor::get_strides_from_shape;
pub use crate::tensor::{Tensor, TensorError};

/// Type alias for a 1-dimensional tensor (vector).
///
/// Useful for representing 1D data such as time series or feature vectors.
pub type Tensor1<T, A> = Tensor<T, 1, A>;

/// Type alias for a 2-dimensional tensor (matrix).
///
/// Commonly used for matrices, grayscale images (H×W), or batches of vectors.
pub type Tensor2<T, A> = Tensor<T, 2, A>;

/// Type alias for a 3-dimensional tensor.
///
/// Commonly used for RGB images (H×W×C) or batches of grayscale images (N×H×W).
pub type Tensor3<T, A> = Tensor<T, 3, A>;

/// Type alias for a 4-dimensional tensor.
///
/// Commonly used for batches of RGB images (N×H×W×C) or video data (N×T×H×W).
pub type Tensor4<T, A> = Tensor<T, 4, A>;

/// Convenience type for 2D tensors allocated on the CPU.
///
/// Equivalent to `Tensor2<T, CpuAllocator>`.
pub type CpuTensor2<T> = Tensor2<T, CpuAllocator>;
