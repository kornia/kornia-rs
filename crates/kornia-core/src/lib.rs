#![deny(missing_docs)]
//! Core module containing the tensor and storage implementations.

/// allocator module containing the memory management utilities.
pub mod allocator;

/// tensor module containing the tensor and storage implementations.
pub mod tensor;

/// serde module containing the serialization and deserialization utilities.
pub mod serde;

/// storage module containing the storage implementations.
pub mod storage;

/// view module containing the view implementations.
pub mod view;

pub use crate::allocator::{CpuAllocator, TensorAllocator};
//pub use crate::storage::SafeTensorType;
pub(crate) use crate::tensor::get_strides_from_shape;
pub use crate::tensor::{Tensor, TensorError};

/// Type alias for a 1-dimensional tensor.
pub type Tensor1<T> = Tensor<T, 1>;

/// Type alias for a 2-dimensional tensor.
pub type Tensor2<T> = Tensor<T, 2>;

/// Type alias for a 3-dimensional tensor.
pub type Tensor3<T> = Tensor<T, 3>;

/// Type alias for a 4-dimensional tensor.
pub type Tensor4<T> = Tensor<T, 4>;
