#![deny(missing_docs)]
#![doc = env!("CARGO_PKG_DESCRIPTION")]

/// allocator module containing the memory management utilities.
pub mod allocator;

/// bincode module containing the serialization and deserialization utilities.
#[cfg(feature = "bincode")]
pub mod bincode;

/// tensor module containing the tensor and storage implementations.
pub mod tensor;

/// serde module containing the serialization and deserialization utilities.
#[cfg(feature = "serde")]
pub mod serde;

/// storage module containing the storage implementations.
pub mod storage;

/// view module containing the view implementations.
pub mod view;

/// parent deallocator module for tensor
pub mod parent_deallocator;

pub use crate::allocator::{CpuAllocator, TensorAllocator};
pub(crate) use crate::tensor::get_strides_from_shape;
pub use crate::tensor::{Tensor, TensorError};
pub use parent_deallocator::ParentDeallocator;

/// Type alias for a 1-dimensional tensor.
pub type Tensor1<T, A> = Tensor<T, 1, A>;

/// Type alias for a 2-dimensional tensor.
pub type Tensor2<T, A> = Tensor<T, 2, A>;

/// Type alias for a 3-dimensional tensor.
pub type Tensor3<T, A> = Tensor<T, 3, A>;

/// Type alias for a 4-dimensional tensor.
pub type Tensor4<T, A> = Tensor<T, 4, A>;

/// Type alias for a 2-dimensional tensor with CPU allocator.
pub type CpuTensor2<T> = Tensor2<T, CpuAllocator>;
