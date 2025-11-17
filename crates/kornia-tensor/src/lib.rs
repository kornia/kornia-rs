#![deny(missing_docs)]
#![doc = env!("CARGO_PKG_DESCRIPTION")]

/// allocator module containing the memory management utilities.
pub mod allocator;

/// backend module containing device operation abstractions.
pub mod backend;

/// bincode module containing the serialization and deserialization utilities.
#[cfg(feature = "bincode")]
pub mod bincode;

/// device module containing device abstraction.
pub mod device;

/// device_marker module containing zero-cost device type markers.
pub mod device_marker;

/// serde module containing the serialization and deserialization utilities.
#[cfg(feature = "serde")]
pub mod serde;

/// storage module containing the storage implementations.
pub mod storage;

/// tensor module containing the tensor and storage implementations.
pub mod tensor;

/// view module containing the view implementations.
pub mod view;

pub use crate::allocator::{CpuAllocator, TensorAllocator};
#[cfg(feature = "cuda")]
pub use crate::allocator::CudaAllocator;
pub use crate::backend::{Backend, CpuBackend};
#[cfg(feature = "cuda")]
pub use crate::backend::CudaBackend;
pub use crate::device::Device;
pub use crate::device_marker::{Cpu, DeviceMarker};
#[cfg(feature = "cuda")]
pub use crate::device_marker::Cuda;
pub(crate) use crate::tensor::get_strides_from_shape;
pub use crate::tensor::{Tensor, TensorError};

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
