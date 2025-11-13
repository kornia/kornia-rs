// Re-export allocators from kornia-tensor
pub use kornia_tensor::CpuAllocator;
#[cfg(feature = "cuda")]
pub use kornia_tensor::CudaAllocator;
use kornia_tensor::TensorAllocator;

/// A marker trait for allocating and deallocating memory for images.
pub trait ImageAllocator: TensorAllocator {}

impl ImageAllocator for CpuAllocator {}

#[cfg(feature = "cuda")]
impl ImageAllocator for CudaAllocator {}
