// Re-export the CpuAllocator and ForeignAllocator from kornia-tensor
pub use kornia_tensor::allocator::ForeignAllocator;
pub use kornia_tensor::CpuAllocator;
use kornia_tensor::TensorAllocator;

/// A marker trait for allocating and deallocating memory for images.
pub trait ImageAllocator: TensorAllocator {}

impl ImageAllocator for CpuAllocator {}

impl ImageAllocator for ForeignAllocator {}
