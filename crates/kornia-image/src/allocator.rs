use kornia_tensor::TensorAllocator;

pub use kornia_tensor::CpuAllocator;

/// A marker trait for allocating and deallocating memory for images.
pub trait ImageAllocator: TensorAllocator {}

impl ImageAllocator for CpuAllocator {}
