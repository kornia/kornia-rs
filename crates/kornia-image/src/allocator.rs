// Re-export the CpuAllocator from kornia-tensor
use kornia_tensor::allocator::TensorAllocatorError;
pub use kornia_tensor::CpuAllocator;
use kornia_tensor::TensorAllocator;

/// A marker trait for allocating and deallocating memory for images.
pub trait ImageAllocator: TensorAllocator {}

impl ImageAllocator for CpuAllocator {}

/// A no-op allocator for wrapping foreign-owned memory (numpy, CUDA, mmap, etc.)
/// as Rust Image objects without taking ownership.
#[derive(Clone)]
pub struct ForeignAllocator;

impl TensorAllocator for ForeignAllocator {
    fn alloc(&self, _layout: std::alloc::Layout) -> Result<*mut u8, TensorAllocatorError> {
        panic!("ForeignAllocator: alloc called on foreign-memory allocator");
    }

    fn dealloc(&self, _ptr: *mut u8, _layout: std::alloc::Layout) {}
}

impl ImageAllocator for ForeignAllocator {}
