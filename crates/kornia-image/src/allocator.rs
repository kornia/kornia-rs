// Re-export the CpuAllocator from kornia-tensor
use kornia_tensor::allocator::TensorAllocatorError;
pub use kornia_tensor::CpuAllocator;
use kornia_tensor::TensorAllocator;

/// A marker trait for allocating and deallocating memory for images.
pub trait ImageAllocator: TensorAllocator {}

impl ImageAllocator for CpuAllocator {}

/// A no-op allocator for wrapping foreign-owned memory as Rust Image objects.
///
/// This allocator is used when Rust code needs to view memory owned by an
/// external system (numpy, CUDA, mmap, etc.) without taking ownership.
/// `alloc` panics because this allocator is only meant to wrap foreign memory;
/// new allocations should never be requested through it. `dealloc` is a no-op
/// because the foreign system retains ownership of the underlying buffer.
#[derive(Clone)]
pub struct ForeignAllocator;

impl TensorAllocator for ForeignAllocator {
    fn alloc(&self, _layout: std::alloc::Layout) -> Result<*mut u8, TensorAllocatorError> {
        panic!("ForeignAllocator: alloc called on foreign-memory allocator");
    }

    fn dealloc(&self, _ptr: *mut u8, _layout: std::alloc::Layout) {
        // no-op: foreign system owns the memory
    }
}

impl ImageAllocator for ForeignAllocator {}
