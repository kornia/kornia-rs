#![cfg(feature = "gpu-cubecl")]

use std::alloc::Layout;

use kornia_tensor::allocator::TensorAllocator;
use kornia_tensor::backend::cubecl::new_cuda_allocator;

/// Run with: cargo test -p kornia-tensor --features gpu-cubecl -- --ignored
#[test]
#[ignore = "requires a CUDA device"]
fn alloc_dealloc_roundtrip_cuda() {
    let allocator = new_cuda_allocator();
    let layout = Layout::from_size_align(1024, 8).expect("valid layout");
    let ptr = allocator.alloc(layout).expect("alloc");
    assert!(!ptr.is_null(), "device pointer must not be null");
    allocator.dealloc(ptr, layout);
}

#[test]
#[ignore = "requires a CUDA device"]
fn zst_alloc_is_noop() {
    let allocator = new_cuda_allocator();
    let layout = Layout::from_size_align(0, 8).expect("valid layout");
    let ptr = allocator.alloc(layout).expect("zst alloc");
    assert!(!ptr.is_null(), "zst device pointer must not be null");
    allocator.dealloc(ptr, layout);
}
