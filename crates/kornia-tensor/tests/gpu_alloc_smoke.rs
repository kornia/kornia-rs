#![cfg(feature = "gpu-cubecl")]

use std::alloc::Layout;

use kornia_tensor::allocator::TensorAllocator;
use kornia_tensor::backend::cubecl::new_cuda_allocator;
use kornia_tensor::resource::MemoryDomain;

/// Run with: cargo test -p kornia-tensor --features gpu-cubecl -- --ignored
#[test]
#[ignore = "requires a CUDA device"]
fn alloc_dealloc_roundtrip_cuda() {
    let allocator = new_cuda_allocator();
    let layout = Layout::from_size_align(1024, 8).expect("valid layout");
    let resource = allocator.allocate(layout).expect("allocate");
    assert_eq!(resource.len_bytes(), 1024);
    assert!(
        matches!(resource.domain(), MemoryDomain::Device { .. }),
        "expected Device domain, got {:?}",
        resource.domain()
    );
    // resource drops here → GpuResource::drop → Backend::dealloc
}

#[test]
#[ignore = "requires a CUDA device"]
fn zst_alloc_is_noop() {
    let allocator = new_cuda_allocator();
    let layout = Layout::from_size_align(0, 8).expect("valid layout");
    let resource = allocator.allocate(layout).expect("zst allocate");
    assert_eq!(resource.len_bytes(), 0);
    // resource drops here — no dealloc call (size == 0)
}
