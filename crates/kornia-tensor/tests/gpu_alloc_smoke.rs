#![cfg(feature = "gpu-cubecl")]

use std::alloc::Layout;

use cubecl::Runtime;
use cubecl_cuda::CudaRuntime;
use kornia_tensor::allocator::TensorAllocator;
use kornia_tensor::backend::GpuAllocator;
use kornia_tensor::backend::cubecl::CubeclBackend;

#[test]
fn alloc_dealloc_roundtrip_cuda() {
    let device = <CudaRuntime as Runtime>::Device::default();
    let client = CudaRuntime::client(&device);
    let backend = CubeclBackend::<CudaRuntime>::new(client);
    let allocator = GpuAllocator::new(backend);

    let layout = Layout::from_size_align(1024, 8).expect("valid layout");
    let ptr = allocator.alloc(layout).expect("alloc");
    assert!(!ptr.is_null(), "device pointer must not be null");
    allocator.dealloc(ptr, layout);
}
