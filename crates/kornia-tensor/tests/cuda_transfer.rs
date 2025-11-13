#![cfg(feature = "cuda")]

use kornia_tensor::{CpuAllocator, CudaAllocator, Tensor};

#[test]
fn test_cuda_allocator_creation() -> Result<(), Box<dyn std::error::Error>> {
    let allocator = CudaAllocator::new(0)?;
    assert_eq!(allocator.device_id(), 0);
    Ok(())
}

#[test]
fn test_cpu_to_cuda_transfer() -> Result<(), Box<dyn std::error::Error>> {
    // Create tensor on CPU
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let cpu_tensor = Tensor::<f32, 1, _>::from_shape_vec([4], data, CpuAllocator)?;

    // Transfer to CUDA
    let cuda_alloc = CudaAllocator::new(0)?;
    let cuda_tensor = cpu_tensor.to_device(cuda_alloc)?;

    // Verify device
    assert!(cuda_tensor.is_gpu());
    assert!(!cuda_tensor.is_cpu());

    Ok(())
}

#[test]
fn test_cuda_to_cpu_transfer() -> Result<(), Box<dyn std::error::Error>> {
    // Create tensor on CPU
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let cpu_tensor = Tensor::<f32, 1, _>::from_shape_vec([4], data.clone(), CpuAllocator)?;

    // Transfer to CUDA
    let cuda_alloc = CudaAllocator::new(0)?;
    let cuda_tensor = cpu_tensor.to_device(cuda_alloc)?;

    // Transfer back to CPU
    let cpu_tensor_back = cuda_tensor.to_cpu()?;

    // Verify data integrity
    assert_eq!(cpu_tensor_back.as_slice(), data.as_slice());

    Ok(())
}

#[test]
fn test_roundtrip_transfer() -> Result<(), Box<dyn std::error::Error>> {
    let original_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let cpu_tensor = Tensor::<f32, 2, _>::from_shape_vec(
        [2, 3],
        original_data.clone(),
        CpuAllocator,
    )?;

    let cuda_alloc = CudaAllocator::new(0)?;

    // Multiple round trips
    for _ in 0..5 {
        let cuda_tensor = cpu_tensor.to_device(cuda_alloc.clone())?;
        let cpu_tensor_back = cuda_tensor.to_cpu()?;
        assert_eq!(cpu_tensor_back.as_slice(), original_data.as_slice());
    }

    Ok(())
}

#[test]
fn test_different_dtypes() -> Result<(), Box<dyn std::error::Error>> {
    let cuda_alloc = CudaAllocator::new(0)?;

    // Test u8
    {
        let data = vec![1u8, 2, 3, 4];
        let cpu_tensor = Tensor::<u8, 1, _>::from_shape_vec([4], data.clone(), CpuAllocator)?;
        let cuda_tensor = cpu_tensor.to_device(cuda_alloc.clone())?;
        let cpu_back = cuda_tensor.to_cpu()?;
        assert_eq!(cpu_back.as_slice(), data.as_slice());
    }

    // Test i32
    {
        let data = vec![-1i32, 0, 1, 2];
        let cpu_tensor = Tensor::<i32, 1, _>::from_shape_vec([4], data.clone(), CpuAllocator)?;
        let cuda_tensor = cpu_tensor.to_device(cuda_alloc.clone())?;
        let cpu_back = cuda_tensor.to_cpu()?;
        assert_eq!(cpu_back.as_slice(), data.as_slice());
    }

    // Test f64
    {
        let data = vec![1.5f64, 2.5, 3.5, 4.5];
        let cpu_tensor = Tensor::<f64, 1, _>::from_shape_vec([4], data.clone(), CpuAllocator)?;
        let cuda_tensor = cpu_tensor.to_device(cuda_alloc.clone())?;
        let cpu_back = cuda_tensor.to_cpu()?;
        assert_eq!(cpu_back.as_slice(), data.as_slice());
    }

    Ok(())
}

#[test]
fn test_multidimensional_transfer() -> Result<(), Box<dyn std::error::Error>> {
    let cuda_alloc = CudaAllocator::new(0)?;

    // 3D tensor
    let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
    let cpu_tensor = Tensor::<f32, 3, _>::from_shape_vec([2, 3, 4], data.clone(), CpuAllocator)?;

    let cuda_tensor = cpu_tensor.to_device(cuda_alloc)?;
    let cpu_back = cuda_tensor.to_cpu()?;

    assert_eq!(cpu_back.as_slice(), data.as_slice());
    assert_eq!(cpu_back.shape, [2, 3, 4]);

    Ok(())
}

#[test]
#[should_panic(expected = "Cannot access GPU tensor as slice")]
fn test_cannot_access_gpu_tensor_as_slice() {
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let cpu_tensor = Tensor::<f32, 1, _>::from_shape_vec([4], data, CpuAllocator).unwrap();

    let cuda_alloc = CudaAllocator::new(0).unwrap();
    let cuda_tensor = cpu_tensor.to_device(cuda_alloc).unwrap();

    // This should panic
    let _ = cuda_tensor.as_slice();
}

#[test]
fn test_large_transfer() -> Result<(), Box<dyn std::error::Error>> {
    // Test with a larger tensor (1MB of f32 data)
    let size = 256 * 1024; // 256K elements = 1MB
    let data: Vec<f32> = (0..size).map(|x| x as f32).collect();
    let cpu_tensor = Tensor::<f32, 1, _>::from_shape_vec([size], data.clone(), CpuAllocator)?;

    let cuda_alloc = CudaAllocator::new(0)?;
    let cuda_tensor = cpu_tensor.to_device(cuda_alloc)?;
    let cpu_back = cuda_tensor.to_cpu()?;

    // Verify a subset of the data
    assert_eq!(cpu_back.as_slice()[0], 0.0);
    assert_eq!(cpu_back.as_slice()[size - 1], (size - 1) as f32);
    assert_eq!(cpu_back.as_slice().len(), size);

    Ok(())
}

