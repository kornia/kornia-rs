use kornia_image::{Image, ImageSize};
use kornia_tensor::{CpuAllocator, CudaAllocator, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    println!("=== CUDA Tensor Transfer Example ===\n");

    // Example 1: Simple tensor transfer
    example_tensor_transfer()?;

    // Example 2: Image transfer
    example_image_transfer()?;

    // Example 3: Round-trip transfer
    example_roundtrip_transfer()?;

    println!("\n=== All examples completed successfully! ===");
    Ok(())
}

fn example_tensor_transfer() -> Result<(), Box<dyn std::error::Error>> {
    println!("Example 1: Basic Tensor CPU ↔ CUDA Transfer");
    println!("--------------------------------------------");

    // Create a tensor on CPU
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let cpu_tensor = Tensor::<f32, 2, _>::from_shape_vec([2, 3], data, CpuAllocator)?;

    println!("Created CPU tensor with shape {:?}", cpu_tensor.shape);
    println!("Data on CPU: {:?}", cpu_tensor.as_slice());
    println!("Device: {}", cpu_tensor.device());

    // Transfer to CUDA
    let cuda_alloc = CudaAllocator::new(0)?;
    let cuda_tensor = cpu_tensor.to_device(cuda_alloc)?;
    
    println!("\nTransferred to CUDA");
    println!("Device: {}", cuda_tensor.device());
    println!("Is GPU: {}", cuda_tensor.is_gpu());

    // Transfer back to CPU
    let cpu_tensor_back = cuda_tensor.to_cpu()?;
    println!("\nTransferred back to CPU");
    println!("Data on CPU: {:?}", cpu_tensor_back.as_slice());
    println!("Device: {}", cpu_tensor_back.device());

    // Verify data integrity
    assert_eq!(cpu_tensor.as_slice(), cpu_tensor_back.as_slice());
    println!("\n✓ Data integrity verified!");

    Ok(())
}

fn example_image_transfer() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n\nExample 2: Image CPU ↔ CUDA Transfer");
    println!("-------------------------------------");

    // Create a small RGB image on CPU
    let size = ImageSize {
        width: 4,
        height: 4,
    };
    let data: Vec<u8> = (0..48).collect(); // 4x4x3 = 48 pixels
    let cpu_image = Image::<u8, 3, _>::new(size, data, CpuAllocator)?;

    println!("Created CPU image {}x{}", size.width, size.height);
    println!("Device: {}", cpu_image.device());
    println!("First few pixels: {:?}", &cpu_image.as_slice()[..9]);

    // Transfer to CUDA
    let cuda_alloc = CudaAllocator::new(0)?;
    let cuda_image = cpu_image.to_device(cuda_alloc)?;
    
    println!("\nTransferred image to CUDA");
    println!("Device: {}", cuda_image.device());

    // Transfer back to CPU
    let cpu_image_back = cuda_image.to_cpu()?;
    println!("\nTransferred back to CPU");
    println!("First few pixels: {:?}", &cpu_image_back.as_slice()[..9]);

    // Verify data integrity
    assert_eq!(cpu_image.as_slice(), cpu_image_back.as_slice());
    println!("\n✓ Image data integrity verified!");

    Ok(())
}

fn example_roundtrip_transfer() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n\nExample 3: Multiple Round-trip Transfers");
    println!("-----------------------------------------");

    // Create initial data
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let mut tensor = Tensor::<f32, 1, _>::from_shape_vec([4], data, CpuAllocator)?;

    println!("Initial data: {:?}", tensor.as_slice());

    let cuda_alloc = CudaAllocator::new(0)?;

    // Perform multiple round trips
    for i in 1..=3 {
        // CPU -> CUDA
        let cuda_tensor = tensor.to_device(cuda_alloc.clone())?;
        println!("\nRound-trip {}: Transferred to CUDA", i);
        
        // CUDA -> CPU
        tensor = cuda_tensor.to_cpu()?;
        println!("Round-trip {}: Transferred back to CPU", i);
        println!("Data: {:?}", tensor.as_slice());
    }

    // Verify final data matches original
    assert_eq!(tensor.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
    println!("\n✓ All round-trips successful with data integrity!");

    Ok(())
}

