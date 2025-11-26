//! Device marker types for compile-time device dispatch.
//!
//! This module provides zero-sized types that enable the compiler to generate
//! optimized, device-specific code paths without runtime overhead.
//!
//! # Design
//!
//! The device is encoded in the type system, allowing the compiler to:
//! - Generate specialized code for each device via monomorphization
//! - Prevent mixing tensors from different devices at compile-time
//! - Eliminate runtime device checks in hot paths
//!
//! # Examples
//!
//! ```
//! use kornia_tensor::{Tensor2, Cpu, DeviceMarker, TensorError};
//!
//! // Generic function works with any device
//! fn scale<D: DeviceMarker>(tensor: &Tensor2<f32, D>, factor: f32) -> Result<Tensor2<f32, D>, TensorError>
//! where
//!     D: DeviceMarker,
//! {
//!     // Compiler generates device-specific code
//!     tensor.map(|&x| x * factor)
//! }
//!
//! let cpu_tensor = Tensor2::<f32, Cpu>::zeros([10, 10]).unwrap();
//! let scaled = scale(&cpu_tensor, 2.0).unwrap();
//! ```

use crate::{allocator::TensorAllocator, device::Device, TensorError};

/// Marker trait for device types.
///
/// This trait is sealed and can only be implemented by built-in device types
/// (`Cpu` and `Cuda<ID>`). It enables zero-cost dispatch by making the device
/// part of the type system.
///
/// # Type Safety
///
/// The device marker prevents mixing tensors from different devices at compile time,
/// ensuring type safety across CPU and GPU operations.
///
/// # Examples
///
/// ```
/// use kornia_tensor::{Tensor2, Cpu, DeviceMarker, TensorError};
///
/// fn process<D: DeviceMarker>(tensor: &Tensor2<f32, D>) -> Result<Tensor2<f32, D>, TensorError> {
///     // Compiler generates device-specific code
///     tensor.map(|&x| x * 2.0)
/// }
///
/// let tensor = Tensor2::<f32, Cpu>::zeros([5, 5]).unwrap();
/// let result = process(&tensor).unwrap();
/// ```
pub trait DeviceMarker: private::Sealed + Clone + Send + Sync + 'static {
    /// The allocator type for this device.
    type Allocator: TensorAllocator;
    
    /// Returns an allocator for this device.
    ///
    /// # Errors
    ///
    /// Returns an error if the device is not available or initialization fails.
    /// For CPU, this always succeeds. For CUDA, this may fail if:
    /// - The device ID is invalid
    /// - CUDA drivers are not installed
    /// - The device is already in use
    fn allocator() -> Result<Self::Allocator, TensorError>;
    
    /// Returns device information.
    ///
    /// This is used internally for debugging and device queries.
    fn device_info() -> Device;
}

mod private {
    /// Seals the DeviceMarker trait to prevent external implementations.
    ///
    /// This ensures only built-in device types can implement `DeviceMarker`,
    /// maintaining the library's safety and performance guarantees.
    pub trait Sealed {}
    
    impl Sealed for super::Cpu {}
    
    #[cfg(feature = "cuda")]
    impl<const ID: usize> Sealed for super::Cuda<ID> {}
}

/// Marker trait for CPU-only operations.
///
/// This trait is implemented only for `Cpu`, enabling compile-time type safety
/// for operations that are only valid on CPU devices (e.g., `as_slice()`).
///
/// # Examples
///
/// ```
/// use kornia_tensor::{Tensor2, Cpu, CpuDevice};
///
/// fn process_cpu<D: CpuDevice>(tensor: &Tensor2<f32, D>) -> &[f32] {
///     // This only compiles for CPU tensors
///     tensor.as_slice()
/// }
///
/// let tensor = Tensor2::<f32, Cpu>::zeros([10, 10]).unwrap();
/// let slice = process_cpu(&tensor);
/// ```
pub trait CpuDevice: DeviceMarker {}

/// Zero-sized type representing CPU device.
///
/// This is the default device for tensor operations. CPU tensors can be
/// accessed directly via `as_slice()` and `as_slice_mut()`.
///
/// # Examples
///
/// ```
/// use kornia_tensor::{Tensor2, Cpu};
///
/// // Explicit CPU device
/// let tensor = Tensor2::<f32, Cpu>::zeros([100, 100]).unwrap();
/// assert!(tensor.is_cpu());
///
/// // Default device is also CPU
/// let tensor2 = Tensor2::<f32, Cpu>::zeros([100, 100]).unwrap();
/// assert!(tensor2.is_cpu());
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Cpu;

impl CpuDevice for Cpu {}

impl DeviceMarker for Cpu {
    type Allocator = crate::CpuAllocator;
    
    fn allocator() -> Result<Self::Allocator, TensorError> {
        Ok(crate::CpuAllocator)
    }
    
    fn device_info() -> Device {
        Device::Cpu
    }
}

/// Zero-sized type representing CUDA device with compile-time device ID.
///
/// The device ID is encoded in the type, enabling the compiler to generate
/// device-specific code for multi-GPU systems. This ensures tensors from
/// different GPUs cannot be accidentally mixed.
///
/// # Examples
///
/// ```no_run
/// # #[cfg(feature = "cuda")]
/// # {
/// use kornia_tensor::{Tensor2, Cuda};
///
/// // Tensor on GPU 0
/// let tensor0 = Tensor2::<f32, Cuda<0>>::zeros([100, 100])?;
/// assert!(tensor0.is_gpu());
///
/// // Tensor on GPU 1
/// let tensor1 = Tensor2::<f32, Cuda<1>>::zeros([100, 100])?;
///
/// // Type system prevents mixing devices
/// // This would be a compile error:
/// // let mixed = tensor0.add(&tensor1);
/// # }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Device Transfers
///
/// Use `to_device()` with type annotations to transfer between devices:
///
/// ```no_run
/// # #[cfg(feature = "cuda")]
/// # {
/// use kornia_tensor::{Tensor2, Cpu, Cuda};
///
/// let cpu_tensor = Tensor2::<f32, Cpu>::zeros([10, 10])?;
/// let gpu_tensor: Tensor2<f32, Cuda<0>> = cpu_tensor.to_device()?;
/// let back_to_cpu: Tensor2<f32, Cpu> = gpu_tensor.to_cpu()?;
/// # }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[cfg(feature = "cuda")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Cuda<const DEVICE_ID: usize = 0>;

#[cfg(feature = "cuda")]
impl<const ID: usize> DeviceMarker for Cuda<ID> {
    type Allocator = crate::CudaAllocator;
    
    fn allocator() -> Result<Self::Allocator, TensorError> {
        crate::CudaAllocator::new(ID).map_err(TensorError::StorageError)
    }
    
    fn device_info() -> Device {
        Device::Cuda { device_id: ID }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_device_marker() {
        let alloc = Cpu::allocator();
        assert!(alloc.is_ok());
        
        let device = Cpu::device_info();
        assert_eq!(device, Device::Cpu);
        assert!(device.is_cpu());
    }

    #[test]
    fn test_cpu_traits() {
        // Test that Cpu implements required traits
        let cpu1 = Cpu;
        let cpu2 = cpu1; // Copy
        let cpu3 = cpu1.clone(); // Clone
        assert_eq!(cpu1, cpu2); // PartialEq
        assert_eq!(cpu2, cpu3); // Eq
        
        // Debug
        let debug_str = format!("{:?}", cpu1);
        assert_eq!(debug_str, "Cpu");
        
        // Default
        let default_cpu = Cpu::default();
        assert_eq!(default_cpu, Cpu);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_device_marker() {
        // Test CUDA device marker (may fail if CUDA not available)
        let device = Cuda::<0>::device_info();
        match device {
            Device::Cuda { device_id } => assert_eq!(device_id, 0),
            _ => panic!("Expected CUDA device"),
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_traits() {
        let cuda1 = Cuda::<0>;
        let cuda2 = cuda1; // Copy
        let cuda3 = cuda1.clone(); // Clone
        assert_eq!(cuda1, cuda2); // PartialEq
        assert_eq!(cuda2, cuda3); // Eq
        
        // Debug
        let debug_str = format!("{:?}", cuda1);
        assert!(debug_str.contains("Cuda"));
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_different_cuda_devices_not_equal() {
        let _cuda0 = Cuda::<0>;
        let _cuda1 = Cuda::<1>;
        // These are different types, so this comparison wouldn't compile
        // assert_ne!(cuda0, cuda1);
        
        // But we can verify their device info differs
        let info0 = Cuda::<0>::device_info();
        let info1 = Cuda::<1>::device_info();
        assert_ne!(info0, info1);
    }
}

