/// Device type enumeration for tensor allocation.
///
/// Represents different compute devices where tensors can be allocated.
#[derive(Debug, Default, PartialEq)]
pub enum Device {
    /// CPU device
    #[default]
    Cpu,
    /// CUDA device with device ID
    #[cfg(feature = "cuda")]
    Cuda {
        /// The CUDA device ID
        device_id: usize
    },
    /// Metal device with device ID
    #[cfg(feature = "metal")]
    Metal {
        /// The Metal device ID
        device_id: usize
    },
    /// Vulkan device with device ID
    #[cfg(feature = "vulkan")]
    Vulkan {
        /// The Vulkan device ID
        device_id: usize
    },
}

impl AsRef<str> for Device {
    fn as_ref(&self) -> &str {
        match self {
            Device::Cpu => "cpu",
            #[cfg(feature = "cuda")]
            Device::Cuda { .. } => "cuda",
            #[cfg(feature = "metal")]
            Device::Metal { .. } => "metal",
            #[cfg(feature = "vulkan")]
            Device::Vulkan { .. } => "vulkan",
        }
    }
}

impl Device {
    /// Returns true if this device is CPU.
    #[inline]
    pub fn is_cpu(&self) -> bool {
        matches!(self, Device::Cpu)
    }

    /// Returns true if this device is CUDA.
    #[cfg(feature = "cuda")]
    #[inline]
    pub fn is_cuda(&self) -> bool {
        matches!(self, Device::Cuda { .. })
    }

    /// Returns true if this device is CUDA.
    #[cfg(not(feature = "cuda"))]
    #[inline]
    pub fn is_cuda(&self) -> bool {
        false
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_cpu() {
        let device = Device::Cpu;
        assert_eq!(device.as_ref(), "cpu");
        assert!(device.is_cpu());
        assert!(!device.is_cuda());
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_device_cuda() {
        let device = Device::Cuda { device_id: 0 };
        assert_eq!(device.as_ref(), "cuda");
        assert_eq!(format!("{:?}", device), "Cuda { device_id: 0 }");
        assert!(!device.is_cpu());
        assert!(device.is_cuda());
    }

    #[test]
    fn test_device_default() {
        let device = Device::default();
        assert_eq!(device, Device::Cpu);
        assert!(device.is_cpu());
        assert!(!device.is_cuda());
    }
}

