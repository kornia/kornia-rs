/// Device type enumeration for tensor allocation.
///
/// Represents different compute devices where tensors can be allocated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Device {
    /// CPU device
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

impl Device {
    /// Returns the device type as a string.
    pub fn device_type(&self) -> &str {
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

    /// Returns the device ID if applicable.
    pub fn device_id(&self) -> Option<usize> {
        match self {
            Device::Cpu => None,
            #[cfg(feature = "cuda")]
            Device::Cuda { device_id } => Some(*device_id),
            #[cfg(feature = "metal")]
            Device::Metal { device_id } => Some(*device_id),
            #[cfg(feature = "vulkan")]
            Device::Vulkan { device_id } => Some(*device_id),
        }
    }

    /// Returns true if the device is CPU.
    pub fn is_cpu(&self) -> bool {
        matches!(self, Device::Cpu)
    }

    /// Returns true if the device is a GPU.
    pub fn is_gpu(&self) -> bool {
        !self.is_cpu()
    }

    /// Creates a CUDA device with the specified device ID.
    #[cfg(feature = "cuda")]
    pub fn cuda(device_id: usize) -> Self {
        Device::Cuda { device_id }
    }

    /// Creates a Metal device with the specified device ID.
    #[cfg(feature = "metal")]
    pub fn metal(device_id: usize) -> Self {
        Device::Metal { device_id }
    }

    /// Creates a Vulkan device with the specified device ID.
    #[cfg(feature = "vulkan")]
    pub fn vulkan(device_id: usize) -> Self {
        Device::Vulkan { device_id }
    }
}

impl Default for Device {
    fn default() -> Self {
        Device::Cpu
    }
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Device::Cpu => write!(f, "cpu"),
            #[cfg(feature = "cuda")]
            Device::Cuda { device_id } => write!(f, "cuda:{}", device_id),
            #[cfg(feature = "metal")]
            Device::Metal { device_id } => write!(f, "metal:{}", device_id),
            #[cfg(feature = "vulkan")]
            Device::Vulkan { device_id } => write!(f, "vulkan:{}", device_id),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_cpu() {
        let device = Device::Cpu;
        assert_eq!(device.device_type(), "cpu");
        assert_eq!(device.device_id(), None);
        assert!(device.is_cpu());
        assert!(!device.is_gpu());
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_device_cuda() {
        let device = Device::cuda(0);
        assert_eq!(device.device_type(), "cuda");
        assert_eq!(device.device_id(), Some(0));
        assert!(!device.is_cpu());
        assert!(device.is_gpu());
        assert_eq!(format!("{}", device), "cuda:0");
    }

    #[test]
    fn test_device_default() {
        let device = Device::default();
        assert_eq!(device, Device::Cpu);
    }
}

