use candle_core::{DType, Device};

#[cfg(feature = "cuda")]
fn cuda_supports_bf16(device_id: usize) -> bool {
    use cudarc::driver::{sys::CUdevice_attribute, CudaDevice};

    CudaDevice::new(device_id)
        .and_then(|dev| {
            dev.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        })
        .map(|major| major >= 8)
        .unwrap_or(false)
}

pub fn get_device_and_dtype() -> (Device, DType) {
    #[cfg(feature = "cuda")]
    {
        match Device::cuda_if_available(0) {
            Ok(device) => {
                let dtype = if cuda_supports_bf16(0) {
                    log::info!("GPU supports BF16, using BF16");
                    DType::BF16
                } else {
                    log::warn!(
                        "GPU does not support BF16, falling back to FP16. \
                        Note: FP16 may overflow in attention logits and produce lower-quality results."
                    );
                    DType::F16
                };
                (device, dtype)
            }
            Err(e) => {
                log::warn!("CUDA not available, defaulting to CPU: {:?}", e);
            }
        }
    }

    (Device::Cpu, DType::F32)
}
