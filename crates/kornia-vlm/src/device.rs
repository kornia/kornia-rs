use candle_core::{Device, DType};

#[cfg(feature = "cuda")]
fn cuda_supports_bf16(device_id: usize) -> bool {
    use cudarc::driver::{sys::CUdevice_attribute, CudaDevice};

    if let Ok(dev) = CudaDevice::new(device_id) {
        if let Ok(major) =
            dev.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        {
            return major >= 8;
        }
    }
    false
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
                return (device, dtype);
            }
            Err(e) => {
                log::warn!("CUDA not available, defaulting to CPU: {e:?}");
            }
        }
    }

    // CPU fallback (covers both: no CUDA feature OR CUDA not available)
    (Device::Cpu, DType::F32)
}