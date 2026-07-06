use candle_core::{DType, Device};

// helper function to know if the gpu supports bf16
#[cfg(feature = "cuda")]
fn cuda_supports_bf16(ordinal: usize) -> bool {
    use cudarc::driver::safe::CudaContext;

    match CudaContext::new(ordinal) {
        Ok(ctx) => {
            let major = ctx
                .attribute(
                    cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                )
                .unwrap_or(0);

            major >= 8
        }
        Err(_) => false,
    }
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
                        "GPU does not support BF16. Using FP16 instead; numerical stability may be slightly reduced on some models."
                    );
                    DType::F16
                };
                return (device, dtype); // ← explicit return so the CPU fallback below applies on Err
            }
            Err(e) => {
                log::warn!("CUDA not available: {:?}", e);
                // falls through to CPU fallback below
            }
        }
    }
    // MPS path (Apple Silicon)
    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        match Device::new_metal(0) {
            Ok(device) => {
                log::info!("Using Apple Metal (MPS) backend with F16");
                return (device, DType::F16);
            }
            Err(e) => {
                log::warn!("Metal unavailable: {:?}", e);
            }
        }
    }

    // CPU fallback (also used when cuda feature is disabled)
    log::info!("Using CPU with F32");
    (Device::Cpu, DType::F32)
}
