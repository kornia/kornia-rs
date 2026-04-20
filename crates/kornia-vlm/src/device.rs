use candle_core::{DType, Device};
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
                return (device, dtype); // ← explicit return so the CPU fallback below applies on Err
            }
            Err(e) => {
                log::warn!("CUDA not available, defaulting to CPU: {:?}", e);
                // falls through to CPU fallback below
            }
        }
    }

    // CPU fallback (also used when cuda feature is disabled)
    log::info!("Using CPU with F32");
    (Device::Cpu, DType::F32)
}
