//! Runtime client initialization helpers. Feature-gated per backend so the crate
//! still compiles on machines without CUDA when built with `--no-default-features --features cpu`.

#[cfg(feature = "cuda")]
pub use cubecl_cuda::CudaRuntime;
#[cfg(feature = "cpu")]
pub use cubecl_cpu::CpuRuntime;

use cubecl::Runtime;
use cubecl::client::ComputeClient;

/// Initialize a cubecl-cuda compute client.
///
/// Returns `Err` if no CUDA device is available. cubecl-cuda 0.10-pre currently
/// panics on missing devices, so we wrap the call in `catch_unwind` to give the
/// bench/test a graceful skip path. Revisit if upstream exposes a fallible ctor.
#[cfg(feature = "cuda")]
pub fn init_cuda() -> Result<ComputeClient<CudaRuntime>, String> {
    std::panic::catch_unwind(|| {
        let device = <CudaRuntime as Runtime>::Device::default();
        CudaRuntime::client(&device)
    })
    .map_err(|_| "no CUDA device available".to_string())
}

/// Initialize a cubecl-cpu compute client.
#[cfg(feature = "cpu")]
pub fn init_cpu() -> ComputeClient<CpuRuntime> {
    let device = <CpuRuntime as Runtime>::Device::default();
    CpuRuntime::client(&device)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "cpu")]
    #[test]
    fn cpu_client_initializes() {
        let _client = init_cpu();
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_client_initializes_or_skips() {
        match init_cuda() {
            Ok(_) => eprintln!("cuda client OK"),
            Err(msg) => eprintln!("skipping: {msg}"),
        }
    }
}
