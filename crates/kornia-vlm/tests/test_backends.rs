use kornia_vlm::backends::VlmBackend;

#[test]
fn test_default_is_candle() {
    let b = VlmBackend::default();
    assert_eq!(b, VlmBackend::Candle);
    assert_eq!(b.name(), "candle");
    assert!(!b.is_onnx());
}

#[test]
#[cfg(feature = "onnx")]
fn test_onnx_cpu() {
    let b = VlmBackend::OnnxCpu;
    assert_eq!(b.name(), "onnxruntime-cpu");
    assert!(b.is_onnx());
}

#[test]
#[cfg(feature = "onnx")]
fn test_onnx_cuda() {
    let b = VlmBackend::OnnxCuda { device_id: 0 };
    assert_eq!(b.name(), "onnxruntime-cuda");
    assert!(b.is_onnx());
}

#[test]
#[cfg(feature = "onnx")]
fn test_tensorrt() {
    let b = VlmBackend::TensorRt { device_id: 0 };
    assert_eq!(b.name(), "onnxruntime-tensorrt");
    assert!(b.is_onnx());
}
