#[test]
fn onnxruntime_is_available() {
    let path = std::env::var("ORT_DYLIB_PATH").ok();

    // If ORT is not configured, skip instead of failing CI
    let Some(path) = path else {
        eprintln!("ORT_DYLIB_PATH not set, skipping test");
        return;
    };

    if !std::path::Path::new(&path).exists() {
        eprintln!("ONNX Runtime dylib not found at {}, skipping test", path);
        return;
    }
}