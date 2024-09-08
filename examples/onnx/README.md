Example showing how to use ONNX models with OnnxRuntime for RTDETR.

Download the onnxruntime dylib from the [onnxruntime releases page](https://github.com/microsoft/onnxruntime/releases) and extract it.

Export first the path to the onnxruntime dylib:

```bash
export ORT_DYLIB_PATH=/path/to/onnxruntime/build/Linux/RelWithDebInfo/libonnxruntime.so
```

Then run the example:

```bash
cargo run --example onnx-rtdetr -- --image-path img.jpg --model-path rtdetr.onnx --ort-dylib-path $ORT_DYLIB_PATH
```
