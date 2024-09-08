An example showing how to use the RTDETR model with the `kornia::dnn` module and the webcam with the `kornia::io` module with the ability to cancel the feed after a certain amount of time. This example will display the webcam feed in a [`rerun`](https://github.com/rerun-io/rerun) window.

NOTE: This example requires the gstremer backend to be enabled. To enable the gstreamer backend, use the `gstreamer` feature flag when building the `kornia` crate and its dependencies.

## Prerequisites

Maily you need to download onnxruntime from: <https://github.com/microsoft/onnxruntime/releases>

## Usage

```bash
Usage: rtdetr [OPTIONS] --model-path <MODEL_PATH>

Options:
  -c, --camera-id <CAMERA_ID>              [default: 0]
  -f, --fps <FPS>                          [default: 5]
  -m, --model-path <MODEL_PATH>
  -n, --num-threads <NUM_THREADS>          [default: 8]
  -s, --score-threshold <SCORE_THRESHOLD>  [default: 0.75]
  -h, --help                               Print help
```

Example:

```bash
ORT_DYLIB_PATH=/path/to/libonnxruntime.so cargo run --bin rtdetr --release -- --camera-id 0 --model-path rtdetr.onnx --num-threads 8 --score-threshold 0.75
```
