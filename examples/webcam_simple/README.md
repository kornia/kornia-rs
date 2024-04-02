An example showing how to use the webcam with the `kornia::io` module. This example will display the webcam feed in a [`rerun`](https://github.com/rerun-io/rerun) window.

NOTE: This example requires the gstremer backend to be enabled. To enable the gstreamer backend, use the `gstreamer` feature flag when building the `kornia` crate and its dependencies.

```bash
cargo run --release -- --camera-id 0
```
