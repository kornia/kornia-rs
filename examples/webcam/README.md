An example showing how to use the webcam with the `kornia::io` module with the ability to cancel the feed after a certain amount of time. This example will display the webcam feed in a [`rerun`](https://github.com/rerun-io/rerun) window.

NOTE: This example requires the gstremer backend to be enabled. To enable the gstreamer backend, use the `gstreamer` feature flag when building the `kornia` crate and its dependencies.

```bash
Usage: webcam [OPTIONS]

Options:
  -c, --camera-id <CAMERA_ID>  [default: 0]
  -f, --fps <FPS>              [default: 30]
  -d, --duration <DURATION>
  -h, --help                   Print help
```

Example:

```bash
cargo run --release -- --camera-id 0 --duration 5 --fps 30
```

TODO: add output
