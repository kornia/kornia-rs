An example showing how to apply morphological operations (dilate, erode, open, close) to binary images using the `kornia-imgproc` crate.

NOTE: Build release version first
```bash
cargo build --release
```

```bash
cargo run --release -- -i <IMAGE_PATH> [OPTIONS]

Options:
  -i, --image-path <IMAGE_PATH>  Path to the input image
  -s, --kernel-size <SIZE>       Kernel size (default: 5)
  -k, --kernel-shape <SHAPE>     Kernel shape: box, cross, ellipse (default: box)
  -h, --help                     Print help
```

Output:
This example will display all morphological operations in a [`rerun`](https://github.com/rerun-io/rerun) window: