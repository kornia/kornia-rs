An example showing how to use detect FAST features on an image.

```bash
Usage: fast_detector --image-path <image-path> [--threshold <threshold>] [--arc-length <arc-length>] [--min-distance <min-distance>]

Detect FAST features on an image.

Options:
  --image-path      path to the image to detect FAST features on
  --threshold       threshold for the FAST detector
  --arc-length      arc length for the FAST detector
  --min-distance    minimum distance between detected keypoints
  --help, help      display usage information
```

Example:

```bash
cargo run -p fast_detector --release -- --image-path ./path/to/image.jpg
```

![Demo](https://github.com/user-attachments/assets/c04436f6-1cf6-4e53-89a8-2c2b79a56035)
