An example showing how to use detect FAST features on an image.

```bash
Usage: fast_detector [OPTIONS] --image-path <IMAGE_PATH>

Options:
  -i, --image-path <IMAGE_PATH>
  -t, --threshold <THRESHOLD>    [default: 10]
  -a, --arc-length <ARC_LENGTH>  [default: 5]
  -h, --help                     Print help
```

Example:

```bash
cargo run --bin fast_detector --release -- --image-path ./data/fast_detector/test.jpg
```
