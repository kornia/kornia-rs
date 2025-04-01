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
![Screenshot from 2025-04-01 02-12-32](https://github.com/user-attachments/assets/8e81073b-d246-4dc4-b64e-d7595c66ffd3)
