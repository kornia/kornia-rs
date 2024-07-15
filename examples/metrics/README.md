Example how to compute the metrics (mse, psnr) of an image using the Kornia Rust image processing library.

```bash
Usage: metrics --image-path <IMAGE_PATH>

Options:
  -i, --image-path <IMAGE_PATH>
  -h, --help                     Print help
```

Example:

```bash
cargo run -p metrics -- --image-path path/to/image.jpg
```

Output:

```bash
MSE error: 0.020697484
PSNR error: -23.751804
```
