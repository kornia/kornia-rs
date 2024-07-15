Example to perform image normalization by computing the mean and standard deviation of the image and then normalizing the image using the computed values.

```bash
Usage: normalize_ii --image-path <IMAGE_PATH>

Options:
  -i, --image-path <IMAGE_PATH>
  -h, --help                     Print help
```

Example:

```bash
cargo run -p normalize_ii -- --image-path path/to/image.jpg
```

Output:

```bash
min: 0.0, max: 1.0
```
