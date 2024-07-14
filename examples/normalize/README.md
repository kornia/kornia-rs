Example showing how to normalize and compute the min and max of an image.

```bash
Usage: normalize --image-path <IMAGE_PATH>

Options:
  -i, --image-path <IMAGE_PATH>
  -h, --help                     Print help
```

Example:

```bash
cargo run -p normalize -- --image-path path/to/image.jpg
```

output:

```bash
min: 0.0, max: 255.0
```
