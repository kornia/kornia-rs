An example showing how to apply different filters to an image.

Usage:

```bash
Usage: filters --filter <filter> --kx <kx> --ky <ky> [--sigma-x <sigma-x>] [--sigma-y <sigma-y>]

Apply a separable filter to an image

Options:
  --filter          the filter to apply
  --kx              the kernel size for the horizontal filter
  --ky              the kernel size for the vertical filter
  --sigma-x         the sigma for the gaussian filter
  --sigma-y         the sigma for the gaussian filter
  --help, help      display usage information
```

Example:

```bash
cargo run --bin filters --release -- --filter sobel --kx 3 --ky 3
```
