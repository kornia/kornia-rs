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

![Screenshot from 2024-12-27 16-01-47](https://github.com/user-attachments/assets/8a4f1b16-bb01-493a-8a89-758b89d82633)
