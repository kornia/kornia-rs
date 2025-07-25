# AprilTag Decoding Example

This example demonstrates how to detect and decode AprilTags in images using the
`kornia-apriltag` crate.

```txt
Usage: cargo run -p apriltag --release -- <path> [-k <kind...>] [-d <decode-sharpening>] [-r] [-m <min-white-black-difference>]

Detects AprilTags in an image

Positional Arguments:
  path              image path

Options:
  -k, --kind        apriltag family kind to detect
  -d, --decode-sharpening
                    sharpening factor for decoding
  -r, --refine-edges-enabled
                    enable edge refinement during detection
  -m, --min-white-black-difference
                    minimum difference between white and black for detection
  --help, help      display usage information
```

## Example

```bash
cargo run -p apriltag --release -- "path/to/image"
```

![Demo](https://github.com/user-attachments/assets/e515381c-eb82-4e44-bd40-fe063ffd36c1)
