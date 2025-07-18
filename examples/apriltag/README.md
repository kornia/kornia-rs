# AprilTag Decoding Example

This example demonstrates how to detect and decode AprilTags in images using the
`kornia-apriltag` crate.

```txt
Usage: cargo run -p apriltag --release -- <path> [-k <kind...>] [-d <decode-sharpening>] [-r] [-m <min-white-black-difference>]

Detects AprilTags in a image

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

// TODO: Add Example Image
