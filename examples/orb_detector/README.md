An example showing how to use ORB detector on an image.

```bash
Usage: orb_detector <example_kind> -f <image-path>

ORB Detector

Positional Arguments:
  example_kind      possible values: static, webcam

Options:
  -f, --image-path  path to the image
  --help, help      display usage information
```

Example:

```bash
cargo run -p orb_detector --release -- webcam --image-path /path/to/image.png
```

![Demo](https://github.com/user-attachments/assets/037b7a82-cf6b-4d5c-abd2-3d5ed4df915b)
