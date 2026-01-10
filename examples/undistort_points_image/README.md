An example showing how to use `undistort_points` function provided by `kornia-imgproc` crate for image undistortion.

NOTE: Build release version first
```bash
cargo build --release
```

```bash
Usage: cargo run --release -- -i <IMAGE_PATH>

Options:
  -i, --image-path <IMAGE_PATH>
  -h, --help                     Print help
```

Output:
This example will display the distorted and undistorted image in a [`rerun`](https://github.com/rerun-io/rerun) window.
![](https://github.com/user-attachments/assets/8b989eb8-746e-4693-b6d5-42f4ff37f7fc)
