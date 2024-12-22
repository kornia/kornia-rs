Example showing how to read a PLY file from OpenSplat and visualize it using Rerun.io

```bash
Usage: ply_rerun --ply-path <ply-path> --ply-type <ply-type>

Read a PLY file and log it to Rerun

Options:
  --ply-path        path to the PLY file
  --ply-type        property type to read
  --help            display usage information
```

Example:

```bash
cargo run -p ply_rerun -- --ply-path banana.ply --ply-type default
```

NOTE: Get the `banana.ply` file from [here](https://drive.google.com/file/d/12lmvVWpFlFPL6nxl2e2d-4u4a31RCSKT/view?usp=sharing).

Output:

![rerun_banana](https://github.com/user-attachments/assets/e6c926b0-77bd-4073-acdd-508f39da0cf6)
