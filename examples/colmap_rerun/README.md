Example showing how to load a [COLMAP dataset](https://colmap.github.io/) and visualize it using Rerun.io

```bash
Usage: colmap-rerun --colmap-path <COLMAP_PATH>

Options:
  -p, --colmap-path <COLMAP_PATH>
  -h, --help                     Print help
```

Example:

```bash
cargo run -p colmap-rerun -- --colmap-path path/to/colmap/dataset/sparse
```

NOTE: The COLMAP dataset can be downloaded from [here](https://colmap.github.io/datasets.html).

Output:

![Screenshot from 2024-11-29 15-07-37](https://github.com/user-attachments/assets/e780a631-1537-4b16-93e1-3d3680b33316)
