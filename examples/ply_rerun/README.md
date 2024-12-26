Example showing how to read a PLY file from ClouCompare and visualize it using Rerun.io

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
cargo run -p ply_rerun -- --ply-path room-bin.ply --ply-type default
```

Output:

https://github.com/user-attachments/assets/ead3b767-4b57-4e3f-8bcf-0f93fe6ab962

NOTE: download the data from https://github.com/kornia/data/raw/refs/heads/main/pointcloud/room_bin.ply
