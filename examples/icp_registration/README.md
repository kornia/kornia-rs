# Point Cloud Registration using ICP

## Usage

Example how to register two point clouds using the Iterative Closest Point (ICP) algorithm.

```bash
Usage: icp_registration -s <source-path> -t <target-path>

Example of ICP registration

Options:
  -s, --source-path path to the source point cloud
  -t, --target-path path to the target point cloud
  --help            display usage information
```

## Example

```bash
cargo run --bin icp_registration --release -- -s cloud_bin_0.pcd -t cloud_bin_1.pcd
```

NOTE: download the point clouds from [here](https://github.com/kornia/data/tree/main/pointcloud).

## Output

https://github.com/user-attachments/assets/fb7d481e-2ffd-4fda-8aca-8dcdac06e65f

