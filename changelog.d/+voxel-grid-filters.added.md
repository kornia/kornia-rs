`kornia_3d::voxel` and `kornia_3d::pointcloud::filter`: voxel grids and point cloud outlier filters.
`VoxelLayout` (voxel size and origin) maps points to `i32` voxel keys, negative keys included; `VoxelGrid<T>` is a sparse FxHash voxel map with insert/get/remove/iteration and 6/18/26-neighbour iteration.
`voxel_downsample` keeps one point per occupied voxel (`DownsampleMode::Centroid`, `First` or `NearestToCentre`; colors and normals carried along) in ascending key order, so the output is deterministic; `VoxelDownsampler` reuses its buffers across calls.
`voxel_neighbour_counts` and `remove_sparse_voxels` count and filter on occupied neighbour voxels.
`remove_radius_outliers` and `remove_statistical_outliers` (with `*_inlier_indices` forms) follow PCL's radius and statistical outlier removal, searched on a uniform cell grid (a cell list, with the `kiddo` k-d tree as fallback) and parallel over the cells.
With the `cuda` feature, `CudaVoxelFilter` runs the downsampling and the neighbour count on device-resident points (stable radix sort by key, then segment reduce) with the same output, in the same order, as the CPU.
