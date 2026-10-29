`kornia_3d::voxel`: voxel grids and voxel downsampling on the host.
`VoxelLayout` (voxel size and origin) maps points to `i32` voxel keys, negative keys included; `VoxelGrid<T>` is a sparse FxHash voxel map with insert/get/remove/iteration and 6/18/26-neighbour iteration.
`voxel_downsample` keeps one point per occupied voxel (`DownsampleMode::Centroid`, `First` or `NearestToCentre`; colors and normals carried along) in ascending key order, so the output is deterministic; `VoxelDownsampler` reuses its buffers across calls.
`voxel_neighbour_counts` and `remove_sparse_voxels` count and filter on occupied neighbour voxels.
`kornia_3d::pointcloud::filter::select_points` copies the points of a cloud at given indices, with their colors and normals.
