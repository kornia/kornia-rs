`kornia_3d::depth`: depth images to 3D.
`unproject_depth` / `unproject_depth_into` turn a `u16` (with a `metres_per_unit` multiplier) or `f32` depth image and `PinholeCamera` intrinsics into a `PointCloud`, skipping `0` / `NaN` and masked pixels, optionally with each point's source pixel index; the `_into` form reuses the caller's buffers.
`register_depth` reprojects a depth image into a second pinhole camera given a `dst_from_src` `Pose3d`: nearest-pixel projection, a z-buffer so the nearer point wins, and an optional `splat_radius` that closes the holes of an upsampled destination grid.
With the `cuda` feature, `CudaDepthProjector` runs both on device-resident images and produces the same points (same order) and the same depth values as the CPU.
