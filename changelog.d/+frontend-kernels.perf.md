`pyrdown_u8` on single-channel images uses a 5-row ring of horizontally filtered rows (the same bytes as before,
and as `cv::pyrDown`): 1.37 -> 0.19 ms on 752x480, one thread. New `pyrdown_separable_f32` and
`scharr_spatial_gradient_separable_f32` evaluate the same kernels separably (10 operations per output instead of
25 or 18): 5.1 -> 0.36 ms and 12.6 -> 0.72 ms, 1.3x faster than OpenCV's `pyrDown` / two `Scharr` calls, with
results identical to OpenCV and to the 3x3/5x5 versions on integer-valued input.
