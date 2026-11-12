`LkPyramidU8` and `calc_optical_flow_pyr_lk_u8`: sparse pyramidal Lucas-Kanade on u8 images with OpenCV's
fixed-point scheme (reflect-101 padded u8 levels, i16 Scharr derivatives, Q14 bilinear weights, exact integer window
sums, oscillation half-step) and OpenCV's status rules (status 0 when a window leaves the image, on a small minimum
eigenvalue, or when the final window leaves the image). On 1000 EuRoC points (250 at the border) it agrees with
`cv::calcOpticalFlowPyrLK` on every status and to 7.6e-6 px median (0.06 px max), and runs 2.1x faster than OpenCV
on one thread (46.5 vs 99.2 ms including both pyramid builds).
