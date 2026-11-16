`calc_optical_flow_pyr_lk` with `BorderMode::Clamp` or `BorderMode::Mirror` no longer reports status 1 for a
track that left the image: as in `cv::calcOpticalFlowPyrLK`, a point is lost when its window's floored top-left
corner leaves `[-win_size, size)` at the finest level. Tracking also skips pyramid levels smaller than the 21x21
window, as `cv::buildOpticalFlowPyramid` does; on small images those levels pushed border points off the image
(a point at (0, 65.5) on a 96x72 image now tracks to within 0.65 px of OpenCV instead of 15 px). With
`BorderMode::Reject`, such a level made every point fail the in-image check, so all points were lost; they are
now tracked on the levels that fit the window.
