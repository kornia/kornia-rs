title:	feat(imgproc): map points back through a perspective warp
state:	OPEN
author:	contributor
labels:	enhancement
number:	9998
--
Add a function that maps keypoints detected in a perspective-warped (rectified) image back to the original image's pixel coordinates, using the same homography given to warp_perspective.
