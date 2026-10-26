`kornia_imgproc::depth`: depth-image filtering for `u16` (invalid = `0`) and `f32` (invalid = `NaN`) images, on CPU and, with the `cuda` feature, on device-resident images (bit-identical to the CPU):
- `median_blur_depth` — 3×3 / 5×5 median of the valid samples only; equals `cv2.medianBlur` on images with no invalid sample, and never mixes the invalid marker into the result.
- `flying_pixel_mask` — flags valid pixels farther than `max(abs_threshold, rel_threshold * d)` from the median of the valid samples in their 3×3 window, or with fewer than `min_valid` valid taps.
- `invalidate_masked_depth` — writes the invalid marker where the mask is set.
