//! Erosion of the boundaries between the regions of a label map.
//!
//! A label map assigns an integer label to every pixel (a segmentation, a
//! connected-component labelling, a superpixel map). Pixels close to a
//! pixel with a different label are where the labelling is least reliable;
//! [`erode_label_boundaries`] overwrites them with a caller-chosen value so
//! that each region shrinks away from its neighbours.

pub(crate) mod kernels;

use kornia_image::{Image, ImageError};

use crate::color::kernel_common::{check_size, sealed};
use crate::connected_components::Connectivity;

/// Largest supported distance threshold: distances are carried as `u16`
/// and capped at `threshold + 1`, which must stay representable.
const MAX_THRESHOLD: usize = u16::MAX as usize - 1;

/// Label pixel types: `u8` and `u16` single-channel label maps. Sealed.
pub trait LabelPixel: sealed::Sealed + Copy + PartialEq + Send + Sync + 'static {
    #[doc(hidden)]
    fn erode_label_boundaries_impl(
        src: &Image<Self, 1>,
        dst: &mut Image<Self, 1>,
        radius: usize,
        connectivity: Connectivity,
        fill: Self,
    ) -> Result<(), ImageError>;
}

/// Overwrite every pixel within `radius` of a differently labelled pixel.
///
/// A pixel `p` is written as `fill` when some pixel `q` of the image with
/// `label(q) != label(p)` lies at distance at most `radius` from `p`, where
/// the distance is
///
/// * Manhattan (`|dx| + |dy|`) for [`Connectivity::Four`], and
/// * Chebyshev (`max(|dx|, |dy|)`) for [`Connectivity::Eight`].
///
/// Every other pixel keeps its label. Both sides of a boundary are written,
/// so a boundary between two regions becomes a band about `2 * radius`
/// pixels wide. The image border is not a boundary: pixels outside the
/// image do not exist, so they never differ from anything. `radius = 0` is
/// the identity, and an image holding a single label is returned unchanged
/// for every radius.
///
/// # Algorithm
///
/// With `S` the set of pixels that have a differently labelled neighbour
/// under `connectivity`, the rule above is equivalent to
/// `distance(p, S) <= radius - 1` under the matching metric (along a
/// shortest path from `p` to `q` the label changes between two neighbours,
/// the first of which is in `S`). That distance is computed exactly with a
/// separable, capped distance transform: a vertical then a horizontal
/// two-sweep min-plus pass (Manhattan), or the same passes on the thresholded
/// vertical distance (Chebyshev). The cost per pixel is bounded independently
/// of `radius`: small radii replace the horizontal two-sweep pass by a direct
/// window of at most eight shifted minima, and no structuring element is
/// ever scanned.
///
/// # Reference
///
/// * scikit-image `skimage.segmentation.find_boundaries(label_img,
///   connectivity, mode="thick")` marks exactly the set `S` above, with
///   `connectivity=1` ↔ [`Connectivity::Four`] and `connectivity=2` ↔
///   [`Connectivity::Eight`]; it equals this op with `radius = 1`.
/// * OpenCV: `S` is `cv2.dilate(labels, k) != cv2.erode(labels, k)` with a
///   3×3 cross (`Four`) or rectangle (`Eight`) kernel, and the result for a
///   larger radius is that mask dilated by a diamond (`Four`) or a
///   `(2 * radius - 1)`-sided square (`Eight`), then used to write `fill`.
///
/// Departures: the `inner` / `outer` (one-sided) and `subpixel` modes of
/// `find_boundaries` are not offered, the boundary is grown to `radius`
/// instead of being fixed at one pixel, and the result is the label map with
/// `fill` written in place of a boolean mask.
///
/// # Arguments
///
/// * `src` - Single-channel `u8` or `u16` label map.
/// * `dst` - Pre-allocated output with the same size as `src`.
/// * `radius` - Distance, in pixels, from a differently labelled pixel up to
///   which pixels are overwritten; `0` copies `src`.
/// * `connectivity` - Neighbourhood and metric: [`Connectivity::Four`]
///   (Manhattan) or [`Connectivity::Eight`] (Chebyshev).
/// * `fill` - Value written to the eroded pixels (e.g. an "unknown" label).
///
/// # Returns
///
/// `Ok(())` on success; `dst` holds the eroded label map.
///
/// # Errors
///
/// * [`ImageError::InvalidImageSize`] if `src` and `dst` sizes differ.
/// * [`ImageError::InvalidKernelLength`] if the distance threshold
///   `min(radius - 1, largest in-image distance)` exceeds 65534; only images
///   with a side above 32767 pixels can reach it.
/// * [`ImageError::MixedResidency`] / [`ImageError::Cuda`] for device images
///   (feature `cuda`) on mixed residency or a launch failure.
///
/// # Example
///
/// ```rust
/// use kornia_image::{Image, ImageSize};
/// use kornia_imgproc::connected_components::Connectivity;
/// use kornia_imgproc::morphology::erode_label_boundaries;
///
/// // One row, two regions: 1 1 1 2 2 2.
/// let size = ImageSize { width: 6, height: 1 };
/// let src = Image::<u8, 1>::new(size, vec![1, 1, 1, 2, 2, 2])?;
/// let mut dst = Image::<u8, 1>::from_size_val(size, 0)?;
///
/// erode_label_boundaries(&src, &mut dst, 1, Connectivity::Four, 0)?;
/// assert_eq!(dst.as_slice(), &[1, 1, 0, 0, 2, 2]);
///
/// erode_label_boundaries(&src, &mut dst, 2, Connectivity::Four, 0)?;
/// assert_eq!(dst.as_slice(), &[1, 0, 0, 0, 0, 2]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn erode_label_boundaries<T: LabelPixel>(
    src: &Image<T, 1>,
    dst: &mut Image<T, 1>,
    radius: usize,
    connectivity: Connectivity,
    fill: T,
) -> Result<(), ImageError> {
    T::erode_label_boundaries_impl(src, dst, radius, connectivity, fill)
}

/// Distance threshold `radius - 1`, clamped to the largest distance between
/// two pixels of a `width × height` image (a larger threshold selects the
/// same pixels). `None` when there is nothing to do: `radius == 0` or an
/// empty image.
pub(crate) fn distance_threshold(
    width: usize,
    height: usize,
    radius: usize,
    connectivity: Connectivity,
) -> Result<Option<u16>, ImageError> {
    if radius == 0 || width == 0 || height == 0 {
        return Ok(None);
    }
    let max_distance = match connectivity {
        Connectivity::Four => (width - 1) + (height - 1),
        Connectivity::Eight => (width - 1).max(height - 1),
    };
    let threshold = (radius - 1).min(max_distance);
    if threshold > MAX_THRESHOLD {
        return Err(ImageError::InvalidKernelLength(radius, radius));
    }
    Ok(Some(threshold as u16))
}

macro_rules! impl_label_pixel {
    ($t:ty) => {
        impl LabelPixel for $t {
            fn erode_label_boundaries_impl(
                src: &Image<$t, 1>,
                dst: &mut Image<$t, 1>,
                radius: usize,
                connectivity: Connectivity,
                fill: $t,
            ) -> Result<(), ImageError> {
                check_size(src, dst)?;
                let (w, h) = (src.cols(), src.rows());
                let threshold = distance_threshold(w, h, radius, connectivity)?;
                crate::try_device!(src, dst, |stream| {
                    cuda_adapter::erode_label_boundaries(
                        src,
                        dst,
                        threshold,
                        connectivity,
                        fill,
                        stream,
                    )
                });
                match threshold {
                    None => dst.as_slice_mut().copy_from_slice(src.as_slice()),
                    Some(threshold) => kernels::erode_label_boundaries::<$t>(
                        src.as_slice(),
                        dst.as_slice_mut(),
                        w,
                        h,
                        threshold,
                        connectivity,
                        fill,
                    ),
                }
                Ok(())
            }
        }
    };
}

impl_label_pixel!(u8);
impl_label_pixel!(u16);

#[cfg(feature = "cuda")]
mod cuda_adapter {
    use std::sync::Arc;

    use cudarc::driver::CudaStream;
    use kornia_image::{Image, ImageError};

    use super::Connectivity;
    use crate::cuda::dispatch::{device_slices, untyped_device_err};
    use crate::cuda::label_boundaries::{launch_erode_label_boundaries, CudaLabelPixel};

    pub(super) fn erode_label_boundaries<T: CudaLabelPixel>(
        src: &Image<T, 1>,
        dst: &mut Image<T, 1>,
        threshold: Option<u16>,
        connectivity: Connectivity,
        fill: T,
        stream: &Arc<CudaStream>,
    ) -> Result<(), ImageError> {
        let (w, h) = (src.cols(), src.rows());
        let ctx = stream.context();
        let (s, d) = device_slices!(src, dst);
        launch_erode_label_boundaries(ctx, stream, s, d, w, h, threshold, connectivity, fill)
            .map_err(|e| ImageError::Cuda(e.to_string()))
    }
}

#[cfg(test)]
mod tests;

#[cfg(all(test, feature = "cuda"))]
mod cuda_tests;
