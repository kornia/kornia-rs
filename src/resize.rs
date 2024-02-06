use crate::image::{Image, ImageSize};
use ndarray::{stack, Array2, Array3, Zip};

/// Create a meshgrid of x and y coordinates
///
/// # Arguments
///
/// * `x` - A 1D array of x coordinates
/// * `y` - A 1D array of y coordinates
///
/// # Returns
///
/// A tuple of 2D arrays of shape (height, width) containing the x and y coordinates
///
/// # Example
///
/// ```
/// let x = ndarray::Array::linspace(0., 4., 5).insert_axis(ndarray::Axis(0));
/// let y = ndarray::Array::linspace(0., 3., 4).insert_axis(ndarray::Axis(0));
/// let (xx, yy) = kornia_rs::resize::meshgrid(&x, &y);
/// assert_eq!(xx.shape(), &[4, 5]);
/// assert_eq!(yy.shape(), &[4, 5]);
/// assert_eq!(xx[[0, 0]], 0.);
/// assert_eq!(xx[[0, 4]], 4.);
/// ```
pub fn meshgrid(x: &Array2<f32>, y: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
    // create the meshgrid of x and y coordinates
    let nx = x.len_of(ndarray::Axis(1));
    let ny = y.len_of(ndarray::Axis(1));

    // broadcast the x and y coordinates to create a 2D grid, and then transpose the y coordinates
    // to create the meshgrid of x and y coordinates of shape (height, width)
    let xx = x.broadcast((ny, nx)).unwrap().to_owned();
    let yy = y.broadcast((nx, ny)).unwrap().t().to_owned();

    (xx, yy)
}

/// Kernel for bilinear interpolation
///
/// # Arguments
///
/// * `image` - The input image container.
/// * `u` - The x coordinate of the pixel to interpolate.
/// * `v` - The y coordinate of the pixel to interpolate.
/// * `c` - The channel of the pixel to interpolate.
///
/// # Returns
///
/// The interpolated pixel value.
// TODO: add support for other data types. Maybe use a trait? or template?
fn bilinear_interpolation(image: &Array3<u8>, u: f32, v: f32, c: usize) -> f32 {
    let (height, width, _) = image.dim();
    let iu = u.trunc() as usize;
    let iv = v.trunc() as usize;

    let frac_u = u.fract();
    let frac_v = v.fract();
    let val00 = image[[iv, iu, c]] as f32;
    let val01 = if iu + 1 < width {
        image[[iv, iu + 1, c]] as f32
    } else {
        val00
    };
    let val10 = if iv + 1 < height {
        image[[iv + 1, iu, c]] as f32
    } else {
        val00
    };
    let val11 = if iu + 1 < width && iv + 1 < height {
        image[[iv + 1, iu + 1, c]] as f32
    } else {
        val00
    };

    let frac_uu = 1. - frac_u;
    let frac_vv = 1. - frac_v;

    val00 * frac_uu * frac_vv
        + val01 * frac_u * frac_vv
        + val10 * frac_uu * frac_v
        + val11 * frac_u * frac_v
}

/// Kernel for nearest neighbor interpolation
///
/// # Arguments
///
/// * `image` - The input image container.
/// * `u` - The x coordinate of the pixel to interpolate.
/// * `v` - The y coordinate of the pixel to interpolate.
/// * `c` - The channel of the pixel to interpolate.
///
/// # Returns
///
/// The interpolated pixel value.
fn nearest_neighbor_interpolation(image: &Array3<u8>, u: f32, v: f32, c: usize) -> f32 {
    let (height, width, _) = image.dim();

    let iu = u.round() as usize;
    let iv = v.round() as usize;

    let iu = iu.clamp(0, width - 1);
    let iv = iv.clamp(0, height - 1);

    image[[iv, iu, c]] as f32
}

#[derive(Debug, Clone, Copy)]
pub enum InterpolationMode {
    Bilinear,
    NearestNeighbor,
}

// TODO: implement builder pattern
pub struct ResizeOptions {
    pub interpolation: InterpolationMode,
}

impl Default for ResizeOptions {
    fn default() -> Self {
        ResizeOptions {
            interpolation: InterpolationMode::Bilinear,
        }
    }
}

/// Resize an image to a new size
///
/// The function resizes an image to a new size using the specified interpolation mode.
///
/// # Arguments
///
/// * `image` - The input image container.
/// * `new_size` - The new size of the image.
/// * `optional_args` - Optional arguments for the resize operation.
///
/// # Returns
///
/// The resized image.
pub fn resize<T, const CHANNELS: usize>(
    image: &Image<T, CHANNELS>,
    new_size: ImageSize,
    optional_args: ResizeOptions,
) -> Image<T, CHANNELS>
where
    T: num_traits::FromPrimitive + std::fmt::Debug + Send + Sync + Copy,
{
    let image_size = image.image_size();

    // create the output image
    let mut output = ndarray::Array3::<T>::zeros((new_size.height, new_size.width, CHANNELS));

    // create a grid of x and y coordinates for the output image
    // and interpolate the values from the input image.
    let x = ndarray::Array::linspace(0., (image_size.width - 1) as f32, new_size.width)
        .insert_axis(ndarray::Axis(0));
    let y = ndarray::Array::linspace(0., (image_size.height - 1) as f32, new_size.height)
        .insert_axis(ndarray::Axis(0));

    // create the meshgrid of x and y coordinates, arranged in a 2D grid of shape (height, width)
    let (xx, yy) = meshgrid(&x, &y);

    // TODO: benchmark this
    // stack the x and y coordinates into a single array of shape (height, width, 2)
    let xy = stack![ndarray::Axis(2), xx, yy];

    // iterate over the output image and interpolate the pixel values

    Zip::from(xy.rows())
        .and(output.rows_mut())
        .par_for_each(|uv, mut out| {
            assert_eq!(uv.len(), 2);
            let (u, v) = (uv[0], uv[1]);

            // compute the pixel values for each channel
            let pixels = (0..image.num_channels()).map(|k| match optional_args.interpolation {
                InterpolationMode::Bilinear => bilinear_interpolation(&image.data, u, v, k),
                //InterpolationMode::NearestNeighbor => {
                //    nearest_neighbor_interpolation(&image.data, u, v, k)
                //}
            });

            // write the pixel values to the output image
            for (k, pixel) in pixels.enumerate() {
                out[k] = pixel as u8;
            }
        });

    Image { data: output }
}

#[cfg(test)]
mod tests {

    #[test]
    fn resize_smoke_ch3() {
        use crate::image::{Image, ImageSize};
        let image = Image::from_shape_vec([4, 5, 3], vec![0; 4 * 5 * 3]);
        let image_resized = super::resize(
            &image,
            ImageSize {
                width: 2,
                height: 3,
            },
            super::ResizeOptions::default(),
        );
        assert_eq!(image_resized.num_channels(), 3);
        assert_eq!(image_resized.image_size().width, 2);
        assert_eq!(image_resized.image_size().height, 3);
    }

    #[test]
    fn resize_smoke_ch1() {
        use crate::image::{Image, ImageSize};
        let image = Image::from_shape_vec([4, 5, 1], vec![0; 4 * 5 * 1]);
        let image_resized = super::resize(
            &image,
            ImageSize {
                width: 2,
                height: 3,
            },
            super::ResizeOptions::default(),
        );
        assert_eq!(image_resized.num_channels(), 1);
        assert_eq!(image_resized.image_size().width, 2);
        assert_eq!(image_resized.image_size().height, 3);
    }

    #[test]
    fn meshgrid() {
        let x = ndarray::Array::linspace(0., 4., 5).insert_axis(ndarray::Axis(0));
        let y = ndarray::Array::linspace(0., 3., 4).insert_axis(ndarray::Axis(0));
        let (xx, yy) = super::meshgrid(&x, &y);
        assert_eq!(xx.shape(), &[4, 5]);
        assert_eq!(yy.shape(), &[4, 5]);
        assert_eq!(xx[[0, 0]], 0.);
        assert_eq!(xx[[0, 4]], 4.);
        assert_eq!(yy[[0, 0]], 0.);
        assert_eq!(yy[[3, 0]], 3.);
    }
}