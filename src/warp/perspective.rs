use crate::interpolation::{interpolate_pixel, InterpolationMode};
use crate::{
    image::{Image, ImageSize},
    interpolation::meshgrid,
};
use anyhow::Result;
use ndarray::stack;

// flat representation of a 3x3 matrix
pub type PerspectiveMatrix = [f32; 9];

#[rustfmt::skip]
fn determinant3x3(m: &PerspectiveMatrix) -> f32 {
    m[0] * (m[4] * m[8] - m[5] * m[7]) -
    m[1] * (m[3] * m[8] - m[5] * m[6]) +
    m[2] * (m[3] * m[7] - m[4] * m[6])
}

#[rustfmt::skip]
fn adjugate3x3(m: &PerspectiveMatrix) -> PerspectiveMatrix {
    [
        m[4] * m[8] - m[5] * m[7],  // [0, 0]
        m[2] * m[7] - m[1] * m[8],  // [0, 1]
        m[1] * m[5] - m[2] * m[4],  // [0, 2]
        m[5] * m[6] - m[3] * m[8],  // [1, 0]
        m[0] * m[8] - m[2] * m[6],  // [1, 1]
        m[2] * m[3] - m[0] * m[5],  // [1, 2]
        m[3] * m[7] - m[4] * m[6],  // [2, 0]
        m[1] * m[6] - m[0] * m[7],  // [2, 1]
        m[0] * m[4] - m[1] * m[3],  // [2, 2]
    ]
}

fn inverse_perspective_matrix(m: PerspectiveMatrix) -> Result<PerspectiveMatrix> {
    let det = determinant3x3(&m);

    if det == 0.0 {
        return Err(anyhow::anyhow!("Matrix is singular and cannot be inverted"));
    }

    let adj = adjugate3x3(&m);
    let inv_det = 1.0 / det;

    let mut inv_m = [0.0; 9];
    for i in 0..9 {
        inv_m[i] = adj[i] * inv_det;
    }

    Ok(inv_m)
}

// implement later as batched operation
fn transform_point(x: f32, y: f32, m: PerspectiveMatrix) -> (f32, f32) {
    let w = m[6] * x + m[7] * y + m[8];
    let x = (m[0] * x + m[1] * y + m[2]) / w;
    let y = (m[3] * x + m[4] * y + m[5]) / w;
    (x, y)
}

/// Applies a perspective transformation to an image.
///
/// * `src` - The input image with shape (height, width, channels).
/// * `m` - The 3x3 perspective transformation matrix src -> dst.
/// * `new_size` - The size of the output image.
/// * `interpolation` - The interpolation mode to use.
///
/// # Returns
///
/// The output image with shape (new_height, new_width, channels).
///
/// # Example
///
/// ```
/// use kornia_rs::image::{Image, ImageSize};
/// use kornia_rs::warp::warp_perspective;
///
/// let src = Image::<f32, 1>::new(
///   ImageSize {
///     width: 4,
///     height: 5,
///   },
///   vec![0.0f32; 4 * 5]
/// ).unwrap();
///
/// let m = [1.0, 0.0, -1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
///
/// let dst = warp_perspective(&src, m, ImageSize {
///     width: 2,
///     height: 3,
///   },
///   kornia_rs::interpolation::InterpolationMode::Bilinear
/// ).unwrap();
///
/// assert_eq!(dst.size().width, 2);
/// assert_eq!(dst.size().height, 3);
/// ```
pub fn warp_perspective<const CHANNELS: usize>(
    src: &Image<f32, CHANNELS>,
    m: PerspectiveMatrix,
    new_size: ImageSize,
    interpolation: InterpolationMode,
) -> Result<Image<f32, CHANNELS>> {
    // inverse perspective matrix
    // TODO: allow later to skip the inverse calculation if user provides it
    let inv_m = inverse_perspective_matrix(m)?;

    // allocate the output image
    let mut dst = Image::from_size_val(new_size, 0.0)?;

    // create a grid of x and y coordinates for the output image
    // TODO: make this re-useable
    let x = ndarray::Array::range(0.0, new_size.width as f32, 1.0).insert_axis(ndarray::Axis(0));
    let y = ndarray::Array::range(0.0, new_size.height as f32, 1.0).insert_axis(ndarray::Axis(0));

    // create the meshgrid of x and y coordinates, arranged in a 2D grid of shape (height, width)
    let (xx, yy) = meshgrid(&x, &y);

    // stack the x and y coordinates into a single array of shape (height, width, 2)
    let xy = stack![ndarray::Axis(2), xx, yy];

    // iterate over the output image and find the corresponding position in the input image

    ndarray::Zip::from(xy.rows())
        .and(dst.data.rows_mut())
        .par_for_each(|uv, mut out| {
            assert_eq!(uv.len(), 2);
            let (u, v) = (uv[0], uv[1]);

            // find corresponding position in src image
            let (u_src, v_src) = transform_point(u, v, inv_m);

            // TODO: allow for multi-channel images
            // interpolate the pixel value
            let pixels = (0..src.num_channels())
                .map(|c| interpolate_pixel(&src.data, u_src, v_src, c, interpolation));

            for (c, pixel) in pixels.enumerate() {
                out[c] = pixel;
            }
        });

    Ok(dst)
}

#[cfg(test)]
mod tests {
    use anyhow::Result;

    #[test]
    fn inverse_perspective_matrix() -> Result<()> {
        let m = [1.0, 0.0, -1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
        let expected = [1.0, 0.0, 1.0, 0.0, 1.0, -1.0, 0.0, 0.0, 1.0];
        let inv_m = super::inverse_perspective_matrix(m)?;
        assert_eq!(inv_m, expected);
        Ok(())
    }

    #[test]
    fn transform_point() {
        let m = [1.0, 0.0, -1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
        let (x, y) = super::transform_point(1.0, 1.0, m);
        let (x_expected, y_expected) = (0.0, 2.0);
        assert_eq!(x, x_expected);
        assert_eq!(y, y_expected);
    }

    #[test]
    fn warp_perspective_identity() -> Result<()> {
        use crate::image::{Image, ImageSize};
        let image: Image<f32, 3> = Image::from_size_val(
            ImageSize {
                width: 4,
                height: 5,
            },
            0.0f32,
        )?;

        // identity matrix
        let m = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

        let image_transformed = super::warp_perspective(
            &image,
            m,
            ImageSize {
                width: 2,
                height: 3,
            },
            super::InterpolationMode::Bilinear,
        )?;

        assert_eq!(image_transformed.num_channels(), 3);
        assert_eq!(image_transformed.size().width, 2);
        assert_eq!(image_transformed.size().height, 3);

        Ok(())
    }

    #[test]
    fn warp_perspective_hflip() -> Result<()> {
        use crate::image::{Image, ImageSize};
        let image = Image::<_, 1>::new(
            ImageSize {
                width: 2,
                height: 3,
            },
            vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0],
        )?;

        let image_expected = vec![1.0, 0.0, 3.0, 2.0, 5.0, 4.0];

        // flip matrix
        let m = [-1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

        let image_transformed = super::warp_perspective(
            &image,
            m,
            ImageSize {
                width: 2,
                height: 3,
            },
            super::InterpolationMode::Bilinear,
        )?;

        assert_eq!(image_transformed.num_channels(), 1);
        assert_eq!(image_transformed.size().width, 2);
        assert_eq!(image_transformed.size().height, 3);

        assert_eq!(image_transformed.data.as_slice().expect(""), image_expected);

        Ok(())
    }

    #[test]
    fn test_warp_perspective_resize() -> Result<()> {
        use crate::image::{Image, ImageSize};
        let image = Image::<_, 1>::new(
            ImageSize {
                width: 4,
                height: 4,
            },
            vec![
                0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                15.0,
            ],
        )?;

        // resize matrix (from get_perspective_transform)
        let m = [0.3333, 0.0, 0.0, 0.0, 0.3333, 0.0, 0.0, 0.0, 1.0];

        let image_expected = vec![0.0, 3.0, 12.0, 15.0];

        let image_transformed = super::warp_perspective(
            &image,
            m,
            ImageSize {
                width: 2,
                height: 2,
            },
            super::InterpolationMode::Bilinear,
        )?;

        let image_resized = crate::resize::resize_native(
            &image,
            ImageSize {
                width: 2,
                height: 2,
            },
            super::InterpolationMode::Bilinear,
        )?;

        assert_eq!(image_transformed.num_channels(), 1);
        assert_eq!(image_transformed.size().width, 2);
        assert_eq!(image_transformed.size().height, 2);

        assert_eq!(image_transformed.data.as_slice().expect(""), image_expected);
        assert_eq!(
            image_transformed.data.as_slice().expect(""),
            image_resized.data.as_slice().expect("")
        );

        Ok(())
    }
}
