use std::f32::consts::PI;

use crate::image::{Image, ImageSize};
use crate::interpolation::meshgrid;
use crate::interpolation::{interpolate_pixel, InterpolationMode};
use anyhow::Result;
use ndarray::stack;

type AffineMatrix = (f32, f32, f32, f32, f32, f32);

/// Inverts a 2x3 affine transformation matrix.
///
/// Arguments:
///
/// * `m` - The 2x3 affine transformation matrix.
///
/// Returns:
///
/// The inverted 2x3 affine transformation matrix.
pub fn invert_affine_transform(m: AffineMatrix) -> AffineMatrix {
    let (a, b, c, d, e, f) = m;

    // follow OpenCV: check for determinant == 0
    // https://github.com/opencv/opencv/blob/4.9.0/modules/imgproc/src/imgwarp.cpp#L2765
    let determinant = a * e - b * d;
    let inv_determinant = if determinant != 0.0 {
        1.0 / determinant
    } else {
        0.0
    };

    let new_a = e * inv_determinant;
    let new_b = -b * inv_determinant;
    let new_d = -d * inv_determinant;
    let new_e = a * inv_determinant;
    let new_c = -(new_a * c + new_b * f);
    let new_f = -(new_d * c + new_e * f);

    (new_a, new_b, new_c, new_d, new_e, new_f)
}

/// Returns a 2x3 rotation matrix for a 2D rotation around a center point.
///
/// The rotation matrix is defined as:
///
/// | alpha  beta  tx |
/// | -beta  alpha ty |
///
/// where:
///
/// alpha = scale * cos(angle)
/// beta = scale * sin(angle)
/// tx = (1 - alpha) * center.x - beta * center.y
/// ty = beta * center.x + (1 - alpha) * center.y
///
/// # Arguments
///
/// * `center` - The center point of the rotation.
/// * `angle` - The angle of rotation in degrees.
/// * `scale` - The scale factor.
///
/// # Example
///
/// ```
/// use kornia_rs::warp::get_rotation_matrix2d;
///
/// let center = (0.0, 0.0);
/// let angle = 90.0;
/// let scale = 1.0;
/// let rotation_matrix = get_rotation_matrix2d(center, angle, scale);
/// ```
pub fn get_rotation_matrix2d(center: (f32, f32), angle: f32, scale: f32) -> AffineMatrix {
    let angle = angle * PI / 180.0f32;
    let alpha = scale * angle.cos();
    let beta = scale * angle.sin();

    let tx = (1.0 - alpha) * center.0 - beta * center.1;
    let ty = beta * center.0 + (1.0 - alpha) * center.1;
    (alpha, beta, tx, -beta, alpha, ty)
}

/// Applies an affine transformation to an image.
///
/// # Arguments
///
/// * `src` - The input image with shape (height, width, channels).
/// * `m` - The 2x3 affine transformation matrix.
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
/// use kornia_rs::warp::warp_affine;
///
/// let src = kornia_rs::image::Image::<_, 3>::from_size_val(
///    kornia_rs::image::ImageSize {
///       width: 4,
///      height: 5,
///  },
///  1f32,
/// ).unwrap();
///
/// let m = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0);
/// let new_size = kornia_rs::image::ImageSize {
///    width: 4,
///   height: 5,
/// };
///
/// let output = warp_affine(&src, m, new_size, kornia_rs::interpolation::InterpolationMode::Nearest).unwrap();
///
/// assert_eq!(output.size().width, 4);
/// assert_eq!(output.size().height, 5);
/// ```
pub fn warp_affine<const CHANNELS: usize>(
    src: &Image<f32, CHANNELS>,
    m: AffineMatrix,
    new_size: ImageSize,
    interpolation: InterpolationMode,
) -> Result<Image<f32, CHANNELS>> {
    // invert affine transform matrix to find corresponding positions in src from dst
    let m_inv = invert_affine_transform(m);

    // create the output image
    let mut output = Image::from_size_val(new_size, 0.0)?;

    // create a grid of x and y coordinates for the output image
    // TODO: make this re-useable
    let x = ndarray::Array::range(0.0, new_size.width as f32, 1.0).insert_axis(ndarray::Axis(0));
    let y = ndarray::Array::range(0.0, new_size.height as f32, 1.0).insert_axis(ndarray::Axis(0));

    // create the meshgrid of x and y coordinates, arranged in a 2D grid of shape (height, width)
    let (xx, yy) = meshgrid(&x, &y);

    // TODO: benchmark this
    // stack the x and y coordinates into a single array of shape (height, width, 2)
    let xy = stack![ndarray::Axis(2), xx, yy];

    // iterate over the output image and interpolate the pixel values

    ndarray::Zip::from(xy.rows())
        .and(output.data.rows_mut())
        .par_for_each(|uv, mut out| {
            assert_eq!(uv.len(), 2);
            let (u, v) = (uv[0], uv[1]);

            // find corresponding position in src image
            let u_src = m_inv.0 * u + m_inv.1 * v + m_inv.2;
            let v_src = m_inv.3 * u + m_inv.4 * v + m_inv.5;

            // TODO: remove -- this is already done in interpolate_pixel
            if u_src < 0.0
                || u_src > (src.width() - 1) as f32
                || v_src < 0.0
                || v_src > (src.height() - 1) as f32
            {
                return;
            }

            // compute the pixel values for each channel
            let pixels = (0..src.num_channels())
                .map(|k| interpolate_pixel(&src.data, u_src, v_src, k, interpolation));

            // write the pixel values to the output image
            for (k, pixel) in pixels.enumerate() {
                out[k] = pixel;
            }
        });

    Ok(output)
}

#[cfg(test)]
mod tests {
    use anyhow::Result;

    #[test]
    fn warp_affine_smoke_ch3() -> Result<()> {
        use crate::image::{Image, ImageSize};
        let image = Image::<_, 3>::new(
            ImageSize {
                width: 4,
                height: 5,
            },
            vec![0f32; 4 * 5 * 3],
        )?;
        let image_transformed = super::warp_affine(
            &image,
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
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
    fn warp_affine_smoke_ch1() -> Result<()> {
        use crate::image::{Image, ImageSize};
        let image = Image::<_, 1>::new(
            ImageSize {
                width: 4,
                height: 5,
            },
            vec![0f32; 4 * 5],
        )?;
        let image_transformed = super::warp_affine(
            &image,
            (1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            ImageSize {
                width: 2,
                height: 3,
            },
            super::InterpolationMode::Nearest,
        )?;
        assert_eq!(image_transformed.num_channels(), 1);
        assert_eq!(image_transformed.size().width, 2);
        assert_eq!(image_transformed.size().height, 3);
        Ok(())
    }

    #[test]
    fn warp_affine_correctness_identity() -> Result<()> {
        use crate::image::{Image, ImageSize};
        let image = Image::<_, 1>::new(
            ImageSize {
                width: 4,
                height: 5,
            },
            (0..20).map(|x| x as f32).collect(),
        )?;
        let image_transformed = super::warp_affine(
            &image,
            (1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            ImageSize {
                width: 4,
                height: 5,
            },
            super::InterpolationMode::Nearest,
        )?;
        assert_eq!(image_transformed.data, image.data);
        assert_eq!(image_transformed.size(), image.size());
        Ok(())
    }

    #[test]
    fn warp_affine_correctness_rot90() -> Result<()> {
        use crate::image::{Image, ImageSize};
        let image = Image::<_, 1>::new(
            ImageSize {
                width: 2,
                height: 2,
            },
            vec![0.0f32, 1.0f32, 2.0f32, 3.0f32],
        )?;
        let image_transformed = super::warp_affine(
            &image,
            super::get_rotation_matrix2d((0.5, 0.5), 90.0, 1.0),
            ImageSize {
                width: 2,
                height: 2,
            },
            super::InterpolationMode::Nearest,
        )?;
        assert_eq!(
            image_transformed.data,
            ndarray::array![[[1.0f32], [3.0f32]], [[0.0f32], [2.0f32]]]
        );
        Ok(())
    }
}
