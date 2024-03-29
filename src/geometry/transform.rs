use crate::image::{Image, ImageSize};
use crate::resize::{interpolate, meshgrid, InterpolationMode};
use anyhow::Result;
use ndarray::stack;

type AffineMatrix = (f32, f32, f32, f32, f32, f32);

fn invert_affine_transform(m: AffineMatrix) -> AffineMatrix {
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

            // compute the pixel values for each channel
            let pixels = (0..src.num_channels())
                .map(|k| interpolate(&src.data, u_src, v_src, k, interpolation));

            // write the pixel values to the output image
            for (k, pixel) in pixels.enumerate() {
                out[k] = pixel;
            }
        });

    Ok(output)
}

#[cfg(test)]
mod tests {

    #[test]
    fn warp_affine_smoke_ch3() {
        use crate::image::{Image, ImageSize};
        let image = Image::<_, 3>::new(
            ImageSize {
                width: 4,
                height: 5,
            },
            vec![0f32; 4 * 5 * 3],
        )
        .unwrap();
        let image_transformed = super::warp_affine(
            &image,
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            ImageSize {
                width: 2,
                height: 3,
            },
            super::InterpolationMode::Bilinear,
        )
        .unwrap();
        assert_eq!(image_transformed.num_channels(), 3);
        assert_eq!(image_transformed.size().width, 2);
        assert_eq!(image_transformed.size().height, 3);
    }

    #[test]
    fn warp_affine_smoke_ch1() {
        use crate::image::{Image, ImageSize};
        let image = Image::<_, 1>::new(
            ImageSize {
                width: 4,
                height: 5,
            },
            vec![0f32; 4 * 5],
        )
        .unwrap();
        let image_transformed = super::warp_affine(
            &image,
            (1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            ImageSize {
                width: 2,
                height: 3,
            },
            super::InterpolationMode::Nearest,
        )
        .unwrap();
        assert_eq!(image_transformed.num_channels(), 1);
        assert_eq!(image_transformed.size().width, 2);
        assert_eq!(image_transformed.size().height, 3);
    }

    #[test]
    fn warp_affine_correctness_identity() {
        use crate::image::{Image, ImageSize};
        let image = Image::<_, 1>::new(
            ImageSize {
                width: 4,
                height: 5,
            },
            (0..20).map(|x| x as f32).collect(),
        )
        .unwrap();
        let image_transformed = super::warp_affine(
            &image,
            (1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            ImageSize {
                width: 4,
                height: 5,
            },
            super::InterpolationMode::Nearest,
        )
        .unwrap();
        assert_eq!(image_transformed.data, image.data);
    }

    #[test]
    fn warp_affine_correctness_rot90() {
        use crate::image::{Image, ImageSize};
        let image = Image::<_, 1>::new(
            ImageSize {
                width: 2,
                height: 2,
            },
            vec![0.0f32, 1.0f32, 2.0f32, 3.0f32],
        )
        .unwrap();
        let image_transformed = super::warp_affine(
            &image,
            (0.0, -1.0, 0.5, 1.0, 0.0, 0.5),
            ImageSize {
                width: 2,
                height: 2,
            },
            super::InterpolationMode::Nearest,
        )
        .unwrap();
        assert_eq!(image_transformed.data, ndarray::array![[[1.0f32], [3.0f32]], [[0.0f32], [2.0f32]]]);
    }
}
