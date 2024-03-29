use crate::image::{Image, ImageSize};
use crate::resize::{self, interpolate};
use anyhow::Result;
use ndarray::{stack, Array2};

fn invert_affine_transform(m: Array2<f32>) -> Array2<f32> {
    assert_eq!(m.dim(), (2, 3));

    let a = m[[0, 0]];
    let b = m[[0, 1]];
    let c = m[[0, 2]];
    let d = m[[1, 0]];
    let e = m[[1, 1]];
    let f = m[[1, 2]];

    let inv_determinant = a * e - b * d;
    let new_a = e * inv_determinant;
    let new_b = -b * inv_determinant;
    let new_d = -d * inv_determinant;
    let new_e = a * inv_determinant;
    let new_c = -(new_a * c + new_b * f);
    let new_f = -(new_d * c + new_e * f);

    ndarray::array![[new_a, new_b, new_c], [new_d, new_e, new_f]]
}

pub fn warp_affine<const CHANNELS: usize>(
    src: &Image<f32, CHANNELS>,
    m: Array2<f32>,
    new_size: ImageSize,
    interpolation: resize::InterpolationMode,
) -> Result<Image<f32, CHANNELS>> {
    // invert affine transform matrix to find corresponding positions in src from dst
    let m_inv = invert_affine_transform(m);

    // create the output image
    let mut output = Image::from_size_val(new_size, 0.0)?;

    // create a grid of x and y coordinates for the output image
    let x = ndarray::Array::range(0.0, new_size.width as f32, 1.0).insert_axis(ndarray::Axis(0));
    let y = ndarray::Array::range(0.0, new_size.height as f32, 1.0).insert_axis(ndarray::Axis(0));

    // create the meshgrid of x and y coordinates, arranged in a 2D grid of shape (height, width)
    let (xx, yy) = resize::meshgrid(&x, &y);

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
            let u_src = m_inv[[0, 0]] * u + m_inv[[0, 1]] * v + m_inv[[0, 2]];
            let v_src = m_inv[[1, 0]] * u + m_inv[[1, 1]] * v + m_inv[[1, 2]];

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
