use cubecl::prelude::*;

/// GPU CubeCL kernel for applying an affine transformation to an image.
///
/// # Arguments
///
/// * `src` - Input image tensor.
/// * `dst` - Output image tensor.
/// * `m_inv` - Inverse 2x3 affine matrix tensor.
/// * `src_cols` - Width of the source image.
/// * `src_rows` - Height of the source image.
/// * `dst_cols` - Width of the target image.
/// * `dst_rows` - Height of the target image.
/// * `channels` - Number of image channels.
#[cube(launch_unchecked)]
pub fn warp_affine_kernel<F: Float>(
    src: &Tensor<F>,
    dst: &mut Tensor<F>,
    m_inv: &Tensor<F>,
    src_cols: u32,
    src_rows: u32,
    dst_cols: u32,
    dst_rows: u32,
    channels: u32,
) {
    let x = ABSOLUTE_POS_X;
    let y = ABSOLUTE_POS_Y;

    if x >= dst_cols || y >= dst_rows {
        return;
    }

    let x_f = F::cast_from(x);
    let y_f = F::cast_from(y);

    let m0 = m_inv[0];
    let m1 = m_inv[1];
    let m2 = m_inv[2];
    let m3 = m_inv[3];
    let m4 = m_inv[4];
    let m5 = m_inv[5];

    let src_x = m0 * x_f + m1 * y_f + m2;
    let src_y = m3 * x_f + m4 * y_f + m5;

    // Bilinear interpolation
    if src_x >= F::new(0.0)
        && src_x < F::cast_from(src_cols)
        && src_y >= F::new(0.0)
        && src_y < F::cast_from(src_rows)
    {
        let x0 = F::floor(src_x);
        let y0 = F::floor(src_y);
        let x1 = x0 + F::new(1.0);
        let y1 = y0 + F::new(1.0);

        let dx = src_x - x0;
        let dy = src_y - y0;

        let ix0 = u32::cast_from(x0);
        let iy0 = u32::cast_from(y0);
        let ix1 = u32::min(ix0 + 1, src_cols - 1);
        let iy1 = u32::min(iy0 + 1, src_rows - 1);

        for c in 0..channels {
            let idx00 = (iy0 * src_cols + ix0) * channels + c;
            let idx10 = (iy0 * src_cols + ix1) * channels + c;
            let idx01 = (iy1 * src_cols + ix0) * channels + c;
            let idx11 = (iy1 * src_cols + ix1) * channels + c;

            let val00 = src[idx00];
            let val10 = src[idx10];
            let val01 = src[idx01];
            let val11 = src[idx11];

            let val0 = val00 * (F::new(1.0) - dx) + val10 * dx;
            let val1 = val01 * (F::new(1.0) - dx) + val11 * dx;
            let val = val0 * (F::new(1.0) - dy) + val1 * dy;

            let dst_idx = (y * dst_cols + x) * channels + c;
            dst[dst_idx] = val;
        }
    }
}

/// GPU CubeCL kernel for applying a perspective transformation to an image.
///
/// # Arguments
///
/// * `src` - Input image tensor.
/// * `dst` - Output image tensor.
/// * `m_inv` - Inverse 3x3 perspective matrix tensor.
/// * `src_cols` - Width of the source image.
/// * `src_rows` - Height of the source image.
/// * `dst_cols` - Width of the target image.
/// * `dst_rows` - Height of the target image.
/// * `channels` - Number of image channels.
#[cube(launch_unchecked)]
pub fn warp_perspective_kernel<F: Float>(
    src: &Tensor<F>,
    dst: &mut Tensor<F>,
    m_inv: &Tensor<F>,
    src_cols: u32,
    src_rows: u32,
    dst_cols: u32,
    dst_rows: u32,
    channels: u32,
) {
    let x = ABSOLUTE_POS_X;
    let y = ABSOLUTE_POS_Y;

    if x >= dst_cols || y >= dst_rows {
        return;
    }

    let x_f = F::cast_from(x);
    let y_f = F::cast_from(y);

    let m0 = m_inv[0];
    let m1 = m_inv[1];
    let m2 = m_inv[2];
    let m3 = m_inv[3];
    let m4 = m_inv[4];
    let m5 = m_inv[5];
    let m6 = m_inv[6];
    let m7 = m_inv[7];
    let m8 = m_inv[8];

    let w = m6 * x_f + m7 * y_f + m8;
    let src_x = (m0 * x_f + m1 * y_f + m2) / w;
    let src_y = (m3 * x_f + m4 * y_f + m5) / w;

    // Bilinear interpolation
    if src_x >= F::new(0.0)
        && src_x < F::cast_from(src_cols)
        && src_y >= F::new(0.0)
        && src_y < F::cast_from(src_rows)
    {
        let x0 = F::floor(src_x);
        let y0 = F::floor(src_y);
        let x1 = x0 + F::new(1.0);
        let y1 = y0 + F::new(1.0);

        let dx = src_x - x0;
        let dy = src_y - y0;

        let ix0 = u32::cast_from(x0);
        let iy0 = u32::cast_from(y0);
        let ix1 = u32::min(ix0 + 1, src_cols - 1);
        let iy1 = u32::min(iy0 + 1, src_rows - 1);

        for c in 0..channels {
            let idx00 = (iy0 * src_cols + ix0) * channels + c;
            let idx10 = (iy0 * src_cols + ix1) * channels + c;
            let idx01 = (iy1 * src_cols + ix0) * channels + c;
            let idx11 = (iy1 * src_cols + ix1) * channels + c;

            let val00 = src[idx00];
            let val10 = src[idx10];
            let val01 = src[idx01];
            let val11 = src[idx11];

            let val0 = val00 * (F::new(1.0) - dx) + val10 * dx;
            let val1 = val01 * (F::new(1.0) - dx) + val11 * dx;
            let val = val0 * (F::new(1.0) - dy) + val1 * dy;

            let dst_idx = (y * dst_cols + x) * channels + c;
            dst[dst_idx] = val;
        }
    }
}
