use kornia_image::Image;
use rayon::prelude::*;

pub(crate) fn laplacian_u8_to_i16<const C: usize>(src: &Image<u8, C>, dst: &mut Image<i16, C>) {
    let rows = src.rows();
    let cols = src.cols();
    let row_stride = cols * C;

    let src_slice = src.as_slice();
    let dst_slice = dst.as_slice_mut();

    dst_slice
        .par_chunks_mut(row_stride)
        .enumerate()
        .for_each(|(y, dst_row)| {
            for x in 0..cols {
                let x_left = x.saturating_sub(1);
                let x_right = (x + 1).min(cols - 1);
                let y_up = y.saturating_sub(1);
                let y_down = (y + 1).min(rows - 1);

                for c in 0..C {
                    let v_center = src_slice[(y * cols + x) * C + c] as i16;
                    let v_up = src_slice[(y_up * cols + x) * C + c] as i16;
                    let v_down = src_slice[(y_down * cols + x) * C + c] as i16;
                    let v_left = src_slice[(y * cols + x_left) * C + c] as i16;
                    let v_right = src_slice[(y * cols + x_right) * C + c] as i16;

                    // Laplacian 3x3: [0, 1, 0; 1, -4, 1; 0, 1, 0]
                    let val = v_up + v_down + v_left + v_right - 4 * v_center;
                    dst_row[x * C + c] = val;
                }
            }
        });
}
