use kornia_image::Image;
use rayon::prelude::*;

#[inline(always)]
fn integral_image_two_pass<T, const C: usize>(src: &Image<T, C>, dst: &mut Image<f32, C>)
where
    T: Copy + Into<f32> + Sync + Send,
{
    let rows = src.rows();
    let cols = src.cols();
    let row_stride = cols * C;

    let src_slice = src.as_slice();
    let dst_slice = dst.as_slice_mut();

    // Pass 1: Horizontal prefix sum (independent per row)
    // We compute R(x, y) = src(x, y) + R(x-1, y)
    src_slice
        .par_chunks(row_stride)
        .zip(dst_slice.par_chunks_mut(row_stride))
        .for_each(|(src_row, dst_row)| {
            if C == 1 {
                let mut sum = 0.0;
                for (s, d) in src_row.iter().zip(dst_row.iter_mut()) {
                    sum += (*s).into();
                    *d = sum;
                }
            } else if C == 3 {
                let mut sum0 = 0.0;
                let mut sum1 = 0.0;
                let mut sum2 = 0.0;
                for i in (0..src_row.len()).step_by(3) {
                    sum0 += src_row[i].into();
                    sum1 += src_row[i + 1].into();
                    sum2 += src_row[i + 2].into();
                    dst_row[i] = sum0;
                    dst_row[i + 1] = sum1;
                    dst_row[i + 2] = sum2;
                }
            } else {
                let mut sums = vec![0.0; C];
                for i in (0..src_row.len()).step_by(C) {
                    for c in 0..C {
                        sums[c] += src_row[i + c].into();
                    }
                    dst_row[i..(C + i)].copy_from_slice(&sums[..C]);
                }
            }
        });

    // Pass 2: Vertical prefix sum
    // I(x, y) = R(x, y) + I(x, y-1)
    // Since each column is independent, we can chunk horizontally,
    // but we can't easily mutably borrow disjoint vertical strips without unsafe.
    // However, if we just iterate sequentially over rows, we can do it safely.
    // For large images, we can use unsafe to split the buffer into vertical strips for Rayon.
    // For now, to be safe and simple, we do a sequential vertical sum over rows.
    // Modern CPUs will prefetch this row-by-row sweep very efficiently.
    for y in 1..rows {
        let (prev_rows, curr_rows) = dst_slice.split_at_mut(y * row_stride);
        let prev_row = &prev_rows[(y - 1) * row_stride..];
        let curr_row = &mut curr_rows[..row_stride];

        for (c, p) in curr_row.iter_mut().zip(prev_row.iter()) {
            *c += *p;
        }
    }
}

pub(crate) fn integral_image_u8_to_f32<const C: usize>(
    src: &Image<u8, C>,
    dst: &mut Image<f32, C>,
) {
    integral_image_two_pass(src, dst);
}

pub(crate) fn integral_image_f32_to_f32<const C: usize>(
    src: &Image<f32, C>,
    dst: &mut Image<f32, C>,
) {
    integral_image_two_pass(src, dst);
}
