use crate::{quad::Quad, utils::Pixel};
use kornia_image::{allocator::ImageAllocator, Image};

/// TODO
pub fn decode_tags() {}

/// TODO
pub fn refine_edges<A: ImageAllocator>(
    src: &Image<Pixel, 1, A>,
    quad: &mut Quad,
    reversed_border: bool,
) {
    let src_slice = src.as_slice();
    let mut lines: [[f32; 4]; 4] = Default::default();

    (0..4).for_each(|edge| {
        let a = edge;
        let b = (edge + 1) & 3;

        let mut nx = quad.corners[b].y - quad.corners[a].y;
        let mut ny = quad.corners[b].x - quad.corners[a].y;
        let mag = (nx * nx + ny * ny).sqrt();
        nx /= mag;
        ny /= mag;

        if reversed_border {
            nx = -nx;
            ny = -ny;
        }

        let nsamples = 16.max((mag / 8.0) as usize);

        let mut mx = 0f32;
        let mut my = 0f32;
        let mut mxx = 0f32;
        let mut mxy = 0f32;
        let mut myy = 0f32;
        let mut n = 0f32;

        (0..nsamples).for_each(|s| {
            let alpha = (1 + s) as f32 / (nsamples + 1) as f32;
            let x0 = alpha * quad.corners[a].x + (1.0 - alpha) * quad.corners[b].x;
            let y0 = alpha * quad.corners[a].y + (1.0 - alpha) * quad.corners[b].y;

            let mut mn = 0f32;
            let mut m_count = 0f32;

            const RANGE: usize = 2; // TODO: Make it tuneable. It will depend on the downscaling factor of the image preprocessing.

            let steps_per_unit = 4;
            let step_length = 1.0 / steps_per_unit as f32;
            let max_steps = 2 * steps_per_unit * RANGE + 1;
            let delta = 0.5f32;

            (0..max_steps).for_each(|step| {
                let n = step_length * step as f32 - RANGE as f32;
                let grange = 1f32;

                let x1 = x0 + (n + grange) * nx - delta;
                let y1 = y0 + (n + grange) * ny - delta;

                let x1i = x1.trunc() as isize;
                let y1i = y1.trunc() as isize;
                let a1 = x1.fract();
                let b1 = y1.fract();

                if x1i < 0
                    || x1i + 1 >= src.width() as isize
                    || y1i < 0
                    || y1i + 1 >= src.height() as isize
                {
                    return;
                }

                let x1i = x1i as usize;
                let y1i = y1i as usize;

                let x2 = x0 + (n - grange) * nx - delta;
                let y2 = y0 + (n - grange) * ny - delta;

                let x2i = x2.trunc() as isize;
                let y2i = y2.trunc() as isize;
                let a2 = x2.fract();
                let b2 = y2.fract();

                if x2i < 0
                    || x2i + 1 >= src.width() as isize
                    || y2i < 0
                    || y2i + 1 >= src.height() as isize
                {
                    return;
                }

                let x2i = x2i as usize;
                let y2i = y2i as usize;

                let g1 = (1.0 - a1) * (1.0 - b1) * src_slice[y1i * src.width() + x1i] as u8 as f32
                    + a1 * (1.0 - b1) * src_slice[y1i * src.width() + x1i + 1] as u8 as f32
                    + (1.0 - a1) * b1 * src_slice[(y1i + 1) * src.width() + x1i] as u8 as f32
                    + a1 * b1 * src_slice[(y1i + 1) * src.width() + x1i + 1] as u8 as f32;

                let g2 = (1.0 - a2) * (1.0 - b2) * src_slice[y2i * src.width() + x2i] as u8 as f32
                    + a2 * (1.0 - b2) * src_slice[y2i * src.width() + x2i + 1] as u8 as f32
                    + (1.0 - a2) * b2 * src_slice[(y2i + 1) * src.width() + x2i] as u8 as f32
                    + a2 * b2 * src_slice[(y2i + 1) * src.width() + x2i + 1] as u8 as f32;

                if g1 < g2 {
                    return;
                }

                let weight = (g2 - g1) * (g2 - g1);

                mn += weight * n;
                m_count += weight;
            });

            if m_count == 0.0 {
                return;
            }

            let n0 = mn / m_count;

            let best_x = x0 + n0 * nx;
            let best_y = y0 + n0 * ny;

            mx += best_x;
            my += best_y;
            mxx += best_x * best_x;
            mxy += best_x * best_y;
            myy += best_y * best_y;
            n += 1.0;
        });

        let ex = mx / n;
        let ey = my / n;
        let cxx = mxx / n - ex * ex;
        let cxy = mxy / n - ex * ex;
        let cyy = myy / n - ey * ey;

        let normal_theta = 0.5 * (-2.0 * cxy).atan2(cyy - cxx);
        nx = normal_theta.cos();
        ny = normal_theta.sin();

        lines[edge][0] = ex;
        lines[edge][1] = ey;
        lines[edge][2] = nx;
        lines[edge][3] = ny;
    });

    // now refit the corners of the quad
    (0..4).for_each(|i| {
        let a00 = lines[i][3];
        let a01 = -lines[(i + 1) & 3][3];
        let a10 = -lines[i][2];
        let a11 = lines[(i + 1) & 3][2];
        let b0 = -lines[i][0] + lines[(i + 1) & 3][0];
        let b1 = -lines[i][1] + lines[(i + 1) & 3][1];

        let det = a00 * a11 - a10 * a01;

        if det.abs() > 0.001 {
            let w00 = a11 / det;
            let w01 = -a01 / det;

            let l0 = w00 * b0 + w01 * b1;

            quad.corners[(i + 1) & 3].x = lines[i][0] + l0 * a00;
            quad.corners[(i + 1) & 3].y = lines[i][1] + l0 * a00;
        }
    });
}
