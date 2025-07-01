use std::{f32::consts::PI, ops::ControlFlow};

use crate::{
    family::TagFamily,
    quad::Quad,
    utils::{
        homography_compute, inverse_3x3_matrix, matrix_3x3_cholesky,
        matrix_3x3_lower_triange_inverse, matrix_3x3_mul, value_for_pixel, Point2d,
    },
};
use kornia_image::{allocator::ImageAllocator, Image};

/// TODO
#[derive(Debug, Default, Clone)]
pub struct GrayModel {
    /// TODO
    pub a: [[f32; 3]; 3],
    /// TODO
    pub b: [f32; 3],
    /// TODO
    pub c: [f32; 3],
}

impl GrayModel {
    /// TODO
    pub fn add(&mut self, x: f32, y: f32, gray: f32) {
        self.a[0][0] += x * x;
        self.a[0][1] += x * y;
        self.a[0][2] += x;
        self.a[1][1] += y * y;
        self.a[1][2] += y;
        self.a[2][2] += 1.0;

        self.b[0] += x * gray;
        self.b[1] += y * gray;
        self.b[2] += gray;
    }

    /// TODO
    pub fn solve(&mut self) {
        let mut l = [0f32; 9];
        matrix_3x3_cholesky(&self.a, &mut l);

        let mut m = [0f32; 9];
        matrix_3x3_lower_triange_inverse(&l, &mut m);

        let tmp = [
            m[0] * self.b[0],
            m[3] * self.b[0] + m[4] * self.b[1],
            m[6] * self.b[0] + m[7] * self.b[1] + m[8] * self.b[2],
        ];

        self.c[0] = m[0] * tmp[0] + m[3] * tmp[1] + m[6] * tmp[2];
        self.c[1] = m[4] * tmp[1] + m[7] * tmp[2];
        self.c[2] = m[8] * tmp[2];
    }

    /// TODO
    pub fn interpolate(&self, x: f32, y: f32) -> f32 {
        self.c[0] * x + self.c[1] * y + self.c[2]
    }
}

/// TODO
#[derive(Debug, Default, Clone)]
pub struct QuickDecodeEntry {
    /// TODO
    pub rcode: usize,
    /// TODO
    pub id: u16,
    /// TODO
    pub hamming: u8,
    /// TODO
    pub rotation: u8,
}

/// TODO
pub struct QuickDecode(Vec<QuickDecodeEntry>);

/// TODO
#[derive(Debug, Clone, PartialEq)]
pub struct Detection<'a> {
    /// TODO
    pub tag_family: &'a TagFamily,
    /// TODO
    pub id: u16,
    /// TODO
    pub hamming: u8,
    /// TODO
    pub decision_margin: f32,
    /// TODO
    pub center: Point2d<f32>,
    /// TODO
    pub quad: Quad,
}

impl QuickDecode {
    /// TODO
    // TODO: Support multiple tag familes. The current logic needs to be changed then
    pub fn new(tag_family: &TagFamily) -> Self {
        let ncodes = tag_family.code_data.len();
        let capacity =
            ncodes + tag_family.nbits * ncodes + ncodes * tag_family.nbits * (tag_family.nbits - 1);

        let mut quick_decode = Self(vec![
            QuickDecodeEntry {
                rcode: usize::MAX,
                ..Default::default()
            };
            capacity * 3
        ]);

        tag_family
            .code_data
            .iter()
            .enumerate()
            .for_each(|(i, code)| {
                quick_decode.add(*code, i as u16, 0);

                // add hamming 1
                (0..tag_family.nbits).for_each(|j| {
                    quick_decode.add(code ^ (1 << j), i as u16, 1);
                });

                // add hamming 2
                (0..tag_family.nbits).for_each(|j| {
                    (0..j).for_each(|k| {
                        quick_decode.add(code ^ (1 << j) ^ (1 << k), i as u16, 2);
                    });
                });
            });

        quick_decode
    }

    /// TODO
    pub fn add(&mut self, code: usize, id: u16, hamming: u8) {
        let mut bucket = code % self.0.len();

        // TODO: Use iterators instead
        while self.0[bucket].rcode != usize::MAX {
            bucket = (bucket + 1) % self.0.len();
        }

        self.0[bucket].rcode = code;
        self.0[bucket].id = id;
        self.0[bucket].hamming = hamming;
    }
}

/// TODO
// TODO: Add support for multiple tag families
pub fn decode_tags<'a, A: ImageAllocator>(
    src: &Image<u8, 1, A>,
    quads: &mut [Quad],
    tag_family: &'a TagFamily,
    quick_decode: &mut QuickDecode,
    refine_edges_param: bool,
) -> Vec<Detection<'a>> {
    let mut detections = Vec::new();

    quads.iter_mut().for_each(|quad| {
        if refine_edges_param {
            refine_edges(src, quad, tag_family.reversed_border);
        }

        if !quad_update_homographies(quad) {
            return;
        }

        if tag_family.reversed_border != quad.reversed_border {
            return;
        }

        let mut quad_clone = quad.clone();
        let mut entry = QuickDecodeEntry::default();

        let decision_margin = quad_decode(src, tag_family, &quad_clone, &mut entry, quick_decode);

        if let Some(decision_margin) = decision_margin {
            if decision_margin >= 0.0 && entry.hamming < u8::MAX {
                let theta = entry.rotation as f32 * PI / 2.0;
                let c = theta.cos();
                let s = theta.sin();

                // Fix the rotation of our homography to properly orient the tag
                #[rustfmt::skip]
                let r = [
                    c,  -s,   0.0,
                    s,   c,   0.0,
                    0.0, 0.0, 1.0,
                ];

                quad_clone.h = matrix_3x3_mul(&quad_clone.h, &r);
                let center = quad_clone.homography_project(0.0, 0.0);

                let detection = Detection {
                    tag_family,
                    id: entry.id,
                    hamming: entry.hamming,
                    decision_margin,
                    quad: quad_clone,
                    center,
                };

                detections.push(detection);
            }
        }
    });

    detections
}

/// TODO
pub fn refine_edges<A: ImageAllocator>(
    src: &Image<u8, 1, A>,
    quad: &mut Quad,
    reversed_border: bool,
) {
    let src_slice = src.as_slice();
    let mut lines: [[f32; 4]; 4] = Default::default();

    (0..4).for_each(|edge| {
        let a = edge;
        let b = (edge + 1) & 3;

        let mut nx = quad.corners[b].y - quad.corners[a].y;
        let mut ny = -quad.corners[b].x + quad.corners[a].x;
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

                let g1 = (1.0 - a1) * (1.0 - b1) * src_slice[y1i * src.width() + x1i] as f32
                    + a1 * (1.0 - b1) * src_slice[y1i * src.width() + x1i + 1] as f32
                    + (1.0 - a1) * b1 * src_slice[(y1i + 1) * src.width() + x1i] as f32
                    + a1 * b1 * src_slice[(y1i + 1) * src.width() + x1i + 1] as f32;

                let g2 = (1.0 - a2) * (1.0 - b2) * src_slice[y2i * src.width() + x2i] as f32
                    + a2 * (1.0 - b2) * src_slice[y2i * src.width() + x2i + 1] as f32
                    + (1.0 - a2) * b2 * src_slice[(y2i + 1) * src.width() + x2i] as f32
                    + a2 * b2 * src_slice[(y2i + 1) * src.width() + x2i + 1] as f32;

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

/// TODO
pub fn quad_update_homographies(quad: &mut Quad) -> bool {
    let mut corr_arr: [[f32; 4]; 4] = Default::default();

    // TODO: Directly assign the array instead of iterating
    (0..4).for_each(|i| {
        corr_arr[i][0] = if i == 0 || i == 3 { -1.0 } else { 1.0 };
        corr_arr[i][1] = if i == 0 || i == 1 { -1.0 } else { 1.0 };
        corr_arr[i][2] = quad.corners[i].x;
        corr_arr[i][3] = quad.corners[i].y;
    });

    if let Some(h) = homography_compute(corr_arr) {
        quad.h_inv = inverse_3x3_matrix(&h);
        quad.h = h;

        return true;
    }

    false
}

/// TODO
// returns the decision margin
pub fn quad_decode<A: ImageAllocator>(
    src: &Image<u8, 1, A>,
    tag_family: &TagFamily,
    quad: &Quad,
    entry: &mut QuickDecodeEntry,
    quick_decode: &mut QuickDecode,
) -> Option<f32> {
    #[rustfmt::skip]
    let patterns = [
        // left white column
        -0.5, 0.5,
        0.0, 1.0,
        1.0,

        // left black column
        0.5, 0.5,
        0.0, 1.0,
        0.0,

        // right white column
        tag_family.width_at_border as f32 + 0.5, 0.5,
        0.0, 1.0,
        1.0,

        // right black column
        tag_family.width_at_border as f32 - 0.5, 0.5,
        0.0, 1.0,
        0.0,

        // top white row
        0.5, -0.5,
        1.0, 0.0,
        1.0,

        // top black row
        0.5, 0.5,
        1.0, 0.0,
        0.0,

        // bottom white row
        0.5, tag_family.width_at_border as f32 + 0.5,
        1.0, 0.0,
        0.0,

        // bottom black row
        0.5, tag_family.width_at_border as f32 - 0.5,
        1.0, 0.0,
        0.0,
    ];

    let src_slice = src.as_slice();

    let mut white_model = GrayModel::default();
    let mut black_model = GrayModel::default();

    (0..patterns.len() / 5).for_each(|pattern_idx| {
        let pattern_start = pattern_idx * 5;
        let pattern = &patterns[pattern_start..pattern_start + 5];
        let is_white = match pattern[4] {
            1.0 => true,
            0.0 => false,
            _ => unreachable!(),
        };

        (0..tag_family.width_at_border).for_each(|i| {
            let tagx01 = (pattern[0] + i as f32 * pattern[2]) / tag_family.width_at_border as f32;
            let tagy01 = (pattern[1] + i as f32 * pattern[3]) / tag_family.width_at_border as f32;

            let tagx = 2.0 * (tagx01 - 0.5);
            let tagy = 2.0 * (tagy01 - 0.5);

            let p = quad.homography_project(tagx, tagy);

            if p.x.trunc() < 0.0 || p.y.trunc() < 0.0 {
                return;
            }

            let p = Point2d {
                x: p.x as usize,
                y: p.y as usize,
            };

            if p.x >= src.width() || p.y >= src.height() {
                return;
            }

            let v = src_slice[p.y * src.width() + p.x] as f32;

            if is_white {
                white_model.add(tagx, tagy, v);
            } else {
                black_model.add(tagx, tagy, v);
            }
        });
    });

    white_model.solve();
    if tag_family.width_at_border > 1 {
        black_model.solve();
    } else {
        black_model.c[0] = 0.0;
        black_model.c[1] = 0.0;
        black_model.c[2] = black_model.b[2] / 4.0;
    }

    if (white_model.interpolate(0.0, 0.0) - black_model.interpolate(0.0, 0.0) < 0.0)
        != tag_family.reversed_border
    {
        return None;
    }

    let mut black_score = 0f32;
    let mut white_score = 0f32;
    let mut black_score_count = 1usize;
    let mut white_score_count = 1usize;

    // TODO: Avoid this allocation every time
    let mut values = vec![0f32; tag_family.total_width * tag_family.total_width];

    let min_coord = (tag_family.width_at_border as isize - tag_family.total_width as isize) / 2;

    (0..tag_family.nbits).for_each(|i| {
        let bit_x = tag_family.bit_x[i];
        let bit_y = tag_family.bit_y[i];

        let tag_x01 = (bit_x as f32 + 0.5) / tag_family.width_at_border as f32;
        let tag_y01 = (bit_y as f32 + 0.5) / tag_family.width_at_border as f32;

        let tag_x = 2.0 * (tag_x01 - 0.5);
        let tag_y = 2.0 * (tag_y01 - 0.5);

        let p = quad.homography_project(tag_x, tag_y);

        let Some(v) = value_for_pixel(src, p) else {
            return;
        };

        let thresh =
            (black_model.interpolate(tag_x, tag_y) + white_model.interpolate(tag_x, tag_y)) / 2.0;

        values[(tag_family.total_width as isize * (bit_y as isize - min_coord) + bit_x as isize
            - min_coord) as usize] = v - thresh;
    });

    sharpen(&mut values, tag_family.total_width);

    let mut rcode = 0usize;
    (0..tag_family.nbits).for_each(|i| {
        let bit_y = tag_family.bit_y[i];
        let bit_x = tag_family.bit_x[i];

        rcode <<= 1;

        let v = values[((bit_y as isize - min_coord) * tag_family.total_width as isize
            + bit_x as isize
            - min_coord) as usize];

        if v > 0.0 {
            white_score += v;
            white_score_count += 1;
            rcode |= 1;
        } else {
            black_score -= v;
            black_score_count += 1;
        }
    });

    quick_decode_codeword(tag_family, &mut rcode, entry, quick_decode);

    Some((white_score / white_score_count as f32).min(black_score / black_score_count as f32))
}

const DECODE_SHARPENING: f32 = 0.25; // TODO: Make it tuneable in sharpen function

/// TODO
pub fn sharpen(values: &mut [f32], size: usize) {
    // TODO: Avoid allocation
    let mut sharpened = vec![0f32; size * size];

    #[rustfmt::skip]
    const KERNEL: [f32; 9] = [
         0.0, -1.0,  0.0,
        -1.0,  4.0, -1.0,
         0.0, -1.0,  0.0,
    ];

    (0..size as isize).for_each(|y| {
        (0..size as isize).for_each(|x| {
            sharpened[(y * size as isize + x) as usize] = 0.0;

            (0..3isize).for_each(|i| {
                (0..3isize).for_each(|j| {
                    if (y + i - 1) < 0
                        || (y + i - 1) > size as isize - 1
                        || (x + j - 1) < 0
                        || (x + j - 1) > size as isize - 1
                    {
                        return;
                    }
                    sharpened[(y * size as isize + x) as usize] += values
                        [((y + i - 1) * size as isize + (x + j - 1)) as usize]
                        * KERNEL[(i * 3 + j) as usize];
                });
            });
        });
    });

    (0..size).for_each(|y| {
        (0..size).for_each(|x| {
            values[y * size + x] += DECODE_SHARPENING * sharpened[y * size + x];
        });
    });
}

/// TODO
pub fn quick_decode_codeword(
    tag_family: &TagFamily,
    rcode: &mut usize,
    entry: &mut QuickDecodeEntry,
    quick_decode: &mut QuickDecode,
) {
    if let ControlFlow::Break(_) = (0..4).try_for_each(|ridx| {
        let mut bucket = *rcode % quick_decode.0.len();

        while quick_decode.0[bucket].rcode != usize::MAX {
            if quick_decode.0[bucket].rcode == *rcode {
                *entry = quick_decode.0[bucket].clone();
                entry.rotation = ridx;

                return ControlFlow::Break(());
            }

            bucket = (bucket + 1) % quick_decode.0.len();
        }

        *rcode = rotate_90(*rcode, tag_family.nbits);

        ControlFlow::Continue(())
    }) {
        return;
    }

    entry.rcode = 0;
    entry.id = u16::MAX;
    entry.hamming = 255;
    entry.rotation = 0;
}

fn rotate_90(mut w: usize, num_bits: usize) -> usize {
    let mut p = num_bits;
    let mut l = 0;

    if num_bits % 4 == 1 {
        p -= 1;
        l = 1;
    }

    w = ((w >> l) << (p / 4 + l)) | (w >> (3 * p / 4 + l) << l) | (w & l);
    w &= (1 << num_bits) - 1;

    w
}
