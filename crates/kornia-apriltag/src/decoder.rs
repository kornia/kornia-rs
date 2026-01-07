use std::{f32::consts::PI, ops::ControlFlow};

use crate::{
    family::{TagFamily, TagFamilyKind},
    quad::Quad,
    utils::{
        matrix_3x3_cholesky, matrix_3x3_lower_triangle_inverse, matrix_3x3_mul, value_for_pixel,
        Point2d,
    },
    DecodeTagsConfig,
};
use kornia_image::{allocator::ImageAllocator, Image};

/// Represents a model for grayscale interpolation using a quadratic surface.
/// The model fits a function of the form f(x, y) = c[0]*x + c[1]*y + c[2].
#[derive(Debug, Default, Clone, PartialEq)]
struct GrayModel {
    /// The 3x3 matrix of accumulated quadratic terms.
    a: [[f32; 3]; 3],
    /// The vector of accumulated linear terms.
    b: [f32; 3],
    /// The solved coefficients for the quadratic model.
    c: [f32; 3],
}

impl GrayModel {
    /// Adds a new data point (x, y, gray) to the quadratic model.
    fn add(&mut self, x: f32, y: f32, gray: f32) {
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

    /// Solves the quadratic model to find the coefficients.
    fn solve(&mut self) {
        let mut l = [0f32; 9];
        matrix_3x3_cholesky(&self.a, &mut l);

        let mut m = [0f32; 9];
        matrix_3x3_lower_triangle_inverse(&l, &mut m);

        let tmp = [
            m[0] * self.b[0],
            m[3] * self.b[0] + m[4] * self.b[1],
            m[6] * self.b[0] + m[7] * self.b[1] + m[8] * self.b[2],
        ];

        self.c[0] = m[0] * tmp[0] + m[3] * tmp[1] + m[6] * tmp[2];
        self.c[1] = m[4] * tmp[1] + m[7] * tmp[2];
        self.c[2] = m[8] * tmp[2];
    }

    /// Interpolates the grayscale value at the given (x, y) using the solved coefficients.
    fn interpolate(&self, x: f32, y: f32) -> f32 {
        self.c[0] * x + self.c[1] * y + self.c[2]
    }
}

/// Holds a pair of grayscale quadratic models for white and black regions.
#[derive(Debug, Default, PartialEq)]
pub struct GrayModelPair {
    white_model: GrayModel,
    black_model: GrayModel,
}

impl GrayModelPair {
    /// Creates a new `GrayModelPair` with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Resets both the white and black grayscale models to their initial state.
    pub fn reset(&mut self) {
        for i in 0..3 {
            for j in 0..3 {
                self.white_model.a[i][j] = 0.0;
                self.black_model.a[i][j] = 0.0;
            }
            self.white_model.b[i] = 0.0;
            self.white_model.c[i] = 0.0;
            self.black_model.b[i] = 0.0;
            self.black_model.c[i] = 0.0;
        }
    }
}

/// Represents an entry in the quick decode table for tag decoding.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct QuickDecodeEntry {
    /// The raw code value associated with the tag.
    pub rcode: usize,
    /// The decoded tag ID.
    pub id: u16,
    /// The Hamming distance for this code.
    pub hamming: u8,
    /// The rotation (in 90-degree increments) of the tag.
    pub rotation: u8,
}

/// Sentinel value used to mark unoccupied slots in the quick decode hash table.
const SLOT_EMPTY: u16 = u16::MAX;

/// A table for fast lookup of decoded tag codes and their associated metadata.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct QuickDecode {
    table: Vec<u16>,
}

impl QuickDecode {
    /// Creates a new `QuickDecode` table for fast lookup of decoded tag codes and their associated metadata.
    ///
    /// # Arguments
    ///
    /// * `nbits` - Number of bits in the tag code.
    /// * `code_data` - Slice of code values to populate the table.
    ///
    /// # Returns
    ///
    /// A new `QuickDecode` instance with precomputed entries for all codes and their Hamming neighbors.
    pub fn new(nbits: usize, code_data: &[usize]) -> Self {
        let ncodes = code_data.len();
        let entries_needed = ncodes // Hamming 0
            + nbits * ncodes // Hamming 1
            + ncodes * nbits * (nbits - 1) / 2; // Hamming 2

        let capacity = entries_needed * 3;

        let mut quick_decode = Self {
            table: vec![SLOT_EMPTY; capacity],
        };

        code_data.iter().enumerate().for_each(|(i, code)| {
            let id = i as u16;
            quick_decode.add(*code, id);

            // add hamming 1
            (0..nbits).for_each(|j| {
                quick_decode.add(code ^ (1 << j), id);
            });

            // add hamming 2
            (0..nbits).for_each(|j| {
                (0..j).for_each(|k| {
                    quick_decode.add(code ^ (1 << j) ^ (1 << k), id);
                });
            });
        });
        quick_decode
    }

    /// Adds a new entry to the quick decode table.
    ///
    /// # Arguments
    ///
    /// * `code` - The code value to add.
    /// * `id` - The tag ID associated with the code.
    fn add(&mut self, code: usize, id: u16) {
        let len = self.table.len();
        let mut bucket = code % len;

        while self.table[bucket] != SLOT_EMPTY {
            bucket = (bucket + 1) % len;
        }

        self.table[bucket] = id;
    }

    /// Decodes an observed code using the quick decode table.
    ///
    /// # Arguments
    ///
    /// * `observed_code` - The code value to decode.
    /// * `valid_codes` - Slice of valid codes for the tag family.
    ///
    /// # Returns
    ///
    /// Returns `Some(QuickDecodeEntry)` if the code is found within a Hamming distance of 2, or `None` otherwise.
    pub fn decode(&self, observed_code: usize, valid_codes: &[usize]) -> Option<QuickDecodeEntry> {
        let len = self.table.len();
        let mut bucket = observed_code % len;

        loop {
            let candidate_id = self.table[bucket];

            if candidate_id == SLOT_EMPTY {
                return None;
            }

            let perfect_code = valid_codes[candidate_id as usize];
            let delta = observed_code ^ perfect_code;

            let hamming = delta.count_ones();

            if hamming <= 2 {
                return Some(QuickDecodeEntry {
                    rcode: observed_code,
                    id: candidate_id,
                    hamming: hamming as u8,
                    rotation: 0,
                });
            }

            bucket = (bucket + 1) % len;
        }
    }
}

/// Represents a detected tag in the image, including its family, ID, decoding quality, and geometric information.
#[derive(Debug, Clone, PartialEq)]
pub struct Detection {
    /// Reference to the tag family this detection belongs to.
    pub tag_family_kind: TagFamilyKind,
    /// The decoded tag ID.
    pub id: u16,
    /// The Hamming distance of the detected code to the closest valid code.
    pub hamming: u8,
    /// The decision margin indicating the confidence of the detection.
    pub decision_margin: f32,
    /// The center point of the detected tag in image coordinates.
    pub center: Point2d<f32>,
    /// The quadrilateral representing the detected tag's corners in the image.
    pub quad: Quad,
}

/// Buffer used for storing intermediate values during the sharpening process.
#[derive(Debug, PartialEq, Clone)]
pub struct SharpeningBuffer {
    values: Vec<f32>,
    sharpened: Vec<f32>,
}

impl SharpeningBuffer {
    /// Creates a new `SharpeningBuffer` with the specified length.
    ///
    /// # Arguments
    ///
    /// * `len` - The length of the buffer arrays.
    ///
    /// # Returns
    ///
    /// A new `SharpeningBuffer` instance with allocated buffers of size `len`.
    pub fn new(len: usize) -> Self {
        Self {
            values: vec![0.0; len],
            sharpened: vec![0.0; len],
        }
    }

    /// Resets the buffer values to zero.
    pub fn reset(&mut self) {
        (0..self.values.len()).for_each(|i| {
            self.values[i] = 0.0;
            // No need to reset `sharpened` here as it is reset for every quad
        });
    }
}

/// Decodes tags from a grayscale image using detected quadrilaterals and configuration parameters.
///
/// # Arguments
///
/// * `src` - Reference to the grayscale source image.
/// * `quads` - Mutable slice of detected quadrilaterals to process.
/// * `config` - Reference to the tag decoding configuration.
/// * `gray_model_pair` - Mutable reference to a pair of grayscale models for white and black regions.
/// * `sharpening_buffer` - Mutable reference to a buffer used for sharpening intermediate values.
///
/// # Returns
///
/// Returns a vector of `Detection` containing information about each successfully decoded tag.
pub fn decode_tags<A: ImageAllocator>(
    src: &Image<u8, 1, A>,
    quads: &mut [Quad],
    config: &mut DecodeTagsConfig,
    gray_model_pair: &mut GrayModelPair,
) -> Vec<Detection> {
    // TODO: Avoid allocations on every call
    let mut detections = Vec::new();

    quads.iter_mut().for_each(|quad| {
        if config.refine_edges_enabled {
            refine_edges(src, quad);
        }

        if !quad.update_homographies() {
            return;
        }

        config.tag_families.iter_mut().for_each(|family| {
            if family.reversed_border != quad.reversed_border {
                return;
            }

            let mut entry = QuickDecodeEntry::default();

            let decision_margin = quad_decode(
                src,
                family,
                quad,
                config.decode_sharpening,
                &mut entry,
                gray_model_pair,
            );

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

                    quad.homography = matrix_3x3_mul(&quad.homography, &r);
                    let center = quad.homography_project(0.0, 0.0);

                    let detection = Detection {
                        tag_family_kind: family.into(),
                        id: entry.id,
                        hamming: entry.hamming,
                        decision_margin,
                        center,
                        quad: std::mem::take(quad), // TODO: Should we take or copy the value? Benchmark it
                    };

                    detections.push(detection);
                }
            }
        });
    });

    detections
}

/// Refines the edges of a quadrilateral in the image by adjusting its corners based on local image gradients.
///
/// # Arguments
///
/// * `src` - Reference to the grayscale source image.
/// * `quad` - Mutable reference to the quadrilateral to refine.
// TODO: Consider moving this somewhere in kornia-imgproc.
fn refine_edges<A: ImageAllocator>(src: &Image<u8, 1, A>, quad: &mut Quad) {
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

        if quad.reversed_border {
            nx = -nx;
            ny = -ny;
        }

        let nsamples = 16.max((mag / 8.0) as usize);

        let mut mx = 0.0;
        let mut my = 0.0;
        let mut mxx = 0.0;
        let mut mxy = 0.0;
        let mut myy = 0.0;
        let mut n = 0.0;

        (0..nsamples).for_each(|s| {
            let alpha = (1 + s) as f32 / (nsamples + 1) as f32;
            let x0 = alpha * quad.corners[a].x + (1.0 - alpha) * quad.corners[b].x;
            let y0 = alpha * quad.corners[a].y + (1.0 - alpha) * quad.corners[b].y;

            let mut mn = 0.0;
            let mut m_count = 0.0;

            const RANGE: f32 = 2.0; // TODO: Make it tuneable. It will depend on the downscaling factor of the image preprocessing.
            const STEPS_PER_UNIT: usize = 4;
            const STEP_LENGTH: f32 = 1.0 / STEPS_PER_UNIT as f32;
            const MAX_STEPS: usize = 2 * STEPS_PER_UNIT * RANGE as usize + 1;
            const DELTA: f32 = 0.5;

            const GRANGE: f32 = 1.0;

            (0..MAX_STEPS).for_each(|step| {
                let n = -RANGE + STEP_LENGTH * step as f32;

                let x1 = x0 + (n + GRANGE) * nx - DELTA;
                let y1 = y0 + (n + GRANGE) * ny - DELTA;

                let (x1i, a1) = (x1.trunc(), x1.fract());
                let (y1i, b1) = (y1.trunc(), y1.fract());

                if x1i < 0.0
                    || y1i < 0.0
                    || x1i + 1.0 >= src.width() as f32
                    || y1i + 1.0 >= src.height() as f32
                {
                    return;
                }

                let x2 = x0 + (n - GRANGE) * nx - DELTA;
                let y2 = y0 + (n - GRANGE) * ny - DELTA;

                let (x2i, a2) = (x2.trunc(), x2.fract());
                let (y2i, b2) = (y2.trunc(), y2.fract());

                if x2i < 0.0
                    || y2i < 0.0
                    || x2i + 1.0 >= src.width() as f32
                    || y2i + 1.0 >= src.height() as f32
                {
                    return;
                }

                let (x1i, x2i, y1i, y2i) = (x1i as usize, x2i as usize, y1i as usize, y2i as usize);

                let top_left_idx = y1i * src.width() + x1i;
                let bottom_left_idx = (y1i + 1) * src.width() + x1i;

                let g1 = (1.0 - a1) * (1.0 - b1) * src_slice[top_left_idx] as f32
                    + a1 * (1.0 - b1) * src_slice[top_left_idx + 1] as f32
                    + (1.0 - a1) * b1 * src_slice[bottom_left_idx] as f32
                    + a1 * b1 * src_slice[bottom_left_idx + 1] as f32;

                let top_left_idx = y2i * src.width() + x2i;
                let bottom_left_idx = (y2i + 1) * src.width() + x2i;

                let g2 = (1.0 - a2) * (1.0 - b2) * src_slice[top_left_idx] as f32
                    + a2 * (1.0 - b2) * src_slice[top_left_idx + 1] as f32
                    + (1.0 - a2) * b2 * src_slice[bottom_left_idx] as f32
                    + a2 * b2 * src_slice[bottom_left_idx + 1] as f32;

                if g1 < g2 {
                    return;
                }

                let weight = (g2 - g1).powi(2);

                mn += weight * n;
                m_count += weight;
            });

            if m_count <= 0.0 {
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
        let cxy = mxy / n - ex * ey;
        let cyy = myy / n - ey * ey;

        let normal_theta = 0.5 * (-2.0 * cxy).atan2(cyy - cxx);
        nx = normal_theta.cos();
        ny = normal_theta.sin();

        lines[edge][0] = ex;
        lines[edge][1] = ey;
        lines[edge][2] = nx;
        lines[edge][3] = ny;
    });

    // Now refit the corners of the quad
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
            quad.corners[(i + 1) & 3].y = lines[i][1] + l0 * a10;
        }
    });
}

/// Decodes a tag from a given quadrilateral in the image, using the provided tag family and quick decode table.
///
/// # Arguments
///
/// * `src` - Reference to the grayscale source image.
/// * `tag_family` - Reference to the tag family used for decoding.
/// * `quad` - Reference to the quadrilateral representing the tag in the image.
/// * `decode_sharpening` - Sharpening factor applied during decoding.
/// * `entry` - Mutable reference to a `QuickDecodeEntry` to store the decoding result.
/// * `gray_model_pair` - A mutable reference to a `GrayModelPair`.
/// * `sharpening_buffer` - A mutable reference to a `SharpeningBuffer`.
///
/// # Returns
///
/// Returns `Some(f32)` containing the decision margin if decoding is successful, or `None` otherwise.
fn quad_decode<A: ImageAllocator>(
    src: &Image<u8, 1, A>,
    tag_family: &mut TagFamily,
    quad: &Quad,
    decode_sharpening: f32,
    entry: &mut QuickDecodeEntry,
    gray_model_pair: &mut GrayModelPair,
) -> Option<f32> {
    struct Pattern {
        start_x: f32,
        start_y: f32,
        step_x: f32,
        step_y: f32,
        is_white: bool,
    }

    #[rustfmt::skip]
    let patterns = [
        // left white column
        Pattern {
            start_x: -0.5,
            start_y: 0.5,
            step_x: 0.0,
            step_y: 1.0,
            is_white: true
        },

        // left black column
        Pattern {
            start_x: 0.5,
            start_y: 0.5,
            step_x: 0.0,
            step_y: 1.0,
            is_white: false
        },

        // right white column
        Pattern {
            start_x: tag_family.width_at_border as f32 + 0.5,
            start_y: 0.5,
            step_x: 0.0,
            step_y: 1.0,
            is_white: true,
        },

        // right black column
        Pattern {
            start_x: tag_family.width_at_border as f32 - 0.5,
            start_y: 0.5,
            step_x: 0.0,
            step_y: 1.0,
            is_white: false,
        },

        // top white row
        Pattern {
            start_x: 0.5,
            start_y: -0.5,
            step_x: 1.0,
            step_y: 0.0,
            is_white: true,
        },

        // top black row
        Pattern {
            start_x: 0.5,
            start_y: 0.5,
            step_x: 1.0,
            step_y: 0.0,
            is_white: false
        },

        // bottom white row
        Pattern {
            start_x: 0.5,
            start_y: tag_family.width_at_border as f32 + 0.5,
            step_x: 1.0,
            step_y: 0.0,
            is_white: false,
        },

        // bottom black row
        Pattern {
            start_x: 0.5,
            start_y: tag_family.width_at_border as f32 - 0.5,
            step_x: 1.0,
            step_y: 0.0,
            is_white: false,
        }
    ];

    let src_slice = src.as_slice();

    patterns.iter().for_each(|pattern| {
        (0..tag_family.width_at_border).for_each(|i| {
            let tagx01 =
                (pattern.start_x + i as f32 * pattern.step_x) / tag_family.width_at_border as f32;
            let tagy01 =
                (pattern.start_y + i as f32 * pattern.step_y) / tag_family.width_at_border as f32;

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

            if pattern.is_white {
                gray_model_pair.white_model.add(tagx, tagy, v);
            } else {
                gray_model_pair.black_model.add(tagx, tagy, v);
            }
        });
    });

    gray_model_pair.white_model.solve();
    if tag_family.width_at_border > 1 {
        gray_model_pair.black_model.solve();
    } else {
        gray_model_pair.black_model.c[0] = 0.0;
        gray_model_pair.black_model.c[1] = 0.0;
        gray_model_pair.black_model.c[2] = gray_model_pair.black_model.b[2] / 4.0;
    }

    if (gray_model_pair.white_model.interpolate(0.0, 0.0)
        - gray_model_pair.black_model.interpolate(0.0, 0.0)
        < 0.0)
        != tag_family.reversed_border
    {
        return None;
    }

    let mut black_score = 0f32;
    let mut white_score = 0f32;
    let mut black_score_count = 1usize;
    let mut white_score_count = 1usize;

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

        let thresh = (gray_model_pair.black_model.interpolate(tag_x, tag_y)
            + gray_model_pair.white_model.interpolate(tag_x, tag_y))
            / 2.0;

        tag_family.sharpening_buffer.values[(tag_family.total_width as isize
            * (bit_y as isize - min_coord)
            + bit_x as isize
            - min_coord) as usize] = v - thresh;
    });

    sharpen(
        &mut tag_family.sharpening_buffer,
        decode_sharpening,
        tag_family.total_width,
    );

    let mut rcode = 0usize;
    (0..tag_family.nbits).for_each(|i| {
        let bit_y = tag_family.bit_y[i];
        let bit_x = tag_family.bit_x[i];

        rcode <<= 1;

        let v = tag_family.sharpening_buffer.values[((bit_y as isize - min_coord)
            * tag_family.total_width as isize
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

    // Reset the Sharpening Buffer for the next iteration
    tag_family.sharpening_buffer.reset();

    quick_decode_codeword(tag_family, rcode, entry);

    Some((white_score / white_score_count as f32).min(black_score / black_score_count as f32))
}

/// Applies a sharpening filter to the input values using a Laplacian kernel.
///
/// # Arguments
///
/// * `sharpening_buffer` - Mutable reference of `SharpeningBuffer`.
/// * `decode_sharpening` - The sharpening factor to apply.
/// * `size` - The width and height of the square buffer.
pub fn sharpen(sharpening_buffer: &mut SharpeningBuffer, decode_sharpening: f32, size: usize) {
    #[rustfmt::skip]
    const KERNEL: [f32; 9] = [
         0.0, -1.0,  0.0,
        -1.0,  4.0, -1.0,
         0.0, -1.0,  0.0,
    ];

    (0..size).for_each(|y| {
        let idy = y * size;

        (0..size).for_each(|x| {
            let idx = idy + x;
            sharpening_buffer.sharpened[idx] = 0.0;

            (0..3).for_each(|i| {
                let yi = y + i;

                if yi == 0 || (yi - 1) > size - 1 {
                    return;
                }

                let kernel_row_offset = 3 * i;
                let buffer_row_offset = (yi - 1) * size;

                (0..3).for_each(|j| {
                    let xj = x + j;
                    if xj == 0 || (xj - 1) > size - 1 {
                        return;
                    }

                    sharpening_buffer.sharpened[idx] += sharpening_buffer.values
                        [buffer_row_offset + (xj - 1)]
                        * KERNEL[kernel_row_offset + j];
                });
            });
        });
    });

    sharpening_buffer
        .values
        .iter_mut()
        .enumerate()
        .for_each(|(i, v)| {
            *v += decode_sharpening * sharpening_buffer.sharpened[i];
        });
}

/// Attempts to decode a codeword using the quick decode table for the given tag family.
///
/// # Arguments
///
/// * `tag_family` - Reference to the tag family containing the quick decode table.
/// * `rcode` - The codeword to look up.
/// * `entry` - Mutable reference to a `QuickDecodeEntry` to store the result.
fn quick_decode_codeword(tag_family: &TagFamily, mut rcode: usize, entry: &mut QuickDecodeEntry) {
    if let ControlFlow::Break(_) = (0..4).try_for_each(|ridx| {
        if let Some(mut decoded) = tag_family.quick_decode.decode(rcode, &tag_family.code_data) {
            decoded.rotation = ridx as u8;
            *entry = decoded;

            return ControlFlow::Break(());
        }

        rcode = rotate_90(rcode, tag_family.nbits);

        ControlFlow::Continue(())
    }) {
        return;
    }

    entry.rcode = 0;
    entry.id = u16::MAX;
    entry.hamming = 255;
    entry.rotation = 0;
}

/// Rotates the bits of a codeword by 90 degrees for tag decoding.
///
/// # Arguments
///
/// * `w` - The codeword to rotate.
/// * `num_bits` - The number of bits in the codeword.
///
/// # Returns
///
/// The rotated codeword as a `usize`.
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

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::{
        quad::fit_quads,
        segmentation::{find_connected_components, find_gradient_clusters},
        threshold::{adaptive_threshold, TileMinMax},
        union_find::UnionFind,
        utils::Pixel,
    };
    use kornia_image::allocator::CpuAllocator;
    use kornia_io::png::read_image_png_mono8;

    const EPSILON: f32 = 0.0001;

    #[test]
    fn test_decode_tags() -> Result<(), Box<dyn std::error::Error>> {
        let mut config = DecodeTagsConfig::new(vec![TagFamilyKind::Tag36H11]);
        config.downscale_factor = 1;
        let src = read_image_png_mono8("../../tests/data/apriltag.png")?;

        let mut bin = Image::from_size_val(src.size(), Pixel::Skip, CpuAllocator)?;
        let mut tile_min_max = TileMinMax::new(bin.size(), 4);
        let mut uf = UnionFind::new(bin.as_slice().len());
        let mut clusters = HashMap::new();
        let mut gray_model_pair = GrayModelPair::default();

        adaptive_threshold(&src, &mut bin, &mut tile_min_max, 20)?;
        find_connected_components(&bin, &mut uf)?;
        find_gradient_clusters(&bin, &mut uf, &mut clusters);

        let mut quads = fit_quads(&bin, &mut clusters, &config);

        for quad in &mut quads {
            for corner in &mut quad.corners {
                corner.x = corner.x.round();
                corner.y = corner.y.round();
            }
        }

        let tags = decode_tags(&src, &mut quads, &mut config, &mut gray_model_pair);

        assert_eq!(tags.len(), 1);
        assert_eq!(tags[0].id, 23);
        assert!((tags[0].center.x - 15.0).abs() < EPSILON);
        assert!((tags[0].center.y - 15.0).abs() < EPSILON);
        assert_eq!(tags[0].hamming, 0);
        assert_eq!(tags[0].tag_family_kind, TagFamilyKind::Tag36H11);

        Ok(())
    }

    #[test]
    fn test_gray_model() {
        let mut gm = GrayModel::default();

        gm.add(10.0, 10.0, 5.0);
        let expected_add_gm = GrayModel {
            a: [[100.0, 100.0, 10.0], [0.0, 100.0, 10.0], [0.0, 0.0, 1.0]],
            b: [50.0, 50.0, 5.0],
            c: [0.0, 0.0, 0.0],
        };

        assert_eq!(gm, expected_add_gm);

        gm.add(7.0, 15.0, 3.0);
        gm.solve();
        let expected_solve_gm = GrayModel {
            a: [[149.0, 205.0, 17.0], [0.0, 325.0, 25.0], [0.0, 0.0, 2.0]],
            b: [71.0, 95.0, 8.0],
            c: [0.562500, -0.062500, 0.0],
        };

        assert_eq!(gm.a, expected_solve_gm.a);
        assert_eq!(gm.b, expected_solve_gm.b);
        assert!((gm.c[0] - expected_solve_gm.c[0]).abs() < EPSILON); // Account for precision errors
        assert!((gm.c[1] - expected_solve_gm.c[1]).abs() < EPSILON);
        assert!((gm.c[2] - expected_solve_gm.c[2]).abs() < EPSILON);

        assert!((gm.interpolate(5.0, 3.0) - 2.62500).abs() < EPSILON);

        let mut pair = GrayModelPair {
            white_model: gm,
            black_model: expected_add_gm,
        };
        pair.reset();

        assert_eq!(pair, GrayModelPair::default())
    }

    #[test]
    fn test_refine_edges() -> Result<(), Box<dyn std::error::Error>> {
        let src = read_image_png_mono8("../../tests/data/apriltag.png")?;

        let mut quad = Quad {
            corners: [
                Point2d { x: 25.0, y: 5.0 },
                Point2d { x: 25.0, y: 25.0 },
                Point2d { x: 5.0, y: 25.0 },
                Point2d { x: 5.0, y: 5.0 },
            ],
            reversed_border: false,
            homography: [0.0; 9],
        };

        refine_edges(&src, &mut quad);
        let expected_corners = [
            Point2d {
                x: 26.612904,
                y: 3.387097,
            },
            Point2d {
                x: 26.612904,
                y: 26.612904,
            },
            Point2d {
                x: 3.387097,
                y: 26.612904,
            },
            Point2d {
                x: 3.387097,
                y: 3.387096,
            },
        ];

        for (i, expected) in expected_corners.iter().enumerate() {
            assert!(
                (quad.corners[i].x - expected.x).abs() <= EPSILON,
                "Got {}, Expected {}",
                quad.corners[i].x,
                expected.x
            );
            assert!(
                (quad.corners[i].y - expected.y).abs() <= EPSILON,
                "Got {}, Expected {}",
                quad.corners[i].y,
                expected.y
            );
        }
        Ok(())
    }

    #[test]
    fn test_quad_update_homographies() {
        let mut quad = Quad {
            corners: [
                Point2d { x: 3.0, y: 3.0 },
                Point2d { x: 27.0, y: 3.0 },
                Point2d { x: 3.0, y: 20.0 },
                Point2d { x: 4.0, y: 22.0 },
            ],
            reversed_border: false,
            homography: [0.0; 9],
        };

        quad.update_homographies();
        let expected_homographies = [
            -0.675192, 4.672634, 3.0, 0.368286, 23.455243, 22.826087, 0.122762, 1.209719, 1.0,
        ];

        for (i, expected) in expected_homographies.iter().enumerate() {
            assert!((quad.homography[i] - expected).abs() < EPSILON);
        }
    }

    #[test]
    fn test_quad_decode() -> Result<(), Box<dyn std::error::Error>> {
        let mut tag_family = TagFamily::tag36_h11();
        let src = read_image_png_mono8("../../tests/data/apriltag.png")?;

        let quad = Quad {
            corners: [
                Point2d {
                    x: 26.612904,
                    y: 3.387097,
                },
                Point2d {
                    x: 26.612904,
                    y: 26.612904,
                },
                Point2d {
                    x: 3.387097,
                    y: 26.612904,
                },
                Point2d {
                    x: 3.387097,
                    y: 3.387096,
                },
            ],
            reversed_border: false,
            homography: [
                -0.0, -11.612904, 15.0, 11.612904, -0.000001, 15.0, -0.0, -0.0, 1.0,
            ],
        };

        let mut entry = QuickDecodeEntry::default();
        let mut gray_model_pair = GrayModelPair::default();

        let d = quad_decode(
            &src,
            &mut tag_family,
            &quad,
            0.25,
            &mut entry,
            &mut gray_model_pair,
        );

        assert_eq!(d, Some(225.50317));

        Ok(())
    }

    #[test]
    fn test_sharpen() {
        let mut sharpening_buffer = SharpeningBuffer {
            values: vec![
                255.0, 255.0, 127.0, 0.0, 0.0, 0.0, 0.0, -1.0, 127.0, 63.75, 0.0, 63.75, 127.5,
                -1.0, 127.5, 127.5,
            ],
            sharpened: vec![0.0; 16],
        };

        sharpen(&mut sharpening_buffer, 0.25, 4);
        let expected_values = [
            446.25, 414.5, 190.25, -31.5, -95.5, -79.6875, -31.5, -17.9375, 206.1875, 96.0, -63.75,
            95.875, 223.5, -81.6875, 223.375, 207.1875,
        ];

        assert_eq!(sharpening_buffer.values, expected_values)
    }

    #[test]
    fn test_quick_decode_codeword() {
        let tag_family = TagFamily::tag36_h11();
        let rcode = 52087007497;

        let mut quick_decode_entry = QuickDecodeEntry::default();
        quick_decode_codeword(&tag_family, rcode, &mut quick_decode_entry);

        let expected_decode_entry = QuickDecodeEntry {
            rcode: 52087007497,
            id: 85,
            hamming: 2,
            rotation: 0,
        };

        assert_eq!(quick_decode_entry, expected_decode_entry)
    }

    #[test]
    fn test_rotate_90() {
        assert_eq!(rotate_90(52087007497, 36), 5390865284);
        assert_eq!(rotate_90(42087007497, 36), 39351620409)
    }
}
