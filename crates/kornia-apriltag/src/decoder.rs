use std::{f32::consts::PI, ops::ControlFlow};

use crate::{
    errors::AprilTagError,
    family::{TagFamily, TagFamilyKind},
    quad::Quad,
    utils::value_for_pixel,
    DecodeTagsConfig,
};
use kornia_algebra::Mat3F32;
use kornia_image::{allocator::ImageAllocator, Image};

/// Represents a model for grayscale interpolation using a quadratic surface.
/// The model fits a function of the form f(x, y) = c.x*x + c.y*y + c.z.
#[derive(Debug, Clone, PartialEq)]
struct GrayModel {
    /// The 3x3 matrix of accumulated quadratic terms.
    a: Mat3F32,
    /// The vector of accumulated linear terms.
    b: kornia_algebra::Vec3F32,
    /// The solved coefficients for the quadratic model.
    c: kornia_algebra::Vec3F32,
}

impl Default for GrayModel {
    fn default() -> Self {
        Self {
            a: Mat3F32::ZERO,
            b: kornia_algebra::Vec3F32::ZERO,
            c: kornia_algebra::Vec3F32::ZERO,
        }
    }
}

impl GrayModel {
    /// Adds a new data point (x, y, gray) to the quadratic model.
    fn add(&mut self, x: f32, y: f32, gray: f32) {
        // Update A matrix (symmetric)
        // [x^2  xy  x]
        // [xy   y^2 y]
        // [x    y   1]

        // Upper triangle element skipped to match legacy behavior
        // We use column-major indexing for Mat3F32
        // col 0: x^2, xy, x
        self.a.x_axis.x += x * x;
        self.a.x_axis.y += x * y;
        self.a.x_axis.z += x;

        // col 1: xy, y^2, y
        self.a.y_axis.y += y * y;
        self.a.y_axis.z += y;

        // col 2: x, y, 1
        self.a.z_axis.z += 1.0;

        // Update b vector
        // [x*gray, y*gray, gray]
        self.b.x += x * gray;
        self.b.y += y * gray;
        self.b.z += gray;
    }

    /// Solves the quadratic model to find the coefficients.
    fn solve(&mut self) -> Result<(), AprilTagError> {
        // Solve Ac = b for c.
        // A is symmetric positive definite (ideally).

        // We use the explicit inverse implementation to match legacy behavior exactly,
        // as the matrix can be ill-conditioned and substitution yields different results due to precision.

        if let Some(l) = kornia_algebra::linalg::cholesky::cholesky_3x3(&self.a) {
            // Use the explicit inverse solver from algebra which matches legacy behavior
            self.c = kornia_algebra::linalg::cholesky::cholesky_solve_3x3(&l, &self.b);
            Ok(())
        } else {
            // Matrix is undefined or not positive definite.
            self.c = kornia_algebra::Vec3F32::ZERO;
            Err(AprilTagError::GrayModelUnderdetermined)
        }
    }

    /// Interpolates the grayscale value at the given (x, y) using the solved coefficients.
    fn interpolate(&self, x: f32, y: f32) -> f32 {
        self.c.x * x + self.c.y * y + self.c.z
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
        self.white_model.a = Mat3F32::ZERO;
        self.black_model.a = Mat3F32::ZERO;

        self.white_model.b = kornia_algebra::Vec3F32::ZERO;
        self.white_model.c = kornia_algebra::Vec3F32::ZERO;
        self.black_model.b = kornia_algebra::Vec3F32::ZERO;
        self.black_model.c = kornia_algebra::Vec3F32::ZERO;
    }
}

/// Represents an entry in the quick decode table for tag decoding.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct QuickDecodeEntry {
    /// The raw observed code from the image.
    pub rcode: usize,
    /// The decoded Tag ID.
    pub id: u16,
    /// The Hamming distance found (0, 1, 2, or 3).
    pub hamming: u8,
    /// Rotation
    pub rotation: u8,
}

/// A lookup table for fast Hamming distance decoding.
#[derive(Debug, Clone, PartialEq)]
pub struct QuickDecode {
    /// Flattened storage for IDs.
    chunk_ids: [Vec<u16>; 4],

    /// Index lookup table.
    chunk_offsets: [Vec<u16>; 4],

    /// The number of bits to shift right to extract each chunk.
    shifts: [usize; 4],

    /// Bitmask to isolate the value of a single chunk.
    chunk_mask: usize,

    /// The maximum allowed Hamming distance.
    max_hamming: u32,
}

impl QuickDecode {
    /// Creates a new QuickDecode table.
    pub fn new(
        nbits: usize,
        valid_codes: &[usize],
        allowed_errors: u8,
    ) -> Result<Self, AprilTagError> {
        if valid_codes.len() > u16::MAX as usize {
            return Err(AprilTagError::TooManyCodes(valid_codes.len()));
        }

        if allowed_errors >= 4 {
            return Err(AprilTagError::InvalidAllowedErrors(allowed_errors));
        }

        let chunk_size = nbits.div_ceil(4);
        let capacity = 1 << chunk_size;
        let chunk_mask = capacity - 1;
        let shifts = [0, chunk_size, chunk_size * 2, chunk_size * 3];

        let mut offsets = [
            vec![0u16; capacity + 1],
            vec![0u16; capacity + 1],
            vec![0u16; capacity + 1],
            vec![0u16; capacity + 1],
        ];

        for &code in valid_codes {
            for i in 0..4 {
                let val = (code >> shifts[i]) & chunk_mask;
                offsets[i][val + 1] += 1;
            }
        }

        for offset in &mut offsets {
            for j in 0..capacity {
                offset[j + 1] += offset[j];
            }
        }

        let mut cursors = [
            offsets[0].clone(),
            offsets[1].clone(),
            offsets[2].clone(),
            offsets[3].clone(),
        ];

        let mut ids = [
            vec![0u16; valid_codes.len()],
            vec![0u16; valid_codes.len()],
            vec![0u16; valid_codes.len()],
            vec![0u16; valid_codes.len()],
        ];

        for (idx, &code) in valid_codes.iter().enumerate() {
            let id = idx as u16;
            for i in 0..4 {
                let val = (code >> shifts[i]) & chunk_mask;
                let write_pos = cursors[i][val] as usize;
                ids[i][write_pos] = id;
                cursors[i][val] += 1;
            }
        }

        Ok(Self {
            chunk_ids: ids,
            chunk_offsets: offsets,
            shifts,
            chunk_mask,
            max_hamming: allowed_errors as u32,
        })
    }

    /// Sets the maximum allowed Hamming distance for decoding.
    ///
    /// # Arguments
    ///
    /// * `max_hamming` - The maximum number of bit errors to tolerate (0-3).
    ///
    /// # Errors
    ///
    /// Returns `AprilTagError::InvalidAllowedErrors` if `max_hamming` is greater than 3.
    pub fn set_max_hamming(&mut self, max_hamming: u8) -> Result<(), AprilTagError> {
        if max_hamming > 3 {
            return Err(AprilTagError::InvalidAllowedErrors(max_hamming));
        }
        self.max_hamming = max_hamming as u32;
        Ok(())
    }

    /// Returns the current maximum allowed Hamming distance.
    pub fn max_hamming(&self) -> u8 {
        self.max_hamming as u8
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
    /// Returns `Some(QuickDecodeEntry)` if the code is found within `max_hamming`, or `None` otherwise.
    pub fn decode(&self, observed_code: usize, valid_codes: &[usize]) -> Option<QuickDecodeEntry> {
        // Closure to check Hamming distance
        let check = |id: u16| -> Option<u8> {
            let perfect = valid_codes[id as usize];
            let dist = (observed_code ^ perfect).count_ones();
            if dist <= self.max_hamming {
                Some(dist as u8)
            } else {
                None
            }
        };

        for i in 0..4 {
            let val = (observed_code >> self.shifts[i]) & self.chunk_mask;

            let start = self.chunk_offsets[i][val];
            let end = self.chunk_offsets[i][val + 1];
            let candidates = &self.chunk_ids[i][(start as usize)..(end as usize)];
            for &id in candidates {
                if let Some(dist) = check(id) {
                    return Some(QuickDecodeEntry {
                        rcode: observed_code,
                        id,
                        hamming: dist,
                        rotation: 0,
                    });
                }
            }
        }

        None
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
    pub center: kornia_algebra::Vec2F32,
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
                    let r = Mat3F32::from_cols_array(&[
                        c, s, 0.0, // col 0
                        -s, c, 0.0, // col 1
                        0.0, 0.0, 1.0, // col 2
                    ]);

                    quad.homography *= r;
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

            let px = p.x as usize;
            let py = p.y as usize;

            if px >= src.width() || py >= src.height() {
                return;
            }

            let v = src_slice[py * src.width() + px] as f32;

            if pattern.is_white {
                gray_model_pair.white_model.add(tagx, tagy, v);
            } else {
                gray_model_pair.black_model.add(tagx, tagy, v);
            }
        });
    });

    gray_model_pair.white_model.solve().ok()?;
    if tag_family.width_at_border > 1 {
        gray_model_pair.black_model.solve().ok()?;
    } else {
        gray_model_pair.black_model.c.x = 0.0;
        gray_model_pair.black_model.c.y = 0.0;
        gray_model_pair.black_model.c.z = gray_model_pair.black_model.b.z / 4.0;
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
    use crate::{family::TagFamilyKind, DecodeTagsConfig};
    use crate::{
        quad::fit_quads,
        segmentation::{find_connected_components, find_gradient_clusters},
        threshold::{adaptive_threshold, TileMinMax},
        union_find::UnionFind,
        utils::Pixel,
    };
    use kornia_algebra::Vec2F32;
    use kornia_image::allocator::CpuAllocator;
    use kornia_io::png::read_image_png_mono8;

    const EPSILON: f32 = 0.0001;

    #[test]
    fn test_decode_tags() -> Result<(), Box<dyn std::error::Error>> {
        let mut config = DecodeTagsConfig::new(vec![TagFamilyKind::Tag36H11])?;
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
            a: Mat3F32::from_cols_array(&[
                100.0, 100.0, 10.0, // col 0
                0.0, 100.0, 10.0, // col 1
                0.0, 0.0, 1.0, // col 2
            ]),
            b: kornia_algebra::Vec3F32::new(50.0, 50.0, 5.0),
            c: kornia_algebra::Vec3F32::ZERO,
        };

        assert_eq!(gm.a, expected_add_gm.a);
        assert_eq!(gm.b, expected_add_gm.b);
        assert_eq!(gm.c, expected_add_gm.c);

        gm.add(7.0, 15.0, 3.0);
        let _ = gm.solve();
        let expected_solve_gm = GrayModel {
            a: Mat3F32::from_cols_array(&[
                149.0, 205.0, 17.0, // col 0
                0.0, 325.0, 25.0, // col 1
                0.0, 0.0, 2.0, // col 2
            ]),
            b: kornia_algebra::Vec3F32::new(71.0, 95.0, 8.0),
            c: kornia_algebra::Vec3F32::new(0.562500, -0.062500, 0.0),
        };

        const EPSILON: f32 = 1e-5;

        // Check A
        let diff_a = gm.a - expected_solve_gm.a;
        assert!(diff_a.x_axis.x.abs() < EPSILON);
        assert!(diff_a.x_axis.y.abs() < EPSILON);
        assert!(diff_a.x_axis.z.abs() < EPSILON);
        assert!(diff_a.y_axis.x.abs() < EPSILON);
        assert!(diff_a.y_axis.y.abs() < EPSILON);
        assert!(diff_a.y_axis.z.abs() < EPSILON);
        assert!(diff_a.z_axis.x.abs() < EPSILON);
        assert!(diff_a.z_axis.y.abs() < EPSILON);
        assert!(diff_a.z_axis.z.abs() < EPSILON);

        // Check B
        let diff_b = gm.b - expected_solve_gm.b;
        assert!(diff_b.x.abs() < EPSILON);
        assert!(diff_b.y.abs() < EPSILON);
        assert!(diff_b.z.abs() < EPSILON);

        // Check C
        assert!((gm.c.x - expected_solve_gm.c.x).abs() < EPSILON); // Account for precision errors
        assert!((gm.c.y - expected_solve_gm.c.y).abs() < EPSILON);
        assert!((gm.c.z - expected_solve_gm.c.z).abs() < EPSILON);

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
                kornia_algebra::Vec2F32::new(25.0, 5.0),
                kornia_algebra::Vec2F32::new(25.0, 25.0),
                kornia_algebra::Vec2F32::new(5.0, 25.0),
                kornia_algebra::Vec2F32::new(5.0, 5.0),
            ],
            reversed_border: false,
            homography: Mat3F32::IDENTITY,
        };

        refine_edges(&src, &mut quad);
        let expected_corners = [
            Vec2F32::new(26.612904, 3.387097),
            Vec2F32::new(26.612904, 26.612904),
            Vec2F32::new(3.387097, 26.612904),
            Vec2F32::new(3.387097, 3.387096),
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
                Vec2F32::new(3.0, 3.0),
                Vec2F32::new(27.0, 3.0),
                Vec2F32::new(3.0, 20.0),
                Vec2F32::new(4.0, 22.0),
            ],
            reversed_border: false,
            homography: Mat3F32::IDENTITY,
        };

        quad.update_homographies();
        let expected = Mat3F32::from_cols_array(&[
            -0.675192, 0.368286, 0.122762, // col 0
            4.672634, 23.455243, 1.209719, // col 1
            3.0, 22.826087, 1.0, // col 2
        ]);

        let h_arr: [f32; 9] = quad.homography.into();
        let e_arr: [f32; 9] = expected.into();

        for (i, e) in e_arr.iter().enumerate() {
            assert!((h_arr[i] - e).abs() < EPSILON);
        }
    }

    #[test]
    fn test_quad_decode() -> Result<(), Box<dyn std::error::Error>> {
        let mut tag_family = TagFamily::tag36_h11()?;
        let src = read_image_png_mono8("../../tests/data/apriltag.png")?;

        let quad = Quad {
            corners: [
                Vec2F32::new(26.612904, 3.387097),
                Vec2F32::new(26.612904, 26.612904),
                Vec2F32::new(3.387097, 26.612904),
                Vec2F32::new(3.387097, 3.387096),
            ],
            reversed_border: false,
            homography: Mat3F32::from_cols_array(&[
                -0.0, 11.612904, -0.0, // col 0
                -11.612904, -0.000001, -0.0, // col 1
                15.0, 15.0, 1.0, // col 2
            ]),
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
    fn test_quick_decode_codeword() -> Result<(), Box<dyn std::error::Error>> {
        let tag_family = TagFamily::tag36_h11()?;
        let rcode = 52087007497;

        let mut quick_decode_entry = QuickDecodeEntry::default();
        quick_decode_codeword(&tag_family, rcode, &mut quick_decode_entry);

        let expected_decode_entry = QuickDecodeEntry {
            rcode: 52087007497,
            id: 85,
            hamming: 2,
            rotation: 0,
        };

        assert_eq!(quick_decode_entry, expected_decode_entry);
        Ok(())
    }

    #[test]
    fn test_rotate_90() {
        assert_eq!(rotate_90(52087007497, 36), 5390865284);
        assert_eq!(rotate_90(42087007497, 36), 39351620409)
    }

    #[test]
    fn test_gray_model_solve_failure() {
        let mut model = GrayModel::default();

        // Add redundant or insufficient points to create a singular matrix
        // For example, adding the same point multiple times or points that are collinear
        model.add(0.0, 0.0, 100.0);
        model.add(0.0, 0.0, 100.0);
        model.add(0.0, 0.0, 100.0);

        let result = model.solve();
        assert!(result.is_err());
        match result {
            Err(AprilTagError::GrayModelUnderdetermined) => {}
            _ => panic!("Expected GrayModelUnderdetermined error"),
        }
    }

    #[test]
    fn test_configurable_max_hamming() -> Result<(), Box<dyn std::error::Error>> {
        // Get the valid code for tag id 0
        let family = TagFamily::tag36_h11()?;
        let valid_code = family.code_data[0];

        // Flip 2 bits to create a code with hamming distance 2
        let code_with_2_errors = valid_code ^ 0b11;

        // With max_hamming=1, it should not decode
        let family = TagFamily::tag36_h11()?.with_max_hamming(1)?;
        assert_eq!(family.max_hamming(), 1);
        let mut entry = QuickDecodeEntry::default();
        quick_decode_codeword(&family, code_with_2_errors, &mut entry);
        assert_eq!(entry.id, u16::MAX); // Not found

        // With max_hamming=2 (default), it should decode
        let family = TagFamily::tag36_h11()?;
        assert_eq!(family.max_hamming(), 2);
        quick_decode_codeword(&family, code_with_2_errors, &mut entry);
        assert_eq!(entry.id, 0);
        assert_eq!(entry.hamming, 2);

        // Flip 1 bit - should decode with both max_hamming=1 and max_hamming=2
        let code_with_1_error = valid_code ^ 0b1;

        let family = TagFamily::tag36_h11()?.with_max_hamming(1)?;
        quick_decode_codeword(&family, code_with_1_error, &mut entry);
        assert_eq!(entry.id, 0);
        assert_eq!(entry.hamming, 1);

        let family = TagFamily::tag36_h11()?;
        quick_decode_codeword(&family, code_with_1_error, &mut entry);
        assert_eq!(entry.id, 0);
        assert_eq!(entry.hamming, 1);

        // Invalid max_hamming should return an error
        let result = TagFamily::tag36_h11()?.with_max_hamming(4);
        assert!(result.is_err());

        Ok(())
    }
}
