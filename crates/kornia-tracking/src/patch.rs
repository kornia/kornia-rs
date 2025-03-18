//! This module implements the Pattern52 patch structure for keypoint tracking.
//! It provides functions to compute the Jacobian, mean intensity, and residuals of a patch.

use kornia_imgproc::interpolation::{interpolate_pixel,InterpolationMode};
use kornia_image::Image;
use crate::image_utilities;
use glam::{Mat3, Vec2, Vec3};
type GrayImage = Image<f32, 1>;
/// The size of the 52-element pattern.
pub const PATTERN52_SIZE: usize = 52;

/// A 52x3 matrix used for storing patch data.
pub type Matrix52x3 = [[f32; 3]; PATTERN52_SIZE];

/// A 3x52 matrix used for storing patch data.
pub type Matrix3x52 = [[f32; PATTERN52_SIZE]; 3];

/// A structure representing a patch using a 52-element pattern.
/// It stores intensity data, position and precomputed matrices for tracking.
pub struct Pattern52 {
    /// Indicates if the patch is valid (i.e. not an all-black patch).
    pub valid: bool,
    /// Mean intensity of the patch.
    pub mean: f32,
    /// Position of the patch as a vector (x, y).
    pub pos: Vec2,
    /// Contrast data of the patch. Negative values indicate an invalid point.
    pub data: [f32; PATTERN52_SIZE],
    /// A 3×52 matrix computed as H⁻¹ * Jᵀ.
    pub h_se2_inv_j_se2_t: Matrix3x52,
    /// Scale factor to downscale the patch.
    pub pattern_scale_down: f32,
}

impl Pattern52 {
    /// The raw 52-point pattern.
    pub const PATTERN_RAW: [[f32; 2]; PATTERN52_SIZE] = [
        [-3.0, 7.0],
        [-1.0, 7.0],
        [1.0, 7.0],
        [3.0, 7.0],
        [-5.0, 5.0],
        [-3.0, 5.0],
        [-1.0, 5.0],
        [1.0, 5.0],
        [3.0, 5.0],
        [5.0, 5.0],
        [-7.0, 3.0],
        [-5.0, 3.0],
        [-3.0, 3.0],
        [-1.0, 3.0],
        [1.0, 3.0],
        [3.0, 3.0],
        [5.0, 3.0],
        [7.0, 3.0],
        [-7.0, 1.0],
        [-5.0, 1.0],
        [-3.0, 1.0],
        [-1.0, 1.0],
        [1.0, 1.0],
        [3.0, 1.0],
        [5.0, 1.0],
        [7.0, 1.0],
        [-7.0, -1.0],
        [-5.0, -1.0],
        [-3.0, -1.0],
        [-1.0, -1.0],
        [1.0, -1.0],
        [3.0, -1.0],
        [5.0, -1.0],
        [7.0, -1.0],
        [-7.0, -3.0],
        [-5.0, -3.0],
        [-3.0, -3.0],
        [-1.0, -3.0],
        [1.0, -3.0],
        [3.0, -3.0],
        [5.0, -3.0],
        [7.0, -3.0],
        [-5.0, -5.0],
        [-3.0, -5.0],
        [-1.0, -5.0],
        [1.0, -5.0],
        [3.0, -5.0],
        [5.0, -5.0],
        [-3.0, -7.0],
        [-1.0, -7.0],
        [1.0, -7.0],
        [3.0, -7.0],
    ];

    /// Sets the Jacobian (j_se2) and patch data based on the provided grayscale image.
    /// It computes patch gradients and normalizes the collected values.
    pub fn set_data_jac_se2(
        &mut self,
        greyscale_image: &GrayImage,
        j_se2: &mut Matrix52x3,
    ) {
        let mut num_valid_points = 0;
        let mut sum: f32 = 0.0;
        let mut grad_sum_se2 = Vec3::ZERO;

        // Initialize the 2x3 matrix (jw_se2) for computing the spatial Jacobian.
        let mut jw_se2: [[f32; 3]; 2] = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ];

        // Iterate over each point in the pattern.
        for (i, pattern_pos) in Self::PATTERN_RAW.iter().enumerate() {
            let p = self.pos
                + Vec2::new(
                    pattern_pos[0] / self.pattern_scale_down,
                    pattern_pos[1] / self.pattern_scale_down,
                );
            jw_se2[0][2] = -pattern_pos[1] / self.pattern_scale_down;
            jw_se2[1][2] = pattern_pos[0] / self.pattern_scale_down;

            // Check if the point is within image bounds.
            if image_utilities::inbound(greyscale_image, p.x, p.y, 2) {
                let val_grad = image_utilities::image_grad(greyscale_image, p.x, p.y);

                self.data[i] = val_grad.x;
                sum += val_grad.x;
                let grad2 = Vec2::new(val_grad.y, val_grad.z);
                let mut re = [0.0; 3];
                for j in 0..3 {
                    re[j] = grad2.x * jw_se2[0][j] + grad2.y * jw_se2[1][j];
                }
                j_se2[i] = re;
                grad_sum_se2 += Vec3::from(re);
                num_valid_points += 1;
            } else {
                self.data[i] = -1.0;
            }
        }

        self.mean = sum / num_valid_points as f32;
        let mean_inv = num_valid_points as f32 / sum;

        // Normalize Jacobian rows using computed gradient sum.
        for i in 0..Self::PATTERN_RAW.len() {
            if self.data[i] >= 0.0 {
                for j in 0..3 {
                    j_se2[i][j] -= grad_sum_se2[j] * self.data[i] / sum;
                }
                self.data[i] *= mean_inv;
            } else {
                j_se2[i] = [0.0; 3];
            }
        }
        for row in j_se2.iter_mut() {
            for v in row.iter_mut() {
                *v *= mean_inv;
            }
        }

        // Compute h_se2 = j_se2ᵀ * j_se2 (3×3 matrix).
        let mut h_se2 = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..PATTERN52_SIZE {
                    h_se2[i][j] += j_se2[k][i] * j_se2[k][j];
                }
            }
        }
        // Convert h_se2 into a glam::Mat3.
        let h_mat = Mat3::from_cols(
            Vec3::new(h_se2[0][0], h_se2[1][0], h_se2[2][0]),
            Vec3::new(h_se2[0][1], h_se2[1][1], h_se2[2][1]),
            Vec3::new(h_se2[0][2], h_se2[1][2], h_se2[2][2]),
        );
        // Invert h_mat.
        let h_inv = h_mat.inverse();
        // Compute h_se2_inv_j_se2_t = h_inv * j_se2ᵀ.
        let mut h_inv_j_t: Matrix3x52 = [[0.0; PATTERN52_SIZE]; 3];
        for i in 0..3 {
            for k in 0..PATTERN52_SIZE {
                // For each row i and each original j_se2 row k, compute dot product with h_inv's i-th row.
                let row_i = Vec3::new(h_inv.x_axis[i], h_inv.y_axis[i], h_inv.z_axis[i]);
                let vec = Vec3::from(j_se2[k]);
                h_inv_j_t[i][k] = row_i.dot(vec);
            }
        }
        self.h_se2_inv_j_se2_t = h_inv_j_t;
    }

    /// Creates a new Pattern52 given a grayscale image and patch position.
    /// It computes the internal data necessary for tracking.
    pub fn new(greyscale_image: &GrayImage, px: f32, py: f32) -> Pattern52 {
        let mut j_se2: Matrix52x3 = [[0.0; 3]; PATTERN52_SIZE];
        let mut p = Pattern52 {
            valid: false,
            mean: 1.0,
            pos: Vec2::new(px, py),
            data: [0.0; PATTERN52_SIZE],
            h_se2_inv_j_se2_t: [[0.0; PATTERN52_SIZE]; 3],
            pattern_scale_down: 2.0,
        };
        p.set_data_jac_se2(greyscale_image, &mut j_se2);
        // Compute h_se2 = j_se2ᵀ * j_se2 as before.
        let mut h_se2 = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..PATTERN52_SIZE {
                    h_se2[i][j] += j_se2[k][i] * j_se2[k][j];
                }
            }
        }
        let h_mat = Mat3::from_cols(
            Vec3::new(h_se2[0][0], h_se2[1][0], h_se2[2][0]),
            Vec3::new(h_se2[0][1], h_se2[1][1], h_se2[2][1]),
            Vec3::new(h_se2[0][2], h_se2[1][2], h_se2[2][2]),
        );
        let h_inv = h_mat.inverse();
        let mut h_inv_j_t: Matrix3x52 = [[0.0; PATTERN52_SIZE]; 3];
        for i in 0..3 {
            for k in 0..PATTERN52_SIZE {
                let row_i = Vec3::new(h_inv.x_axis[i], h_inv.y_axis[i], h_inv.z_axis[i]);
                let vec = Vec3::from(j_se2[k]);
                h_inv_j_t[i][k] = row_i.dot(vec);
            }
        }
        p.h_se2_inv_j_se2_t = h_inv_j_t;
        // Validate patch quality.
        p.valid = p.mean > f32::EPSILON &&
            p.h_se2_inv_j_se2_t.iter().all(|row| row.iter().all(|&x| x.is_finite())) &&
            p.data.iter().all(|&x| x.is_finite());
        p
    }

    /// Computes the residual between the measured image intensities and the transformed patch.
    /// Returns an optional vector of residuals if the patch can be normalized.
    pub fn residual(
        &self,
        greyscale_image: &GrayImage,
        transformed_pattern: &[[f32; PATTERN52_SIZE]; 2],
    ) -> Option<[f32; PATTERN52_SIZE]> {
        let mut sum: f32 = 0.0;
        let mut num_valid_points = 0;
        let mut residual = [0.0; PATTERN52_SIZE];
        for i in 0..PATTERN52_SIZE {
            if image_utilities::inbound(
                greyscale_image,
                transformed_pattern[0][i],
                transformed_pattern[1][i],
                2,
            ) {
                let p = interpolate_pixel(
                    greyscale_image,
                    transformed_pattern[0][i],
                    transformed_pattern[1][i],
                    0,
                    InterpolationMode::Bilinear,
                );
                residual[i] = p;
                sum += residual[i];
                num_valid_points += 1;
            } else {
                residual[i] = -1.0;
            }
        }

        if sum < f32::EPSILON {
            return None;
        }

        let mut num_residuals = 0;
        for i in 0..PATTERN52_SIZE {
            if residual[i] >= 0.0 && self.data[i] >= 0.0 {
                residual[i] = num_valid_points as f32 * residual[i] / sum - self.data[i];
                num_residuals += 1;
            } else {
                residual[i] = 0.0;
            }
        }
        if num_residuals > PATTERN52_SIZE / 2 {
            Some(residual)
        } else {
            None
        }
    }
}
