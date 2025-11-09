//! Infinitesimal Plane-based Pose Estimation (IPPE)
//!
//! This module provides routines to estimate two candidate poses for a planar
//! object from 2D to 3D correspondences, following the IPPE method.
//!
//! References:
//! - T. Collins and A. Bartoli, "Infinitesimal Plane-based Pose Estimation"
//! - OpenCV implementation reference: modules/calib3d/src/ippe.hpp/.cpp
//!
//! This file contains original Rust code inspired by the algorithms described
//! in the references above. The OpenCV implementation is BSD-3-Clause; see the
//! OpenCV project license for details.

use crate::pnp::{PnPError, PnPResult};
use glam::{Mat3A, Vec3, Vec3A};
use kornia_lie::so3::SO3;

/// Two IPPE pose solutions sorted by reprojection error (best first).
///
/// Note: The current implementation returns the same solution twice `(best, second)`
/// as a temporary placeholder until the IPPE-specific second solution is implemented.
pub type IPPEResult = (PnPResult, PnPResult);

/// Marker type for the IPPE solver.
pub struct IPPE;

impl IPPE {
    /// Estimate two candidate poses for a square planar object from its four image corners.
    ///
    /// The four object points are implicitly defined as a square of side length `square_length`
    /// on the plane z=0 with the following order (object coordinates):
    ///  - p0 = [-L/2,  L/2, 0]
    ///  - p1 = [ L/2,  L/2, 0]
    ///  - p2 = [ L/2, -L/2, 0]
    ///  - p3 = [-L/2, -L/2, 0]
    ///
    /// The `image_points_norm` are the corresponding undistorted, normalized image coordinates
    /// (i.e. pixel coordinates premultiplied by K^{-1}).
    pub fn solve_square(
        image_points_norm: &[[f32; 2]; 4],
        square_length: f32,
    ) -> Result<IPPEResult, PnPError> {
        // Build canonical 2D square points (object plane) in the order required above.
        let h = (square_length as f64) / 2.0;
        let src: [[f64; 2]; 4] = [[-h, h], [h, h], [h, -h], [-h, -h]];

        // Target points are normalized image coordinates.
        let dst: [[f64; 2]; 4] = [
            [image_points_norm[0][0] as f64, image_points_norm[0][1] as f64],
            [image_points_norm[1][0] as f64, image_points_norm[1][1] as f64],
            [image_points_norm[2][0] as f64, image_points_norm[2][1] as f64],
            [image_points_norm[3][0] as f64, image_points_norm[3][1] as f64],
        ];

// Estimate homography mapping object-plane points to normalized image points.
        let mut hmat = [[0.0f64; 3]; 3];
        kornia_3d::pose::homography_4pt2d(&src, &dst, &mut hmat)
            .map_err(|e| PnPError::SvdFailed(e.to_string()))?;

        // Decompose homography into pose (assuming K = I because points are normalized).
        let (r_mat, t_vec) = decompose_h_normalized(&hmat);

        // Convert to arrays for result types and compute Rodrigues vector.
        let r32 = [
            [r_mat.x_axis.x, r_mat.y_axis.x, r_mat.z_axis.x],
            [r_mat.x_axis.y, r_mat.y_axis.y, r_mat.z_axis.y],
            [r_mat.x_axis.z, r_mat.y_axis.z, r_mat.z_axis.z],
        ];
        let t32 = [t_vec.x, t_vec.y, t_vec.z];

        let rvec_v = SO3::from_matrix(&r_mat).log();
        let rvec = [rvec_v.x, rvec_v.y, rvec_v.z];

        // Compute RMS reprojection error under normalized intrinsics (K=I).
        let obj3d = square_object_points(square_length);
        let rmse1 = rmse_normalized(&obj3d, image_points_norm, &r32, &t32)?;

        // For now, return the same solution twice as a placeholder; a subsequent
        // iteration will implement the Jacobian-based second solution specific to IPPE.
        let best = PnPResult {
            rotation: r32,
            translation: t32,
            rvec,
            reproj_rmse: Some(rmse1),
            num_iterations: None,
            converged: Some(true),
        };

        let second = best.clone();
        Ok((best, second))
    }
}

/// Compute the homography-based pose decomposition assuming normalized image coordinates (K = I).
fn decompose_h_normalized(h: &[[f64; 3]; 3]) -> (Mat3A, Vec3) {
    // Columns of H (cast to f32 vectors).
    let h1 = Vec3::new(h[0][0] as f32, h[1][0] as f32, h[2][0] as f32);
    let h2 = Vec3::new(h[0][1] as f32, h[1][1] as f32, h[2][1] as f32);
    let h3 = Vec3::new(h[0][2] as f32, h[1][2] as f32, h[2][2] as f32);

    let n1 = h1.length();
    let n2 = h2.length();
    let s = 1.0 / (n1 * n2).sqrt(); // scale so that ||r1|| ≈ ||r2|| ≈ 1

    let r1 = h1 * s;
    let r2 = h2 * s;
    let r3 = r1.cross(r2);

    // Orthonormalize R via projection onto SO(3).
    let r_mat = Mat3A::from_cols(Vec3A::from(r1), Vec3A::from(r2), Vec3A::from(r3));
    let r_proj = SO3::from_matrix(&r_mat).matrix();

    let t = h3 * s;
    (r_proj, t)
}

/// Generate the 3D object points of a square in the canonical order and z=0 plane.
fn square_object_points(square_length: f32) -> [[f32; 3]; 4] {
    let h = square_length / 2.0;
    [
        [-h, h, 0.0],
        [h, h, 0.0],
        [h, -h, 0.0],
        [-h, -h, 0.0],
    ]
}

/// Root-mean-square reprojection error for normalized image coordinates (K = I).
fn rmse_normalized(
    points_world: &[[f32; 3]; 4],
    points_norm: &[[f32; 2]; 4],
    r: &[[f32; 3]; 3],
    t: &[f32; 3],
) -> Result<f32, PnPError> {
    let r_mat = Mat3A::from_cols(
        Vec3A::new(r[0][0], r[1][0], r[2][0]),
        Vec3A::new(r[0][1], r[1][1], r[2][1]),
        Vec3A::new(r[0][2], r[1][2], r[2][2]),
    );
    let t_vec = Vec3::new(t[0], t[1], t[2]);

    let mut sum_sq = 0.0f32;
    for (pw, uv) in points_world.iter().zip(points_norm.iter()) {
        let pw_v = Vec3::from_array(*pw);
        let pc = r_mat * pw_v + t_vec;
        // Prevent division by zero or near-zero depth.
        if pc.z.abs() < 1e-6 {
            return Err(PnPError::InvalidPose(
                "projection has near-zero depth along z axis",
            ));
        }
        let inv_z = 1.0 / pc.z;
        let u = pc.x * inv_z;
        let v = pc.y * inv_z;
        let du = u - uv[0];
        let dv = v - uv[1];
        sum_sq += du.mul_add(du, dv * dv);
    }
    Ok((sum_sq / (points_world.len() as f32)).sqrt())
}
