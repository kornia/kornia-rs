//! Rigid alignment utilities (Kabsch / Umeyama)

// We use the f64 version of SVD for higher precision, then cast back to f32
use glam::{DMat3, DVec3, Vec3};
use thiserror::Error;

/// Rotation (R), translation (t), and scale (s) output of Umeyama without scaling (s = 1).
pub type UmeyamaOutput = ([[f32; 3]; 3], [f32; 3], f32);

/// Error type for Umeyama rigid alignment operations.
#[derive(Debug, Error)]
pub enum UmeyamaError {
    /// Source and destination arrays must have the same length
    #[error("Source and destination arrays must have the same length")]
    MismatchedInputLengths,
}

/// Result type alias for Umeyama.
pub type UmeyamaResult = Result<UmeyamaOutput, UmeyamaError>;

/// Umeyama/Kabsch algorithm without scale.
/// Returns (R, t, s) where s == 1.0.
pub fn umeyama(src: &[Vec3], dst: &[Vec3]) -> UmeyamaResult {
    if src.len() != dst.len() {
        return Err(UmeyamaError::MismatchedInputLengths);
    }
    let n = src.len() as f64;

    // 1. Compute Centroids (f64 precision)
    let mut mu_s = [0.0f64; 3];
    let mut mu_d = [0.0f64; 3];
    for i in 0..src.len() {
        mu_s[0] += src[i].x as f64;
        mu_s[1] += src[i].y as f64;
        mu_s[2] += src[i].z as f64;

        mu_d[0] += dst[i].x as f64;
        mu_d[1] += dst[i].y as f64;
        mu_d[2] += dst[i].z as f64;
    }
    for x in &mut mu_s {
        *x /= n;
    }
    for x in &mut mu_d {
        *x /= n;
    }

    // 2. Compute Covariance Matrix H = D * S^T (f64 precision)
    // H_ij = Sum( (dst_i - mu_d) * (src_j - mu_s) )
    let mut h = [[0.0f64; 3]; 3];
    for i in 0..src.len() {
        let sc = [
            (src[i].x as f64) - mu_s[0],
            (src[i].y as f64) - mu_s[1],
            (src[i].z as f64) - mu_s[2],
        ];
        let dc = [
            (dst[i].x as f64) - mu_d[0],
            (dst[i].y as f64) - mu_d[1],
            (dst[i].z as f64) - mu_d[2],
        ];

        // H[row][col] += Dest[row] * Source[col]
        for (r, &dc_r) in dc.iter().enumerate() {
            for (c, &sc_c) in sc.iter().enumerate() {
                h[r][c] += dc_r * sc_c;
            }
        }
    }

    // Average the covariance
    for row in &mut h {
        for val in row {
            *val /= n;
        }
    }

    // 3. Convert to glam and Run SVD (f64)
    // We manually construct DMat3 because 'h' is a primitive array
    let h_f64 = DMat3::from_cols(
        DVec3::new(h[0][0], h[1][0], h[2][0]),
        DVec3::new(h[0][1], h[1][1], h[2][1]),
        DVec3::new(h[0][2], h[1][2], h[2][2]),
    );

    // Call our new High-Precision SVD
    let svd_result = crate::svd::svd3_f64(h_f64);

    let u = svd_result.u;
    let v = svd_result.v;

    // 4. Calculate Rotation R = U * V^T (Standard Kabsch)
    let d = (u * v.transpose()).determinant();

    let r_glam_f64 = if d < 0.0 {
        // Handle Reflection: R = U * diag(1, 1, -1) * V^T
        let correction = DMat3::from_diagonal(DVec3::new(1.0, 1.0, -1.0));
        u * correction * v.transpose()
    } else {
        u * v.transpose()
    };

    // 6. Convert to Output Format
    let r_arr = [
        [
            r_glam_f64.x_axis.x as f32,
            r_glam_f64.y_axis.x as f32,
            r_glam_f64.z_axis.x as f32,
        ],
        [
            r_glam_f64.x_axis.y as f32,
            r_glam_f64.y_axis.y as f32,
            r_glam_f64.z_axis.y as f32,
        ],
        [
            r_glam_f64.x_axis.z as f32,
            r_glam_f64.y_axis.z as f32,
            r_glam_f64.z_axis.z as f32,
        ],
    ];

    let mut t = [0.0; 3];
    for r in 0..3 {
        t[r] = (mu_d[r] - (r_glam_f64.row(r).dot(DVec3::from_slice(&mu_s)))) as f32;
    }

    Ok((r_arr, t, 1.0))
}
