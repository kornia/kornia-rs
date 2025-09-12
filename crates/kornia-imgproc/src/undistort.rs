use anyhow::{bail, Result};
use ndarray::{arr1, s, stack, Array1, Array2, Axis};

/// Computes the ideal point coordinates from the observed point coordinates in a vectorized manner.
pub fn undistort_points(
    src_points: &Array2<f64>,
    camera_matrix: &Array2<f64>,
    dist_coeffs: &Array1<f64>,
    r_matrix: &Option<Array2<f64>>,
    p_matrix: &Option<Array2<f64>>,
) -> Result<Array2<f64>> {
    // --- 1. Validation ---
    if src_points.ndim() != 2 || src_points.shape()[1] != 2 {
        bail!("Input points must be an Nx2 array.");
    }
    if camera_matrix.shape() != [3, 3] {
        bail!("Camera matrix must be a 3x3 array.");
    }

    // --- 2. Extract Coefficients ---
    let fx = camera_matrix[[0, 0]];
    let fy = camera_matrix[[1, 1]];
    let cx = camera_matrix[[0, 2]];
    let cy = camera_matrix[[1, 2]];

    let k1 = dist_coeffs.get(0).copied().unwrap_or(0.0);
    let k2 = dist_coeffs.get(1).copied().unwrap_or(0.0);
    let p1 = dist_coeffs.get(2).copied().unwrap_or(0.0);
    let p2 = dist_coeffs.get(3).copied().unwrap_or(0.0);
    let k3 = dist_coeffs.get(4).copied().unwrap_or(0.0);
    let k4 = dist_coeffs.get(5).copied().unwrap_or(0.0);
    let k5 = dist_coeffs.get(6).copied().unwrap_or(0.0);
    let k6 = dist_coeffs.get(7).copied().unwrap_or(0.0);
    let s1 = dist_coeffs.get(8).copied().unwrap_or(0.0);
    let s2 = dist_coeffs.get(9).copied().unwrap_or(0.0);
    let s3 = dist_coeffs.get(10).copied().unwrap_or(0.0);
    let s4 = dist_coeffs.get(11).copied().unwrap_or(0.0);

    // --- 3. Normalize Distorted Points (Vectorized) ---
    let u = src_points.column(0).to_owned();
    let v = src_points.column(1).to_owned();
    let x_distorted = (u - cx) / fx;
    let y_distorted = (v - cy) / fy;

    // --- 4. Iteratively Find Undistorted Points (Vectorized) ---
    let mut x = x_distorted.clone();
    let mut y = y_distorted.clone();

    for _ in 0..5 {
        let r2 = &x * &x + &y * &y;
        let r4 = &r2 * &r2;
        let r6 = &r4 * &r2;
        
        let radial_numerator = 1.0 + k1 * &r2 + k2 * &r4 + k3 * &r6;
        let radial_denominator = 1.0 + k4 * &r2 + k5 * &r4 + k6 * &r6;
        let radial_dist = &radial_numerator / &radial_denominator;
        let d_tan_x = 2.0 * p1 * &x * &y + p2 * (&r2 + 2.0 * &x * &x);
        let d_tan_y = p1 * (&r2 + 2.0 * &y * &y) + 2.0 * p2 * &x * &y;
        let d_prism_x = s1 * &r2 + s2 * &r4;
        let d_prism_y = s3 * &r2 + s4 * &r4;

        x = (&x_distorted - &d_tan_x - &d_prism_x) / &radial_dist;
        y = (&y_distorted - &d_tan_y - &d_prism_y) / &radial_dist;
    }

    // --- 5. Apply Rectification (R) and New Projection (P) ---
    let ones = Array1::ones(src_points.nrows());
    let undistorted_homo = stack(Axis(1), &[x.view(), y.view(), ones.view()])?;

    let identity = Array2::eye(3);
    let r_mat = r_matrix.as_ref().unwrap_or(&identity);

    // Create an owned Array2<f64> that holds the final transformation matrix.
    let final_transform_matrix = if let Some(p) = p_matrix {
        let p_3x3 = p.slice(s![.., ..3]);
        // The result of dot product is an owned Array2, which is what we want.
        p_3x3.dot(r_mat)
    } else {
        // If no P, the transform is just R. Clone it to create an owned array
        // so that this branch has the same type as the `if` branch.
        r_mat.to_owned()
    };

    // Now, apply the single, final transformation. The types are simple and will compile.
    let projected_homo = undistorted_homo.dot(&final_transform_matrix.t());

    // --- 6. Final Perspective Divide and Output Formatting ---
    let mut dst_points = Array2::zeros((src_points.nrows(), 2));

    let final_x = projected_homo.column(0);
    let final_y = projected_homo.column(1);
    let w = projected_homo.column(2);

    ndarray::azip!((
        mut dst_row in dst_points.rows_mut(),
        &x_i in &final_x,
        &y_i in &final_y,
        &w_i in &w
    ) {
        let w_inv = if w_i.abs() > 1e-6 { 1.0 / w_i } else { 0.0 };
        dst_row.assign(&arr1(&[x_i * w_inv, y_i * w_inv]));
    });

    Ok(dst_points)
}

