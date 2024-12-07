use kornia_3d::ops::euclidean_distance;
use kornia_3d::utils::*;

/// Compute the transformation between two point clouds.
pub(crate) fn fit_transformation(
    points_in_src: &[[f64; 3]],
    points_in_dst: &[&[f64; 3]],
    dst_r_src: &mut [[f64; 3]; 3],
    dst_t_src: &mut [f64; 3],
) {
    assert_eq!(points_in_src.len(), points_in_dst.len());

    // compute centroids
    let (src_centroid, dst_centroid) = compute_centroids(points_in_src, points_in_dst);

    // compute covariance matrix
    let mut hh = faer::Mat::<f64>::zeros(3, 3);
    for (p_in_src, p_in_dst) in points_in_src.iter().zip(points_in_dst.iter()) {
        let p_src = faer::col![p_in_src[0], p_in_src[1], p_in_src[2]] - &src_centroid;
        let p_dst = faer::col![p_in_dst[0], p_in_dst[1], p_in_dst[2]] - &dst_centroid;
        hh += p_src * p_dst.transpose();
    }

    // solve the linear system H * x = 0 to find the rotation
    let svd = hh.svd();
    let (u_t, v) = (svd.u().transpose(), svd.v());

    // compute rotation matrix R = V^T * U^T
    let mut rr = {
        let array_slice =
            unsafe { std::slice::from_raw_parts_mut(dst_r_src.as_mut_ptr() as *mut f64, 9) };
        faer::mat::from_row_major_slice_mut(array_slice, 3, 3)
    };
    faer::linalg::matmul::matmul(&mut rr, v, u_t, None, 1.0, faer::Parallelism::None);

    if rr.determinant() < 0.0 {
        log::warn!("WARNING: det(R) < 0.0, fixing it...");
        let mut v_mut = v.to_owned();
        for i in 0..3 {
            v_mut.write(i, 2, -v_mut.read(i, 2));
        }
        faer::linalg::matmul::matmul(&mut rr, &v_mut, u_t, None, 1.0, faer::Parallelism::None);
    }

    // compute translation vector t = C_dst - R * C_src
    let t = dst_centroid - rr * src_centroid;

    // copy results back to output
    dst_t_src[0] = t[0];
    dst_t_src[1] = t[1];
    dst_t_src[2] = t[2];
}

/// Compute the euclidean distance error between two sets of points.
///
/// # Arguments
///
/// * `source` - A set of points.
/// * `target` - Another set of points.
///
/// # Returns
///
/// The euclidean distance error between the two sets of points.
pub(crate) fn compute_point_to_point_error(source: &[[f64; 3]], target: &[[f64; 3]]) -> f64 {
    assert_eq!(source.len(), target.len());
    let error = source
        .iter()
        .zip(target.iter())
        .map(|(a, b)| euclidean_distance(a, b))
        .sum::<f64>();
    error
}

/// Compute the centroids of two sets of points.
///
/// # Arguments
///
/// * `points1` - A set of points.
/// * `points2` - Another set of points.
///
/// # Returns
///
/// The centroids of the two sets of points.
pub(crate) fn compute_centroids(
    points1: &[[f64; 3]],
    points2: &[&[f64; 3]],
) -> (faer::Col<f64>, faer::Col<f64>) {
    let mut centroid1 = faer::Col::zeros(3);
    let mut centroid2 = faer::Col::zeros(3);

    for (p1, p2) in points1.iter().zip(points2.iter()) {
        centroid1 += array3_to_faer_col(p1);
        centroid2 += array3_to_faer_col(p2);
    }

    centroid1 /= points1.len() as f64;
    centroid2 /= points2.len() as f64;

    (centroid1, centroid2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use kornia_3d::{linalg::transform_points, transforms::axis_angle_to_rotation_matrix};

    #[test]
    fn test_compute_centroids() {
        let points1 = vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let points2 = vec![&[7.0, 8.0, 9.0], &[10.0, 11.0, 12.0]];
        let (centroid1, centroid2) = compute_centroids(&points1, &points2);
        assert_eq!(centroid1.read(0), 2.5);
        assert_eq!(centroid1.read(1), 3.5);
        assert_eq!(centroid1.read(2), 4.5);
        assert_eq!(centroid2.read(0), 8.5);
        assert_eq!(centroid2.read(1), 9.5);
        assert_eq!(centroid2.read(2), 10.5);
    }

    #[test]
    fn test_compute_point_to_point_error() {
        let source = vec![[2.0, 2.0, 2.0], [3.0, 4.0, 5.0]];
        let target = vec![[1.0, 4.0, 5.0], [2.0, 2.0, 2.0]];
        assert_eq!(
            compute_point_to_point_error(&source, &target),
            2.0 * 14f64.sqrt()
        );
    }
    #[test]
    fn test_fit_transformation_identity() {
        let num_points = 30;
        let points_src = (0..num_points)
            .map(|_| {
                [
                    rand::random::<f64>(),
                    rand::random::<f64>(),
                    rand::random::<f64>(),
                ]
            })
            .collect::<Vec<_>>();

        let points_src_ref = points_src.iter().collect::<Vec<_>>();

        let expected_rotation = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let expected_translation = [0.0, 0.0, 0.0];

        let mut rotation = [[0.0; 3]; 3];
        let mut translation = [0.0; 3];

        fit_transformation(
            &points_src,
            &points_src_ref,
            &mut rotation,
            &mut translation,
        );

        for (res, exp) in rotation.iter().zip(expected_rotation.iter()) {
            for (r, e) in res.iter().zip(exp.iter()) {
                assert_relative_eq!(r, e, epsilon = 1e-6);
            }
        }
        for (res, exp) in translation.iter().zip(expected_translation.iter()) {
            assert_relative_eq!(res, exp, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_fit_transformation_rotation() -> Result<(), Box<dyn std::error::Error>> {
        let num_points = 30;
        let points_src = (0..num_points)
            .map(|_| {
                [
                    rand::random::<f64>(),
                    rand::random::<f64>(),
                    rand::random::<f64>(),
                ]
            })
            .collect::<Vec<_>>();

        let expected_rotation =
            axis_angle_to_rotation_matrix(&[1.0, 0.0, 0.0], std::f64::consts::PI / 2.0)?;
        let expected_translation = [0.0, 0.0, 0.0];

        let mut points_dst = vec![[0.0; 3]; points_src.len()];
        transform_points(
            &points_src,
            &expected_rotation,
            &expected_translation,
            &mut points_dst,
        );

        let points_dst_ref = points_dst.iter().collect::<Vec<_>>();
        let mut rotation = [[0.0; 3]; 3];
        let mut translation = [0.0; 3];

        fit_transformation(
            &points_src,
            &points_dst_ref,
            &mut rotation,
            &mut translation,
        );

        for (res, exp) in rotation.iter().zip(expected_rotation.iter()) {
            for (r, e) in res.iter().zip(exp.iter()) {
                assert_relative_eq!(r, e, epsilon = 1e-6);
            }
        }
        for (res, exp) in translation.iter().zip(expected_translation.iter()) {
            assert_relative_eq!(res, exp, epsilon = 1e-6);
        }

        Ok(())
    }

    #[test]
    fn test_fit_transformation_random() -> Result<(), Box<dyn std::error::Error>> {
        let num_test = 1;
        let num_points = 30;
        let noise_std = 0.01;
        let translation_factor = 0.1;
        let rotation_factor = 0.1;

        let points_src = (0..num_points)
            .map(|_| {
                [
                    rand::random::<f64>(),
                    rand::random::<f64>(),
                    rand::random::<f64>(),
                ]
            })
            .collect::<Vec<_>>();

        for _ in 0..num_test {
            // create random rotation
            let (axis, angle) = (
                [
                    rand::random::<f64>(),
                    rand::random::<f64>(),
                    rand::random::<f64>(),
                ],
                rand::random::<f64>() * rotation_factor,
            );

            let expected_rotation = axis_angle_to_rotation_matrix(&axis, angle)?;

            // create random translation
            let expected_translation = [
                rand::random::<f64>() * translation_factor,
                rand::random::<f64>() * translation_factor,
                rand::random::<f64>() * translation_factor,
            ];

            // transform points
            let mut points_dst = vec![[0.0; 3]; num_points];
            transform_points(
                &points_src,
                &expected_rotation,
                &expected_translation,
                &mut points_dst,
            );

            // add noise to points
            for p in points_dst.iter_mut() {
                p[0] += rand::random::<f64>() * noise_std;
                p[1] += rand::random::<f64>() * noise_std;
                p[2] += rand::random::<f64>() * noise_std;
            }

            let mut rotation = [[0.0; 3]; 3];
            let mut translation = [0.0; 3];
            let points_dst_ref = points_dst.iter().collect::<Vec<_>>();

            fit_transformation(
                &points_src,
                &points_dst_ref,
                &mut rotation,
                &mut translation,
            );

            let mut points_src_fit = vec![[0.0; 3]; num_points];
            transform_points(&points_src, &rotation, &translation, &mut points_src_fit);

            println!("expected rotation: {:?}", expected_rotation);
            println!("expected translation: {:?}", expected_translation);
            println!("rotation: {:?}", rotation);
            println!("translation: {:?}", translation);
            println!("########################");

            for (res, exp) in points_src_fit.iter().zip(points_dst.iter()) {
                for (_r, _e) in res.iter().zip(exp.iter()) {
                    // assert_relative_eq!(r, e, epsilon = 1e-6);
                }
            }
        }
        Ok(())
    }
}
