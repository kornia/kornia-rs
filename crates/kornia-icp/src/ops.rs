use kornia_3d::ops::euclidean_distance;
use kornia_3d::utils::*;

/// Compute the transformation between two point clouds.
pub(crate) fn fit_transformation(
    points_in_src: &[[f64; 3]],
    points_in_dst: &[[f64; 3]],
    dst_r_src: &mut [[f64; 3]; 3],
    dst_t_src: &mut [f64; 3],
) {
    assert_eq!(points_in_src.len(), points_in_dst.len());

    // compute centroids
    let (src_centroid, dst_centroid) = compute_centroids(points_in_src, points_in_dst);

    // compute covariance matrix
    let mut hh = faer::Mat::<f64>::zeros(3, 3);
    for (p_in_src, p_in_dst) in points_in_src.iter().zip(points_in_dst.iter()) {
        let p_src = array3_to_faer_col(p_in_src) - &src_centroid;
        let p_dst = array3_to_faer_col(p_in_dst) - &dst_centroid;
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

    // fix the determinant of R in case it is negative as it's a reflection matrix
    if rr.determinant() < 0.0 {
        log::warn!("WARNING: det(R) < 0.0, fixing it...");
        let v_neg = {
            let mut v_neg = v.to_owned();
            v_neg.col_mut(2).copy_from(-v.col(2));
            v_neg
        };
        faer::linalg::matmul::matmul(&mut rr, &v_neg, u_t, None, 1.0, faer::Parallelism::None);
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
    points2: &[[f64; 3]],
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

pub(crate) fn find_correspondences(
    source: &[[f64; 3]],
    target: &[[f64; 3]],
    kdtree: &kiddo::float::kdtree::KdTree<f64, usize, 3, 32, u16>,
) -> (Vec<[f64; 3]>, Vec<[f64; 3]>, Vec<f64>) {
    // find nearest neighbors for each point in source
    let nn_results = source
        .iter()
        .map(|p| kdtree.nearest_one::<kiddo::SquaredEuclidean>(p))
        .collect::<Vec<_>>();

    // compute median distance
    let mut distances = nn_results.iter().map(|nn| nn.distance).collect::<Vec<_>>();
    distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_dist = distances[distances.len() / 2];

    // compute median absolute deviation
    let dmed = distances
        .iter()
        .map(|d| (d - median_dist).abs())
        .collect::<Vec<_>>();
    let mad = dmed[dmed.len() / 2];
    let sigma_d = 1.4826 * mad;

    // put the correspondences in a vector
    let res = nn_results
        .iter()
        .enumerate()
        .filter(|(_, nn)| nn.distance <= median_dist + 3.0 * sigma_d)
        .map(|(i, nn)| (source[i], target[nn.item], nn.distance))
        .collect::<Vec<_>>();

    // unzip the results to separate points and distances
    let (points_in_src, tmp): (Vec<_>, Vec<_>) =
        res.into_iter().map(|(a, b, c)| (a, (b, c))).unzip();
    let (points_in_dst, distances) = tmp.into_iter().unzip();

    (points_in_src, points_in_dst, distances)
}

pub(crate) fn update_transformation(
    rr: &mut [[f64; 3]; 3],
    tt: &mut [f64; 3],
    rr_delta: &[[f64; 3]; 3],
    tt_delta: &[f64; 3],
) {
    // update the translation
    tt[0] += tt_delta[0];
    tt[1] += tt_delta[1];
    tt[2] += tt_delta[2];

    // update the rotation R' = R * R_delta
    let rr_mat_ref = {
        let array_slice = unsafe { std::slice::from_raw_parts_mut(rr.as_mut_ptr() as *mut f64, 9) };
        faer::mat::from_row_major_slice_mut(array_slice, 3, 3)
    };

    let rr_mat = rr_mat_ref.to_owned();
    let rr_delta_mat = {
        let array_slice = unsafe { std::slice::from_raw_parts(rr_delta.as_ptr() as *const f64, 9) };
        faer::mat::from_row_major_slice(array_slice, 3, 3)
    };

    faer::linalg::matmul::matmul(
        rr_mat_ref,
        rr_mat,
        rr_delta_mat,
        None,
        1.0,
        faer::Parallelism::None,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use kornia_3d::{linalg::transform_points3d, transforms::axis_angle_to_rotation_matrix};

    fn create_random_points(num_points: usize) -> Vec<[f64; 3]> {
        (0..num_points)
            .map(|_| {
                [
                    rand::random::<f64>(),
                    rand::random::<f64>(),
                    rand::random::<f64>(),
                ]
            })
            .collect()
    }

    fn create_random_rotation(factor: f64) -> Result<[[f64; 3]; 3], &'static str> {
        let (axis, angle) = (
            [
                rand::random::<f64>(),
                rand::random::<f64>(),
                rand::random::<f64>(),
            ],
            rand::random::<f64>() * factor,
        );
        axis_angle_to_rotation_matrix(&axis, angle)
    }

    fn create_random_translation(factor: f64) -> [f64; 3] {
        [
            rand::random::<f64>() * factor,
            rand::random::<f64>() * factor,
            rand::random::<f64>() * factor,
        ]
    }

    #[test]
    fn test_compute_centroids() {
        let points1 = vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let points2 = vec![[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]];
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
        let points_src = create_random_points(num_points);
        let points_dst = points_src.clone();

        let expected_rotation = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let expected_translation = [0.0, 0.0, 0.0];

        let mut rotation = [[0.0; 3]; 3];
        let mut translation = [0.0; 3];

        fit_transformation(&points_src, &points_dst, &mut rotation, &mut translation);

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
        let points_src = create_random_points(num_points);

        let expected_rotation =
            axis_angle_to_rotation_matrix(&[1.0, 0.0, 0.0], std::f64::consts::PI / 2.0)?;
        let expected_translation = [0.0, 0.0, 0.0];

        let mut points_dst = vec![[0.0; 3]; points_src.len()];
        transform_points3d(
            &points_src,
            &expected_rotation,
            &expected_translation,
            &mut points_dst,
        );

        let mut rotation = [[0.0; 3]; 3];
        let mut translation = [0.0; 3];

        fit_transformation(&points_src, &points_dst, &mut rotation, &mut translation);

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
        let num_test = 10;
        let num_points = 30;
        let translation_factor = 0.1;
        let rotation_factor = 0.1;

        let points_src = create_random_points(num_points);

        for _ in 0..num_test {
            // create random rotation and translation
            let expected_rotation = create_random_rotation(rotation_factor)?;
            let expected_translation = create_random_translation(translation_factor);

            // transform points
            let mut points_dst = vec![[0.0; 3]; num_points];
            transform_points3d(
                &points_src,
                &expected_rotation,
                &expected_translation,
                &mut points_dst,
            );

            let mut rotation = [[0.0; 3]; 3];
            let mut translation = [0.0; 3];

            fit_transformation(&points_src, &points_dst, &mut rotation, &mut translation);

            let mut points_src_fit = vec![[0.0; 3]; num_points];
            transform_points3d(&points_src, &rotation, &translation, &mut points_src_fit);

            for (res, exp) in points_src_fit.iter().zip(points_dst.iter()) {
                for (r, e) in res.iter().zip(exp.iter()) {
                    assert_relative_eq!(r, e, epsilon = 1e-6);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn test_find_correspondences() -> Result<(), Box<dyn std::error::Error>> {
        let points_src = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ];
        let points_dst = vec![[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]];

        let mut kdtree = kiddo::float::kdtree::KdTree::<f64, usize, 3, 32, u16>::new();
        for (i, p) in points_dst.iter().enumerate() {
            kdtree.add(p, i);
        }

        let (points_in_src, points_in_dst, distances) =
            find_correspondences(&points_src, &points_dst, &kdtree);

        assert_eq!(points_in_src.len(), points_in_dst.len());
        assert_eq!(points_in_src.len(), 4);
        assert_eq!(distances[0], 1.0);
        assert_eq!(distances[1], 0.0);
        assert_eq!(distances[2], 1.0);
        assert_eq!(distances[3], 0.0);

        Ok(())
    }
}
