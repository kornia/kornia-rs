use crate::linalg;
use kiddo::immutable::float::kdtree::ImmutableKdTree;
use kornia_algebra::{Mat3F64, Vec3F64, SO3F64};

/// Compute the transformation between two point clouds.
pub(crate) fn fit_transformation(
    points_in_src: &[[f64; 3]],
    points_in_dst: &[[f64; 3]],
    dst_r_src: &mut [[f64; 3]; 3],
    dst_t_src: &mut [f64; 3],
) -> Option<()> {
    assert_eq!(points_in_src.len(), points_in_dst.len());

    // compute centroids
    let (src_centroid, dst_centroid) = compute_centroids(points_in_src, points_in_dst);

    let cs = Vec3F64::new(src_centroid[0], src_centroid[1], src_centroid[2]);
    let cd = Vec3F64::new(dst_centroid[0], dst_centroid[1], dst_centroid[2]);

    // Kabsch: with the cross-covariance H = Σ (p_dst − c_dst)(p_src − c_src)ᵀ, the best rotation
    // is the one nearest to H (also when the SVD's U·Vᵀ is a reflection).
    let mut hh = Mat3F64::ZERO;
    for (p_in_src, p_in_dst) in points_in_src.iter().zip(points_in_dst.iter()) {
        let p_src = Vec3F64::from(*p_in_src) - cs;
        let p_dst = Vec3F64::from(*p_in_dst) - cd;
        hh += Mat3F64::from_cols(p_dst * p_src.x, p_dst * p_src.y, p_dst * p_src.z);
    }
    let rr = SO3F64::from_matrix_nearest(&hh)?.matrix();

    // compute translation vector t = C_dst - R * C_src
    let t = cd - rr * cs;

    // copy results back to output: `dst_r_src` is row-major, and Rᵀ's columns are R's rows
    *dst_r_src = rr.transpose().to_cols_array_2d();
    *dst_t_src = t.into();
    Some(())
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
        centroid1 += faer::col![p1[0], p1[1], p1[2]];
        centroid2 += faer::col![p2[0], p2[1], p2[2]];
    }

    centroid1 /= points1.len() as f64;
    centroid2 /= points2.len() as f64;

    (centroid1, centroid2)
}

pub(crate) fn find_correspondences(
    source: &[[f64; 3]],
    target: &[[f64; 3]],
    kdtree: &ImmutableKdTree<f64, u32, 3, 32>,
) -> (Vec<[f64; 3]>, Vec<[f64; 3]>, Vec<f64>) {
    // find nearest neighbors for each point in source
    let nn_results = source
        .iter()
        .map(|p| kdtree.nearest_one::<kiddo::SquaredEuclidean>(p))
        .collect::<Vec<_>>();

    let mut distances = nn_results.iter().map(|nn| nn.distance).collect::<Vec<_>>();
    if distances.is_empty() {
        return (Vec::new(), Vec::new(), Vec::new());
    }

    let mid_dist = distances.len() / 2;
    distances.select_nth_unstable_by(mid_dist, |a, b| a.total_cmp(b));
    let median_dist = distances[mid_dist];

    let mut dmed = nn_results
        .iter()
        .map(|nn| (nn.distance - median_dist).abs())
        .collect::<Vec<_>>();

    let mid_mad = dmed.len() / 2;
    dmed.select_nth_unstable_by(mid_mad, |a, b| a.total_cmp(b));
    let mad = dmed[mid_mad];

    let sigma_d = 1.4826 * mad;
    let threshold = median_dist + 3.0 * sigma_d;

    let res = nn_results
        .iter()
        .enumerate()
        .filter(|(_, nn)| nn.distance <= threshold)
        .map(|(i, nn)| (source[i], target[nn.item as usize], nn.distance))
        .collect::<Vec<_>>();

    let mut points_in_src = Vec::with_capacity(res.len());
    let mut points_in_dst = Vec::with_capacity(res.len());
    let mut distances_vec = Vec::with_capacity(res.len());

    for (src, dst, dist) in res {
        points_in_src.push(src);
        points_in_dst.push(dst);
        distances_vec.push(dist);
    }

    (points_in_src, points_in_dst, distances_vec)
}

pub(crate) fn update_transformation(
    rr: &mut [[f64; 3]; 3],
    tt: &mut [f64; 3],
    rr_delta: &[[f64; 3]; 3],
    tt_delta: &[f64; 3],
) {
    // Avoid cloning by passing a mutable reference directly
    linalg::matmul33(&rr.clone(), rr_delta, rr);

    // Update translation vector
    tt[0] += tt_delta[0];
    tt[1] += tt_delta[1];
    tt[2] += tt_delta[2];
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{linalg::transform_points3d, transforms::axis_angle_to_rotation_matrix};
    use approx::assert_relative_eq;
    use kiddo::immutable::float::kdtree::ImmutableKdTree;

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
        let c1: Vec<f64> = centroid1.as_ref().iter().copied().collect();
        let c2: Vec<f64> = centroid2.as_ref().iter().copied().collect();
        assert_eq!(c1, vec![2.5, 3.5, 4.5]);
        assert_eq!(c2, vec![8.5, 9.5, 10.5]);
    }

    #[test]
    fn test_fit_transformation_identity() -> Result<(), &'static str> {
        let num_points = 30;
        let points_src = create_random_points(num_points);
        let points_dst = points_src.clone();

        let expected_rotation = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let expected_translation = [0.0, 0.0, 0.0];

        let mut rotation = [[0.0; 3]; 3];
        let mut translation = [0.0; 3];

        fit_transformation(&points_src, &points_dst, &mut rotation, &mut translation)
            .ok_or("SVD failed")?;

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
        )?;

        let mut rotation = [[0.0; 3]; 3];
        let mut translation = [0.0; 3];

        fit_transformation(&points_src, &points_dst, &mut rotation, &mut translation)
            .ok_or("SVD failed")?;

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
            )?;

            let mut rotation = [[0.0; 3]; 3];
            let mut translation = [0.0; 3];

            fit_transformation(&points_src, &points_dst, &mut rotation, &mut translation)
                .ok_or("SVD failed")?;

            let mut points_src_fit = vec![[0.0; 3]; num_points];
            transform_points3d(&points_src, &rotation, &translation, &mut points_src_fit)?;

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

        let kdtree = ImmutableKdTree::new_from_slice(&points_dst);

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
