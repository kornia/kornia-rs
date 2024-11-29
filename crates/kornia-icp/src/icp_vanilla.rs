use kornia_3d::pointcloud::{PointCloud, Vec3};

fn euclidean_distance(a: &Vec3, b: &Vec3) -> f32 {
    ((a.x - b.x).powi(2) + (a.y - b.y).powi(2) + (a.z - b.z).powi(2)).sqrt()
}

fn find_closest_points_bruteforce<'a>(
    source: &'a PointCloud,
    target: &'a PointCloud,
) -> Vec<&'a Vec3> {
    let closest_points = source
        .points()
        .iter()
        .map(|src_point| {
            target
                .points()
                .iter()
                .min_by(|a, b| {
                    let dist_a = euclidean_distance(src_point, a);
                    let dist_b = euclidean_distance(src_point, b);
                    dist_a.partial_cmp(&dist_b).unwrap()
                })
                .unwrap()
        })
        .collect::<Vec<_>>();
    closest_points
}

fn compute_centroid(points: &Vec<&Vec3>) -> Vec3 {
    let mut centroid = faer::Mat::<f32>::zeros(1, 3);
    points.iter().for_each(|p| {
        let p_mat = faer::mat![[p.x, p.y, p.z]];
        centroid += p_mat;
    });
    let centroid = centroid / points.len() as f32;
    Vec3 {
        x: centroid.read(0, 0),
        y: centroid.read(0, 1),
        z: centroid.read(0, 2),
    }
}

// https://medium.com/@michaelscheinfeild/icp-iterative-closest-point-algorithm-32ecaf58e9da
fn compute_transformation(
    source: &PointCloud,
    target: &Vec<&Vec3>,
    rotation: &mut [[f32; 3]; 3],
    translation: &mut [f32; 3],
) -> Result<(), Box<dyn std::error::Error>> {
    let len = source.points().len();
    let dim = 3;

    let mut source_mat = faer::Mat::<f32>::zeros(len, dim);
    source.points().iter().enumerate().for_each(|(i, p)| {
        source_mat.write(i, 0, p.x);
        source_mat.write(i, 1, p.y);
        source_mat.write(i, 2, p.z);
    });

    let mut target_mat = faer::Mat::<f32>::zeros(target.len(), dim);
    target.iter().enumerate().for_each(|(i, p)| {
        target_mat.write(i, 0, p.x);
        target_mat.write(i, 1, p.y);
        target_mat.write(i, 2, p.z);
    });

    let mut cov = faer::Mat::<f32>::zeros(dim, dim);
    faer::linalg::matmul::matmul(
        &mut cov,
        &source_mat,
        &target_mat,
        None,
        1.0,
        faer::Parallelism::None,
    );

    let svd = cov.svd();
    let (u, v_t) = (svd.u(), svd.v().transpose());

    let mut rotation_mat = faer::Mat::<f32>::zeros(dim, dim);
    faer::linalg::matmul::matmul(
        &mut rotation_mat,
        &u,
        &v_t,
        None,
        1.0,
        faer::Parallelism::None,
    );

    // copy rotation matrix to output
    for i in 0..dim {
        for j in 0..dim {
            rotation[i][j] = rotation_mat.read(i, j);
        }
    }

    // copy translation vector to output
    for i in 0..dim {
        translation[i] = svd.s().read(i);
    }

    Ok(())
}

fn compute_error(
    source: &PointCloud,
    target: &PointCloud,
) -> Result<f32, Box<dyn std::error::Error>> {
    Ok(0.0)
}

fn transform_points(
    source: &PointCloud,
    rotation: &[[f32; 3]; 3],
    translation: &[f32; 3],
) -> Result<PointCloud, Box<dyn std::error::Error>> {
    Ok(PointCloud::new(vec![], None, None))
}

pub fn icp_vanilla(
    source: &PointCloud,
    target: &PointCloud,
    rotation: &mut [[f32; 3]; 3],
    translation: &mut [f32; 3],
    max_iterations: usize,
    tolerance: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut current_source = source.clone();
    for _ in 0..max_iterations {
        let closest_points = find_closest_points_bruteforce(&current_source, target);
        compute_transformation(&current_source, &closest_points, rotation, translation)?;
        let transformed = transform_points(&current_source, &rotation, &translation)?;

        let error = compute_error(&transformed, &target)?;

        current_source = transformed;

        if error < tolerance {
            break;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_closest_points_bruteforce() {
        let pc_src = PointCloud::new(
            vec![Vec3 {
                x: 1.0,
                y: 2.0,
                z: 3.0,
            }],
            None,
            None,
        );
        let pc_tgt = PointCloud::new(
            vec![
                Vec3 {
                    x: 4.0,
                    y: 5.0,
                    z: 6.0,
                },
                Vec3 {
                    x: 7.0,
                    y: 8.0,
                    z: 9.0,
                },
                Vec3 {
                    x: 10.0,
                    y: 11.0,
                    z: 12.0,
                },
            ],
            None,
            None,
        );

        let closest_points = find_closest_points_bruteforce(&pc_src, &pc_tgt);
        println!("{:?}", closest_points);
    }

    #[test]
    fn test_icp_vanilla_identity() -> Result<(), Box<dyn std::error::Error>> {
        let pc_src = PointCloud::new(
            vec![
                Vec3 {
                    x: 1.0,
                    y: 2.0,
                    z: 3.0,
                },
                Vec3 {
                    x: 4.0,
                    y: 5.0,
                    z: 6.0,
                },
            ],
            None,
            None,
        );
        let pc_tgt = PointCloud::new(
            vec![
                Vec3 {
                    x: 1.0,
                    y: 2.0,
                    z: 3.0,
                },
                Vec3 {
                    x: 4.0,
                    y: 5.0,
                    z: 6.0,
                },
            ],
            None,
            None,
        );

        let mut rotation = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let mut translation = [0.0, 0.0, 0.0];

        icp_vanilla(&pc_src, &pc_tgt, &mut rotation, &mut translation, 10, 1e-6)?;

        assert_eq!(
            rotation,
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        );
        assert_eq!(translation, [0.0, 0.0, 0.0]);

        Ok(())
    }
}
