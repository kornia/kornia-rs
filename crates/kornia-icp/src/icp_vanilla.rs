use crate::pointcloud::{PointCloud, PointCloudError};

fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

fn find_closest_points_bruteforce(
    source: &PointCloud,
    target: &PointCloud,
) -> Result<Vec<(usize, usize)>, PointCloudError> {
    let mut closest_points = Vec::new();
    source
        .as_slice()
        .chunks_exact(3)
        .enumerate()
        .for_each(|(i, src_point)| {
            let closest_tgt_point_idx = target
                .as_slice()
                .chunks_exact(3)
                .map(|tgt_point| euclidean_distance(src_point, tgt_point))
                .enumerate()
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .unwrap()
                .0;
            closest_points.push((i, closest_tgt_point_idx));
        });
    Ok(closest_points)
}

pub fn icp_vanilla(
    source: &PointCloud,
    target: &PointCloud,
    rotation: &mut [[f32; 3]; 3],
    translation: &mut [f32; 3],
    max_iterations: usize,
) -> Result<(), PointCloudError> {
    for _ in 0..max_iterations {
        let closest_points = find_closest_points_bruteforce(source, target)?;

        //let src_centroid = source
        //    .as_slice()
        //    .chunks_exact(3)
        //    .map(|p| p.to_vec())
        //    .collect::<Vec<_>>()
        //    .iter()
        //    .sum::<Vec<_>>()
        //    / source.num_points() as f32;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_icp_vanilla_identity() -> Result<(), PointCloudError> {
        let pc_src = PointCloud::from_vec(vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])?;
        let pc_tgt = PointCloud::from_vec(vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])?;
        let mut rotation = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let mut translation = [0.0, 0.0, 0.0];

        icp_vanilla(&pc_src, &pc_tgt, &mut rotation, &mut translation, 10)?;

        assert_eq!(
            rotation,
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        );
        assert_eq!(translation, [0.0, 0.0, 0.0]);

        Ok(())
    }
}
