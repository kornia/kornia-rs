use glam::Vec3;

/// A point cloud with points, colors, and normals.
#[derive(Debug, Clone)]
pub struct PointCloud {
    // The points in the point cloud.
    points: Vec<[f64; 3]>,
    // The colors of the points.
    colors: Option<Vec<[u8; 3]>>,
    // The normals of the points.
    normals: Option<Vec<[f64; 3]>>,
}

impl PointCloud {
    /// Create a new point cloud from points, colors (optional), and normals (optional).
    pub fn new(
        points: Vec<[f64; 3]>,
        colors: Option<Vec<[u8; 3]>>,
        normals: Option<Vec<[f64; 3]>>,
    ) -> Self {
        Self {
            points,
            colors,
            normals,
        }
    }

    /// Get the number of points in the point cloud.
    #[inline]
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Check if the point cloud is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Get as reference the points in the point cloud.
    pub fn points(&self) -> &Vec<[f64; 3]> {
        &self.points
    }

    /// Get as reference the colors of the points in the point cloud.
    pub fn colors(&self) -> Option<&Vec<[u8; 3]>> {
        self.colors.as_ref()
    }

    /// Get as reference the normals of the points in the point cloud.
    pub fn normals(&self) -> Option<&Vec<[f64; 3]>> {
        self.normals.as_ref()
    }

    /// Convert a point from [f64; 3] to Vec3.
    fn point_to_vec3(point: &[f64; 3]) -> Vec3 {
        Vec3::new(point[0] as f32, point[1] as f32, point[2] as f32)
    }

    /// Convert a Vec3 to [f64; 3].
    fn vec3_to_point(vec: Vec3) -> [f64; 3] {
        [vec.x as f64, vec.y as f64, vec.z as f64]
    }

    /// Get the minimum bound of the point cloud.
    pub fn get_min_bound(&self) -> Vec3 {
        if self.points.is_empty() {
            return Vec3::ZERO;
        }
        self.points()
            .iter()
            .map(|&point| Self::point_to_vec3(&point))
            .fold(Self::point_to_vec3(&self.points[0]), |a, b| a.min(b))
    }

    /// Get the maximum bound of the point cloud.
    pub fn get_max_bound(&self) -> Vec3 {
        if self.points.is_empty() {
            return Vec3::ZERO;
        }
        self.points()
            .iter()
            .map(|&point| Self::point_to_vec3(&point))
            .fold(Self::point_to_vec3(&self.points[0]), |a, b| a.max(b))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pointcloud() {
        let pointcloud = PointCloud::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            Some(vec![[255, 0, 0], [0, 255, 0]]),
            Some(vec![[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
        );

        assert_eq!(pointcloud.len(), 2);
        assert_eq!(pointcloud.points().len(), 2);

        if let Some(colors) = pointcloud.colors() {
            assert_eq!(colors.len(), 2);
        }
        if let Some(normals) = pointcloud.normals() {
            assert_eq!(normals.len(), 2);
        }

        if let Some(p0) = pointcloud.points().first() {
            assert_eq!(p0[0], 0.0);
            assert_eq!(p0[1], 0.0);
            assert_eq!(p0[2], 0.0);
        }

        if let Some(p1) = pointcloud.points().last() {
            assert_eq!(p1[0], 1.0);
            assert_eq!(p1[1], 0.0);
            assert_eq!(p1[2], 0.0);
        }
    }
}
