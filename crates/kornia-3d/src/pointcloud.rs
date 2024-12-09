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
