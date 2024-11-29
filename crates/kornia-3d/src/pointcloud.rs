/// Simple 3D vector with x, y, and z coordinates.
#[derive(Debug, Clone)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

/// A point cloud with points, colors, and normals.
#[derive(Debug, Clone)]
pub struct PointCloud {
    // The points in the point cloud.
    points: Vec<Vec3>,
    // The colors of the points.
    colors: Option<Vec<Vec3>>,
    // The normals of the points.
    normals: Option<Vec<Vec3>>,
}

impl PointCloud {
    /// Create a new point cloud from points, colors (optional), and normals (optional).
    pub fn new(points: Vec<Vec3>, colors: Option<Vec<Vec3>>, normals: Option<Vec<Vec3>>) -> Self {
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
    pub fn points(&self) -> &Vec<Vec3> {
        &self.points
    }

    /// Get as reference the colors of the points in the point cloud.
    pub fn colors(&self) -> Option<&Vec<Vec3>> {
        self.colors.as_ref()
    }

    /// Get as reference the normals of the points in the point cloud.
    pub fn normals(&self) -> Option<&Vec<Vec3>> {
        self.normals.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pointcloud() {
        let pointcloud = PointCloud::new(
            vec![
                Vec3 {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                },
                Vec3 {
                    x: 1.0,
                    y: 0.0,
                    z: 0.0,
                },
            ],
            Some(vec![
                Vec3 {
                    x: 1.0,
                    y: 0.0,
                    z: 0.0,
                },
                Vec3 {
                    x: 0.0,
                    y: 1.0,
                    z: 0.0,
                },
            ]),
            Some(vec![
                Vec3 {
                    x: 0.0,
                    y: 1.0,
                    z: 0.0,
                },
                Vec3 {
                    x: 1.0,
                    y: 0.0,
                    z: 0.0,
                },
            ]),
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
            assert_eq!(p0.x, 0.0);
            assert_eq!(p0.y, 0.0);
            assert_eq!(p0.z, 0.0);
        }

        if let Some(p1) = pointcloud.points().last() {
            assert_eq!(p1.x, 1.0);
            assert_eq!(p1.y, 0.0);
            assert_eq!(p1.z, 0.0);
        }
    }
}
