/// Outlier removal, and the errors of index selection, on point clouds.
pub mod filter;

mod view;

use std::borrow::Cow;

use filter::{check_attributes, FilterError};
pub use view::PointCloudView;

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

    /// A view of the points at `indices`, with their colors and normals,
    /// borrowed from this cloud (nothing is copied).
    ///
    /// The indices and the attribute lengths are checked once here, so the
    /// view's accessors cannot fail on a valid position. Call
    /// [`PointCloudView::to_point_cloud`] for an owned copy.
    ///
    /// # Arguments
    ///
    /// * `indices` - Indices of the points to select, in view order (repeats
    ///   allowed): a `Vec<u32>` is moved into the view, a `&[u32]` (or
    ///   `&Vec<u32>`, `&[u32; N]`) is borrowed.
    ///
    /// # Returns
    ///
    /// A view with one point per index.
    ///
    /// # Errors
    ///
    /// * [`FilterError::IndexOutOfRange`] if an index is not a point of the
    ///   cloud (the first such index).
    /// * [`FilterError::AttributeLengthMismatch`] if the colors or normals do
    ///   not have one entry per point.
    ///
    /// # Example
    ///
    /// ```
    /// use kornia_3d::pointcloud::PointCloud;
    ///
    /// let cloud = PointCloud::new(vec![[0.0; 3], [1.0; 3], [2.0; 3]], None, None);
    /// let picked = cloud.select(&[2, 0])?;
    /// assert_eq!(picked.points().collect::<Vec<_>>(), vec![&[2.0; 3], &[0.0; 3]]);
    /// assert!(cloud.select(vec![3]).is_err());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn select<'a>(
        &'a self,
        indices: impl Into<Cow<'a, [u32]>>,
    ) -> Result<PointCloudView<'a>, FilterError> {
        check_attributes(self)?;
        let indices = indices.into();
        let len = self.len();
        if let Some(&index) = indices.iter().find(|&&i| i as usize >= len) {
            return Err(FilterError::IndexOutOfRange { index, len });
        }
        Ok(PointCloudView::from_checked(self, indices))
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
