//! [`PointCloudView`]: a borrowed subset of a [`PointCloud`], given by point
//! indices, without copying the points.

use std::borrow::Cow;

use super::PointCloud;

/// A subset of a [`PointCloud`]: the cloud, borrowed, and the indices of the
/// selected points, in view order (repeats allowed). No point, color or
/// normal is copied; [`to_point_cloud`](Self::to_point_cloud) makes the
/// owned copy when one is needed.
///
/// Built by [`PointCloud::select`], which checks every index and the
/// attribute lengths once, so the accessors never fail on a valid position.
/// The voxel reductions ([`crate::voxel::VoxelView::first`],
/// [`crate::voxel::VoxelView::nearest`]) and the filters that keep a subset
/// of their input return one too.
///
/// # Example
///
/// ```
/// use kornia_3d::pointcloud::PointCloud;
///
/// let cloud = PointCloud::new(
///     vec![[0.0; 3], [1.0; 3], [2.0; 3]],
///     Some(vec![[10; 3], [20; 3], [30; 3]]),
///     None,
/// );
/// let view = cloud.select(vec![2, 0])?;
/// assert_eq!(view.len(), 2);
/// assert_eq!(view.point(0), Some(&[2.0; 3]));
/// let colors: Vec<_> = view.colors().into_iter().flatten().copied().collect();
/// assert_eq!(colors, vec![[30; 3], [10; 3]]);
/// assert!(view.normals().is_none());
/// let owned = view.to_point_cloud();
/// assert_eq!(owned.points(), &vec![[2.0; 3], [0.0; 3]]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
pub struct PointCloudView<'a> {
    cloud: &'a PointCloud,
    /// Every entry is `< cloud.len()`, and the cloud's colors and normals,
    /// when present, have one entry per point.
    indices: Cow<'a, [u32]>,
}

impl<'a> PointCloudView<'a> {
    /// Wraps indices already checked against `cloud` (and a cloud whose
    /// attribute lengths are checked).
    pub(crate) fn from_checked(cloud: &'a PointCloud, indices: Cow<'a, [u32]>) -> Self {
        Self { cloud, indices }
    }

    /// Number of points in the view.
    ///
    /// # Arguments
    ///
    /// This method takes no arguments beyond `self`.
    ///
    /// # Returns
    ///
    /// The number of selected indices (repeats counted).
    ///
    /// # Errors
    ///
    /// This method is infallible.
    #[inline]
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    /// Whether the view selects no point.
    ///
    /// # Arguments
    ///
    /// This method takes no arguments beyond `self`.
    ///
    /// # Returns
    ///
    /// `true` when [`len`](Self::len) is `0`.
    ///
    /// # Errors
    ///
    /// This method is infallible.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// The cloud the view selects from.
    ///
    /// # Arguments
    ///
    /// This method takes no arguments beyond `self`.
    ///
    /// # Returns
    ///
    /// The borrowed source cloud.
    ///
    /// # Errors
    ///
    /// This method is infallible.
    #[inline]
    pub fn cloud(&self) -> &'a PointCloud {
        self.cloud
    }

    /// The selected point indices into [`cloud`](Self::cloud), in view order.
    ///
    /// # Arguments
    ///
    /// This method takes no arguments beyond `self`.
    ///
    /// # Returns
    ///
    /// One index per point of the view.
    ///
    /// # Errors
    ///
    /// This method is infallible.
    #[inline]
    pub fn indices(&self) -> &[u32] {
        &self.indices
    }

    /// Takes the index list out of the view, so a caller can reuse its
    /// buffer (for example as the `dst` of the next `*_into` call).
    ///
    /// # Arguments
    ///
    /// This method takes no arguments beyond `self`.
    ///
    /// # Returns
    ///
    /// The indices, moved out when the view owns them, copied when it
    /// borrows them.
    ///
    /// # Errors
    ///
    /// This method is infallible.
    pub fn into_indices(self) -> Vec<u32> {
        self.indices.into_owned()
    }

    /// The point at position `i` of the view.
    ///
    /// # Arguments
    ///
    /// * `i` - Position in the view, `0..len()`.
    ///
    /// # Returns
    ///
    /// The point, or `None` when `i >= len()`.
    ///
    /// # Errors
    ///
    /// This method is infallible.
    #[inline]
    pub fn point(&self, i: usize) -> Option<&'a [f64; 3]> {
        let cloud = self.cloud;
        self.indices
            .get(i)
            .and_then(|&j| cloud.points().get(j as usize))
    }

    /// The color of the point at position `i` of the view.
    ///
    /// # Arguments
    ///
    /// * `i` - Position in the view, `0..len()`.
    ///
    /// # Returns
    ///
    /// The color, or `None` when the cloud has no colors or `i >= len()`.
    ///
    /// # Errors
    ///
    /// This method is infallible.
    #[inline]
    pub fn color(&self, i: usize) -> Option<&'a [u8; 3]> {
        let colors = self.cloud.colors()?;
        self.indices.get(i).and_then(|&j| colors.get(j as usize))
    }

    /// The normal of the point at position `i` of the view.
    ///
    /// # Arguments
    ///
    /// * `i` - Position in the view, `0..len()`.
    ///
    /// # Returns
    ///
    /// The normal, or `None` when the cloud has no normals or `i >= len()`.
    ///
    /// # Errors
    ///
    /// This method is infallible.
    #[inline]
    pub fn normal(&self, i: usize) -> Option<&'a [f64; 3]> {
        let normals = self.cloud.normals()?;
        self.indices.get(i).and_then(|&j| normals.get(j as usize))
    }

    /// The points of the view, in view order, borrowed from the cloud.
    ///
    /// # Arguments
    ///
    /// This method takes no arguments beyond `self`.
    ///
    /// # Returns
    ///
    /// An iterator over [`len`](Self::len) points; it does not allocate.
    ///
    /// # Errors
    ///
    /// This method is infallible.
    pub fn points(&self) -> impl ExactSizeIterator<Item = &'a [f64; 3]> + '_ {
        gather(&self.indices, self.cloud.points())
    }

    /// The colors of the view's points, in view order, borrowed from the
    /// cloud.
    ///
    /// # Arguments
    ///
    /// This method takes no arguments beyond `self`.
    ///
    /// # Returns
    ///
    /// An iterator over [`len`](Self::len) colors (it does not allocate), or
    /// `None` when the cloud has no colors.
    ///
    /// # Errors
    ///
    /// This method is infallible.
    pub fn colors(&self) -> Option<impl ExactSizeIterator<Item = &'a [u8; 3]> + '_> {
        let colors = self.cloud.colors()?;
        Some(gather(&self.indices, colors))
    }

    /// The normals of the view's points, in view order, borrowed from the
    /// cloud.
    ///
    /// # Arguments
    ///
    /// This method takes no arguments beyond `self`.
    ///
    /// # Returns
    ///
    /// An iterator over [`len`](Self::len) normals (it does not allocate),
    /// or `None` when the cloud has no normals.
    ///
    /// # Errors
    ///
    /// This method is infallible.
    pub fn normals(&self) -> Option<impl ExactSizeIterator<Item = &'a [f64; 3]> + '_> {
        let normals = self.cloud.normals()?;
        Some(gather(&self.indices, normals))
    }

    /// Copies the view into an owned cloud: its points, with their colors
    /// and normals when the source has them, in view order.
    ///
    /// # Arguments
    ///
    /// This method takes no arguments beyond `self`.
    ///
    /// # Returns
    ///
    /// A cloud with [`len`](Self::len) points.
    ///
    /// # Errors
    ///
    /// This method is infallible.
    pub fn to_point_cloud(&self) -> PointCloud {
        let points = self.points().copied().collect();
        let colors = self.colors().map(|c| c.copied().collect());
        let normals = self.normals().map(|n| n.copied().collect());
        PointCloud::new(points, colors, normals)
    }
}

/// The entries of `values` at `indices`. Every index is in range (the
/// [`PointCloudView`] invariant).
fn gather<'i, 'v: 'i, T>(
    indices: &'i [u32],
    values: &'v [T],
) -> impl ExactSizeIterator<Item = &'v T> + 'i {
    indices.iter().map(move |&i| &values[i as usize])
}

#[cfg(test)]
mod tests;
