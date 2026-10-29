//! Index selection on point clouds: [`select_points`] copies the points at
//! given indices, with their colors and normals.

use super::PointCloud;

/// Errors of the point cloud filters.
#[derive(Debug, thiserror::Error)]
pub enum FilterError {
    /// A selected index is not a point of the cloud.
    #[error("index {index} is out of range for {len} points")]
    IndexOutOfRange {
        /// Offending index.
        index: u32,
        /// Points in the cloud.
        len: usize,
    },

    /// The colors or normals of a cloud do not have one entry per point.
    #[error("{attribute} has {len} entries for {expected} points")]
    AttributeLengthMismatch {
        /// `"colors"` or `"normals"`.
        attribute: &'static str,
        /// Entries of the attribute.
        len: usize,
        /// Points in the cloud.
        expected: usize,
    },
}

/// Checks that `cloud`'s colors and normals, when present, have one entry
/// per point.
pub(crate) fn check_attributes(cloud: &PointCloud) -> Result<(), FilterError> {
    let expected = cloud.len();
    if let Some(c) = cloud.colors() {
        if c.len() != expected {
            return Err(FilterError::AttributeLengthMismatch {
                attribute: "colors",
                len: c.len(),
                expected,
            });
        }
    }
    if let Some(n) = cloud.normals() {
        if n.len() != expected {
            return Err(FilterError::AttributeLengthMismatch {
                attribute: "normals",
                len: n.len(),
                expected,
            });
        }
    }
    Ok(())
}

/// Copies the points of `cloud` at `indices`, with their colors and normals.
///
/// # Arguments
///
/// * `cloud` - Source cloud.
/// * `indices` - Indices of the points to keep, in output order (repeats
///   allowed).
///
/// # Returns
///
/// A cloud with one point per index.
///
/// # Errors
///
/// * [`FilterError::IndexOutOfRange`] if an index is not a point of `cloud`.
/// * [`FilterError::AttributeLengthMismatch`] if `cloud`'s colors or normals
///   do not have one entry per point.
///
/// # Example
///
/// ```
/// use kornia_3d::pointcloud::{filter::select_points, PointCloud};
///
/// let cloud = PointCloud::new(vec![[0.0; 3], [1.0; 3], [2.0; 3]], None, None);
/// let picked = select_points(&cloud, &[2, 0])?;
/// assert_eq!(picked.points(), &vec![[2.0; 3], [0.0; 3]]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn select_points(cloud: &PointCloud, indices: &[u32]) -> Result<PointCloud, FilterError> {
    check_attributes(cloud)?;
    let len = cloud.len();
    if let Some(&index) = indices.iter().find(|&&i| i as usize >= len) {
        return Err(FilterError::IndexOutOfRange { index, len });
    }
    let pick = |i: &u32| *i as usize;
    let points = indices.iter().map(|i| cloud.points()[pick(i)]).collect();
    let colors = cloud
        .colors()
        .map(|c| indices.iter().map(|i| c[pick(i)]).collect());
    let normals = cloud
        .normals()
        .map(|n| indices.iter().map(|i| n[pick(i)]).collect());
    Ok(PointCloud::new(points, colors, normals))
}

#[cfg(test)]
mod tests;
