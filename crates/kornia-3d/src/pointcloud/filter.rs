//! Errors of the point cloud filters and of index selection
//! ([`PointCloud::select`]).

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

#[cfg(test)]
mod tests;
