//! Rigid AprilGrid target: a planar grid of AprilTags with a KNOWN metric layout.
//!
//! Exploiting the known layout is what removes the single-planar-tag rotation degeneracy: every tag
//! corner across the board becomes a *fixed* metric anchor, and their spatial spread constrains
//! rotation far better than one small tag. The board is the world frame; corners are held fixed in
//! bundle adjustment (gauge + absolute scale come from the board, no free reference frame needed).

use kornia_algebra::Vec3F64;

/// A planar grid of AprilTags with a known metric layout (all corners on the `z = 0` board plane).
///
/// Layout convention (documented and self-consistent; map a physical board's id/corner numbering to
/// this at the detection boundary):
/// - Tags are row-major: `tag_id = row * cols + col`, `col` increasing along `+x`, `row` along `+y`.
/// - Tag `0`'s bottom-left corner sits at the board origin `(0, 0, 0)`.
/// - Adjacent tag origins are `tag_size_m + tag_spacing_m` apart (`tag_spacing_m` is the inter-tag gap).
/// - Each tag's 4 corners are returned in aruco winding `(TL, TR, BR, BL)`, matching
///   [`crate::TagObservation`].
#[derive(Debug, Clone, Copy)]
pub struct AprilGridBoard {
    /// Number of tag rows.
    pub rows: usize,
    /// Number of tag columns.
    pub cols: usize,
    /// Tag side length (black square) in metres.
    pub tag_size_m: f64,
    /// Gap between adjacent tags in metres.
    pub tag_spacing_m: f64,
}

impl AprilGridBoard {
    /// New board from an explicit metric inter-tag gap.
    pub fn new(rows: usize, cols: usize, tag_size_m: f64, tag_spacing_m: f64) -> Self {
        Self {
            rows,
            cols,
            tag_size_m,
            tag_spacing_m,
        }
    }

    /// New board from the Kalibr convention, where spacing is a RATIO of the tag size
    /// (`gap = spacing_ratio * tag_size`).
    pub fn from_kalibr(rows: usize, cols: usize, tag_size_m: f64, spacing_ratio: f64) -> Self {
        Self::new(rows, cols, tag_size_m, spacing_ratio * tag_size_m)
    }

    /// Distance between adjacent tag origins (tag size + gap).
    fn pitch(&self) -> f64 {
        self.tag_size_m + self.tag_spacing_m
    }

    /// Whether this tag id is on the board.
    pub fn contains(&self, tag_id: u16) -> bool {
        (tag_id as usize) < self.rows * self.cols
    }

    /// `(col, row)` of a tag id, or `None` if off the board.
    fn grid(&self, tag_id: u16) -> Option<(usize, usize)> {
        let id = tag_id as usize;
        if id >= self.rows * self.cols {
            return None;
        }
        Some((id % self.cols, id / self.cols))
    }

    /// Centre of a tag in the board frame (`z = 0`), or `None` if the id is off the board.
    pub fn tag_center(&self, tag_id: u16) -> Option<Vec3F64> {
        let (col, row) = self.grid(tag_id)?;
        let h = self.tag_size_m / 2.0;
        Some(Vec3F64::new(
            col as f64 * self.pitch() + h,
            row as f64 * self.pitch() + h,
            0.0,
        ))
    }

    /// The 4 corners of a tag in the board frame, aruco order `(TL, TR, BR, BL)`, `z = 0`;
    /// `None` if the id is off the board.
    pub fn object_points(&self, tag_id: u16) -> Option<[Vec3F64; 4]> {
        let (col, row) = self.grid(tag_id)?;
        let s = self.tag_size_m;
        let (x0, y0) = (col as f64 * self.pitch(), row as f64 * self.pitch()); // bottom-left corner
        Some([
            Vec3F64::new(x0, y0 + s, 0.0),     // TL
            Vec3F64::new(x0 + s, y0 + s, 0.0), // TR
            Vec3F64::new(x0 + s, y0, 0.0),     // BR
            Vec3F64::new(x0, y0, 0.0),         // BL
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry_layout_is_consistent() {
        // 2x2 board, 10 cm tags, 2 cm gap → pitch 12 cm.
        let b = AprilGridBoard::new(2, 2, 0.10, 0.02);
        assert!(b.contains(3) && !b.contains(4));
        // tag 0 bottom-left at origin.
        let c0 = b.object_points(0).unwrap();
        assert!((c0[3] - Vec3F64::new(0.0, 0.0, 0.0)).length() < 1e-12); // BL
        assert!((c0[1] - Vec3F64::new(0.10, 0.10, 0.0)).length() < 1e-12); // TR
                                                                           // tag 1 is one column over (+x by pitch).
        let c1 = b.object_points(1).unwrap();
        assert!((c1[3] - Vec3F64::new(0.12, 0.0, 0.0)).length() < 1e-12);
        // tag 2 is one row up (+y by pitch).
        let c2 = b.object_points(2).unwrap();
        assert!((c2[3] - Vec3F64::new(0.0, 0.12, 0.0)).length() < 1e-12);
        // centre = bottom-left + (h, h).
        assert!((b.tag_center(0).unwrap() - Vec3F64::new(0.05, 0.05, 0.0)).length() < 1e-12);
        // Kalibr ratio constructor.
        let bk = AprilGridBoard::from_kalibr(2, 2, 0.10, 0.2);
        assert!((bk.tag_spacing_m - 0.02).abs() < 1e-12);
    }
}
