//! Rigid calibration target as pure geometry: `tag_id → 4 metric world corners`.
//!
//! [`BoardGeometry`] is the data-only interface the multi-camera solver needs from a rigid target —
//! it deliberately does NOT depend on any board/tag *type* from another crate. Exploiting a known
//! rigid layout is what removes the single-planar-tag rotation degeneracy: every corner across the
//! board becomes a *fixed* metric anchor whose spatial spread constrains rotation far better than one
//! small tag. The board is the world frame; corners are held fixed in bundle adjustment (gauge +
//! absolute scale come from the board, no free reference frame needed).
//!
//! Build it from a standard AprilGrid with [`BoardGeometry::april_grid`] /
//! [`BoardGeometry::april_grid_kalibr`], or from arbitrary measured corners with
//! [`BoardGeometry::from_corners`] (supports non-uniform tag sizes / non-planar rigs).

use kornia_algebra::Vec3F64;
use std::collections::BTreeMap;

/// A rigid calibration target given as `tag_id → 4 metric corner points` in the board (world) frame.
///
/// Corners are stored in aruco winding `(TL, TR, BR, BL)`, matching [`crate::TagObservation`]. The
/// board frame is arbitrary but fixed; for an AprilGrid it is the plane `z = 0` with tag `0`'s
/// bottom-left corner at the origin (see [`BoardGeometry::april_grid`]).
#[derive(Debug, Clone, Default)]
pub struct BoardGeometry {
    corners: BTreeMap<u16, [Vec3F64; 4]>,
}

impl BoardGeometry {
    /// Build from explicit `(tag_id, [TL, TR, BR, BL])` metric world corners (aruco winding).
    ///
    /// Use this for boards measured directly, mixed tag sizes, or non-planar rigs.
    pub fn from_corners(corners: impl IntoIterator<Item = (u16, [Vec3F64; 4])>) -> Self {
        Self {
            corners: corners.into_iter().collect(),
        }
    }

    /// Standard planar AprilGrid, row-major (`tag_id = row * cols + col`, `col` along `+x`, `row`
    /// along `+y`), tag `0`'s bottom-left corner at the origin, adjacent tag origins
    /// `tag_size_m + tag_spacing_m` apart. All corners on `z = 0`.
    pub fn april_grid(rows: usize, cols: usize, tag_size_m: f64, tag_spacing_m: f64) -> Self {
        let pitch = tag_size_m + tag_spacing_m;
        let s = tag_size_m;
        let mut corners = BTreeMap::new();
        for row in 0..rows {
            for col in 0..cols {
                let id = (row * cols + col) as u16;
                let (x0, y0) = (col as f64 * pitch, row as f64 * pitch); // bottom-left corner
                corners.insert(
                    id,
                    [
                        Vec3F64::new(x0, y0 + s, 0.0),     // TL
                        Vec3F64::new(x0 + s, y0 + s, 0.0), // TR
                        Vec3F64::new(x0 + s, y0, 0.0),     // BR
                        Vec3F64::new(x0, y0, 0.0),         // BL
                    ],
                );
            }
        }
        Self { corners }
    }

    /// AprilGrid from the Kalibr convention, where spacing is a RATIO of the tag size
    /// (`gap = spacing_ratio * tag_size`).
    pub fn april_grid_kalibr(
        rows: usize,
        cols: usize,
        tag_size_m: f64,
        spacing_ratio: f64,
    ) -> Self {
        Self::april_grid(rows, cols, tag_size_m, spacing_ratio * tag_size_m)
    }

    /// Whether this tag id is on the board.
    pub fn contains(&self, tag_id: u16) -> bool {
        self.corners.contains_key(&tag_id)
    }

    /// The 4 corners of a tag in the board frame, aruco order `(TL, TR, BR, BL)`; `None` if the id
    /// is off the board.
    pub fn object_points(&self, tag_id: u16) -> Option<[Vec3F64; 4]> {
        self.corners.get(&tag_id).copied()
    }

    /// Centre of a tag in the board frame (mean of its 4 corners), or `None` if off the board.
    pub fn tag_center(&self, tag_id: u16) -> Option<Vec3F64> {
        let c = self.corners.get(&tag_id)?;
        Some((c[0] + c[1] + c[2] + c[3]) * 0.25)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn april_grid_layout_is_consistent() {
        // 2x2 board, 10 cm tags, 2 cm gap → pitch 12 cm.
        let b = BoardGeometry::april_grid(2, 2, 0.10, 0.02);
        assert!(b.contains(3) && !b.contains(4));
        // tag 0 bottom-left at origin.
        let c0 = b.object_points(0).unwrap();
        assert!((c0[3] - Vec3F64::new(0.0, 0.0, 0.0)).length() < 1e-12); // BL
        assert!((c0[1] - Vec3F64::new(0.10, 0.10, 0.0)).length() < 1e-12); // TR
                                                                           // tag 1 one column over (+x by pitch); tag 2 one row up (+y by pitch).
        assert!((b.object_points(1).unwrap()[3] - Vec3F64::new(0.12, 0.0, 0.0)).length() < 1e-12);
        assert!((b.object_points(2).unwrap()[3] - Vec3F64::new(0.0, 0.12, 0.0)).length() < 1e-12);
        // centre = bottom-left + (h, h).
        assert!((b.tag_center(0).unwrap() - Vec3F64::new(0.05, 0.05, 0.0)).length() < 1e-12);
        // Kalibr ratio constructor.
        let bk = BoardGeometry::april_grid_kalibr(2, 2, 0.10, 0.2);
        assert!((bk.object_points(1).unwrap()[3] - Vec3F64::new(0.12, 0.0, 0.0)).length() < 1e-12);
    }

    #[test]
    fn from_corners_roundtrips() {
        let sq = [
            Vec3F64::new(0.0, 0.1, 0.0),
            Vec3F64::new(0.1, 0.1, 0.0),
            Vec3F64::new(0.1, 0.0, 0.0),
            Vec3F64::new(0.0, 0.0, 0.0),
        ];
        let b = BoardGeometry::from_corners([(7u16, sq)]);
        assert!(b.contains(7) && !b.contains(0));
        assert!((b.tag_center(7).unwrap() - Vec3F64::new(0.05, 0.05, 0.0)).length() < 1e-12);
    }
}
