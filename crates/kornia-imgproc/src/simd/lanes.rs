//! Lane helpers shared by the `wide` kernels (table remap, corner response,
//! sub-pixel corners, window LK). Loads and stores bounds-check their slice
//! once.

/// Lanes of the 128-bit i16 vectors, and of the i32 and f32 vectors the
/// kernels widen them to.
pub(crate) const LANES: usize = 8;

/// Starts of `LANES`-wide chunks covering `0..len`, the last one shifted
/// back to end at `len`, so the overlap is recomputed with the same values;
/// none when `len < LANES`.
pub(crate) fn lane_starts(len: usize) -> impl Iterator<Item = usize> {
    let last = len.checked_sub(LANES);
    let before = last.map_or(0, |l| l.div_ceil(LANES));
    (0..before).map(|k| k * LANES).chain(last)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lane_starts_cover_without_gaps() {
        assert_eq!(lane_starts(5).count(), 0);
        assert_eq!(lane_starts(8).collect::<Vec<_>>(), [0]);
        assert_eq!(lane_starts(16).collect::<Vec<_>>(), [0, 8]);
        assert_eq!(lane_starts(17).collect::<Vec<_>>(), [0, 8, 9]);
        assert_eq!(lane_starts(23).collect::<Vec<_>>(), [0, 8, 15]);
    }
}
