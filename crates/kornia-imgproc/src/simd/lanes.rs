//! Lane helpers shared by the `wide` kernels (table remap, corner response,
//! sub-pixel corners, window LK). Loads and stores bounds-check their slice
//! once.

use wide::{f32x8, i16x8, i32x8, u8x16};

/// Lanes of the 128-bit i16 vectors, and of the i32 and f32 vectors the
/// kernels widen them to.
pub(crate) const LANES: usize = 8;

/// Eight u8 values as i16 lanes.
#[inline(always)]
pub(crate) fn load8_i16(s: &[u8]) -> i16x8 {
    let s = &s[..LANES];
    i16x8::new(std::array::from_fn(|k| i16::from(s[k])))
}

/// Sixteen u8 values.
#[inline(always)]
pub(crate) fn load16(s: &[u8]) -> u8x16 {
    let mut a = [0u8; 16];
    a.copy_from_slice(&s[..16]);
    u8x16::new(a)
}

/// Eight i32 values.
#[inline(always)]
pub(crate) fn load_i32(s: &[i32]) -> i32x8 {
    let mut a = [0i32; LANES];
    a.copy_from_slice(&s[..LANES]);
    i32x8::new(a)
}

/// Stores eight i32 lanes at the start of `s`.
#[inline(always)]
pub(crate) fn store_i32(s: &mut [i32], v: i32x8) {
    s[..LANES].copy_from_slice(v.as_array());
}

/// Eight f32 values.
#[inline(always)]
pub(crate) fn load_f32(s: &[f32]) -> f32x8 {
    let mut a = [0f32; LANES];
    a.copy_from_slice(&s[..LANES]);
    f32x8::new(a)
}

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
