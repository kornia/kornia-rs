//! CPU-side precompute of bilinear weight tables.
//!
//! For each output coordinate, we store `(src_idx, weight_x256)` where
//! `weight_x256 ∈ [0, 256]` is the fractional weight times 256. The output
//! sample is then computed by the kernel as
//! `((256 - w) * src[idx] + w * src[idx + 1] + 128) >> 8`.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AxisWeight {
    pub src_idx: u32,
    pub weight_x256: u16,
}

/// Compute axis weights for a 1D resize from `src_len` to `dst_len`.
///
/// Uses pixel-centered sampling: output pixel `i` samples at source coordinate
/// `(i + 0.5) * src_len / dst_len - 0.5`, clamped to `[0, src_len - 1]`. This
/// matches `fast_image_resize`'s default sampling convention so cross-impl
/// outputs can be compared within ±1 LSB.
pub fn compute_axis_weights(src_len: u32, dst_len: u32) -> Vec<AxisWeight> {
    assert!(src_len >= 2, "src_len must be at least 2 for bilinear");
    assert!(dst_len >= 1, "dst_len must be at least 1");
    let scale = src_len as f64 / dst_len as f64;
    (0..dst_len)
        .map(|i| {
            let center = (i as f64 + 0.5) * scale - 0.5;
            let center = center.clamp(0.0, (src_len - 1) as f64);
            let idx = center.floor() as u32;
            let frac = center - idx as f64;
            let w = (frac * 256.0).round() as u32;
            // Clamp idx so `idx + 1` never exceeds src_len - 1 (right edge handling).
            let idx_clamped = idx.min(src_len - 2);
            // If we clamped, push the weight to 256 so we sample fully from src[src_len-1].
            let w_final = if idx_clamped < idx { 256 } else { w.min(256) };
            AxisWeight { src_idx: idx_clamped, weight_x256: w_final as u16 }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_2x_downscale_4_to_2() {
        // src=[a,b,c,d], scale=2.0:
        //   dst[0]: center = 0.5*2 - 0.5 = 0.5 → idx=0, w=128
        //   dst[1]: center = 1.5*2 - 0.5 = 2.5 → idx=2, w=128
        let w = compute_axis_weights(4, 2);
        assert_eq!(w.len(), 2);
        assert_eq!(w[0], AxisWeight { src_idx: 0, weight_x256: 128 });
        assert_eq!(w[1], AxisWeight { src_idx: 2, weight_x256: 128 });
    }

    #[test]
    fn right_edge_does_not_overflow() {
        for &(s, d) in &[(8, 4), (1024, 512), (4096, 2048), (513, 256)] {
            let w = compute_axis_weights(s, d);
            for aw in &w {
                assert!(
                    (aw.src_idx as usize) + 1 < s as usize,
                    "src_idx + 1 = {} overflows src_len {} (size {} -> {})",
                    aw.src_idx + 1, s, s, d,
                );
            }
        }
    }

    #[test]
    fn weight_in_range() {
        let w = compute_axis_weights(1024, 512);
        for aw in &w {
            assert!(aw.weight_x256 <= 256);
        }
    }

    #[test]
    fn identity_resize() {
        // src_len == dst_len: output[i] should sample src[i] with weight 0.
        let w = compute_axis_weights(8, 8);
        for (i, aw) in w.iter().enumerate() {
            assert_eq!(aw.src_idx as usize, i.min(6));
            // For i in 0..7: w should be 0 (sampling exactly at integer pos).
            // For i == 7: idx is clamped to 6, weight pushed to 256 (full src[7]).
            if i < 7 {
                assert_eq!(aw.weight_x256, 0, "i={i}: {aw:?}");
            } else {
                assert_eq!(aw.weight_x256, 256, "i={i}: {aw:?}");
            }
        }
    }
}
