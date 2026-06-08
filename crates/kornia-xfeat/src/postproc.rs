//! Post-processing: NMS, top-K, descriptor sampling at sparse keypoint
//! coordinates. Mirrors upstream `Xfeat.detectAndCompute`.

use crate::ops;

/// One detected keypoint with score, in **original image** coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KeyPoint {
    /// Sub-pixel x in original image coords (after rescaling by `rw`).
    pub x: f32,
    /// Sub-pixel y in original image coords.
    pub y: f32,
    /// Final detection score = `K1h(x, y) * H1(x, y)` (heatmap × reliability).
    pub score: f32,
}

/// Run NMS on a `(H, W)` keypoint heatmap, sample reliability at each surviving
/// peak, sort by combined score, take top-K.
#[allow(clippy::too_many_arguments)]
pub fn nms_topk(
    heatmap: &[f32],
    reliability: &[f32],
    h: usize,
    w: usize,
    h_rel: usize,
    w_rel: usize,
    score_threshold: f32,
    top_k: usize,
    rw: f32,
    rh: f32,
) -> Vec<KeyPoint> {
    debug_assert_eq!(heatmap.len(), h * w);
    debug_assert_eq!(reliability.len(), h_rel * w_rel);

    let raw = ops::nms_maxpool_5x5_equality(heatmap, h, w, score_threshold);

    // Match PyTorch's InterpolateSparse2d(bilinear) coordinate convention:
    //   normgrid: g = 2*x/(W-1) - 1
    //   grid_sample(align_corners=False): pos = (g+1)/2 * W_rel - 0.5
    //   → pos = x * W_rel / (W-1) - 0.5
    let x_scale = (w_rel as f32) / ((w as f32) - 1.0);
    let y_scale = (h_rel as f32) / ((h as f32) - 1.0);

    let mut kps: Vec<KeyPoint> = raw
        .into_iter()
        .map(|(kp_score, idx)| {
            let y = (idx / w) as f32;
            let x = (idx % w) as f32;
            let rel = bilinear_sample_single(
                reliability,
                w_rel,
                h_rel,
                x * x_scale - 0.5,
                y * y_scale - 0.5,
            );
            KeyPoint {
                x: x * rw,
                y: y * rh,
                score: kp_score * rel,
            }
        })
        .collect();

    // Sort descending, then truncate.
    kps.sort_unstable_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    if kps.len() > top_k {
        kps.truncate(top_k);
    }
    kps
}

fn bilinear_sample_single(buf: &[f32], w: usize, h: usize, x: f32, y: f32) -> f32 {
    let x0 = x.floor();
    let y0 = y.floor();
    let wx = x - x0;
    let wy = y - y0;
    let x0i = (x0 as isize).clamp(0, w as isize - 1) as usize;
    let y0i = (y0 as isize).clamp(0, h as isize - 1) as usize;
    let x1i = (x0i + 1).min(w - 1);
    let y1i = (y0i + 1).min(h - 1);
    let v00 = buf[y0i * w + x0i];
    let v01 = buf[y0i * w + x1i];
    let v10 = buf[y1i * w + x0i];
    let v11 = buf[y1i * w + x1i];
    let v0 = v00 * (1.0 - wx) + v01 * wx;
    let v1 = v10 * (1.0 - wx) + v11 * wx;
    v0 * (1.0 - wy) + v1 * wy
}

/// Bicubic sample of a `(H, W, C)` NHWC descriptor map at sparse (x, y) pixel
/// coordinates **in the descriptor map's own resolution** (i.e. already
/// divided by 8 from the input scale). Output is `kp.len() * c` f32.
pub fn bicubic_sample_descriptors(
    desc: &[f32],
    h_d: usize,
    w_d: usize,
    c: usize,
    kps_xy_in_desc_space: &[(f32, f32)],
    out: &mut [f32],
) {
    debug_assert_eq!(desc.len(), h_d * w_d * c);
    debug_assert_eq!(out.len(), kps_xy_in_desc_space.len() * c);

    for (i, &(x, y)) in kps_xy_in_desc_space.iter().enumerate() {
        let out_off = i * c;
        bicubic_sample_one(desc, h_d, w_d, c, x, y, &mut out[out_off..out_off + c]);
    }
}

fn bicubic_sample_one(desc: &[f32], h: usize, w: usize, c: usize, x: f32, y: f32, out: &mut [f32]) {
    // PyTorch grid_sample-style bicubic with cubic Hermite (a = -0.75).
    let ix = x.floor() as isize;
    let iy = y.floor() as isize;
    let fx = x - ix as f32;
    let fy = y - iy as f32;

    let cubic = |t: f32| -> [f32; 4] {
        let a = -0.75f32;
        let t1 = 1.0 + t;
        let t2 = t;
        let t3 = 1.0 - t;
        let t4 = 2.0 - t;
        let h1 = a * t1.powi(3) - 5.0 * a * t1.powi(2) + 8.0 * a * t1 - 4.0 * a;
        let h2 = (a + 2.0) * t2.powi(3) - (a + 3.0) * t2.powi(2) + 1.0;
        let h3 = (a + 2.0) * t3.powi(3) - (a + 3.0) * t3.powi(2) + 1.0;
        let h4 = a * t4.powi(3) - 5.0 * a * t4.powi(2) + 8.0 * a * t4 - 4.0 * a;
        [h1, h2, h3, h4]
    };

    let kx = cubic(fx);
    let ky = cubic(fy);

    for (ci, out_ci) in out.iter_mut().enumerate().take(c) {
        let mut acc = 0.0f32;
        for (j, &kyj) in ky.iter().enumerate() {
            let yy = (iy - 1 + j as isize).clamp(0, h as isize - 1) as usize;
            for (i, &kxi) in kx.iter().enumerate() {
                let xx = (ix - 1 + i as isize).clamp(0, w as isize - 1) as usize;
                acc += desc[(yy * w + xx) * c + ci] * kxi * kyj;
            }
        }
        *out_ci = acc;
    }
}
