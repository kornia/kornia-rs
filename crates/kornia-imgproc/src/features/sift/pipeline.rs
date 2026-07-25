//! End-to-end CPU SIFT: base image, octave loop, and the reference's final
//! ordering.

use rayon::prelude::*;

use super::descriptor::{compute_descriptor, descriptor_inputs, DESCR_LEN};
use super::detect::{find_extrema, RawKeypoint};
use super::orient::{assign_orientations, OrientedKeypoint};
use super::params::{gaussian_kernel_f32, gaussian_ksize, SiftConfig};
use super::scalespace::{blur_h_f32, blur_v_f32};

/// Which scale the pyramid starts from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FirstOctave {
    /// `first_octave = -1`: upsample 2x first (the reference's default).
    Double,
    /// `first_octave = 0`: start at the input resolution.
    Native,
}

/// A detected, oriented and described keypoint, in input-image coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SiftKeypoint {
    /// Column, in input-image pixels.
    pub x: f32,
    /// Row, in input-image pixels.
    pub y: f32,
    /// Diameter of the meaningful neighbourhood.
    pub size: f32,
    /// Orientation in degrees.
    pub angle: f32,
    /// Contrast at the refined extremum.
    pub response: f32,
    /// Packed octave field.
    pub octave: i32,
}

/// Host-side result of a full pass.
#[derive(Debug, Clone, Default)]
pub struct SiftFeatures {
    /// One entry per oriented keypoint.
    pub keypoints: Vec<SiftKeypoint>,
    /// Row-major `keypoints.len() * 128` descriptor block.
    pub descriptors: Vec<f32>,
}

/// Separable bilinear 2x upsample forming the base of octave 0.
///
/// This is the reference's ordinary resize, **not** its precise-upscale affine
/// warp — the option defaults to off, and the two differ by a quarter pixel,
/// which shifts every downstream keypoint.
fn upsample2x(src: &[f32], sw: usize, sh: usize, dst: &mut [f32]) {
    let (dw, dh) = (sw * 2, sh * 2);
    // For a 2x upscale the source tap reduces to a parity test with weights
    // {0.75, 0.25}: `d/2 - 0.25` is exact in f32 for any integer `d`.
    let tap = |d: usize, n: usize| -> (usize, f32) {
        let odd = d & 1;
        let mut s = (d >> 1) as isize - ((odd ^ 1) as isize);
        let mut f = if odd == 1 { 0.25f32 } else { 0.75f32 };
        if s < 0 {
            f = 0.0;
            s = 0;
        }
        if s >= n as isize - 1 {
            f = 0.0;
            s = n as isize - 1;
        }
        (s as usize, f)
    };
    dst[..dw * dh]
        .par_chunks_mut(dw)
        .enumerate()
        .for_each(|(y, row)| {
            let (sy, fy) = tap(y, sh);
            let sy1 = (sy + 1).min(sh - 1);
            for (x, o) in row.iter_mut().enumerate() {
                let (sx, fx) = tap(x, sw);
                let sx1 = (sx + 1).min(sw - 1);
                let r0 = src[sy * sw + sx] * (1.0 - fx) + src[sy * sw + sx1] * fx;
                let r1 = src[sy1 * sw + sx] * (1.0 - fx) + src[sy1 * sw + sx1] * fx;
                *o = r0 * (1.0 - fy) + r1 * fy;
            }
        });
}

/// Nearest stride-2 subsample — **not** a blur-and-decimate, which would break
/// parity with the reference.
fn downsample(src: &[f32], sw: usize, sh: usize, dst: &mut [f32], dw: usize, dh: usize) {
    dst[..dw * dh]
        .par_chunks_mut(dw)
        .enumerate()
        .for_each(|(y, row)| {
            let sy = (y * sh) / dh;
            for (x, o) in row.iter_mut().enumerate() {
                *o = src[sy * sw + (x * sw) / dw];
            }
        });
}

/// Detect, orient and describe.
///
/// `src` is a `w * h` f32 grayscale image in 0..255 — the reference's own
/// internal representation. Normalising to 0..1 changes what the contrast
/// threshold means.
pub fn detect_and_compute(
    src: &[f32],
    w: usize,
    h: usize,
    cfg: &SiftConfig,
    first_octave: FirstOctave,
    max_octaves: usize,
) -> SiftFeatures {
    assert_eq!(src.len(), w * h, "source length must be w * h");
    let doubled = first_octave == FirstOctave::Double;
    let (mut cw, mut ch) = if doubled { (w * 2, h * 2) } else { (w, h) };

    let n_layers = cfg.n_octave_layers + 3;
    let n_dog = cfg.n_octave_layers + 2;
    let sigmas = cfg.layer_sigmas();
    let base_sigma = cfg.base_sig_diff(doubled) as f64;
    let base_kernel = gaussian_kernel_f32(gaussian_ksize(base_sigma), base_sigma);
    let layer_kernels: Vec<Vec<f32>> = (1..n_layers)
        .map(|i| gaussian_kernel_f32(gaussian_ksize(sigmas[i]), sigmas[i]))
        .collect();

    let plane = cw * ch;
    let mut base = vec![0.0f32; plane];
    let mut tmp = vec![0.0f32; plane];
    if doubled {
        upsample2x(src, w, h, &mut base);
    } else {
        base.copy_from_slice(src);
    }
    let mut gauss: Vec<Vec<f32>> = (0..n_layers).map(|_| vec![0.0f32; plane]).collect();
    blur_h_f32(&base, &mut tmp, cw, ch, &base_kernel);
    blur_v_f32(&tmp, &mut gauss[0], cw, ch, &base_kernel, None, None);

    // KORNIA_SIFT_STAGES=1 breaks the pass down. Each probe is a plain elapsed
    // measurement -- no synchronisation needed on CPU -- so the total is not
    // inflated the way the CUDA equivalent is.
    let probe = std::env::var("KORNIA_SIFT_STAGES").is_ok();
    let (mut t_blur, mut t_det, mut t_ori, mut t_desc) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let mark = || std::time::Instant::now();

    let n_oct = cfg
        .n_octaves(cw.min(ch), if doubled { -1 } else { 0 })
        .min(max_octaves.max(1));
    let mut dog = vec![0.0f32; plane * n_dog];
    let mut all: Vec<(OrientedKeypoint, usize, usize)> = Vec::new();
    let mut desc: Vec<f32> = Vec::new();

    for octv in 0..n_oct {
        if cw < 16 || ch < 16 {
            break;
        }
        let p = cw * ch;
        let tb = mark();
        for i in 1..n_layers {
            let k = &layer_kernels[i - 1];
            blur_h_f32(&gauss[i - 1], &mut tmp[..p], cw, ch, k);
            let (lo, hi) = gauss.split_at_mut(i);
            blur_v_f32(
                &tmp[..p],
                &mut hi[0][..p],
                cw,
                ch,
                k,
                Some(&lo[i - 1][..p]),
                Some(&mut dog[(i - 1) * p..i * p]),
            );
        }

        if probe {
            t_blur += tb.elapsed().as_secs_f64() * 1e3;
        }

        let td = mark();
        let mut kps: Vec<RawKeypoint> = Vec::new();
        for layer in 1..=cfg.n_octave_layers {
            find_extrema(&dog[..p * n_dog], cw, ch, n_dog, layer, octv, cfg, &mut kps);
        }
        if probe {
            t_det += td.elapsed().as_secs_f64() * 1e3;
        }
        let to = mark();
        // Indexing by layer is the point here: `gauss[layer]` is the Gaussian
        // the keypoints of that layer were found in.
        #[allow(clippy::needless_range_loop)]
        for layer in 1..=cfg.n_octave_layers {
            let img = &gauss[layer][..p];
            // One task per keypoint group: each histogram is independent, and
            // the reference's order within a keypoint is preserved because a
            // single task owns it.
            let group: Vec<&RawKeypoint> = kps.iter().filter(|k| k.layer == layer as i32).collect();
            let oriented: Vec<OrientedKeypoint> = group
                .par_iter()
                .flat_map_iter(|kp| {
                    let mut o = Vec::new();
                    assign_orientations(img, cw, ch, kp, cfg, &mut o);
                    o.into_iter()
                })
                .collect();
            for k in oriented {
                all.push((k, octv, layer));
            }
        }
        if probe {
            t_ori += to.elapsed().as_secs_f64() * 1e3;
        }
        let tds = mark();
        // Descriptors for this octave, before its layers are overwritten.
        let start = desc.len() / DESCR_LEN;
        let todo: Vec<(OrientedKeypoint, usize)> = all[start..]
            .iter()
            .map(|(k, _, layer)| (*k, *layer))
            .collect();
        let mut block = vec![0.0f32; todo.len() * DESCR_LEN];
        block
            .par_chunks_mut(DESCR_LEN)
            .zip(todo.par_iter())
            .for_each(|(out, (k, layer))| {
                let img = &gauss[*layer][..p];
                let (x, y, s, a) = descriptor_inputs(k, octv as i32);
                compute_descriptor(img, cw, ch, x, y, s, a, out);
            });
        desc.extend_from_slice(&block);
        if probe {
            t_desc += tds.elapsed().as_secs_f64() * 1e3;
        }

        let (nw, nh) = (cw / 2, ch / 2);
        if nw == 0 || nh == 0 || octv + 1 >= n_oct {
            break;
        }
        let src_layer = std::mem::take(&mut gauss[cfg.n_octave_layers]);
        downsample(&src_layer[..p], cw, ch, &mut gauss[0], nw, nh);
        gauss[cfg.n_octave_layers] = src_layer;
        cw = nw;
        ch = nh;
    }

    if probe {
        eprintln!(
            "    stages: blur={t_blur:.1} detect={t_det:.1} orient={t_ori:.1} desc={t_desc:.1} (ms)"
        );
    }
    // first_octave = -1 post-processing, then the reference's ordering.
    let scale = if doubled { 0.5f32 } else { 1.0f32 };
    let mut kps: Vec<SiftKeypoint> = all
        .iter()
        .map(|(k, _, _)| {
            let packed = k.octave;
            SiftKeypoint {
                x: k.x * scale,
                y: k.y * scale,
                size: k.size * scale,
                angle: k.angle,
                response: k.response,
                octave: if doubled {
                    (packed & !255) | ((packed - 1) & 255)
                } else {
                    packed
                },
            }
        })
        .collect();

    let order = final_order(&kps, cfg.n_features);
    let n = order.len();
    let mut out_desc = Vec::with_capacity(n * DESCR_LEN);
    for &i in &order {
        out_desc.extend_from_slice(&desc[i * DESCR_LEN..(i + 1) * DESCR_LEN]);
    }
    kps = order.iter().map(|&i| kps[i]).collect();
    SiftFeatures {
        keypoints: kps,
        descriptors: out_desc,
    }
}

/// `removeDuplicatedSorted` then `retainBest`, as indices into the input order.
///
/// The comparator is the reference's, including its descending fields: `size`,
/// `response` and `octave` sort the opposite way to `x`, `y` and `angle`.
fn final_order(kps: &[SiftKeypoint], n_features: usize) -> Vec<usize> {
    if kps.is_empty() {
        return Vec::new();
    }
    let mut order: Vec<usize> = (0..kps.len()).collect();
    order.sort_by(|&a, &b| {
        let (p, q) = (&kps[a], &kps[b]);
        p.x.total_cmp(&q.x)
            .then(p.y.total_cmp(&q.y))
            .then(q.size.total_cmp(&p.size))
            .then(p.angle.total_cmp(&q.angle))
            .then(q.response.total_cmp(&p.response))
            .then(q.octave.cmp(&p.octave))
            .then(a.cmp(&b))
    });
    // The reference compares only these four fields when removing duplicates.
    let mut dedup: Vec<usize> = Vec::with_capacity(order.len());
    for &i in &order {
        if let Some(&p) = dedup.last() {
            let (a, b) = (&kps[p], &kps[i]);
            if a.x == b.x && a.y == b.y && a.size == b.size && a.angle == b.angle {
                continue;
            }
        }
        dedup.push(i);
    }
    if n_features == 0 || dedup.len() <= n_features {
        return dedup;
    }
    let mut rank = dedup.clone();
    rank.sort_by(|&a, &b| kps[b].response.total_cmp(&kps[a].response).then(a.cmp(&b)));
    let cutoff = kps[rank[n_features - 1]].response;
    dedup
        .into_iter()
        .filter(|&i| kps[i].response >= cutoff)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// End-to-end timing and keypoint count, to compare against cv2 directly.
    /// `KORNIA_SIFT_BENCH=1`, image from `KORNIA_SIFT_RAW=<w>x<h>:<path>` (raw
    /// f32 in 0..255).
    #[test]
    fn bench_cpu_pipeline() {
        if std::env::var("KORNIA_SIFT_BENCH").is_err() {
            eprintln!("KORNIA_SIFT_BENCH unset; skipping");
            return;
        }
        let Ok(spec) = std::env::var("KORNIA_SIFT_RAW") else {
            eprintln!("KORNIA_SIFT_RAW unset; skipping");
            return;
        };
        let (dims, path) = spec.split_once(':').expect("WxH:path");
        let (ws, hs) = dims.split_once('x').expect("WxH");
        let (w, h): (usize, usize) = (ws.parse().unwrap(), hs.parse().unwrap());
        let bytes = std::fs::read(path).expect("raw image");
        let img: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .take(w * h)
            .collect();
        assert_eq!(img.len(), w * h);

        let cfg = SiftConfig::default();
        for (name, fo, moct) in [
            ("fo=-1", FirstOctave::Double, usize::MAX),
            ("fo=0", FirstOctave::Native, usize::MAX),
            ("fo=0 4oct", FirstOctave::Native, 4),
        ] {
            let f = detect_and_compute(&img, w, h, &cfg, fo, moct);
            let mut ts = Vec::new();
            for _ in 0..5 {
                let t = std::time::Instant::now();
                let _ = detect_and_compute(&img, w, h, &cfg, fo, moct);
                ts.push(t.elapsed().as_secs_f64() * 1e3);
            }
            ts.sort_by(f64::total_cmp);
            eprintln!(
                "  cpu pipeline {name:10} {:8.1} ms   kp={}",
                ts[2],
                f.keypoints.len()
            );
        }
    }
}
