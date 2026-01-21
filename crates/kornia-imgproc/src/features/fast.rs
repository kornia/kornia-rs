use kornia_image::{allocator::ImageAllocator, Image, ImageError};
use rayon::prelude::*;
use wide::{i8x16, u8x16, CmpGt, CmpLt};

/// FAST-9 corner detector with optional non-maximum suppression and SIMD support.
#[derive(Clone)]
pub struct FastDetector {
    /// Intensity difference threshold.
    pub threshold: u8,

    /// Enable or disable non-maximum suppression.  
    pub nonmax_suppression: bool,
}

/// Using this to reduce a u8x16 vector to its maximum element, because wide doesnt have it yet.
#[inline(always)]
fn reduce_max_u8x16(v: u8x16) -> u8 {
    let arr = v.to_array();
    *arr.iter().max().unwrap()
}

impl FastDetector {
    /// Creates a new FAST detector instance.
    ///
    /// # Arguments
    ///
    /// * `threshold` - Intensity difference threshold.
    /// * `nonmax_suppression` - Enable or disable non-maximum suppression.
    pub fn new(threshold: u8, nonmax_suppression: bool) -> Self {
        Self {
            threshold,
            nonmax_suppression,
        }
    }

    /// Detects FAST-9 keypoints in a grayscale image.
    ///
    /// # Arguments
    ///
    /// * `image` - Input image of type `Image<u8, 1, A>`.
    ///
    /// # Returns
    ///
    /// A vector of `(x, y)` coordinates representing detected keypoints.
    ///
    pub fn detect<A: ImageAllocator>(
        &self,
        image: &Image<u8, 1, A>,
    ) -> Result<Vec<[usize; 2]>, ImageError> {
        let width = image.width();
        let height = image.height();
        let stride = width; // assuming there is no padding
        let img_ptr = image.as_ptr() as usize;

        // Early exit for small images
        if width < 19 || height < 6 {
            return Ok(vec![]);
        }

        // Bresenham circle offsets for FAST-9
        let pattern_offsets: [(isize, isize); 16] = [
            (0, -3),
            (1, -3),
            (2, -2),
            (3, -1),
            (3, 0),
            (3, 1),
            (2, 2),
            (1, 3),
            (0, 3),
            (-1, 3),
            (-2, 2),
            (-3, 1),
            (-3, 0),
            (-3, -1),
            (-2, -2),
            (-1, -3),
        ];

        let mut pixel_off = [0isize; 25];
        for i in 0..16 {
            pixel_off[i] = pattern_offsets[i].1 * stride as isize + pattern_offsets[i].0;
        }

        for i in 0..9 {
            pixel_off[16 + i] = pixel_off[i];
        }

        // parallelising over chunks of rows
        let detect_start_y = 3;
        let detect_end_y = height - 4;
        let chunk_size = 64; //works well for me

        let chunks: Vec<(usize, usize)> = (detect_start_y..detect_end_y)
            .step_by(chunk_size)
            .map(|s| (s, (s + chunk_size).min(detect_end_y)))
            .collect();

        let mut keypoints: Vec<[usize; 2]> = chunks
            .into_par_iter()
            .map(|(start_row, end_row)| {
                let ptr = img_ptr as *const u8;

                // Buffers for fast scores.
                let mut buf = [vec![0u8; width], vec![0u8; width], vec![0u8; width]];

                let mut cpbuf = [
                    Vec::<usize>::with_capacity(128),
                    Vec::<usize>::with_capacity(128),
                    Vec::<usize>::with_capacity(128),
                ];

                let mut local_kps = Vec::new();

                unsafe {
                    if start_row > 3 {
                        detect_row(
                            ptr,
                            start_row - 1,
                            width,
                            stride,
                            self.threshold,
                            self.nonmax_suppression,
                            &pixel_off,
                            &mut buf[0],
                            &mut cpbuf[0],
                        );
                    }

                    detect_row(
                        ptr,
                        start_row,
                        width,
                        stride,
                        self.threshold,
                        self.nonmax_suppression,
                        &pixel_off,
                        &mut buf[1],
                        &mut cpbuf[1],
                    );

                    // NMS
                    for y in start_row..end_row {
                        let i_top = (y - start_row) % 3;
                        let i_mid = (y - start_row + 1) % 3;
                        let i_bot = (y - start_row + 2) % 3;

                        buf[i_bot].fill(0);
                        cpbuf[i_bot].clear();

                        detect_row(
                            ptr,
                            y + 1,
                            width,
                            stride,
                            self.threshold,
                            self.nonmax_suppression,
                            &pixel_off,
                            &mut buf[i_bot],
                            &mut cpbuf[i_bot],
                        );

                        let candidates = &cpbuf[i_mid];
                        let top = &buf[i_top];
                        let mid = &buf[i_mid];
                        let bot = &buf[i_bot];

                        for &x in candidates {
                            let s = mid[x];

                            if !self.nonmax_suppression {
                                local_kps.push([x, y]);
                                continue;
                            }

                            if s > top[x - 1]
                                && s > top[x]
                                && s > top[x + 1]
                                && s > mid[x - 1]
                                && s > mid[x + 1]
                                && s > bot[x - 1]
                                && s > bot[x]
                                && s > bot[x + 1]
                            {
                                local_kps.push([x, y]);
                            }
                        }
                    }
                }

                local_kps
            })
            .flatten()
            .collect();

        // Handle last row separately as NMS cant be applied on it but fast still can.
        let y_last = height - 4;
        let ptr = img_ptr as *const u8;

        let mut row_buf = [vec![0u8; width], vec![0u8; width]];
        let mut row_cp = [Vec::new(), Vec::new()];

        unsafe {
            detect_row(
                ptr,
                y_last - 1,
                width,
                stride,
                self.threshold,
                self.nonmax_suppression,
                &pixel_off,
                &mut row_buf[0],
                &mut row_cp[0],
            );
            detect_row(
                ptr,
                y_last,
                width,
                stride,
                self.threshold,
                self.nonmax_suppression,
                &pixel_off,
                &mut row_buf[1],
                &mut row_cp[1],
            );
        }

        let top = &row_buf[0];
        let mid = &row_buf[1];

        for &x in &row_cp[1] {
            let s = mid[x];

            if !self.nonmax_suppression {
                keypoints.push([x, y_last]);
                continue;
            }

            if s > top[x - 1] && s > top[x] && s > top[x + 1] && s > mid[x - 1] && s > mid[x + 1] {
                keypoints.push([x, y_last]);
            }
        }

        Ok(keypoints)
    }
}

/// Detects FAST-9 corners in a single image row using SIMD.
/// Have kept it unsafe as the calling code ensures safety.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn detect_row(
    img_base: *const u8,
    y: usize,
    width: usize,
    stride: usize,
    threshold: u8,
    compute_score: bool,
    offsets: &[isize; 25],
    score_row: &mut [u8],
    corner_pos: &mut Vec<usize>,
) {
    let row_ptr = img_base.add(y * stride);

    // SIMD constants
    let flip = u8x16::splat(0x80);
    let t_val = u8x16::splat(threshold);
    let k8 = i8x16::splat(8);
    let max_u8 = u8x16::splat(255);

    let mut x = 3;
    let limit = width - 3 - 16;

    while x <= limit {
        let ptr = row_ptr.add(x);
        let v = u8x16::new(std::slice::from_raw_parts(ptr, 16).try_into().unwrap());

        let v_plus_t = v.min(max_u8 - t_val) + t_val;
        let v_minus_t = v.max(t_val) - t_val;

        let v0: i8x16 = std::mem::transmute(v_plus_t ^ flip);
        let v1: i8x16 = std::mem::transmute(v_minus_t ^ flip);

        let load_simd = |k: usize| -> i8x16 {
            let p = ptr.offset(offsets[k]);
            let n = u8x16::new(std::slice::from_raw_parts(p, 16).try_into().unwrap());
            std::mem::transmute(n ^ flip)
        };

        let x0 = load_simd(0);
        let x4 = load_simd(4);
        let x8 = load_simd(8);
        let x12 = load_simd(12);

        let mb0 = v0.simd_lt(x0);
        let mb4 = v0.simd_lt(x4);
        let mb8 = v0.simd_lt(x8);
        let mb12 = v0.simd_lt(x12);

        let md0 = x0.simd_lt(v1);
        let md4 = x4.simd_lt(v1);
        let md8 = x8.simd_lt(v1);
        let md12 = x12.simd_lt(v1);

        let cross_bright = (mb0 & mb4) | (mb4 & mb8) | (mb8 & mb12) | (mb12 & mb0);
        let cross_dark = (md0 & md4) | (md4 & md8) | (md8 & md12) | (md12 & md0);

        if (cross_bright | cross_dark).any() {
            let mut cb = i8x16::ZERO;
            let mut cd = i8x16::ZERO;
            let mut max_b = u8x16::ZERO;
            let mut max_d = u8x16::ZERO;

            for k in 0..25 {
                let xn = load_simd(k);
                let ib = v0.simd_lt(xn);
                let id = xn.simd_lt(v1);

                cb = (cb - ib) & ib;
                cd = (cd - id) & id;

                max_b = max_b.max(std::mem::transmute::<i8x16, u8x16>(cb));
                max_d = max_d.max(std::mem::transmute::<i8x16, u8x16>(cd));
            }

            let mb_i: i8x16 = std::mem::transmute(max_b);
            let md_i: i8x16 = std::mem::transmute(max_d);

            let mask = (mb_i.simd_gt(k8) | md_i.simd_gt(k8)).to_bitmask();

            if mask != 0 {
                let mut m = mask;
                while m != 0 {
                    let i = m.trailing_zeros() as usize;
                    let rx = x + i;

                    corner_pos.push(rx);
                    score_row[rx] = if compute_score {
                        fast_score(row_ptr.add(rx), threshold, offsets)
                    } else {
                        1
                    };
                    m &= !(1 << i);
                }
            }
        }
        x += 16;
    }

    // Scalar fallback for tail pixels as they dont line up for SIMD
    #[allow(clippy::needless_range_loop)]
    for i in x..(width - 3) {
        let ptr = row_ptr.add(i);
        let v = *ptr as i16;
        let t = threshold as i16;

        let get = |idx| *ptr.offset(offsets[idx]) as i16;
        let p0 = get(0);
        let p4 = get(4);
        let p8 = get(8);
        let p12 = get(12);

        let b0 = p0 > v + t;
        let d0 = p0 < v - t;
        let b4 = p4 > v + t;
        let d4 = p4 < v - t;
        let b8 = p8 > v + t;
        let d8 = p8 < v - t;
        let b12 = p12 > v + t;
        let d12 = p12 < v - t;

        if (b12 || b4) && (b8 || b0) || (d12 || d4) && (d8 || d0) {
            let mut cb = 0;
            let mut cd = 0;
            let mut max_b = 0;
            let mut max_d = 0;

            for k in 0..25 {
                let pk = get(k);
                if pk > v + t {
                    cb += 1;
                } else {
                    max_b = max_b.max(cb);
                    cb = 0;
                }
                if pk < v - t {
                    cd += 1;
                } else {
                    max_d = max_d.max(cd);
                    cd = 0;
                }
            }

            if max_b.max(cb) >= 9 || max_d.max(cd) >= 9 {
                corner_pos.push(i);
                score_row[i] = if compute_score {
                    fast_score(ptr, threshold, offsets)
                } else {
                    1
                };
            }
        }
    }
}

/// Computes the FAST-9 score for a pixel
#[inline(always)]
unsafe fn fast_score(center_ptr: *const u8, _threshold: u8, offsets: &[isize; 25]) -> u8 {
    let v = *center_ptr;

    let mut d_bright = [0u8; 32];
    let mut d_dark = [0u8; 32];

    for i in 0..25 {
        let p = *center_ptr.offset(offsets[i]);
        d_bright[i] = p.saturating_sub(v);
        d_dark[i] = v.saturating_sub(p);
    }

    let load = |arr: &[u8; 32], k| -> u8x16 {
        let ptr = arr.as_ptr().add(k);
        let chunk: [u8; 16] = std::slice::from_raw_parts(ptr, 16).try_into().unwrap();
        u8x16::new(chunk)
    };

    let solve_max_arc = |arr: &[u8; 32]| -> u8 {
        let m01 = load(arr, 0).min(load(arr, 1));
        let m23 = load(arr, 2).min(load(arr, 3));
        let m45 = load(arr, 4).min(load(arr, 5));
        let m67 = load(arr, 6).min(load(arr, 7));

        let min_arc = m01.min(m23).min(m45.min(m67)).min(load(arr, 8));
        reduce_max_u8x16(min_arc)
    };

    solve_max_arc(&d_bright).max(solve_max_arc(&d_dark))
}
#[cfg(test)]
mod tests {
    use super::*;
    use kornia_image::{allocator::CpuAllocator, Image, ImageError};

    fn make_image(
        width: usize,
        height: usize,
        data: Vec<u8>,
    ) -> Result<Image<u8, 1, CpuAllocator>, ImageError> {
        Image::new([width, height].into(), data, CpuAllocator)
    }

    fn create_square_pattern() -> Image<u8, 1, CpuAllocator> {
        let width = 50;
        let height = 50;
        let mut data = vec![50u8; width * height];

        for y in 15..35 {
            for x in 15..35 {
                data[y * width + x] = 200;
            }
        }

        make_image(width, height, data).unwrap()
    }

    #[test]
    fn test_fast_detection() {
        let image = create_square_pattern();
        let detector = FastDetector::new(50, false);

        let keypoints = detector.detect(&image).unwrap();

        assert!(
            keypoints.len() > 4,
            "Without NMS, expected clusters (>4), found {}",
            keypoints.len()
        );
    }

    #[test]
    fn test_flat_color_image() {
        let width = 64;
        let height = 64;
        let data = vec![128u8; width * height];
        let image = make_image(width, height, data).unwrap();

        let detector = FastDetector::new(5, true);
        let keypoints = detector.detect(&image).unwrap();

        assert!(
            keypoints.is_empty(),
            "Flat image should contain no keypoints"
        );
    }

    #[test]
    fn test_image_too_small() {
        let width = 18;
        let height = 20;
        let data = vec![0u8; width * height];
        let image = make_image(width, height, data).unwrap();

        let detector = FastDetector::new(10, true);
        let keypoints = detector.detect(&image).unwrap();

        assert!(
            keypoints.is_empty(),
            "Image width < 19 should return empty result immediately"
        );
    }
}
