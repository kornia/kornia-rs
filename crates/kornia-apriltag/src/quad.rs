use crate::{
    segmentation::GradientInfo,
    utils::{homography_compute, Pixel},
    DecodeTagsConfig,
};
use kornia_algebra::{Mat3F32, Vec3F32};
use kornia_image::{allocator::ImageAllocator, Image};
use kornia_imgproc::filter::kernels::gaussian_kernel_1d;
use std::{
    collections::HashMap,
    f32::{self, consts::PI},
    ops::ControlFlow,
};

const SLOPE_OFFSET_BASE: i32 = 2 << 15; // Base value for slope offset calculations.
const SLOPE_OFFSET_DOUBLE: i32 = 2 * SLOPE_OFFSET_BASE; // Double the base value for extended range.

/// Constants defining quadrant boundaries for slope offset logic.
const QUADRANTS: [[i32; 2]; 2] = [
    [-SLOPE_OFFSET_BASE, 0],
    [SLOPE_OFFSET_DOUBLE, SLOPE_OFFSET_BASE],
];

#[derive(Debug, Clone, Copy, PartialEq)]
/// Options for fitting quadrilaterals (quads) to clusters of gradient information.
pub struct FitQuadConfig {
    /// Cosine of the critical angle in radians.
    pub cos_critical_rad: f32,
    /// Maximum mean squared error allowed for line fitting.
    pub max_line_fit_mse: f32,
    /// Maximum number of maxima to consider.
    pub max_nmaxima: usize,
    /// Minimum number of pixels required in a cluster to be considered.
    pub min_cluster_pixels: usize,
}

impl Default for FitQuadConfig {
    fn default() -> Self {
        Self {
            cos_critical_rad: (10.0 * PI / 180.0).cos(),
            max_line_fit_mse: 10.0,
            max_nmaxima: 10,
            min_cluster_pixels: 5,
        }
    }
}

/// Represents a detected quadrilateral in the image, corresponding to a tag candidate.
#[derive(Debug, Clone, PartialEq)]
pub struct Quad {
    /// The four corners of the quadrilateral, in image coordinates.
    ///
    /// Order: [Bottom-left, Bottom-right, Top-right, Top-left]
    pub corners: [kornia_algebra::Vec2F32; 4],
    /// Indicates whether the border is reversed (black border inside white border).
    pub reversed_border: bool,
    /// The 3x3 homography matrix mapping tag coordinates to image coordinates.
    pub homography: Mat3F32,
}

impl Default for Quad {
    fn default() -> Self {
        Self {
            corners: [kornia_algebra::Vec2F32::ZERO; 4],
            reversed_border: false,
            homography: Mat3F32::ZERO,
        }
    }
}

impl Quad {
    /// Projects a point (x, y) from tag coordinates to image coordinates using the quad's homography.
    ///
    /// # Arguments
    ///
    /// * `x` - The x coordinate in tag space.
    /// * `y` - The y coordinate in tag space.
    ///
    /// # Returns
    ///
    /// A `Vec2F32` representing the projected point in image coordinates.
    pub fn homography_project(&self, x: f32, y: f32) -> kornia_algebra::Vec2F32 {
        let p = self.homography * Vec3F32::new(x, y, 1.0);
        kornia_algebra::Vec2F32::new(p.x / p.z, p.y / p.z)
    }

    /// Updates the homography matrix for the quad based on its current corner positions.
    ///
    /// # Returns
    ///
    /// `true` if the homography was successfully computed and updated, `false` otherwise.
    pub fn update_homographies(&mut self) -> bool {
        let corr_arr = [
            [-1.0, -1.0, self.corners[0].x, self.corners[0].y],
            [1.0, -1.0, self.corners[1].x, self.corners[1].y],
            [1.0, 1.0, self.corners[2].x, self.corners[2].y],
            [-1.0, 1.0, self.corners[3].x, self.corners[3].y],
        ];

        if let Some(h) = homography_compute(corr_arr) {
            self.homography = h;
            return true;
        }

        false
    }
}

/// Fits quadrilaterals (quads) to clusters of gradient information in the image.
///
/// # Arguments
///
/// * `src` - The source image.
/// * `clusters` - A mutable reference to a HashMap containing clusters of `GradientInfo`.
/// * `config` - Configuration for decoding tags.
///
/// # Returns
///
/// A vector of detected `Quad` structures.
// TODO: Support multiple tag families
pub fn fit_quads<A: ImageAllocator>(
    src: &Image<Pixel, 1, A>,
    clusters: &mut HashMap<(usize, usize), Vec<GradientInfo>>,
    config: &DecodeTagsConfig,
) -> Vec<Quad> {
    // TODO: Avoid this allocation every time
    let mut quads = Vec::new();

    let max_cluster_len = 4 * (src.width() + src.height());

    clusters.iter_mut().for_each(|(_, cluster)| {
        if cluster.len() < config.fit_quad_config.min_cluster_pixels {
            return;
        }

        // Skip the cluster, if it is too large
        if cluster.len() > max_cluster_len {
            return;
        }

        if let Some(mut quad) = fit_single_quad(
            src,
            cluster,
            config.min_tag_width,
            config.normal_border,
            config.reversed_border,
            &config.fit_quad_config,
        ) {
            if config.downscale_factor > 1 {
                let downscale_factor = config.downscale_factor as f32;
                quad.corners.iter_mut().for_each(|c| {
                    c.x *= downscale_factor;
                    c.y *= downscale_factor;
                });
            }

            quads.push(quad);
        }
    });

    quads
}

/// Fits a single quadrilateral (quad) to a cluster of gradient information in the image.
///
/// # Arguments
///
/// * `src` - The source image.
/// * `cluster` - A mutable slice of `GradientInfo` representing the cluster.
/// * `min_tag_width` - Minimum width of the tag to be considered.
/// * `normal_border` - Indicates if the normal border is expected.
/// * `reversed_border` - Indicates if the reversed border is expected.
/// * `config` - Configuration for quad fitting process
///
/// # Returns
///
/// An `Option<Quad>` containing the detected quadrilateral if successful, or `None` otherwise.
fn fit_single_quad<A: ImageAllocator>(
    src: &Image<Pixel, 1, A>,
    cluster: &mut [GradientInfo],
    min_tag_width: usize,
    normal_border: bool,
    reversed_border: bool,
    config: &FitQuadConfig,
) -> Option<Quad> {
    if cluster.len() < 24 {
        return None;
    }

    let mut x_max = cluster[0].pos.x;
    let mut x_min = x_max;
    let mut y_max = cluster[0].pos.y;
    let mut y_min = y_max;

    // Find the bounding box of the cluster
    cluster.iter().for_each(|GradientInfo { pos, .. }| {
        if pos.x > x_max {
            x_max = pos.x;
        } else if pos.x < x_min {
            x_min = pos.x;
        }

        if pos.y > y_max {
            y_max = pos.y;
        } else if pos.y < y_min {
            y_min = pos.y;
        }
    });

    // Reject clusters whose bounding box is too small
    if (x_max - x_min) * (y_max - y_min) < min_tag_width {
        return None;
    }

    // add some noise to (cx,cy) so that pixels get a more diverse set
    // of theta estimates. This will help us remove more points.
    let cx = (x_min + x_max) as f32 * 0.5 + 0.05118;
    let cy = (y_min + y_max) as f32 * 0.5 - 0.028581;

    let mut dot = 0.0;

    cluster
        .iter_mut()
        .for_each(|GradientInfo { pos, gx, gy, slope }| {
            let mut dx = pos.x as f32 - cx;
            let mut dy = pos.y as f32 - cy;

            // Convert gradient direction enum to numeric values
            let gx_val = *gx as i32;
            let gy_val = *gy as i32;

            dot += dx * gx_val as f32 + dy * gy_val as f32;

            let quadrant = QUADRANTS[(dy > 0.0) as usize][(dx > 0.0) as usize];

            if dy < 0.0 {
                dy = -dy;
                dx = -dx;
            }

            if dx < 0.0 {
                let tmp = dx;
                dx = dy;
                dy = -tmp;
            }

            *slope = quadrant as f32 + dy / dx;
        });

    let mut quad = Quad {
        reversed_border: dot < 0.0,
        ..Default::default()
    };

    // Ensure that the black border is inside the white border
    if !reversed_border && quad.reversed_border {
        return None;
    }
    if !normal_border && !quad.reversed_border {
        return None;
    }

    cluster.sort_by(|a, b| a.slope.total_cmp(&b.slope));

    let lfps = compute_line_fit_prefix_sums(src, cluster);

    let mut indices = [0usize; 4];

    if !quad_segment_maxima(cluster, &lfps, &mut indices, config) {
        return None;
    }

    let mut lines = [[0.0f32; 4]; 4];

    if let ControlFlow::Break(_) = (0..4).try_for_each(|i| {
        let i0 = indices[i];
        let i1 = indices[(i + 1) & 3];

        let mut mse = 0.0f32;
        fit_line(&lfps, i0, i1, Some(&mut lines[i]), None, Some(&mut mse));

        if mse > config.max_line_fit_mse {
            return ControlFlow::Break(());
        }

        ControlFlow::Continue(())
    }) {
        return None;
    };

    if let ControlFlow::Break(_) = (0..4).try_for_each(|i| {
        let a00 = lines[i][3];
        let a01 = -lines[(i + 1) & 3][3];
        let a10 = -lines[i][2];
        let a11 = lines[(i + 1) & 3][2];
        let b0 = -lines[i][0] + lines[(i + 1) & 3][0];
        let b1 = -lines[i][1] + lines[(i + 1) & 3][1];

        let det = a00 * a11 - a10 * a01;

        if det.abs() < 0.001 {
            return ControlFlow::Break(());
        }

        let w00 = a11 / det;
        let w01 = -a01 / det;

        let l0 = w00 * b0 + w01 * b1;

        quad.corners[i].x = lines[i][0] + l0 * a00;
        quad.corners[i].y = lines[i][1] + l0 * a10;

        ControlFlow::Continue(())
    }) {
        return None;
    };

    let mut area = 0.0f32;

    let mut length = [0f32; 3];
    let mut p: f32;

    // Calculate area of triangle formed by points 0, 1, 2
    (0..3).for_each(|i| {
        let idxa = i;
        let idxb = (i + 1) % 3;
        length[i] = ((quad.corners[idxb].x - quad.corners[idxa].x).powi(2)
            + (quad.corners[idxb].y - quad.corners[idxa].y).powi(2))
        .sqrt();
    });

    p = (length[0] + length[1] + length[2]) / 2.0;
    area += (p * (p - length[0]) * (p - length[1]) * (p - length[2])).sqrt();

    // Calculate area of triangle formed by points 2, 3, 0
    (0..3).for_each(|i| {
        let idxs = [2, 3, 0, 2];
        let idxa = idxs[i];
        let idxb = idxs[i + 1];

        length[i] = ((quad.corners[idxb].x - quad.corners[idxa].x).powi(2)
            + (quad.corners[idxb].y - quad.corners[idxa].y).powi(2))
        .sqrt();
    });

    p = (length[0] + length[1] + length[2]) / 2.0;
    area += (p * (p - length[0]) * (p - length[1]) * (p - length[2])).sqrt();

    // Reject quads that are too small
    if area < 0.95 * min_tag_width as f32 * min_tag_width as f32 {
        return None;
    }

    // Reject quads whose cumulative angle change isn't equal to 2PI
    if let ControlFlow::Break(_) = (0..4).try_for_each(|i| {
        let i0 = i;
        let i1 = (i + 1) & 3;
        let i2 = (i + 2) & 3;

        let dx1 = quad.corners[i1].x - quad.corners[i0].x;
        let dy1 = quad.corners[i1].y - quad.corners[i0].y;
        let dx2 = quad.corners[i2].x - quad.corners[i1].x;
        let dy2 = quad.corners[i2].y - quad.corners[i1].y;

        let cos_dtheta =
            (dx1 * dx2 + dy1 * dy2) / ((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2)).sqrt();

        if !(-config.cos_critical_rad..=config.cos_critical_rad).contains(&cos_dtheta)
            || dx1 * dy2 < dy1 * dx2
        {
            return ControlFlow::Break(());
        }

        ControlFlow::Continue(())
    }) {
        return None;
    };

    Some(quad)
}

/// Stores prefix sums for weighted line fitting over a set of points.
#[derive(Default, Debug, Clone)]
struct LineFit {
    /// Weighted sum of x coordinates ($\sum_i w_i x_i$)
    mx: f32,
    /// Weighted sum of y coordinates ($\sum_i w_i y_i$)
    my: f32,
    /// Weighted sum of squared x coordinates ($\sum_i w_i x_i^2$)
    mxx: f32,
    /// Weighted sum of x·y products ($\sum_i w_i x_i y_i$)
    mxy: f32,
    /// Weighted sum of squared y coordinates ($\sum_i w_i y_i^2$)
    myy: f32,
    /// Total weight ($\sum_i w_i$)
    w: f32,
}

/// Computes prefix sums for weighted line fitting over a set of gradient information points.
///
/// # Arguments
///
/// * `src` - The source image.
/// * `gradient_infos` - A slice of `GradientInfo` representing the cluster.
///
/// # Returns
///
/// A vector of `LineFit` structures containing prefix sums for each point.
fn compute_line_fit_prefix_sums<A: ImageAllocator>(
    src: &Image<Pixel, 1, A>,
    gradient_infos: &[GradientInfo],
) -> Vec<LineFit> {
    let src_slice = src.as_slice();
    // TODO: Find a way to avoid allocation
    let mut lfps = vec![LineFit::default(); gradient_infos.len()];

    gradient_infos.iter().enumerate().for_each(|(i, cluster)| {
        if i > 0 {
            lfps[i] = lfps[i - 1].clone();
        }

        let delta = 0.5f32;
        let x = cluster.pos.x as f32 * 0.5 + delta;
        let y = cluster.pos.y as f32 * 0.5 + delta;
        let ix = x as usize;
        let iy = y as usize;
        let mut w = 1.0f32;

        if ix > 0 && ix + 1 < src.width() && iy > 0 && iy + 1 < src.height() {
            let grad_x = src_slice[iy * src.width() + ix + 1] as i32
                - src_slice[iy * src.width() + ix - 1] as i32;

            let grad_y = src_slice[(iy + 1) * src.width() + ix] as i32
                - src_slice[(iy - 1) * src.width() + ix] as i32;

            w = ((grad_x * grad_x + grad_y * grad_y) as f32).sqrt() + 1.0;
        }

        let fx = x;
        let fy = y;

        lfps[i].mx += w * fx;
        lfps[i].my += w * fy;
        lfps[i].mxx += w * fx * fx;
        lfps[i].mxy += w * fx * fy;
        lfps[i].myy += w * fy * fy;
        lfps[i].w += w;
    });

    lfps
}

/// Segments the gradient information into four maxima corresponding to the corners of a quadrilateral.
///
/// This function analyzes the error profile of line fits over the gradient information and finds
/// four maxima (peaks) in the error signal, which are interpreted as the corners of a quad. It
/// returns `true` if a valid segmentation is found and writes the indices of the maxima into `indices`.
///
/// # Arguments
///
/// * `gradient_infos` - Slice of `GradientInfo` representing the cluster.
/// * `lfps` - Slice of `LineFit` prefix sums for the cluster.
/// * `indices` - Mutable reference to an array where the four corner indices will be written.
/// * `config` - Configuration for quad fitting process
///
/// # Returns
///
/// `true` if four valid maxima are found and written to `indices`, `false` otherwise.
fn quad_segment_maxima(
    gradient_infos: &[GradientInfo],
    lfps: &[LineFit],
    indices: &mut [usize; 4],
    config: &FitQuadConfig,
) -> bool {
    debug_assert_eq!(gradient_infos.len(), lfps.len());
    let len = gradient_infos.len();
    let window_size = 20.min(len / 12);

    // can't fit a quad, too few points
    if window_size < 2 {
        return false;
    }

    let mut errors = vec![0.0f32; len];

    (0..len).for_each(|i| {
        fit_line(
            lfps,
            (i + len - window_size) % len,
            (i + window_size) % len,
            None,
            errors.get_mut(i),
            None,
        );
    });

    const SIGMA: f32 = 1.0;
    const CUTOFF: f32 = 0.05;

    let filter_size = (2.0 * ((-(CUTOFF.ln()) * 2.0 * SIGMA * SIGMA).sqrt() + 1.0) + 1.0) as usize;
    let mut smoothed_errors = vec![0.0f32; len];

    let gaussian_kernel = gaussian_kernel_1d(filter_size, SIGMA);

    (0..len).for_each(|iy| {
        let mut acc = 0.0f32;

        (0..filter_size).for_each(|i| {
            let idx = (iy as isize + i as isize - filter_size as isize / 2 + len as isize)
                .rem_euclid(gradient_infos.len() as isize) as usize;
            acc += errors[idx] * gaussian_kernel[i];
        });

        smoothed_errors[iy] = acc;
    });

    errors = smoothed_errors;

    let mut maxima = vec![0usize; len];
    let mut maxima_errs = vec![0.0f32; len];
    let mut nmaxima = 0usize;

    (0..gradient_infos.len()).for_each(|i| {
        if errors[i] > errors[(i + 1) % len] && errors[i] > errors[(i + len - 1) % len] {
            maxima[nmaxima] = i;
            maxima_errs[nmaxima] = errors[i];
            nmaxima += 1;
        }
    });

    if nmaxima < 4 {
        return false;
    }

    if nmaxima > config.max_nmaxima {
        let mut maxima_errs_copy = maxima_errs.clone();

        maxima_errs_copy.sort_by(|a, b| b.total_cmp(a));

        let maxima_thresh = maxima_errs_copy[config.max_nmaxima];
        let mut out = 0usize;
        for i in 0..nmaxima {
            if maxima_errs[i] <= maxima_thresh {
                continue;
            }
            maxima[out] = maxima[i];
            out += 1;
        }

        if out < 4 {
            return false;
        }

        nmaxima = out;
    }

    let mut best_indices = [0usize, 0, 0, 0];
    let mut best_error = f32::INFINITY;

    let mut err01 = 0.0f32;
    let mut err12 = 0.0f32;
    let mut err23 = 0.0f32;
    let mut err30 = 0.0f32;

    let mut mse01 = 0.0f32;
    let mut mse12 = 0.0f32;
    let mut mse23 = 0.0f32;
    let mut mse30 = 0.0f32;

    let mut params01 = [0.0f32; 4];
    let mut params12 = [0.0f32; 4];

    (0..nmaxima - 3).for_each(|m0| {
        let i0 = maxima[m0];

        ((m0 + 1)..(nmaxima - 2)).for_each(|m1| {
            let i1 = maxima[m1];

            fit_line(
                lfps,
                i0,
                i1,
                Some(&mut params01),
                Some(&mut err01),
                Some(&mut mse01),
            );

            if mse01 > config.max_line_fit_mse {
                return;
            }

            ((m1 + 1)..nmaxima - 1).for_each(|m2| {
                let i2 = maxima[m2];

                fit_line(
                    lfps,
                    i1,
                    i2,
                    Some(&mut params12),
                    Some(&mut err12),
                    Some(&mut mse12),
                );

                if mse12 > config.max_line_fit_mse {
                    return;
                }

                let dot = params01[2] * params12[2] + params01[3] * params12[3];
                if dot.abs() > config.cos_critical_rad {
                    return;
                }

                ((m2 + 1)..nmaxima).for_each(|m3| {
                    let i3 = maxima[m3];

                    fit_line(lfps, i2, i3, None, Some(&mut err23), Some(&mut mse23));

                    if mse23 > config.max_line_fit_mse {
                        return;
                    }

                    fit_line(lfps, i3, i0, None, Some(&mut err30), Some(&mut mse30));

                    if mse30 > config.max_line_fit_mse {
                        return;
                    }

                    let err = err01 + err12 + err23 + err30;

                    if err < best_error {
                        best_error = err;
                        best_indices[0] = i0;
                        best_indices[1] = i1;
                        best_indices[2] = i2;
                        best_indices[3] = i3;
                    }
                });
            });
        });
    });

    if best_error.is_infinite() {
        return false;
    }

    best_indices.iter().enumerate().for_each(|(i, b)| {
        indices[i] = *b;
    });

    if (best_error / gradient_infos.len() as f32) < config.max_line_fit_mse {
        return true;
    }

    false
}

/// Fits a line to a segment of points using prefix sums for weighted least squares.
///
/// This function computes the best-fit line parameters for the segment of points between indices `i0` and `i1`
/// (inclusive, possibly wrapping around the end of the array), using the provided prefix sums (`lfps`).
/// Optionally outputs the line parameters, error, and mean squared error.
///
/// # Arguments
///
/// * `lfps` - Slice of `LineFit` prefix sums for the points.
/// * `i0` - Start index of the segment (inclusive).
/// * `i1` - End index of the segment (inclusive).
/// * `lineparm` - Optional mutable reference to an array where the line parameters will be written.
///   The array is [ex, ey, nx, ny], where (ex, ey) is the centroid and (nx, ny) is the direction.
/// * `err` - Optional mutable reference to a float where the error will be written.
/// * `mse` - Optional mutable reference to a float where the mean squared error will be written.
fn fit_line(
    lfps: &[LineFit],
    i0: usize,
    i1: usize,
    lineparm: Option<&mut [f32; 4]>,
    err: Option<&mut f32>,
    mse: Option<&mut f32>,
) {
    if i0 == i1 || !(i0 < lfps.len() && i1 < lfps.len()) {
        return;
    }

    let mut mx: f32;
    let mut my: f32;
    let mut mxx: f32;
    let mut myy: f32;
    let mut mxy: f32;
    let mut w: f32;
    let n: usize;

    if i0 < i1 {
        n = i1 - i0 + 1;

        mx = lfps[i1].mx;
        my = lfps[i1].my;
        mxx = lfps[i1].mxx;
        mxy = lfps[i1].mxy;
        myy = lfps[i1].myy;
        w = lfps[i1].w;

        if i0 > 0 {
            mx -= lfps[i0 - 1].mx;
            my -= lfps[i0 - 1].my;
            mxx -= lfps[i0 - 1].mxx;
            mxy -= lfps[i0 - 1].mxy;
            myy -= lfps[i0 - 1].myy;
            w -= lfps[i0 - 1].w;
        }
    } else if i0 > 0 {
        mx = lfps[lfps.len() - 1].mx - lfps[i0 - 1].mx;
        my = lfps[lfps.len() - 1].my - lfps[i0 - 1].my;
        mxx = lfps[lfps.len() - 1].mxx - lfps[i0 - 1].mxx;
        mxy = lfps[lfps.len() - 1].mxy - lfps[i0 - 1].mxy;
        myy = lfps[lfps.len() - 1].myy - lfps[i0 - 1].myy;
        w = lfps[lfps.len() - 1].w - lfps[i0 - 1].w;

        mx += lfps[i1].mx;
        my += lfps[i1].my;
        mxx += lfps[i1].mxx;
        mxy += lfps[i1].mxy;
        myy += lfps[i1].myy;
        w += lfps[i1].w;

        n = lfps.len() - i0 + i1 + 1;
    } else {
        return;
    }

    if n < 2 {
        return;
    }

    let ex = mx / w;
    let ey = my / w;
    let cxx = mxx / w - ex * ex;
    let cxy = mxy / w - ex * ey;
    let cyy = myy / w - ey * ey;

    let eig_small = 0.5 * (cxx + cyy - ((cxx - cyy) * (cxx - cyy) + 4.0 * cxy * cxy).sqrt());

    if let Some(lineparm) = lineparm {
        lineparm[0] = ex;
        lineparm[1] = ey;

        let eig = 0.5 * (cxx + cyy + ((cxx - cyy) * (cxx - cyy) + 4.0 * cxy * cxy).sqrt());
        let nx1 = cxx - eig;
        let ny1 = cxy;
        let m1 = nx1 * nx1 + ny1 * ny1;
        let nx2 = cxy;
        let ny2 = cyy - eig;
        let m2 = nx2 * nx2 + ny2 * ny2;

        let nx: f32;
        let ny: f32;
        let m: f32;

        if m1 > m2 {
            nx = nx1;
            ny = ny1;
            m = m1;
        } else {
            nx = nx2;
            ny = ny2;
            m = m2;
        }

        let length = m.sqrt();

        if length.abs() < 1e-12 {
            lineparm[2] = 0.0;
            lineparm[3] = 0.0;
        } else {
            lineparm[2] = nx / length;
            lineparm[3] = ny / length;
        }
    }

    if let Some(err) = err {
        *err = n as f32 * eig_small;
    }

    if let Some(mse) = mse {
        *mse = eig_small;
    }
}

#[cfg(test)]
mod tests {
    use kornia_image::allocator::CpuAllocator;
    use kornia_io::png::read_image_png_mono8;

    use crate::{
        family::TagFamilyKind,
        segmentation::{
            find_connected_components, find_gradient_clusters, GradientDirection, GradientInfo,
        },
        threshold::{adaptive_threshold, TileMinMax},
        union_find::UnionFind,
        utils::{Pixel, Point2d},
    };

    use super::*;

    #[test]
    fn test_fit_quads() -> Result<(), Box<dyn std::error::Error>> {
        let src = read_image_png_mono8("../../tests/data/apriltag.png")?;

        let mut bin = Image::from_size_val(src.size(), Pixel::Skip, CpuAllocator)?;
        let mut tile_min_max = TileMinMax::new(src.size(), 4);
        let mut uf = UnionFind::new(src.as_slice().len());
        let mut clusters = HashMap::new();

        adaptive_threshold(&src, &mut bin, &mut tile_min_max, 20)?;
        find_connected_components(&bin, &mut uf)?;
        find_gradient_clusters(&bin, &mut uf, &mut clusters);

        let mut decode_tag_config = DecodeTagsConfig::new(vec![TagFamilyKind::Tag36H11])?;
        decode_tag_config.downscale_factor = 1;

        let quads = fit_quads(&bin, &mut clusters, &decode_tag_config);

        let expected_quad = [[[27, 3], [27, 27], [3, 27], [3, 3]]];

        assert_eq!(quads.len(), expected_quad.len());

        for (Quad { corners: point, .. }, expected_quad) in quads.iter().zip(expected_quad) {
            // We allow a ±1 error here to avoid CI failures due to small precision differences.
            // The expected outputs were generated with C code using 64-bit precision, while our code uses f32.
            assert!((point[0].x as usize).abs_diff(expected_quad[0][0]) <= 1);
            assert!((point[0].y as usize).abs_diff(expected_quad[0][1]) <= 1);

            assert!((point[1].x as usize).abs_diff(expected_quad[1][0]) <= 1);
            assert!((point[1].y as usize).abs_diff(expected_quad[1][1]) <= 1);

            assert!((point[2].x as usize).abs_diff(expected_quad[2][0]) <= 1);
            assert!((point[2].y as usize).abs_diff(expected_quad[2][1]) <= 1);

            assert!((point[3].x as usize).abs_diff(expected_quad[3][0]) <= 1);
            assert!((point[3].y as usize).abs_diff(expected_quad[3][1]) <= 1);
        }

        Ok(())
    }

    #[test]
    fn test_quad_segment_maxima_edge_cases() {
        // Test 1: Edge cases - too few points, empty input, constant slope
        let gradient_infos = vec![
            GradientInfo {
                pos: Point2d { x: 0, y: 0 },
                gx: GradientDirection::TowardsWhite,
                gy: GradientDirection::TowardsBlack,
                slope: 0.0,
            };
            20
        ];
        let lfps = vec![LineFit::default(); 20];
        let mut indices = [0; 4];

        // Should return false because window_size < 2 (20/12 = 1.66... < 2)
        assert!(!quad_segment_maxima(
            &gradient_infos,
            &lfps,
            &mut indices,
            &FitQuadConfig::default()
        ));

        // Test 2: Empty input
        let empty_gradient_infos = Vec::<GradientInfo>::new();
        let empty_lfps = Vec::<LineFit>::new();
        assert!(!quad_segment_maxima(
            &empty_gradient_infos,
            &empty_lfps,
            &mut indices,
            &FitQuadConfig::default()
        ));

        // Test 3: Constant slope (no maxima)
        let mut constant_slope_infos = Vec::new();
        for i in 0..100 {
            constant_slope_infos.push(GradientInfo {
                pos: Point2d { x: i, y: i },
                gx: GradientDirection::TowardsWhite,
                gy: GradientDirection::TowardsBlack,
                slope: 1.0, // Constant slope
            });
        }
        let constant_lfps = vec![LineFit::default(); 100];
        assert!(!quad_segment_maxima(
            &constant_slope_infos,
            &constant_lfps,
            &mut indices,
            &FitQuadConfig::default()
        ));
    }

    #[test]
    fn test_quad_segment_maxima() -> Result<(), Box<dyn std::error::Error>> {
        let src = read_image_png_mono8("../../tests/data/apriltag.png")?;
        let mut bin = Image::from_size_val(src.size(), Pixel::Skip, CpuAllocator)?;
        let mut tile_min_max = TileMinMax::new(src.size(), 4);
        let mut uf = UnionFind::new(src.as_slice().len());
        let mut clusters = HashMap::new();

        adaptive_threshold(&src, &mut bin, &mut tile_min_max, 20)?;
        find_connected_components(&bin, &mut uf)?;
        find_gradient_clusters(&bin, &mut uf, &mut clusters);

        // Find the largest cluster to test with
        let largest_cluster = clusters
            .values()
            .max_by_key(|cluster| cluster.len())
            .unwrap();

        if largest_cluster.len() >= 24 {
            let lfps = compute_line_fit_prefix_sums(&bin, largest_cluster);
            let mut indices = [0; 4];

            let result = quad_segment_maxima(
                largest_cluster,
                &lfps,
                &mut indices,
                &FitQuadConfig::default(),
            );

            assert!(!result);
        }

        Ok(())
    }

    #[test]
    fn test_fit_line() {
        // Create some test LineFit data
        let mut lfps: Vec<LineFit> = Vec::new();

        // Create a simple line of points: (0,0), (1,1), (2,2), (3,3)
        for i in 0..4 {
            let mut lfp = LineFit::default();
            if i > 0 {
                lfp = lfps[i - 1].clone();
            }

            let x = i as f32;
            let y = i as f32;
            let w = 1.0f32;

            lfp.mx += w * x;
            lfp.my += w * y;
            lfp.mxx += w * x * x;
            lfp.mxy += w * x * y;
            lfp.myy += w * y * y;
            lfp.w += w;

            lfps.push(lfp);
        }

        // Test 1: Normal case with all parameters
        let mut lineparm = [0.0f32; 4];
        let mut err = 0.0f32;
        let mut mse = 0.0f32;

        fit_line(
            &lfps,
            0,
            3,
            Some(&mut lineparm),
            Some(&mut err),
            Some(&mut mse),
        );

        // Check that parameters were computed
        assert!(lineparm[0] != 0.0 || lineparm[1] != 0.0); // ex, ey should be computed
        assert!(err >= 0.0); // Error can be zero for perfect fit
        assert!(mse >= 0.0); // MSE can be zero for perfect fit

        // Test 2: Edge case - same indices
        let mut lineparm2 = [0.0f32; 4];
        let mut err2 = 0.0f32;
        let mut mse2 = 0.0f32;

        fit_line(
            &lfps,
            1,
            1,
            Some(&mut lineparm2),
            Some(&mut err2),
            Some(&mut mse2),
        );

        // Should not modify parameters when i0 == i1
        assert_eq!(lineparm2[0], 0.0);
        assert_eq!(lineparm2[1], 0.0);
        assert_eq!(err2, 0.0);
        assert_eq!(mse2, 0.0);

        // Test 3: Edge case - out of bounds indices
        let mut lineparm3 = [0.0f32; 4];
        let mut err3 = 0.0f32;
        let mut mse3 = 0.0f32;

        fit_line(
            &lfps,
            0,
            10,
            Some(&mut lineparm3),
            Some(&mut err3),
            Some(&mut mse3),
        );

        // Should not modify parameters when indices are out of bounds
        assert_eq!(lineparm3[0], 0.0);
        assert_eq!(lineparm3[1], 0.0);
        assert_eq!(err3, 0.0);
        assert_eq!(mse3, 0.0);

        // Test 4: Edge case - i0 > i1 (wrapped case)
        let mut lineparm4 = [0.0f32; 4];
        let mut err4 = 0.0f32;
        let mut mse4 = 0.0f32;

        fit_line(
            &lfps,
            3,
            1,
            Some(&mut lineparm4),
            Some(&mut err4),
            Some(&mut mse4),
        );

        // Should compute parameters for wrapped case
        assert!(lineparm4[0] != 0.0 || lineparm4[1] != 0.0);
        assert!(err4 >= 0.0);
        assert!(mse4 >= 0.0);

        // Test 5: No output parameters (should not crash)
        fit_line(&lfps, 0, 3, None, None, None);
    }

    #[test]
    fn test_compute_lfps() {
        use crate::segmentation::GradientDirection;
        use crate::utils::Pixel;
        use kornia_image::{Image, ImageSize};

        // Create a 4x4 image with all white pixels
        let size = ImageSize {
            width: 4,
            height: 4,
        };
        let img = Image::<Pixel, 1, _>::from_size_val(size, Pixel::White, CpuAllocator).unwrap();

        // Test 1: Single point
        let gradient_infos = vec![GradientInfo {
            pos: Point2d { x: 1, y: 2 },
            gx: GradientDirection::TowardsWhite,
            gy: GradientDirection::TowardsBlack,
            slope: 0.0,
        }];
        let lfps = compute_line_fit_prefix_sums(&img, &gradient_infos);
        assert_eq!(lfps.len(), 1);
        // The weighted mean should be close to the point's position (scaled by 0.5 + delta)
        let expected_x = 1.0 * 0.5 + 0.5;
        let expected_y = 2.0 * 0.5 + 0.5;
        assert!((lfps[0].mx / lfps[0].w - expected_x).abs() < 1e-4);
        assert!((lfps[0].my / lfps[0].w - expected_y).abs() < 1e-4);

        // Test 2: Multiple points in a line
        let gradient_infos = vec![
            GradientInfo {
                pos: Point2d { x: 0, y: 0 },
                gx: GradientDirection::TowardsWhite,
                gy: GradientDirection::TowardsBlack,
                slope: 0.0,
            },
            GradientInfo {
                pos: Point2d { x: 1, y: 1 },
                gx: GradientDirection::TowardsWhite,
                gy: GradientDirection::TowardsBlack,
                slope: 0.0,
            },
            GradientInfo {
                pos: Point2d { x: 2, y: 2 },
                gx: GradientDirection::TowardsWhite,
                gy: GradientDirection::TowardsBlack,
                slope: 0.0,
            },
        ];
        let lfps = compute_line_fit_prefix_sums(&img, &gradient_infos);
        assert_eq!(lfps.len(), 3);
        // The last element should be the sum of all previous
        let last = &lfps[2];
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_w = 0.0;
        for GradientInfo { pos, .. } in gradient_infos.iter().take(3) {
            let x = pos.x as f32 * 0.5 + 0.5;
            let y = pos.y as f32 * 0.5 + 0.5;
            sum_x += x;
            sum_y += y;
            sum_w += 1.0;
        }
        assert!((last.mx - sum_x).abs() < 1e-4);
        assert!((last.my - sum_y).abs() < 1e-4);
        assert!((last.w - sum_w).abs() < 1e-4);

        // Test 3: Empty input
        let gradient_infos = vec![];
        let lfps = compute_line_fit_prefix_sums(&img, &gradient_infos);
        assert_eq!(lfps.len(), 0);
    }
}
