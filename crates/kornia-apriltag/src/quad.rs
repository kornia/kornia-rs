use crate::{family::TagFamily, segmentation::GradientInfo, utils::Pixel};
use kornia_image::{allocator::ImageAllocator, Image};
use std::{collections::HashMap, f32};

/// TODO
#[derive(Debug, Default, Clone)]
pub struct Quad {
    point: [[f32; 2]; 4],
    // reversed_bool: bool,
    // h: Vec<f32>,
    // hinv: Vec<f32>,
}

/// TODO
// TODO: Support multiple tag familes
pub fn fit_quads<A: ImageAllocator>(
    src: &Image<Pixel, 1, A>,
    tag_family: &TagFamily,
    clusters: &mut HashMap<(usize, usize), Vec<GradientInfo>>,
    min_cluster_pixels: usize,
) -> Vec<Quad> {
    // These will be come handy later, once we support more tag familes
    let normal_border = !tag_family.reversed_border;
    let reversed_border = tag_family.reversed_border;

    let mut quads = Vec::new();

    clusters.iter_mut().for_each(|(_, mut cluster)| {
        if cluster.len() < min_cluster_pixels {
            return;
        }

        if cluster.len() > 2 * (2 * src.width() + 2 * src.height()) {
            return;
        }

        if let Some(quad) = fit_quad(
            src,
            &mut cluster,
            tag_family.width_at_border,
            normal_border,
            reversed_border,
        ) {
            quads.push(quad);
        }
    });

    quads
}

const COS_CRITICAL_RAD: f32 = 0.984808; // TODO: Make this tuneable in fit_quad function

/// TODO
pub fn fit_quad<A: ImageAllocator>(
    src: &Image<Pixel, 1, A>,
    cluster: &mut [GradientInfo],
    min_tag_width: usize,
    normal_border: bool,
    reversed_border: bool,
) -> Option<Quad> {
    let mut x_max = cluster[0].pos.x;
    let mut x_min = x_max;
    let mut y_max = cluster[0].pos.y;
    let mut y_min = y_max;

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

    if (x_max - x_min) * (y_max - y_min) < min_tag_width {
        return None;
    }

    let cx = (x_min + x_max) as f32 * 0.5 + 0.05118;
    let cy = (y_min + y_max) as f32 * 0.5 - 0.028581;

    let mut dot = 0.0;

    let quadrants = [[-1 * (2 << 15), 0], [2 * (2 << 15), 2 << 15]];

    cluster
        .iter_mut()
        .for_each(|GradientInfo { pos, gx, gy, slope }| {
            let mut dx = pos.x as f32 - cx;
            let mut dy = pos.y as f32 - cy;

            // TODO: Improve Gradient Direction Logic
            let gx = *gx as i16;
            let gy = *gy as i16;

            dot += dx * gx as f32 + dy * gy as f32;

            let quadrant = quadrants[(dy > 0.0) as usize][(dx > 0.0) as usize];
            if dy < 0.0 {
                dy = -dy;
                dx = -dx;
            }

            if dx < 0.0 {
                let tmp = dx;
                dx = dy;
                dy = tmp;
            }

            *slope = quadrant as f32 + dy / dx;
        });

    let quad_reversed_border = dot < 0.0;
    if !reversed_border && reversed_border {
        return None;
    }
    if !normal_border && !quad_reversed_border {
        return None;
    }

    sort_gradient_info(cluster);

    let lfps = compute_lfps(src, cluster);

    let mut indices = [0usize; 4];

    if !quad_segment_maxima(cluster, &lfps, &mut indices) {
        return None;
    }

    let mut lines = [[0.0f32; 4]; 4];

    let mut should_return_none = false;
    (0..4).for_each(|i| {
        if should_return_none {
            return;
        }

        let i0 = indices[i];
        let i1 = indices[(i + 1) & 3];

        let mut mse = 0.0f32;
        fit_line(&lfps, i0, i1, Some(&mut lines[i]), None, Some(&mut mse));

        if mse > MAX_LINE_FIT_MSE {
            should_return_none = true;
            return;
        }
    });

    if should_return_none {
        return None;
    }

    should_return_none = false;

    let mut quad = Quad::default();

    (0..4).for_each(|i| {
        if should_return_none {
            return;
        }

        let a00 = lines[i][3];
        let a01 = -lines[(i + 1) & 3][3];
        let a10 = -lines[i][2];
        let a11 = lines[(i + 1) & 3][2];
        let b0 = -lines[i][0] + lines[(i + 1) & 3][0];
        let b1 = -lines[i][1] + lines[(i + 1) & 3][1];

        let det = a00 * a11 - a10 * a01;

        if det.abs() < 0.001 {
            should_return_none = true;
            return;
        }

        let w00 = a11 / det;
        let w01 = -a01 / det;

        let l0 = w00 * b0 + w01 * b1;

        quad.point[i][0] = lines[i][0] + l0 * a00;
        quad.point[i][1] = lines[i][1] + l0 * a10;
    });

    if should_return_none {
        return None;
    }

    // Reject quads that are too small
    let mut area = 0.0f32;

    let mut length = [0f32; 3];
    let mut p: f32;

    (0..3).for_each(|i| {
        let idxa = i;
        let idxb = (i + 1) % 3;
        length[i] = ((quad.point[idxb][0] - quad.point[idxa][0]).powi(2)
            + (quad.point[idxb][1] - quad.point[idxa][1]).powi(2))
        .sqrt();
    });

    p = (length[0] + length[1] + length[2]) / 2.0;
    area += (p * (p - length[0]) * (p - length[1]) * (p - length[2])).sqrt();

    (0..3).for_each(|i| {
        let idxs = [2, 3, 0, 2];
        let idxa = idxs[i];
        let idxb = idxs[i + 1];

        length[i] = ((quad.point[idxb][0] - quad.point[idxa][0]).powi(2)
            + (quad.point[idxb][1] - quad.point[idxa][1]).powi(2))
        .sqrt();
    });

    p = (length[0] + length[1] + length[2]) / 2.0;
    area += (p * (p - length[0]) * (p - length[1]) * (p - length[2])).sqrt();

    if area < 0.95 * min_tag_width as f32 * min_tag_width as f32 {
        return None;
    }

    should_return_none = false;
    // Reject quads whose cumulative angle change isn't equal to 2PI
    (0..4).for_each(|i| {
        if should_return_none {
            return;
        }

        let i0 = i;
        let i1 = (i + 1) & 3;
        let i2 = (i + 2) & 3;

        let dx1 = quad.point[i1][0] - quad.point[i0][0];
        let dy1 = quad.point[i1][1] - quad.point[i0][1];
        let dx2 = quad.point[i2][0] - quad.point[i1][0];
        let dy2 = quad.point[i2][1] - quad.point[i1][1];

        let cos_dtheta =
            (dx1 * dx2 + dy1 * dy2) / ((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2)).sqrt();

        if cos_dtheta > COS_CRITICAL_RAD || cos_dtheta < -COS_CRITICAL_RAD || dx1 * dy2 < dy1 * dx2
        {
            should_return_none = true;
            return;
        }
    });

    if should_return_none {
        return None;
    }

    Some(quad)
}

/// TODO
pub fn sort_gradient_info(gradient_infos: &mut [GradientInfo]) {
    match gradient_infos.len() {
        0 | 1 => return,
        2 => sort2(gradient_infos),
        3 => sort3(gradient_infos),
        4 => sort4(gradient_infos),
        5 => sort5(gradient_infos),
        _ => merge_sort(gradient_infos),
    }
}

fn maybe_swap(gradient_infos: &mut [GradientInfo], i: usize, j: usize) {
    if gradient_infos[i].slope > gradient_infos[j].slope {
        gradient_infos.swap(i, j);
    }
}

fn sort2(gradient_infos: &mut [GradientInfo]) {
    maybe_swap(gradient_infos, 0, 1);
}

fn sort3(gradient_infos: &mut [GradientInfo]) {
    maybe_swap(gradient_infos, 0, 1);
    maybe_swap(gradient_infos, 0, 2);
    maybe_swap(gradient_infos, 1, 2);
}

fn sort4(gradient_infos: &mut [GradientInfo]) {
    maybe_swap(gradient_infos, 0, 1);
    maybe_swap(gradient_infos, 2, 3);
    maybe_swap(gradient_infos, 0, 2);
    maybe_swap(gradient_infos, 1, 3);
    maybe_swap(gradient_infos, 1, 2);
}

fn sort5(gradient_infos: &mut [GradientInfo]) {
    maybe_swap(gradient_infos, 0, 1);
    maybe_swap(gradient_infos, 3, 4);
    maybe_swap(gradient_infos, 2, 4);
    maybe_swap(gradient_infos, 2, 3);
    maybe_swap(gradient_infos, 1, 4);
    maybe_swap(gradient_infos, 0, 3);
    maybe_swap(gradient_infos, 0, 2);
    maybe_swap(gradient_infos, 1, 3);
    maybe_swap(gradient_infos, 1, 2);
}

fn merge_sort(gradient_infos: &mut [GradientInfo]) {
    let mut temp = gradient_infos.to_vec();
    merge_sort_rec(gradient_infos, &mut temp);
}

fn merge_sort_rec(gradient_infos: &mut [GradientInfo], temp: &mut [GradientInfo]) {
    let len = gradient_infos.len();

    match len {
        0 | 1 => return,
        2 => sort2(gradient_infos),
        3 => sort3(gradient_infos),
        4 => sort4(gradient_infos),
        5 => sort5(gradient_infos),
        _ => {
            let mid = len / 2;
            let (left, right) = gradient_infos.split_at_mut(mid);
            let (temp_left, temp_right) = temp.split_at_mut(mid);

            merge_sort_rec(left, temp_left);
            merge_sort_rec(right, temp_right);

            merge(left, right, temp);
            gradient_infos.copy_from_slice(temp);
        }
    }
}

fn merge(left: &[GradientInfo], right: &[GradientInfo], out: &mut [GradientInfo]) {
    let mut i = 0;
    let mut j = 0;
    let mut k = 0;
    let n = left.len();
    let m = right.len();

    while i < n && j < m {
        if left[i].slope <= right[j].slope {
            out[k] = left[i];
            i += 1;
        } else {
            out[k] = right[j];
            j += 1;
        }
        k += 1;
    }

    // Copy any remaining elements
    if i < n {
        out[k..(k + n - i)].copy_from_slice(&left[i..n]);
    }
    if j < m {
        out[k..(k + m - j)].copy_from_slice(&right[j..m]);
    }
}

/// TODO
#[derive(Default, Debug, Clone)]
pub struct LineFit {
    mx: f32,
    my: f32,
    mxx: f32,
    mxy: f32,
    myy: f32,
    w: f32,
}

/// TODO
pub fn compute_lfps<A: ImageAllocator>(
    src: &Image<Pixel, 1, A>,
    gradient_infos: &[GradientInfo],
) -> Vec<LineFit> {
    let src_slice = src.as_slice();
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
            let grad_x = src_slice[iy * src.width() + ix + 1] as i16
                - src_slice[iy * src.width() + ix - 1] as i16;

            let grad_y = src_slice[(iy + 1) * src.width() + ix] as i16
                - src_slice[(iy - 1) * src.width() + ix] as i16;

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

const MAX_LINE_FIT_MSE: f32 = 10.0; // TODO: Make this value tuneable in the quad_segment_maxima function

/// TODO
pub fn quad_segment_maxima(
    gradient_infos: &[GradientInfo],
    lfps: &[LineFit],
    indices: &mut [usize; 4],
) -> bool {
    let ksz = 20.min(gradient_infos.len() / 12);

    // can't fit a quad, too few points
    if ksz < 2 {
        return false;
    }

    let mut errors = vec![0.0f32; gradient_infos.len()];

    (0..lfps.len()).for_each(|i| {
        fit_line(
            lfps,
            (i + lfps.len() - ksz) % lfps.len(),
            (i + ksz) % lfps.len(),
            None,
            errors.get_mut(i),
            None,
        );
    });

    let mut y = vec![0.0f32; lfps.len()];
    let sigma = 1.0f32;
    let cutoff = 0.05f32;

    let mut fsz = ((-cutoff.ln() * 2.0 * sigma * sigma).sqrt() + 1.0) as usize;
    fsz = 2 * fsz + 1;

    let mut f = vec![0.0f32; fsz];

    (0..fsz).for_each(|i| {
        let j = i - fsz / 2;
        f[i] = (-(j as f32) * j as f32 / (2.0 * sigma * sigma)).exp();
    });

    (0..gradient_infos.len()).for_each(|iy| {
        let mut acc = 0.0f32;

        (0..fsz).for_each(|i| {
            acc += errors[(iy + i - fsz / 2 + gradient_infos.len()) % gradient_infos.len()];
        });

        y[iy] = acc;
    });

    errors = y;

    let mut maxima = vec![0usize; gradient_infos.len()];
    let mut maxima_errs = vec![0.0f32; gradient_infos.len()];
    let mut nmaxima = 0usize;

    (0..gradient_infos.len()).for_each(|i| {
        if errors[i] > errors[(i + 1) % gradient_infos.len()]
            && errors[i] > errors[(i + gradient_infos.len() - 1) % gradient_infos.len()]
        {
            maxima[nmaxima] = i;
            maxima_errs[nmaxima] = errors[i];
            nmaxima += 1;
        }
    });

    if nmaxima < 4 {
        return false;
    }

    let max_nmaxima = 10; // TODO: Make this value tuneable

    if nmaxima > max_nmaxima {
        let mut maxima_errs_copy = maxima_errs.clone();

        maxima_errs_copy.sort_by(|a, b| b.total_cmp(a));

        let maxima_thresh = maxima_errs_copy[max_nmaxima];
        let mut out = 0usize;
        (0..nmaxima).for_each(|i| {
            if maxima_errs[i] <= maxima_thresh {
                return;
            }
            out += 1;
            maxima[out] = maxima[i];
        });
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

    let max_dot = 0.984808; // TODO: Make this value tuneable

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

            if mse01 > MAX_LINE_FIT_MSE {
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

                if mse12 > MAX_LINE_FIT_MSE {
                    return;
                }

                let dot = params01[2] * params12[2] + params01[3] * params12[3];
                if dot.abs() > max_dot {
                    return;
                }

                ((m2 + 1)..nmaxima).for_each(|m3| {
                    let i3 = maxima[m3];

                    fit_line(lfps, i2, i3, None, Some(&mut err23), Some(&mut mse23));

                    if mse23 > MAX_LINE_FIT_MSE {
                        return;
                    }

                    fit_line(lfps, i3, i0, None, Some(&mut err30), Some(&mut mse30));

                    if mse30 > MAX_LINE_FIT_MSE {
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

    if (best_error / gradient_infos.len() as f32) < MAX_LINE_FIT_MSE {
        return true;
    }

    false
}

/// TODO
pub fn fit_line(
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
