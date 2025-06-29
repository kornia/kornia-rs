use crate::{family::TagFamily, segmentation::GradientInfo, utils::Pixel};
use kornia_image::{allocator::ImageAllocator, Image};
use std::{collections::HashMap, f32};

/// TODO
#[derive(Debug, Default, Clone)]
pub struct Quad {
    /// TODO
    pub point: [[f32; 2]; 4],
    /// TODO
    pub reversed_border: bool,
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

    clusters.iter_mut().for_each(|(_, cluster)| {
        if cluster.len() < min_cluster_pixels {
            return;
        }

        if cluster.len() > 2 * (2 * src.width() + 2 * src.height()) {
            return;
        }

        if let Some(quad) = fit_quad(
            src,
            cluster,
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
    if cluster.len() < 24 {
        return None;
    }

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

    let quadrants = [[-(2 << 15), 0], [2 * (2 << 15), 2 << 15]];

    cluster
        .iter_mut()
        .for_each(|GradientInfo { pos, gx, gy, slope }| {
            let mut dx = pos.x as f32 - cx;
            let mut dy = pos.y as f32 - cy;

            // Convert gradient direction enum to numeric values
            let gx_val = match gx {
                crate::segmentation::GradientDirection::TowardsWhite => 255,
                crate::segmentation::GradientDirection::TowardsBlack => -255,
                crate::segmentation::GradientDirection::None => 0,
            };
            let gy_val = match gy {
                crate::segmentation::GradientDirection::TowardsWhite => 255,
                crate::segmentation::GradientDirection::TowardsBlack => -255,
                crate::segmentation::GradientDirection::None => 0,
            };

            dot += dx * gx_val as f32 + dy * gy_val as f32;

            let quadrant = quadrants[(dy > 0.0) as usize][(dx > 0.0) as usize];
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

    if !reversed_border && quad.reversed_border {
        return None;
    }
    if !normal_border && !quad.reversed_border {
        return None;
    }

    cluster.sort_by(|a, b| a.slope.total_cmp(&b.slope));

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
        }
    });

    if should_return_none {
        return None;
    }

    should_return_none = false;

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

    // Calculate area of triangle formed by points 0, 1, 2
    (0..3).for_each(|i| {
        let idxa = i;
        let idxb = (i + 1) % 3;
        length[i] = ((quad.point[idxb][0] - quad.point[idxa][0]).powi(2)
            + (quad.point[idxb][1] - quad.point[idxa][1]).powi(2))
        .sqrt();
    });

    p = (length[0] + length[1] + length[2]) / 2.0;
    area += (p * (p - length[0]) * (p - length[1]) * (p - length[2])).sqrt();

    // Calculate area of triangle formed by points 2, 3, 0
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

        if !(-COS_CRITICAL_RAD..=COS_CRITICAL_RAD).contains(&cos_dtheta) || dx1 * dy2 < dy1 * dx2 {
            should_return_none = true;
        }
    });

    if should_return_none {
        return None;
    }

    Some(quad)
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

    (0..gradient_infos.len()).for_each(|i| {
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

    let fsz = (2.0 * ((-(cutoff.ln()) * 2.0 * sigma * sigma).sqrt() + 1.0) + 1.0) as usize;

    let mut f = vec![0.0f32; fsz];

    (0..fsz).for_each(|i| {
        let j = i as isize - fsz as isize / 2;
        f[i] = (-(j as f32) * j as f32 / (2.0 * sigma * sigma)).exp();
    });

    (0..gradient_infos.len()).for_each(|iy| {
        let mut acc = 0.0f32;

        (0..fsz).for_each(|i| {
            acc += errors[((iy as isize + i as isize - fsz as isize / 2
                + gradient_infos.len() as isize)
                % gradient_infos.len() as isize) as usize]
                * f[i];
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

#[cfg(test)]
mod tests {
    use kornia_image::allocator::CpuAllocator;
    use kornia_io::png::read_image_png_mono8;

    use crate::{
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

        let quads = fit_quads(&bin, &TagFamily::TAG36_H11, &mut clusters, 5);

        let expected_quad = [[[27, 3], [27, 27], [3, 27], [3, 3]]];

        assert_eq!(quads.len(), expected_quad.len());

        for (Quad { point, .. }, expected_quad) in quads.iter().zip(expected_quad) {
            // We allow a ±1 error here to avoid CI failures due to small precision differences.
            // The expected outputs were generated with C code using 64-bit precision, while our code uses f32.
            assert!((point[0][0] as usize).abs_diff(expected_quad[0][0]) <= 1);
            assert!((point[0][1] as usize).abs_diff(expected_quad[0][1]) <= 1);

            assert!((point[2][0] as usize).abs_diff(expected_quad[2][0]) <= 1);
            assert!((point[2][0] as usize).abs_diff(expected_quad[2][0]) <= 1);

            assert!((point[2][1] as usize).abs_diff(expected_quad[2][1]) <= 1);
            assert!((point[2][1] as usize).abs_diff(expected_quad[2][1]) <= 1);

            assert!((point[3][1] as usize).abs_diff(expected_quad[3][1]) <= 1);
            assert!((point[3][1] as usize).abs_diff(expected_quad[3][1]) <= 1);
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
        
        // Should return false because ksz < 2 (20/12 = 1.66... < 2)
        assert!(!quad_segment_maxima(&gradient_infos, &lfps, &mut indices));

        // Test 2: Empty input
        let empty_gradient_infos = Vec::<GradientInfo>::new();
        let empty_lfps = Vec::<LineFit>::new();
        assert!(!quad_segment_maxima(
            &empty_gradient_infos,
            &empty_lfps,
            &mut indices
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
            &mut indices
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
            let lfps = compute_lfps(&bin, largest_cluster);
            let mut indices = [0; 4];
            
            let result = quad_segment_maxima(largest_cluster, &lfps, &mut indices);
            
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
        
        fit_line(&lfps, 0, 3, Some(&mut lineparm), Some(&mut err), Some(&mut mse));
        
        // Check that parameters were computed
        assert!(lineparm[0] != 0.0 || lineparm[1] != 0.0); // ex, ey should be computed
        assert!(err >= 0.0); // Error can be zero for perfect fit
        assert!(mse >= 0.0); // MSE can be zero for perfect fit
        
        // Test 2: Edge case - same indices
        let mut lineparm2 = [0.0f32; 4];
        let mut err2 = 0.0f32;
        let mut mse2 = 0.0f32;
        
        fit_line(&lfps, 1, 1, Some(&mut lineparm2), Some(&mut err2), Some(&mut mse2));
        
        // Should not modify parameters when i0 == i1
        assert_eq!(lineparm2[0], 0.0);
        assert_eq!(lineparm2[1], 0.0);
        assert_eq!(err2, 0.0);
        assert_eq!(mse2, 0.0);
        
        // Test 3: Edge case - out of bounds indices
        let mut lineparm3 = [0.0f32; 4];
        let mut err3 = 0.0f32;
        let mut mse3 = 0.0f32;
        
        fit_line(&lfps, 0, 10, Some(&mut lineparm3), Some(&mut err3), Some(&mut mse3));
        
        // Should not modify parameters when indices are out of bounds
        assert_eq!(lineparm3[0], 0.0);
        assert_eq!(lineparm3[1], 0.0);
        assert_eq!(err3, 0.0);
        assert_eq!(mse3, 0.0);
        
        // Test 4: Edge case - i0 > i1 (wrapped case)
        let mut lineparm4 = [0.0f32; 4];
        let mut err4 = 0.0f32;
        let mut mse4 = 0.0f32;
        
        fit_line(&lfps, 3, 1, Some(&mut lineparm4), Some(&mut err4), Some(&mut mse4));
        
        // Should compute parameters for wrapped case
        assert!(lineparm4[0] != 0.0 || lineparm4[1] != 0.0);
        assert!(err4 >= 0.0);
        assert!(mse4 >= 0.0);
        
        // Test 5: No output parameters (should not crash)
        fit_line(&lfps, 0, 3, None, None, None);
    }

    #[test]
    fn test_compute_lfps() {
        use kornia_image::{Image, ImageSize};
        use crate::utils::Pixel;
        use crate::segmentation::GradientDirection;

        // Create a 4x4 image with all white pixels
        let size = ImageSize { width: 4, height: 4 };
        let img = Image::<Pixel, 1, _>::from_size_val(size, Pixel::White, CpuAllocator).unwrap();

        // Test 1: Single point
        let gradient_infos = vec![GradientInfo {
            pos: Point2d { x: 1, y: 2 },
            gx: GradientDirection::TowardsWhite,
            gy: GradientDirection::TowardsBlack,
            slope: 0.0,
        }];
        let lfps = compute_lfps(&img, &gradient_infos);
        assert_eq!(lfps.len(), 1);
        // The weighted mean should be close to the point's position (scaled by 0.5 + delta)
        let expected_x = 1.0 * 0.5 + 0.5;
        let expected_y = 2.0 * 0.5 + 0.5;
        assert!((lfps[0].mx / lfps[0].w - expected_x).abs() < 1e-4);
        assert!((lfps[0].my / lfps[0].w - expected_y).abs() < 1e-4);

        // Test 2: Multiple points in a line
        let gradient_infos = vec![
            GradientInfo { pos: Point2d { x: 0, y: 0 }, gx: GradientDirection::TowardsWhite, gy: GradientDirection::TowardsBlack, slope: 0.0 },
            GradientInfo { pos: Point2d { x: 1, y: 1 }, gx: GradientDirection::TowardsWhite, gy: GradientDirection::TowardsBlack, slope: 0.0 },
            GradientInfo { pos: Point2d { x: 2, y: 2 }, gx: GradientDirection::TowardsWhite, gy: GradientDirection::TowardsBlack, slope: 0.0 },
        ];
        let lfps = compute_lfps(&img, &gradient_infos);
        assert_eq!(lfps.len(), 3);
        // The last element should be the sum of all previous
        let last = &lfps[2];
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_w = 0.0;
        for i in 0..3 {
            let x = gradient_infos[i].pos.x as f32 * 0.5 + 0.5;
            let y = gradient_infos[i].pos.y as f32 * 0.5 + 0.5;
            sum_x += x;
            sum_y += y;
            sum_w += 1.0;
        }
        assert!((last.mx - sum_x).abs() < 1e-4);
        assert!((last.my - sum_y).abs() < 1e-4);
        assert!((last.w - sum_w).abs() < 1e-4);

        // Test 3: Empty input
        let gradient_infos = vec![];
        let lfps = compute_lfps(&img, &gradient_infos);
        assert_eq!(lfps.len(), 0);
    }
}
