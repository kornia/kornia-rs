//! Layered FAST detector API: rectangle → grid of cells → multi-level pyramid.
//!
//! The three entry points here (`fast_detect_rect_u8`, `fast_detect_cells_u8`,
//! `fast_detect_pyramid_u8`) let a caller pick the layer that matches their
//! existing scaffolding:
//!
//! * **Rect** — zero-copy FAST over an arbitrary sub-rectangle. Basalt-style
//!   consumers that already own a grid walker plug in here.
//! * **Cells** — adaptive-threshold cell walker + NMS budget per cell. What
//!   both ORB-SLAM3 and Basalt build on top of the raw detector.
//! * **Pyramid** — run `cells` across a caller-supplied pyramid and rescale
//!   back to full-resolution coordinates. Pyramid geometry (scale factor,
//!   number of levels, u8 vs u16) is the caller's problem — we only care
//!   about `&[&Image<u8>]`.
//!
//! Returns are all named structs ([`FastCorner`], [`CellKeypoint`],
//! [`PyramidKeypoint`]) so adding fields later is non-breaking.
use kornia_image::{allocator::ImageAllocator, Image};
use rayon::prelude::*;

use super::fast::fast_detect_rows_u8_serial;

/// Axis-aligned rectangle in image pixel coordinates.
///
/// Used to specify the region-of-interest for [`fast_detect_rect_u8`]. All
/// coordinates are in the coordinate system of the input image (no downscale).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rect {
    /// Column of the top-left corner (inclusive).
    pub x: usize,
    /// Row of the top-left corner (inclusive).
    pub y: usize,
    /// Width in pixels.
    pub w: usize,
    /// Height in pixels.
    pub h: usize,
}

impl Rect {
    /// Right edge (exclusive).
    #[inline]
    pub fn x_end(&self) -> usize {
        self.x + self.w
    }

    /// Bottom edge (exclusive).
    #[inline]
    pub fn y_end(&self) -> usize {
        self.y + self.h
    }
}

/// A single FAST corner — position plus response score.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FastCorner {
    /// Position as `[col, row]` in image pixels.
    pub xy: [f32; 2],
    /// FAST response score (higher = stronger corner). Normalized to roughly
    /// `[0, 16)` — the sum of absolute intensity differences around the
    /// Bresenham ring divided by 255.
    pub response: f32,
}

/// A FAST corner annotated with the cell it was detected in.
///
/// Returned by [`fast_detect_cells_u8`]. The `cell_id` is a row-major index
/// into the grid of cells the image was partitioned into; downstream code can
/// use it to build occupancy masks, do per-cell top-k selection, etc.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CellKeypoint {
    /// Position as `[col, row]` in image pixels.
    pub xy: [f32; 2],
    /// FAST response score.
    pub response: f32,
    /// Row-major cell index the corner falls into.
    pub cell_id: u32,
}

/// A FAST corner detected in a pyramid level, with coordinates rescaled to
/// the full-resolution (level-0) image.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PyramidKeypoint {
    /// Full-resolution position as `[col, row]`.
    pub xy: [f32; 2],
    /// FAST response score at the detection level (not rescaled).
    pub response: f32,
    /// Pyramid level index (0 = full resolution).
    pub level: u8,
    /// Row-major cell index within the detection level.
    pub cell_id: u32,
}

/// Configuration for the cell-based adaptive-threshold FAST detector.
///
/// Mirrors the cell loop both ORB-SLAM3 and Basalt run around `cv::FAST`.
/// Setting `target_per_cell = 1` and a long threshold cascade reproduces
/// Basalt's `detectKeypoints`; setting `target_per_cell = usize::MAX` and a
/// two-step cascade reproduces ORB-SLAM3's ini/min-threshold pattern.
#[derive(Debug, Clone)]
pub struct CellDetectConfig {
    /// Side length of each square cell in pixels.
    pub cell_size: usize,
    /// Ordered thresholds (u8-scale, expressed as `f32` in `[0, 255]`) to
    /// try per cell. For each cell, the detector tries the first threshold;
    /// if fewer than `target_per_cell` corners emerge it falls through to
    /// the next, and so on. Must be non-empty.
    pub threshold_cascade: Vec<f32>,
    /// Maximum corners kept per cell. The top-`target_per_cell` by response
    /// are retained. `usize::MAX` disables per-cell clamping (keep all).
    pub target_per_cell: usize,
    /// Minimum arc length for the FAST segment test (9 or 12).
    pub arc_length: usize,
    /// Pixel border skipped at image edges (≥3, the Bresenham radius).
    pub border: usize,
}

impl Default for CellDetectConfig {
    fn default() -> Self {
        // Defaults target ORB-SLAM3's per-cell pattern: two-tier threshold,
        // keep all corners in a cell (downstream octree NMS picks the top-N).
        Self {
            cell_size: 35,
            threshold_cascade: vec![20.0, 7.0],
            target_per_cell: usize::MAX,
            arc_length: 9,
            border: 3,
        }
    }
}

/// Run FAST over a rectangular sub-region of an image without copying.
///
/// The `rect` is clipped to the image bounds (minus `border`); out-of-range
/// rects yield an empty result. Thresholding matches [`super::FastDetector`]
/// semantics: `threshold` is in the u8 intensity scale `[0, 255]`, converted
/// internally to normalized form.
///
/// This is **Layer 1** of the layered API: it produces a flat list of
/// corners inside the rect, with no cell bookkeeping or NMS. For the usual
/// ORB-SLAM / Basalt cell loop, prefer [`fast_detect_cells_u8`].
pub fn fast_detect_rect_u8<A: ImageAllocator>(
    image: &Image<u8, 1, A>,
    rect: Rect,
    threshold: f32,
    arc_length: usize,
    border: usize,
) -> Vec<FastCorner> {
    let margin = border.max(3);
    let w = image.width();
    let h = image.height();

    let x0 = rect.x.max(margin);
    let y0 = rect.y.max(margin);
    let x1 = rect.x_end().min(w.saturating_sub(margin));
    let y1 = rect.y_end().min(h.saturating_sub(margin));
    if x1 <= x0 || y1 <= y0 {
        return Vec::new();
    }

    // Delegate the row-level NEON kernel. The existing function emits corners
    // over full rows; we filter by column range on the way out. Column
    // filtering is ~1 compare per emitted candidate — cheaper than duplicating
    // the ~600-line row kernel for a column bound.
    let t_norm = (threshold / 255.0).clamp(0.0, 1.0);
    let raw = fast_detect_rows_u8_serial(image, t_norm, arc_length, border, y0..y1);

    let mut out = Vec::with_capacity(raw.len());
    for ([y, x], r) in raw {
        if x >= x0 && x < x1 {
            out.push(FastCorner {
                xy: [x as f32, y as f32],
                response: r,
            });
        }
    }
    out
}

/// Grid dimensions — number of cells along each axis.
#[inline]
fn grid_dims(image_w: usize, image_h: usize, cell_size: usize) -> (usize, usize) {
    let nx = image_w.div_ceil(cell_size);
    let ny = image_h.div_ceil(cell_size);
    (nx, ny)
}

/// Adaptive-threshold cell-based FAST detector.
///
/// Partitions the image into a grid of `cfg.cell_size` × `cfg.cell_size`
/// cells (the final row/column may be smaller). For each cell not flagged
/// in `occupancy`, runs [`fast_detect_rect_u8`] with the first threshold in
/// `cfg.threshold_cascade`; if fewer than `cfg.target_per_cell` corners
/// emerge, tries the next threshold in the cascade, and so on. The top-k by
/// response within each cell are retained.
///
/// `occupancy` (if `Some`) is a row-major boolean slice of length
/// `n_cells_x * n_cells_y`; `true` means "skip this cell". Used by
/// Basalt-style trackers to suppress cells that already contain tracked
/// points without rebuilding the image.
///
/// Cells are processed in parallel via rayon.
pub fn fast_detect_cells_u8<A: ImageAllocator + Sync>(
    image: &Image<u8, 1, A>,
    cfg: &CellDetectConfig,
    occupancy: Option<&[bool]>,
) -> Vec<CellKeypoint> {
    assert!(
        !cfg.threshold_cascade.is_empty(),
        "threshold_cascade must be non-empty"
    );
    assert!(cfg.cell_size > 0, "cell_size must be positive");

    let w = image.width();
    let h = image.height();
    let (nx, ny) = grid_dims(w, h, cfg.cell_size);
    let total_cells = nx * ny;
    if let Some(occ) = occupancy {
        assert_eq!(
            occ.len(),
            total_cells,
            "occupancy length must match grid ({} × {} = {})",
            nx,
            ny,
            total_cells
        );
    }

    (0..total_cells)
        .into_par_iter()
        .flat_map_iter(|cell_id| {
            if occupancy.is_some_and(|occ| occ[cell_id]) {
                return Vec::new().into_iter();
            }
            let cx = cell_id % nx;
            let cy = cell_id / nx;
            let rect = Rect {
                x: cx * cfg.cell_size,
                y: cy * cfg.cell_size,
                w: cfg.cell_size,
                h: cfg.cell_size,
            };

            let mut corners: Vec<FastCorner> = Vec::new();
            for &thr in &cfg.threshold_cascade {
                corners = fast_detect_rect_u8(image, rect, thr, cfg.arc_length, cfg.border);
                if corners.len() >= cfg.target_per_cell {
                    break;
                }
            }

            if cfg.target_per_cell < corners.len() {
                // Partial-sort: keep the top-k by response.
                corners.sort_by(|a, b| {
                    b.response
                        .partial_cmp(&a.response)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                corners.truncate(cfg.target_per_cell);
            }

            let cell_id_u32 = cell_id as u32;
            corners
                .into_iter()
                .map(move |c| CellKeypoint {
                    xy: c.xy,
                    response: c.response,
                    cell_id: cell_id_u32,
                })
                .collect::<Vec<_>>()
                .into_iter()
        })
        .collect()
}

/// Run [`fast_detect_cells_u8`] across a caller-supplied pyramid and rescale
/// all keypoint coordinates back to the full-resolution (level-0) frame.
///
/// `levels[0]` is treated as full resolution; higher indices are coarser.
/// The same `cfg` is used at every level — if different thresholds per
/// level are required, call this function per level and concatenate.
///
/// Rescale uses the exact width/height ratio `levels[0].size / levels[i].size`
/// rather than assuming a fixed factor (1.2×, 2×, …), so ORB-SLAM's 1.2×
/// chain and Basalt's 2× chain both work without special-casing.
pub fn fast_detect_pyramid_u8<A: ImageAllocator + Sync>(
    levels: &[&Image<u8, 1, A>],
    cfg: &CellDetectConfig,
) -> Vec<PyramidKeypoint> {
    if levels.is_empty() {
        return Vec::new();
    }
    let full_w = levels[0].width() as f32;
    let full_h = levels[0].height() as f32;

    let mut out = Vec::new();
    for (lvl, img) in levels.iter().enumerate() {
        let sx = full_w / img.width() as f32;
        let sy = full_h / img.height() as f32;
        let level_u8 = lvl as u8;
        let kps = fast_detect_cells_u8(img, cfg, None);
        out.reserve(kps.len());
        for k in kps {
            out.push(PyramidKeypoint {
                xy: [k.xy[0] * sx, k.xy[1] * sy],
                response: k.response,
                level: level_u8,
                cell_id: k.cell_id,
            });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_image::ImageSize;
    use kornia_tensor::CpuAllocator;

    /// Black background with 5×5 bright squares placed on a regular grid.
    /// Each square has 4 strong FAST corners (center pixel bright, most of the
    /// Bresenham ring outside the square → long dark run), so the detector
    /// fires reliably at every grid position — unlike an axis-aligned
    /// checkerboard, whose T-junctions only produce runs of ~5 same-sign
    /// pixels, below the FAST-9 threshold.
    fn dot_image(w: usize, h: usize, spacing: usize) -> Image<u8, 1, CpuAllocator> {
        let size = ImageSize {
            width: w,
            height: h,
        };
        let mut buf = vec![0u8; w * h];
        let mut cy = spacing;
        while cy + 5 < h {
            let mut cx = spacing;
            while cx + 5 < w {
                for dy in 0..5 {
                    for dx in 0..5 {
                        buf[(cy + dy) * w + (cx + dx)] = 255;
                    }
                }
                cx += spacing;
            }
            cy += spacing;
        }
        Image::from_size_slice(size, &buf, CpuAllocator).unwrap()
    }

    #[test]
    fn rect_clips_to_bounds() {
        let img = dot_image(64, 64, 16);
        // A rect overhanging the image should be clipped, not crash.
        let rect = Rect {
            x: 50,
            y: 50,
            w: 100,
            h: 100,
        };
        let _ = fast_detect_rect_u8(&img, rect, 20.0, 9, 3);
        // Degenerate rect (outside image) → empty.
        let out = fast_detect_rect_u8(
            &img,
            Rect {
                x: 1000,
                y: 1000,
                w: 10,
                h: 10,
            },
            20.0,
            9,
            3,
        );
        assert!(out.is_empty());
    }

    #[test]
    fn cells_detects_in_every_non_border_cell() {
        // Dot image with 16-px spacing places a 5×5 bright square in every
        // 32-px cell of a 128×128 image → every non-edge cell fires.
        let img = dot_image(128, 128, 16);
        let cfg = CellDetectConfig {
            cell_size: 32,
            threshold_cascade: vec![20.0],
            target_per_cell: 1,
            arc_length: 9,
            border: 3,
        };
        let kps = fast_detect_cells_u8(&img, &cfg, None);
        // 128/32 = 4 cells per side → 16 cells. Border cells may miss; we
        // just require the middle 4 cells to fire.
        assert!(
            kps.len() >= 4,
            "expected ≥4 cell keypoints, got {}",
            kps.len()
        );
        // All cell_ids are in range.
        let (nx, ny) = grid_dims(128, 128, 32);
        for k in &kps {
            assert!((k.cell_id as usize) < nx * ny);
        }
    }

    #[test]
    fn cells_respects_occupancy_mask() {
        let img = dot_image(128, 128, 16);
        let (nx, ny) = grid_dims(128, 128, 32);
        // Mark every cell as occupied → zero keypoints out.
        let occ = vec![true; nx * ny];
        let cfg = CellDetectConfig {
            cell_size: 32,
            threshold_cascade: vec![20.0],
            target_per_cell: 1,
            arc_length: 9,
            border: 3,
        };
        let kps = fast_detect_cells_u8(&img, &cfg, Some(&occ));
        assert!(kps.is_empty(), "occupancy-full mask should suppress all");
    }

    #[test]
    fn pyramid_rescales_to_full_res() {
        // A 2-level pyramid with 2× downscale. Corners detected at level 1
        // should have full-res coordinates approximately 2× their level-1
        // coordinates.
        let l0 = dot_image(128, 128, 16);
        let l1 = dot_image(64, 64, 16);
        let levels = [&l0, &l1];
        let cfg = CellDetectConfig {
            cell_size: 32,
            threshold_cascade: vec![20.0],
            target_per_cell: 1,
            arc_length: 9,
            border: 3,
        };
        let kps = fast_detect_pyramid_u8(&levels, &cfg);
        let max_lvl = kps.iter().map(|k| k.level).max().unwrap_or(0);
        assert!(max_lvl >= 1, "expected keypoints on both pyramid levels");
        // A level-1 keypoint's full-res x must fit in [0, 128].
        for k in &kps {
            assert!(k.xy[0] >= 0.0 && k.xy[0] <= 128.0);
            assert!(k.xy[1] >= 0.0 && k.xy[1] <= 128.0);
        }
    }
}
