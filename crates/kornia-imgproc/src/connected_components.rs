//! Connected-component labeling — label-exact with OpenCV's SAUF
//! algorithm (`cv2.connectedComponentsWithAlgorithm(..., cv2.CCL_WU)`).
//!
//! Foreground is any nonzero pixel; background gets label 0; components
//! are numbered 1..N **in raster order of each component's first pixel**
//! — which is exactly the numbering cv2's SAUF produces for both 4- and
//! 8-connectivity (verified empirically; cv2's DEFAULT 8-way algorithm,
//! BBDT, yields the same partition with a different, block-scan-dependent
//! numbering and is therefore not label-comparable).
//!
//! The CPU path is a raster-scan union-find with min-index roots; the
//! CUDA path (`cuda/ccl.rs`) is the Playne–Stephenson label-equivalence
//! fixpoint whose `atomicMin` iteration converges to the same min-index
//! labeling, followed by a device compaction that renumbers roots in
//! index order — so device labels are identical to the CPU's.

use kornia_image::{Image, ImageError};

/// Pixel connectivity for component labeling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Connectivity {
    /// Edge neighbors only (N, S, E, W).
    Four,
    /// Edge + corner neighbors.
    Eight,
}

#[inline]
fn find(parent: &mut [u32], mut x: u32) -> u32 {
    // Path halving.
    while parent[x as usize] != x {
        let p = parent[x as usize];
        parent[x as usize] = parent[p as usize];
        x = parent[x as usize];
    }
    x
}

#[inline]
fn union(parent: &mut [u32], a: u32, b: u32) {
    let (ra, rb) = (find(parent, a), find(parent, b));
    // Min-index root: keeps the canonical labeling deterministic.
    if ra < rb {
        parent[rb as usize] = ra;
    } else if rb < ra {
        parent[ra as usize] = rb;
    }
}

/// Label connected components of the nonzero pixels of `src` into
/// `labels` (0 = background, components numbered 1..N in raster order of
/// first appearance — cv2 SAUF numbering). Returns `N + 1` like cv2's
/// `connectedComponents` (the count includes the background label).
/// Device pairs run the CUDA label-equivalence kernels — label-identical
/// to the CPU path.
pub fn connected_components(
    src: &Image<u8, 1>,
    labels: &mut Image<i32, 1>,
    connectivity: Connectivity,
) -> Result<i32, ImageError> {
    if src.size() != labels.size() {
        return Err(ImageError::InvalidImageSize(
            src.cols(),
            src.rows(),
            labels.cols(),
            labels.rows(),
        ));
    }

    #[cfg(feature = "cuda")]
    if let crate::cuda::dispatch::Residency::Device(exec) =
        crate::cuda::dispatch::pair_residency(src, labels)?
    {
        let mut n = 0i32;
        exec.run(|stream| {
            n = cuda_adapters::ccl_cuda(src, labels, connectivity, stream)?;
            Ok(())
        })?;
        return Ok(n);
    }

    let (w, h) = (src.cols(), src.rows());
    let s = src.as_slice();
    let n_px = w * h;
    let _ = &n_px;

    // Pass 1: RUN-based union-find — horizontal runs collapse to their
    // start index for free (parent chain along the run), and only runs
    // overlapping a run in the previous row union. Far fewer union/find
    // operations than per-pixel scanning; identical min-index roots.
    let mut parent: Vec<u32> = (0..n_px as u32).collect();

    // Stripe-parallel: each stripe unions rows [y0+1, y1) against their
    // predecessor internally (disjoint parent index ranges — safe to
    // split via chunks of the parent array is not possible with a shared
    // Vec, so stripes serialize on a raw pointer with provably disjoint
    // touch sets), then the stripe boundary rows are stitched serially.
    let stripes = rayon::current_num_threads().max(1);
    let rows_per = h.div_ceil(stripes).max(1);
    struct P(*mut u32);
    unsafe impl Send for P {}
    unsafe impl Sync for P {}
    impl P {
        /// Accessor so closures capture the Sync wrapper, not the raw
        /// pointer field (edition-2021 disjoint capture).
        fn get(&self) -> *mut u32 {
            self.0
        }
    }
    let pp = P(parent.as_mut_ptr());
    let bounds: Vec<(usize, usize)> = (0..stripes)
        .map(|k| (k * rows_per, ((k + 1) * rows_per).min(h)))
        .filter(|&(a, b)| a < b)
        .collect();
    use rayon::prelude::*;
    bounds.par_iter().for_each(|&(y0, y1)| {
        // SAFETY: this stripe only touches parent entries of its own rows
        // [y0, y1) — run chaining and unions are between the current row
        // and the previous row, and the first row of a stripe skips the
        // cross-stripe union (done in the stitch pass below).
        let parent = unsafe { std::slice::from_raw_parts_mut(pp.get(), w * h) };
        let mut prev_runs: Vec<(usize, usize)> = Vec::new();
        let mut cur_runs: Vec<(usize, usize)> = Vec::new();
        for y in y0..y1 {
            cur_runs.clear();
            let row = &s[y * w..y * w + w];
            let mut pi = 0usize;
            let mut x = 0;
            while x < w {
                if row[x] == 0 {
                    x += 1;
                    continue;
                }
                let start = x;
                while x < w && row[x] != 0 {
                    x += 1;
                }
                let gs = y * w + start;
                parent[gs + 1..y * w + x].fill(gs as u32);
                cur_runs.push((start, x));
                if y > y0 {
                    let (lo, hi) = if connectivity == Connectivity::Eight {
                        (start.saturating_sub(1), (x + 1).min(w))
                    } else {
                        (start, x)
                    };
                    while pi < prev_runs.len() && prev_runs[pi].1 <= lo {
                        pi += 1;
                    }
                    let mut pj = pi;
                    while pj < prev_runs.len() && prev_runs[pj].0 < hi {
                        union(parent, gs as u32, ((y - 1) * w + prev_runs[pj].0) as u32);
                        pj += 1;
                    }
                    pi = pj.saturating_sub(1).max(pi);
                }
            }
            std::mem::swap(&mut prev_runs, &mut cur_runs);
        }
    });
    // Stitch stripe boundaries (first row of each stripe vs the row
    // above), serial.
    for &(y0, _) in bounds.iter().skip(1) {
        let y = y0;
        let row = &s[y * w..y * w + w];
        let prev = &s[(y - 1) * w..y * w];
        let mut x = 0;
        while x < w {
            if row[x] == 0 {
                x += 1;
                continue;
            }
            let start = x;
            while x < w && row[x] != 0 {
                x += 1;
            }
            let gs = y * w + start;
            let (lo, hi) = if connectivity == Connectivity::Eight {
                (start.saturating_sub(1), (x + 1).min(w))
            } else {
                (start, x)
            };
            let mut px = lo;
            while px < hi {
                if prev[px] != 0 {
                    let rstart = {
                        let mut r = px;
                        while r > 0 && prev[r - 1] != 0 {
                            r -= 1;
                        }
                        r
                    };
                    union(&mut parent, gs as u32, ((y - 1) * w + rstart) as u32);
                    while px < hi && prev[px] != 0 {
                        px += 1;
                    }
                } else {
                    px += 1;
                }
            }
        }
    }

    // Pass 2: compact labels in raster order of the root's first
    // appearance (root = component's min linear index, so this IS the
    // raster order of each component's first pixel). Resolved once per
    // RUN — every pixel of a run shares its root — then the output span
    // is filled.
    let out = labels.as_slice_mut();
    let mut next = 1i32;
    let mut compact: Vec<i32> = vec![0; n_px];
    for y in 0..h {
        let row = &s[y * w..y * w + w];
        let orow = &mut out[y * w..y * w + w];
        let mut x = 0;
        while x < w {
            if row[x] == 0 {
                orow[x] = 0;
                x += 1;
                continue;
            }
            let start = x;
            while x < w && row[x] != 0 {
                x += 1;
            }
            let r = find(&mut parent, (y * w + start) as u32) as usize;
            if compact[r] == 0 {
                compact[r] = next;
                next += 1;
            }
            orow[start..x].fill(compact[r]);
        }
    }
    Ok(next)
}

#[cfg(feature = "cuda")]
mod cuda_adapters {
    use super::*;
    use crate::cuda::ccl::launch_connected_components;
    use crate::cuda::dispatch::untyped_device_err;
    use cudarc::driver::CudaStream;
    use std::sync::Arc;

    fn err(e: impl std::fmt::Display) -> ImageError {
        ImageError::Cuda(e.to_string())
    }

    pub(super) fn ccl_cuda(
        src: &Image<u8, 1>,
        labels: &mut Image<i32, 1>,
        connectivity: Connectivity,
        stream: &Arc<CudaStream>,
    ) -> Result<i32, ImageError> {
        let ctx = stream.context();
        let s = src
            .0
            .as_cudaslice()
            .ok_or_else(|| untyped_device_err("source"))?;
        let d = labels
            .0
            .as_cudaslice_mut()
            .ok_or_else(|| untyped_device_err("destination"))?;
        launch_connected_components(
            ctx,
            stream,
            s,
            d,
            src.cols(),
            src.rows(),
            connectivity == Connectivity::Eight,
        )
        .map_err(err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_image::ImageSize;

    fn sz(w: usize, h: usize) -> ImageSize {
        ImageSize {
            width: w,
            height: h,
        }
    }

    fn run(data: Vec<u8>, w: usize, h: usize, conn: Connectivity) -> (i32, Vec<i32>) {
        let src = Image::<u8, 1>::new(sz(w, h), data).unwrap();
        let mut labels = Image::<i32, 1>::from_size_val(sz(w, h), 0).unwrap();
        let n = connected_components(&src, &mut labels, conn).unwrap();
        (n, labels.as_slice().to_vec())
    }

    #[test]
    fn empty_image_one_label() {
        let (n, lab) = run(vec![0; 12], 4, 3, Connectivity::Eight);
        assert_eq!(n, 1);
        assert!(lab.iter().all(|&l| l == 0));
    }

    #[test]
    fn two_blobs_diagonal_conn_difference() {
        // Two pixels touching diagonally: one component under 8-conn,
        // two under 4-conn.
        #[rustfmt::skip]
        let data = vec![
            255, 0,
            0, 255,
        ];
        let (n8, lab8) = run(data.clone(), 2, 2, Connectivity::Eight);
        assert_eq!(n8, 2);
        assert_eq!(lab8, vec![1, 0, 0, 1]);
        let (n4, lab4) = run(data, 2, 2, Connectivity::Four);
        assert_eq!(n4, 3);
        assert_eq!(lab4, vec![1, 0, 0, 2]);
    }

    #[test]
    fn raster_first_numbering() {
        // Component starting later in raster order gets the higher label,
        // regardless of size.
        #[rustfmt::skip]
        let data = vec![
            0, 255, 0, 0,
            0, 0, 0, 255,
            0, 0, 0, 255,
        ];
        let (n, lab) = run(data, 4, 3, Connectivity::Eight);
        assert_eq!(n, 3);
        assert_eq!(lab[1], 1);
        assert_eq!(lab[7], 2);
        assert_eq!(lab[11], 2);
    }

    #[test]
    fn u_shape_merges_to_one() {
        // U-shape: left and right arms merge through the bottom — the
        // union-find must relabel the provisional right-arm component.
        #[rustfmt::skip]
        let data = vec![
            255, 0, 255,
            255, 0, 255,
            255, 255, 255,
        ];
        let (n, lab) = run(data, 3, 3, Connectivity::Four);
        assert_eq!(n, 2);
        let fg: Vec<i32> = lab.iter().copied().filter(|&l| l != 0).collect();
        assert!(fg.iter().all(|&l| l == 1));
    }

    #[test]
    fn size_mismatch_rejected() {
        let src = Image::<u8, 1>::from_size_val(sz(8, 8), 0).unwrap();
        let mut labels = Image::<i32, 1>::from_size_val(sz(4, 8), 0).unwrap();
        assert!(connected_components(&src, &mut labels, Connectivity::Eight).is_err());
    }
}

#[cfg(all(test, feature = "cuda"))]
mod cuda_tests {
    use super::*;
    use crate::cuda::color::test_utils::{default_stream, pattern_u8};
    use kornia_image::ImageSize;

    fn sz(w: usize, h: usize) -> ImageSize {
        ImageSize {
            width: w,
            height: h,
        }
    }

    /// Device labels must be IDENTICAL to the CPU's (same numbers, not
    /// just the same partition): random binary content at several
    /// densities, both connectivities, odd sizes.
    #[test]
    fn ccl_device_equals_host_label_exact() {
        let stream = default_stream();
        for (w, h) in [(64usize, 48usize), (67, 43), (128, 128), (5, 4)] {
            for thresh in [64u8, 128, 200] {
                for conn in [Connectivity::Four, Connectivity::Eight] {
                    let data: Vec<u8> = pattern_u8(w * h)
                        .into_iter()
                        .map(|v| if v > thresh { 255 } else { 0 })
                        .collect();
                    let src = Image::<u8, 1>::new(sz(w, h), data).unwrap();
                    let mut cpu = Image::<i32, 1>::from_size_val(sz(w, h), 0).unwrap();
                    let n_cpu = connected_components(&src, &mut cpu, conn).unwrap();

                    let d_src = src.to_cuda(&stream).unwrap();
                    let mut d_lab = Image::<i32, 1>::zeros_cuda(sz(w, h), &stream).unwrap();
                    let n_gpu = connected_components(&d_src, &mut d_lab, conn).unwrap();
                    let back = d_lab.to_host_owned().unwrap();
                    assert_eq!(n_cpu, n_gpu, "{w}x{h} t={thresh} {conn:?}");
                    assert_eq!(
                        back.as_slice(),
                        cpu.as_slice(),
                        "{w}x{h} t={thresh} {conn:?}"
                    );
                }
            }
        }
    }
}
