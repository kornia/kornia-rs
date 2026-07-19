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

/// First index >= `x` with a nonzero byte, or `row.len()`. Skips 8 bytes
/// per step through zero regions via u64 loads (little-endian byte order
/// makes `trailing_zeros / 8` the first nonzero lane).
#[inline]
fn next_nonzero(row: &[u8], mut x: usize) -> usize {
    let w = row.len();
    while x + 8 <= w {
        let v = u64::from_le_bytes(row[x..x + 8].try_into().unwrap());
        if v != 0 {
            return x + (v.trailing_zeros() / 8) as usize;
        }
        x += 8;
    }
    while x < w && row[x] == 0 {
        x += 1;
    }
    x
}

/// First index >= `x` with a zero byte, or `row.len()`. The
/// `(v - 0x0101..) & !v & 0x8080..` trick flags lanes that are zero.
#[inline]
fn next_zero(row: &[u8], mut x: usize) -> usize {
    let w = row.len();
    while x + 8 <= w {
        let v = u64::from_le_bytes(row[x..x + 8].try_into().unwrap());
        let zeros = v.wrapping_sub(0x0101_0101_0101_0101) & !v & 0x8080_8080_8080_8080;
        if zeros != 0 {
            return x + (zeros.trailing_zeros() / 8) as usize;
        }
        x += 8;
    }
    while x < w && row[x] != 0 {
        x += 1;
    }
    x
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
    use rayon::prelude::*;

    // The union-find is indexed by RUN id, not pixel index: runs are
    // 50-500x fewer than pixels, so the parent/compact tables stay
    // cache-resident instead of costing two full-image allocations and
    // initializations per call (the previous per-pixel form spent most
    // of its time zeroing and walking 8MB tables at 1080p). Run ids are
    // assigned in raster order, so the min-id root of a component is its
    // first raster run and compact numbering stays cv2-SAUF-exact.

    // Pass 1 (stripe-parallel): each stripe scans its rows, records runs
    // as (start_x, end_x), and unions overlapping runs of consecutive
    // rows in a stripe-LOCAL union-find. Stripe boundaries are stitched
    // serially afterwards on the global table.
    struct StripeRuns {
        y0: usize,
        /// (start_x, end_x) per run, raster order within the stripe.
        runs: Vec<(u32, u32)>,
        /// Local run-count prefix per row: runs of row y0+k are
        /// row_ptr[k]..row_ptr[k+1].
        row_ptr: Vec<u32>,
        /// Stripe-local union-find over local run ids.
        parent: Vec<u32>,
    }
    let stripes = rayon::current_num_threads().max(1);
    let rows_per = h.div_ceil(stripes).max(1);
    let bounds: Vec<(usize, usize)> = (0..stripes)
        .map(|k| (k * rows_per, ((k + 1) * rows_per).min(h)))
        .filter(|&(a, b)| a < b)
        .collect();
    let mut stripe_runs: Vec<StripeRuns> = bounds
        .par_iter()
        .map(|&(y0, y1)| {
            let mut runs: Vec<(u32, u32)> = Vec::new();
            let mut row_ptr: Vec<u32> = Vec::with_capacity(y1 - y0 + 1);
            let mut parent: Vec<u32> = Vec::new();
            row_ptr.push(0);
            let mut prev_row = 0u32..0u32; // local id range of previous row
            for y in y0..y1 {
                let row = &s[y * w..y * w + w];
                let cur_first = runs.len() as u32;
                let mut pi = prev_row.start;
                let mut x = next_nonzero(row, 0);
                while x < w {
                    let start = x;
                    x = next_zero(row, x + 1);
                    let id = runs.len() as u32;
                    runs.push((start as u32, x as u32));
                    parent.push(id);
                    if y > y0 {
                        let (lo, hi) = if connectivity == Connectivity::Eight {
                            (start.saturating_sub(1) as u32, (x + 1).min(w) as u32)
                        } else {
                            (start as u32, x as u32)
                        };
                        while pi < prev_row.end && runs[pi as usize].1 <= lo {
                            pi += 1;
                        }
                        let mut pj = pi;
                        while pj < prev_row.end && runs[pj as usize].0 < hi {
                            union(&mut parent, id, pj);
                            pj += 1;
                        }
                        // The last overlapping prev run may also overlap
                        // the next current run — don't advance past it.
                        pi = pj.saturating_sub(1).max(pi);
                    }
                    x = next_nonzero(row, x + 1);
                }
                prev_row = cur_first..runs.len() as u32;
                row_ptr.push(runs.len() as u32);
            }
            StripeRuns {
                y0,
                runs,
                row_ptr,
                parent,
            }
        })
        .collect();

    // Merge stripe-local forests into one global table (local ids get a
    // per-stripe offset) and stitch each stripe's first row against the
    // row above it.
    let offsets: Vec<u32> = {
        let mut acc = 0u32;
        let mut v = Vec::with_capacity(stripe_runs.len());
        for sr in &stripe_runs {
            v.push(acc);
            acc += sr.runs.len() as u32;
        }
        v
    };
    let n_runs: usize = stripe_runs.iter().map(|sr| sr.runs.len()).sum();
    let mut parent: Vec<u32> = Vec::with_capacity(n_runs);
    for (sr, &off) in stripe_runs.iter_mut().zip(&offsets) {
        parent.extend(sr.parent.iter().map(|&p| p + off));
        sr.parent = Vec::new();
    }
    for k in 1..stripe_runs.len() {
        let (below, above) = {
            let (a, b) = stripe_runs.split_at(k);
            (&a[k - 1], &b[0])
        };
        // Last row of the stripe below vs first row of the stripe above.
        let prev_lo = below.row_ptr[below.row_ptr.len() - 2];
        let prev_hi = below.row_ptr[below.row_ptr.len() - 1];
        let cur_hi = above.row_ptr[1];
        let (prev_off, cur_off) = (offsets[k - 1], offsets[k]);
        let mut pi = prev_lo;
        for cid in 0..cur_hi {
            let (start, end) = above.runs[cid as usize];
            let (lo, hi) = if connectivity == Connectivity::Eight {
                (start.saturating_sub(1), (end + 1).min(w as u32))
            } else {
                (start, end)
            };
            while pi < prev_hi && below.runs[pi as usize].1 <= lo {
                pi += 1;
            }
            let mut pj = pi;
            while pj < prev_hi && below.runs[pj as usize].0 < hi {
                union(&mut parent, cur_off + cid, prev_off + pj);
                pj += 1;
            }
            pi = pj.saturating_sub(1).max(pi);
        }
    }

    // Pass 2a (serial, over runs only): resolve every run's root and
    // assign compact labels in run order — run order IS raster order of
    // each component's first pixel.
    let mut run_label: Vec<i32> = vec![0; n_runs];
    let mut compact: Vec<i32> = vec![0; n_runs];
    let mut next = 1i32;
    for rid in 0..n_runs as u32 {
        let r = find(&mut parent, rid) as usize;
        if compact[r] == 0 {
            compact[r] = next;
            next += 1;
        }
        run_label[rid as usize] = compact[r];
    }

    // Pass 2b (row-parallel): fill the output image from the run spans.
    let out = labels.as_slice_mut();
    out.par_chunks_mut(w).enumerate().for_each(|(y, orow)| {
        let k = y / rows_per;
        let sr = &stripe_runs[k];
        let base = offsets[k];
        let lo = sr.row_ptr[y - sr.y0] as usize;
        let hi = sr.row_ptr[y - sr.y0 + 1] as usize;
        orow.fill(0);
        for rid in lo..hi {
            let (a, b) = sr.runs[rid];
            orow[a as usize..b as usize].fill(run_label[base as usize + rid]);
        }
    });
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
