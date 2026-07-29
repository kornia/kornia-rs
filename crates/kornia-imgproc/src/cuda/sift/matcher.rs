//! Brute-force 128-D descriptor matching on device, with Lowe's ratio test and
//! a mutual nearest-neighbour check.
//!
//! # Why no score matrix
//!
//! The obvious implementation materialises the `n1 x n2` distance matrix and
//! reduces it afterwards. At a few thousand descriptors a side that is tens of
//! megabytes of traffic for a result that is two small arrays. Here each query
//! keeps its best and second-best in registers while the train set streams past,
//! so nothing larger than the output is ever written.
//!
//! # Why no shared-memory tiling
//!
//! The plan called for tiling the train set through shared memory. Measured
//! against the actual sizes that is unnecessary: 2500 descriptors is 1.28 MB and
//! the L2 on this part is 4 MB, so the whole train set stays resident and the
//! re-reads never reach DRAM. Avoiding shared memory also avoids the carveout
//! that has cost this module six measured regressions.
//!
//! # Layout
//!
//! One warp per query descriptor, each lane owning 4 of the 128 dimensions. A
//! warp therefore reads one train descriptor as a single coalesced 512-byte
//! transaction, and the per-lane partial sums collapse in 5 `__shfl_xor` steps.
//!
//! # Distances
//!
//! Everything is squared L2. The ratio test `d_best < ratio * d_second` on true
//! distances is equivalent to `d2_best < ratio^2 * d2_second` on squared ones,
//! so no square root is ever taken.
//!
//! # Why u8 + `__dp4a`, and why it is lossless
//!
//! SIFT descriptors are **exactly** the integers 0..255 stored as f32 (the
//! reference scales, rounds and saturates to uchar; both backends reproduce
//! that bit-for-bit). So a u8 copy loses nothing, and the distance becomes
//! integer arithmetic: `||a-b||^2 = ||a||^2 + ||b||^2 - 2*a.b` with every term
//! exact in i32 (`||a||^2 + ||b||^2 <= 2*128*255^2 = 16.6M < 2^31`). The old
//! f32 kernel's sums were exact integers too (all partials < 2^24), so the
//! distances — and therefore the pairs, including tie behaviour — are
//! IDENTICAL; `__dp4a` just computes them with 4 MACs per instruction over a
//! quarter of the bytes.
//!
//! The one-time quantise pass costs two reads and half a write of each
//! descriptor set and is folded into the same stream. Inputs that are not
//! integer-valued 0..255 are outside the matcher's contract (nothing in this
//! crate produces them); they would be truncated per C cast.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, CudaView};

use super::descriptor::DESCR_LEN;
use super::kernels::get_or_compile;
use super::SiftCudaError;
use crate::cuda::make_config;

/// `i32` slots per emitted match: `query, train`.
pub const MATCH_STRIDE: usize = 2;

fn quantize_src() -> String {
    format!(
        r#"
#define DLEN {dlen}

// f32 descriptors (exact integers 0..255) -> packed u8 rows + i32 squared
// norms. One warp per row: each lane packs its 4 components into one u32, so
// a row is 32 u32 = 128 B, then the self-dot collapses in 5 shuffles.
extern "C" __global__ void __launch_bounds__(128) sift_desc_quantize(
    const float* __restrict__ src, int n,
    unsigned int* __restrict__ packed, int* __restrict__ norm)
{{
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (row >= n) return;

    const float4 v = *(const float4*)(src + (long)row * DLEN + lane * 4);
    const unsigned int b0 = (unsigned int)v.x, b1 = (unsigned int)v.y;
    const unsigned int b2 = (unsigned int)v.z, b3 = (unsigned int)v.w;
    const unsigned int p = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
    packed[(long)row * 32 + lane] = p;

    int nrm = __dp4a(p, p, 0u);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        nrm += __shfl_xor_sync(0xffffffffu, nrm, off);
    if (lane == 0) norm[row] = nrm;
}}
"#,
        dlen = DESCR_LEN,
    )
}

fn best2_src() -> String {
    format!(
        r#"
#define DLEN {dlen}
#define PERLANE {perlane}

// For each query descriptor, the nearest and second-nearest train descriptor,
// as squared L2 distances. One warp per query, over the u8-packed rows.
//
// dist = ||q||^2 + ||t||^2 - 2*q.t, all exact in i32. Every distance the old
// f32 kernel produced was an exact integer below 2^24, so the values — and the
// tie behaviour (strict <, first j wins) — are identical; the outputs are
// still written as f32 for the unchanged ratio kernel downstream.
extern "C" __global__ void __launch_bounds__(128) sift_match_best2(
    const unsigned int* __restrict__ q, const int* __restrict__ qn, int nq,
    const unsigned int* __restrict__ t, const int* __restrict__ tn, int nt,
    int* __restrict__ out_idx, float* __restrict__ out_d1, float* __restrict__ out_d2)
{{
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp_id >= nq) return;

    // This lane's u32 (4 components) of the query, held for the whole scan; a
    // warp reads one 128-byte packed row per candidate in one transaction.
    const unsigned int qv = q[(long)warp_id * 32 + lane];
    const int my_norm = qn[warp_id];

    int best = 2147483647, second = 2147483647;
    int best_j = -1;

    for (int j = 0; j < nt; j++) {{
        int acc = __dp4a(qv, t[(long)j * 32 + lane], 0u);
        // Butterfly reduction: every lane ends with the full dot product, so
        // the best/second bookkeeping needs no broadcast.
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            acc += __shfl_xor_sync(0xffffffffu, acc, off);
        const int d = my_norm + tn[j] - 2 * acc;

        if (d < best) {{ second = best; best = d; best_j = j; }}
        else if (d < second) {{ second = d; }}
    }}

    if (lane == 0) {{
        out_idx[warp_id] = best_j;
        // INT_MAX marks "no second candidate"; the pairs kernel tests against
        // f32 +MAX for that, so map it to the old sentinel explicitly.
        out_d1[warp_id] = (float)best;
        out_d2[warp_id] = second == 2147483647 ? 3.402823466e+38f : (float)second;
    }}
}}
"#,
        dlen = DESCR_LEN,
        perlane = DESCR_LEN / 32,
    )
}

fn pairs_src() -> String {
    format!(
        r#"
// Apply Lowe's ratio and the mutual nearest-neighbour check, appending the
// survivors. `ratio2` is the squared ratio, matching the squared distances.
extern "C" __global__ void sift_match_pairs(
    const int* __restrict__ fwd_idx, const float* __restrict__ fwd_d1,
    const float* __restrict__ fwd_d2, int nq,
    const int* __restrict__ rev_idx, int nt,
    float ratio2, int cross_check,
    int* __restrict__ out_pairs, int* __restrict__ out_count, int max_out)
{{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nq) return;
    const int j = fwd_idx[i];
    if (j < 0 || j >= nt) return;
    // `ratio >= 1` disables the test outright. Leaving it as the arithmetic
    // comparison would NOT be equivalent: with tied best and second distances
    // (identical or all-zero descriptors) `d1 < 1.0 * d2` is false, so a ratio
    // of exactly 1 would reject every match instead of accepting every one.
    //
    // A second-best of +inf means there was only one candidate; the reference
    // keeps such a match rather than dividing by infinity.
    if (ratio2 < 1.0f && fwd_d2[i] < 3.402823466e+38f &&
        !(fwd_d1[i] < ratio2 * fwd_d2[i])) return;
    if (cross_check && rev_idx[j] != i) return;

    const int slot = atomicAdd(out_count, 1);
    if (slot < max_out) {{
        out_pairs[slot * {stride}] = i;
        out_pairs[slot * {stride} + 1] = j;
    }}
}}
"#,
        stride = MATCH_STRIDE
    )
}

/// Scratch for one match call, sized for the largest descriptor set.
pub struct SiftMatcher {
    /// u8-packed descriptor rows (32 u32 per row) and their i32 squared norms,
    /// one pair of buffers per side, refilled by the quantise pass each call.
    q8: CudaSlice<u32>,
    qn: CudaSlice<i32>,
    t8: CudaSlice<u32>,
    tn: CudaSlice<i32>,
    fwd_idx: CudaSlice<i32>,
    fwd_d1: CudaSlice<f32>,
    fwd_d2: CudaSlice<f32>,
    rev_idx: CudaSlice<i32>,
    rev_d1: CudaSlice<f32>,
    rev_d2: CudaSlice<f32>,
    pairs: CudaSlice<i32>,
    count: CudaSlice<i32>,
    cap: usize,
}

impl SiftMatcher {
    /// Allocate for up to `cap` descriptors on either side.
    pub fn new(stream: &Arc<CudaStream>, cap: usize) -> Result<Self, SiftCudaError> {
        if cap == 0 {
            return Err(SiftCudaError::Geometry(
                "matcher capacity must be non-zero".into(),
            ));
        }
        Ok(Self {
            q8: stream.alloc_zeros::<u32>(cap * (DESCR_LEN / 4))?,
            qn: stream.alloc_zeros::<i32>(cap)?,
            t8: stream.alloc_zeros::<u32>(cap * (DESCR_LEN / 4))?,
            tn: stream.alloc_zeros::<i32>(cap)?,
            fwd_idx: stream.alloc_zeros::<i32>(cap)?,
            fwd_d1: stream.alloc_zeros::<f32>(cap)?,
            fwd_d2: stream.alloc_zeros::<f32>(cap)?,
            rev_idx: stream.alloc_zeros::<i32>(cap)?,
            rev_d1: stream.alloc_zeros::<f32>(cap)?,
            rev_d2: stream.alloc_zeros::<f32>(cap)?,
            pairs: stream.alloc_zeros::<i32>(cap * MATCH_STRIDE)?,
            count: stream.alloc_zeros::<i32>(1)?,
            cap,
        })
    }

    /// Match `d1` against `d2`, returning the surviving `(query, train)` pairs.
    ///
    /// `ratio` is Lowe's ratio on true distances (0.8 is the usual value);
    /// `>= 1.0` disables the test. `cross_check` additionally requires each pair
    /// to be a mutual nearest neighbour.
    ///
    /// Both descriptor sets stay on device; only the pair list comes back.
    #[allow(clippy::too_many_arguments)]
    pub fn match_descriptors(
        &mut self,
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        d1: &CudaView<'_, f32>,
        n1: usize,
        d2: &CudaView<'_, f32>,
        n2: usize,
        ratio: f32,
        cross_check: bool,
    ) -> Result<Vec<[i32; 2]>, SiftCudaError> {
        if n1 == 0 || n2 == 0 {
            return Ok(Vec::new());
        }
        if n1 > self.cap || n2 > self.cap {
            return Err(SiftCudaError::Geometry(format!(
                "matcher built for {} descriptors, got {n1} and {n2}",
                self.cap
            )));
        }
        for (d, n) in [(d1, n1), (d2, n2)] {
            if d.len() < n * DESCR_LEN {
                return Err(SiftCudaError::SliceTooSmall {
                    got: d.len(),
                    need: n * DESCR_LEN,
                });
            }
        }

        // One quantise pass per side: exact for integer-valued descriptors
        // (which is all this crate produces), and it shrinks every subsequent
        // candidate read 4x while turning 4 FMAs into one __dp4a.
        launch_quantize(ctx, stream, d1, n1, &mut self.q8, &mut self.qn)?;
        launch_quantize(ctx, stream, d2, n2, &mut self.t8, &mut self.tn)?;

        launch_best2(ctx, stream, n1, n2, self)?;
        // The reverse scan is only needed for the mutual check. Without it
        // `rev_idx` is never read — the pair kernel's test is `cross_check &&
        // rev_idx[j] != i` — so there is nothing to clear either.
        if cross_check {
            launch_best2_rev(ctx, stream, n2, n1, self)?;
        }

        stream.memset_zeros(&mut self.count)?;
        let kernel = get_or_compile(ctx, "sift_match_pairs", pairs_src, "sift_match_pairs")?;
        let (nq, nt) = (n1 as i32, n2 as i32);
        let ratio2 = ratio * ratio;
        let cc = i32::from(cross_check);
        let max_out = self.cap as i32;
        kernel
            .launch_builder(stream)
            .arg(&self.fwd_idx)
            .arg(&self.fwd_d1)
            .arg(&self.fwd_d2)
            .arg(&nq)
            .arg(&self.rev_idx)
            .arg(&nt)
            .arg(&ratio2)
            .arg(&cc)
            .arg(&mut self.pairs)
            .arg(&mut self.count)
            .arg(&max_out)
            .launch_2d(n1 as u32, 1, make_config(n1 as u32, 1, Some((128, 1))))
            .map_err(|e| SiftCudaError::Cuda(e.to_string()))?;

        let n = stream.clone_dtoh(&self.count)?[0].max(0) as usize;
        let n = n.min(self.cap);
        if n == 0 {
            return Ok(Vec::new());
        }
        let flat = stream.clone_dtoh(&self.pairs.slice(0..n * MATCH_STRIDE))?;
        Ok(flat
            .chunks_exact(MATCH_STRIDE)
            .map(|c| [c[0], c[1]])
            .collect())
    }
}

fn launch_quantize(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src: &CudaView<'_, f32>,
    n: usize,
    packed: &mut CudaSlice<u32>,
    norm: &mut CudaSlice<i32>,
) -> Result<(), SiftCudaError> {
    let kernel = get_or_compile(
        ctx,
        "sift_desc_quantize",
        quantize_src,
        "sift_desc_quantize",
    )?;
    let n_i = n as i32;
    let threads = 128u32;
    let blocks = (n as u32).div_ceil(threads / 32);
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };
    kernel
        .launch_builder(stream)
        .arg(src)
        .arg(&n_i)
        .arg(packed)
        .arg(norm)
        .launch_cfg(cfg)
        .map_err(|e| SiftCudaError::Cuda(e.to_string()))
}

/// `swap`: false = forward (q8 vs t8 into fwd_*), true = reverse.
#[allow(clippy::too_many_arguments)]
fn best2_common(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    m: &mut SiftMatcher,
    nq: usize,
    nt: usize,
    swap: bool,
) -> Result<(), SiftCudaError> {
    let kernel = get_or_compile(ctx, "sift_match_best2", best2_src, "sift_match_best2")?;
    let (nq_i, nt_i) = (nq as i32, nt as i32);
    // One warp per query, 4 warps per block.
    let threads = 128u32;
    let warps_per_block = threads / 32;
    let blocks = (nq as u32).div_ceil(warps_per_block);
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };
    let (q, qn, t, tn) = if swap {
        (&m.t8, &m.tn, &m.q8, &m.qn)
    } else {
        (&m.q8, &m.qn, &m.t8, &m.tn)
    };
    let (out_idx, out_d1, out_d2) = if swap {
        (&mut m.rev_idx, &mut m.rev_d1, &mut m.rev_d2)
    } else {
        (&mut m.fwd_idx, &mut m.fwd_d1, &mut m.fwd_d2)
    };
    kernel
        .launch_builder(stream)
        .arg(q)
        .arg(qn)
        .arg(&nq_i)
        .arg(t)
        .arg(tn)
        .arg(&nt_i)
        .arg(out_idx)
        .arg(out_d1)
        .arg(out_d2)
        .launch_cfg(cfg)
        .map_err(|e| SiftCudaError::Cuda(e.to_string()))
}

fn launch_best2(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    n1: usize,
    n2: usize,
    m: &mut SiftMatcher,
) -> Result<(), SiftCudaError> {
    best2_common(ctx, stream, m, n1, n2, false)
}

fn launch_best2_rev(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    n2: usize,
    n1: usize,
    m: &mut SiftMatcher,
) -> Result<(), SiftCudaError> {
    best2_common(ctx, stream, m, n2, n1, true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::color::test_utils::default_stream;

    /// Host brute force, used as the oracle.
    fn cpu_best2(q: &[f32], nq: usize, t: &[f32], nt: usize) -> Vec<(i32, f32, f32)> {
        (0..nq)
            .map(|i| {
                let (mut b, mut s, mut bj) = (f32::MAX, f32::MAX, -1i32);
                for j in 0..nt {
                    let d: f32 = (0..DESCR_LEN)
                        .map(|c| {
                            let v = q[i * DESCR_LEN + c] - t[j * DESCR_LEN + c];
                            v * v
                        })
                        .sum();
                    if d < b {
                        s = b;
                        b = d;
                        bj = j as i32;
                    } else if d < s {
                        s = d;
                    }
                }
                (bj, b, s)
            })
            .collect()
    }

    fn rand_desc(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n * DESCR_LEN)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 33) % 256) as f32
            })
            .collect()
    }

    #[test]
    fn matcher_finds_identical_descriptors() {
        let stream = default_stream();
        let ctx = &stream.context();
        let n = 300;
        let d = rand_desc(n, 12345);
        let dev = stream.clone_htod(&d).unwrap();
        let mut m = SiftMatcher::new(&stream, n).unwrap();

        // Matched against itself, every descriptor's nearest neighbour is itself
        // at distance zero, and the mutual check holds trivially.
        let pairs = m
            .match_descriptors(
                ctx,
                &stream,
                &dev.as_view(),
                n,
                &dev.as_view(),
                n,
                1.0,
                true,
            )
            .unwrap();
        assert_eq!(pairs.len(), n, "self-match should pair every descriptor");
        for [a, b] in &pairs {
            assert_eq!(a, b, "self-match pairs must be the identity");
        }
    }

    #[test]
    fn matcher_matches_host_brute_force() {
        let stream = default_stream();
        let ctx = &stream.context();
        let (n1, n2) = (257usize, 193usize);
        let a = rand_desc(n1, 999);
        let b = rand_desc(n2, 4242);
        let da = stream.clone_htod(&a).unwrap();
        let db = stream.clone_htod(&b).unwrap();
        let mut m = SiftMatcher::new(&stream, n1.max(n2)).unwrap();

        // Ratio 1.0 with no cross-check keeps every query, so the emitted pair
        // set is exactly "each query with its nearest train".
        let pairs = m
            .match_descriptors(
                ctx,
                &stream,
                &da.as_view(),
                n1,
                &db.as_view(),
                n2,
                1.0,
                false,
            )
            .unwrap();
        let want = cpu_best2(&a, n1, &b, n2);
        assert_eq!(pairs.len(), n1);
        let mut got = vec![-1i32; n1];
        for [q, t] in &pairs {
            got[*q as usize] = *t;
        }
        for (i, (bj, _, _)) in want.iter().enumerate() {
            assert_eq!(got[i], *bj, "query {i} matched the wrong descriptor");
        }
    }

    /// A single train descriptor has no second-best, so the ratio test has
    /// nothing to divide by. The match must survive rather than vanish or
    /// produce a NaN comparison.
    #[test]
    fn matcher_single_train_survives_ratio() {
        let stream = default_stream();
        let ctx = &stream.context();
        let q = rand_desc(5, 3);
        let t = rand_desc(1, 88);
        let dq = stream.clone_htod(&q).unwrap();
        let dt = stream.clone_htod(&t).unwrap();
        let mut m = SiftMatcher::new(&stream, 5).unwrap();
        let pairs = m
            .match_descriptors(ctx, &stream, &dq.as_view(), 5, &dt.as_view(), 1, 0.8, false)
            .unwrap();
        assert_eq!(pairs.len(), 5, "every query should match the only train");
        assert!(pairs.iter().all(|[_, t]| *t == 0));
    }

    /// All-zero descriptors make every distance exactly zero — maximal ties in
    /// both the best/second bookkeeping and the mutual check.
    #[test]
    fn matcher_all_zero_descriptors() {
        let stream = default_stream();
        let ctx = &stream.context();
        let n = 64;
        let z = vec![0.0f32; n * DESCR_LEN];
        let dz = stream.clone_htod(&z).unwrap();
        let mut m = SiftMatcher::new(&stream, n).unwrap();
        // best == second == 0, so `0 < 0.64 * 0` is false: all rejected.
        let strict = m
            .match_descriptors(ctx, &stream, &dz.as_view(), n, &dz.as_view(), n, 0.8, false)
            .unwrap();
        assert!(
            strict.is_empty(),
            "identical descriptors cannot pass a ratio test"
        );
        // With the ratio disabled every query still resolves to a real index.
        let loose = m
            .match_descriptors(ctx, &stream, &dz.as_view(), n, &dz.as_view(), n, 1.0, false)
            .unwrap();
        assert_eq!(loose.len(), n);
        assert!(loose.iter().all(|[_, t]| *t >= 0 && (*t as usize) < n));
    }

    /// Cross-check must actually reject: build a train set where one descriptor
    /// is the nearest neighbour of two different queries, so at most one of them
    /// can be mutual.
    #[test]
    fn matcher_cross_check_rejects_non_mutual() {
        let stream = default_stream();
        let ctx = &stream.context();
        let mut q = rand_desc(1, 21);
        // Two queries near the same train point, one nearer than the other.
        let mut near = q.clone();
        near[0] += 3.0;
        q.extend_from_slice(&near);
        let t = {
            let mut t = rand_desc(1, 21);
            t.extend_from_slice(&rand_desc(1, 555));
            t
        };
        let dq = stream.clone_htod(&q).unwrap();
        let dt = stream.clone_htod(&t).unwrap();
        let mut m = SiftMatcher::new(&stream, 4).unwrap();
        let pairs = m
            .match_descriptors(ctx, &stream, &dq.as_view(), 2, &dt.as_view(), 2, 1.0, true)
            .unwrap();
        // Train 0 is nearest to both queries but can only be mutual with the
        // closer one, so exactly one pair may reference it.
        let to_zero = pairs.iter().filter(|[_, t]| *t == 0).count();
        assert!(
            to_zero <= 1,
            "cross-check let a non-mutual pair through: {pairs:?}"
        );
    }

    /// Capacity is a hard limit, not a silent truncation.
    #[test]
    fn matcher_rejects_oversized_input() {
        let stream = default_stream();
        let ctx = &stream.context();
        let d = rand_desc(10, 1);
        let dev = stream.clone_htod(&d).unwrap();
        let mut m = SiftMatcher::new(&stream, 4).unwrap();
        assert!(m
            .match_descriptors(
                ctx,
                &stream,
                &dev.as_view(),
                10,
                &dev.as_view(),
                10,
                0.8,
                false
            )
            .is_err());
    }

    #[test]
    fn matcher_ratio_test_rejects_ambiguous() {
        let stream = default_stream();
        let ctx = &stream.context();
        // Two identical train descriptors make every query ambiguous: best and
        // second-best are equal, so no ratio below 1 can accept them.
        let q = rand_desc(8, 7);
        let mut t = rand_desc(1, 7);
        t.extend_from_within(..);
        let dq = stream.clone_htod(&q).unwrap();
        let dt = stream.clone_htod(&t).unwrap();
        let mut m = SiftMatcher::new(&stream, 8).unwrap();
        let pairs = m
            .match_descriptors(ctx, &stream, &dq.as_view(), 8, &dt.as_view(), 2, 0.8, false)
            .unwrap();
        assert!(
            pairs.is_empty(),
            "duplicated train descriptors must fail the ratio test, got {pairs:?}"
        );
    }
}
