/// Hamming distance between two fixed-size byte descriptors.
///
/// Specialized for `N=32` (ORB's 256-bit descriptor) on aarch64 (NEON) and
/// x86_64 (AVX2). Other widths and CPUs use scalar XOR and popcount.
///
/// # Arguments
///
/// * `a` - First N-byte descriptor.
/// * `b` - Second N-byte descriptor.
///
/// # Returns
///
/// The number of differing bits.
///
/// # Errors
///
/// This function does not return errors.
#[inline]
pub fn hamming_distance<const N: usize>(a: &[u8; N], b: &[u8; N]) -> u32 {
    match hamming_kernel::<N>() {
        #[cfg(target_arch = "x86_64")]
        HammingKernel::Avx2 => {
            if let (Ok(a), Ok(b)) = (a.as_slice().try_into(), b.as_slice().try_into()) {
                // SAFETY: the shared selector confirms AVX2; the converted
                // array references each contain exactly 32 valid bytes.
                return unsafe { hamming32_avx2(a, b) };
            }
        }
        #[cfg(target_arch = "aarch64")]
        HammingKernel::Neon => {
            if let (Ok(a), Ok(b)) = (a.as_slice().try_into(), b.as_slice().try_into()) {
                // SAFETY: NEON is architectural on aarch64; both arrays are 32 bytes.
                return unsafe { NeonHamming::new(a).distance(b) };
            }
        }
        HammingKernel::Scalar => {}
    }
    hamming_distance_scalar(a, b)
}

#[derive(Clone, Copy)]
enum HammingKernel {
    Scalar,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    #[cfg(target_arch = "aarch64")]
    Neon,
}

#[inline]
fn hamming_kernel<const N: usize>() -> HammingKernel {
    if N == 32 {
        #[cfg(target_arch = "x86_64")]
        if crate::simd::cpu_features().has_avx2 {
            return HammingKernel::Avx2;
        }
        #[cfg(target_arch = "aarch64")]
        return HammingKernel::Neon;
    }
    HammingKernel::Scalar
}

#[inline(always)]
fn hamming_distance_scalar<const N: usize>(a: &[u8; N], b: &[u8; N]) -> u32 {
    a.iter().zip(b).map(|(&x, &y)| (x ^ y).count_ones()).sum()
}

#[cfg(target_arch = "x86_64")]
struct Avx2Hamming {
    query: std::arch::x86_64::__m256i,
    lut: std::arch::x86_64::__m256i,
    mask: std::arch::x86_64::__m256i,
}

#[cfg(target_arch = "x86_64")]
impl Avx2Hamming {
    // These helpers inherit the caller's target features. Only the thin
    // target_feature wrappers are called from baseline or downstream code.
    #[inline(always)]
    unsafe fn new(query: &[u8; 32]) -> Self {
        use std::arch::x86_64::*;
        Self {
            // SAFETY: caller guarantees AVX2; query contains 32 readable bytes.
            query: unsafe { _mm256_loadu_si256(query.as_ptr().cast()) },
            // SAFETY: caller guarantees AVX2 for these register-only intrinsics.
            lut: unsafe {
                _mm256_setr_epi8(
                    0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2,
                    2, 3, 2, 3, 3, 4,
                )
            },
            // SAFETY: caller guarantees AVX2.
            mask: unsafe { _mm256_set1_epi8(0x0f) },
        }
    }

    #[inline(always)]
    unsafe fn partial_sums(&self, candidate: &[u8; 32]) -> std::arch::x86_64::__m256i {
        use std::arch::x86_64::*;
        // SAFETY: caller guarantees AVX2. The unaligned load reads exactly
        // the candidate's 32 bytes; the remaining intrinsics use registers.
        unsafe {
            let x = _mm256_xor_si256(self.query, _mm256_loadu_si256(candidate.as_ptr().cast()));
            let lo = _mm256_and_si256(x, self.mask);
            let hi = _mm256_and_si256(_mm256_srli_epi16(x, 4), self.mask);
            let pop = _mm256_add_epi8(
                _mm256_shuffle_epi8(self.lut, lo),
                _mm256_shuffle_epi8(self.lut, hi),
            );
            _mm256_sad_epu8(pop, _mm256_setzero_si256())
        }
    }

    #[inline(always)]
    unsafe fn distance(&self, candidate: &[u8; 32]) -> u32 {
        use std::arch::x86_64::*;
        // SAFETY: caller guarantees AVX2 and candidate contains 32 bytes.
        unsafe {
            let sums = self.partial_sums(candidate);
            let sum = _mm_add_epi64(
                _mm256_castsi256_si128(sums),
                _mm256_extracti128_si256(sums, 1),
            );
            let sum = _mm_add_epi64(sum, _mm_unpackhi_epi64(sum, sum));
            _mm_cvtsi128_si32(sum) as u32
        }
    }

    #[inline(always)]
    unsafe fn distances4(&self, candidates: &[[u8; 32]; 4]) -> [u32; 4] {
        use std::arch::x86_64::*;
        // SAFETY: caller guarantees AVX2. Each typed candidate supplies 32
        // bytes. SAD lanes are <=64 with zero high halves, so horizontal
        // i32 adds reduce four independent descriptors without overflow.
        // The final unaligned store initializes exactly four u32 outputs.
        unsafe {
            let ab = _mm256_hadd_epi32(
                self.partial_sums(&candidates[0]),
                self.partial_sums(&candidates[1]),
            );
            let cd = _mm256_hadd_epi32(
                self.partial_sums(&candidates[2]),
                self.partial_sums(&candidates[3]),
            );
            let halves = _mm256_hadd_epi32(ab, cd);
            let totals = _mm_add_epi32(
                _mm256_castsi256_si128(halves),
                _mm256_extracti128_si256(halves, 1),
            );
            let mut distances = [0; 4];
            _mm_storeu_si128(distances.as_mut_ptr().cast(), totals);
            distances
        }
    }
}

/// AVX2 32-byte Hamming distance via nibble lookup and SAD reduction.
///
/// # Safety
///
/// The caller must ensure AVX2 is available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hamming32_avx2(a: &[u8; 32], b: &[u8; 32]) -> u32 {
    // SAFETY: caller guarantees AVX2; both typed arrays contain 32 bytes.
    unsafe { Avx2Hamming::new(a).distance(b) }
}

#[cfg(target_arch = "aarch64")]
struct NeonHamming {
    query0: std::arch::aarch64::uint8x16_t,
    query1: std::arch::aarch64::uint8x16_t,
}

#[cfg(target_arch = "aarch64")]
impl NeonHamming {
    #[inline(always)]
    unsafe fn new(query: &[u8; 32]) -> Self {
        use std::arch::aarch64::*;
        // SAFETY: NEON is architectural; the two unaligned loads read the
        // complete 32-byte query without exceeding its bounds.
        unsafe {
            Self {
                query0: vld1q_u8(query.as_ptr()),
                query1: vld1q_u8(query.as_ptr().add(16)),
            }
        }
    }

    #[inline(always)]
    unsafe fn distance(&self, candidate: &[u8; 32]) -> u32 {
        use std::arch::aarch64::*;
        // SAFETY: NEON is architectural; both loads stay inside candidate.
        // Each 16-byte popcount sum is at most 128, so u8 reduction is exact.
        unsafe {
            let x0 = veorq_u8(self.query0, vld1q_u8(candidate.as_ptr()));
            let x1 = veorq_u8(self.query1, vld1q_u8(candidate.as_ptr().add(16)));
            vaddvq_u8(vcntq_u8(x0)) as u32 + vaddvq_u8(vcntq_u8(x1)) as u32
        }
    }
}

// One definition of earliest-index best/runner-up updates for all kernels,
// ungated and projection scans, including four-candidate batches.
struct HammingNearest {
    index: usize,
    best: u32,
    second: u32,
}

impl HammingNearest {
    #[inline(always)]
    fn update<const SECOND: bool>(&mut self, index: usize, distance: u32) {
        if distance < self.best {
            if SECOND {
                self.second = self.best;
            }
            self.best = distance;
            self.index = index;
        } else if SECOND && distance < self.second {
            self.second = distance;
        }
    }
}

#[inline(always)]
fn scan_hamming<const N: usize, const SECOND: bool, const GATED: bool>(
    candidates: &[[u8; N]],
    mut distance: impl FnMut(&[u8; N]) -> u32,
    mut distances4: impl FnMut(&[[u8; N]; 4]) -> [u32; 4],
    mut gate: impl FnMut(usize) -> bool,
    mut visit: impl FnMut(usize, u32),
) -> (usize, u32, u32) {
    let mut nearest = HammingNearest {
        index: 0,
        best: u32::MAX,
        second: u32::MAX,
    };
    let mut tail = candidates;
    let mut first_tail_index = 0;
    if !GATED {
        let (batches, remainder) = candidates.as_chunks::<4>();
        for (batch_index, batch) in batches.iter().enumerate() {
            for (offset, distance) in distances4(batch).into_iter().enumerate() {
                let index = batch_index * 4 + offset;
                visit(index, distance);
                nearest.update::<SECOND>(index, distance);
            }
        }
        first_tail_index = batches.len() * 4;
        tail = remainder;
    }
    for (offset, candidate) in tail.iter().enumerate() {
        let index = first_tail_index + offset;
        if GATED && !gate(index) {
            continue;
        }
        let distance = distance(candidate);
        visit(index, distance);
        nearest.update::<SECOND>(index, distance);
    }
    (nearest.index, nearest.best, nearest.second)
}

/// Find the earliest nearest descriptor and optionally its runner-up.
///
/// # Arguments
///
/// * `query` - N-byte query descriptor.
/// * `candidates` - Candidate descriptors in tie-breaking index order.
///
/// # Returns
///
/// `(index, best_distance, second_distance)`. An empty row returns
/// `(0, u32::MAX, u32::MAX)`; `SECOND=false` leaves the runner-up at `u32::MAX`.
///
/// # Errors
///
/// This function does not return errors.
pub(crate) fn hamming_row<const N: usize, const SECOND: bool>(
    query: &[u8; N],
    candidates: &[[u8; N]],
) -> (usize, u32, u32) {
    hamming_row_scan::<N, SECOND, false>(query, candidates, |_| true, |_, _| {})
}

fn hamming_row_scan<const N: usize, const SECOND: bool, const GATED: bool>(
    query: &[u8; N],
    candidates: &[[u8; N]],
    gate: impl FnMut(usize) -> bool,
    visit: impl FnMut(usize, u32),
) -> (usize, u32, u32) {
    // Select once per row, including on CPUs without AVX2. No pair calls
    // go back through the public runtime dispatcher.
    match hamming_kernel::<N>() {
        #[cfg(target_arch = "x86_64")]
        HammingKernel::Avx2 => {
            if let Ok(query) = query.as_slice().try_into() {
                let (candidates, _) = candidates.as_flattened().as_chunks::<32>();
                if GATED {
                    // Sparse projection scans spend most of their time in
                    // scalar gates. Keep that loop in the baseline context;
                    // call the selected distance kernel only for eligible rows.
                    return scan_hamming::<32, SECOND, true>(
                        candidates,
                        |candidate| {
                            // SAFETY: selector confirms AVX2; both arrays are 32 bytes.
                            unsafe { hamming32_avx2(query, candidate) }
                        },
                        |batch| {
                            std::array::from_fn(|i| {
                                // SAFETY: selector confirms AVX2; each array is 32 bytes.
                                unsafe { hamming32_avx2(query, &batch[i]) }
                            })
                        },
                        gate,
                        visit,
                    );
                }
                // SAFETY: the selector confirms AVX2 and N == 32; typed
                // references prevent short loads and truncated descriptors.
                return unsafe {
                    hamming_row_avx2::<SECOND, GATED>(query, candidates, gate, visit)
                };
            }
        }
        #[cfg(target_arch = "aarch64")]
        HammingKernel::Neon => {
            if let Ok(query) = query.as_slice().try_into() {
                let (candidates, _) = candidates.as_flattened().as_chunks::<32>();
                // SAFETY: NEON is architectural and N == 32.
                let kernel = unsafe { NeonHamming::new(query) };
                return scan_hamming::<32, SECOND, GATED>(
                    candidates,
                    |candidate| {
                        // SAFETY: NEON is architectural; candidate is 32 bytes.
                        unsafe { kernel.distance(candidate) }
                    },
                    |batch| {
                        std::array::from_fn(|i| {
                            // SAFETY: NEON is architectural; candidate is 32 bytes.
                            unsafe { kernel.distance(&batch[i]) }
                        })
                    },
                    gate,
                    visit,
                );
            }
        }
        HammingKernel::Scalar => {}
    }
    scan_hamming::<N, SECOND, GATED>(
        candidates,
        |candidate| hamming_distance_scalar(query, candidate),
        |batch| std::array::from_fn(|i| hamming_distance_scalar(query, &batch[i])),
        gate,
        visit,
    )
}

#[cfg(test)]
fn hamming_row_scalar<const N: usize, const SECOND: bool>(
    query: &[u8; N],
    candidates: &[[u8; N]],
) -> (usize, u32, u32) {
    scan_hamming::<N, SECOND, false>(
        candidates,
        |candidate| hamming_distance_scalar(query, candidate),
        |batch| std::array::from_fn(|i| hamming_distance_scalar(query, &batch[i])),
        |_| true,
        |_, _| {},
    )
}

/// # Safety
///
/// The caller must ensure AVX2 is available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hamming_row_avx2<const SECOND: bool, const GATED: bool>(
    query: &[u8; 32],
    candidates: &[[u8; 32]],
    gate: impl FnMut(usize) -> bool,
    visit: impl FnMut(usize, u32),
) -> (usize, u32, u32) {
    // SAFETY: caller guarantees AVX2; the query is exactly 32 bytes.
    let kernel = unsafe { Avx2Hamming::new(query) };
    scan_hamming::<32, SECOND, GATED>(
        candidates,
        |candidate| {
            // SAFETY: caller guarantees AVX2; candidate is exactly 32 bytes.
            unsafe { kernel.distance(candidate) }
        },
        |batch| {
            // SAFETY: caller guarantees AVX2; all candidates are 32 bytes.
            unsafe { kernel.distances4(batch) }
        },
        gate,
        visit,
    )
}

// The caller guarantees query indices fit in the low 32 bits of each key.
fn hamming_rows_mutual<const N: usize, const SECOND: bool>(
    descriptors1: &[[u8; N]],
    descriptors2: &[[u8; N]],
) -> (Vec<(usize, u32, u32)>, Vec<usize>) {
    use rayon::prelude::*;
    debug_assert!(descriptors1.len() <= u32::MAX as usize);
    // Reuse every forward distance for the reverse minima. Each task owns
    // its column keys, so the inner loop needs no atomics or shared writes.
    // Packed (distance, query index) keys keep earliest-index ties when
    // worker minima are merged in any order. Only O(threads * N2) scratch.
    let rows_per_task = descriptors1
        .len()
        .div_ceil(rayon::current_num_threads() * 4);
    let mut fwd = vec![(0, u32::MAX, u32::MAX); descriptors1.len()];
    let reverse = fwd
        .par_chunks_mut(rows_per_task)
        .zip(descriptors1.par_chunks(rows_per_task))
        .enumerate()
        .map(|(chunk_index, (results, queries))| {
            let mut reverse = vec![u64::MAX; descriptors2.len()];
            for (offset, (result, query)) in results.iter_mut().zip(queries).enumerate() {
                let i = chunk_index * rows_per_task + offset;
                *result = hamming_row_scan::<N, SECOND, false>(
                    query,
                    descriptors2,
                    |_| true,
                    |j, distance| {
                        let key = ((distance as u64) << 32) | i as u64;
                        reverse[j] = reverse[j].min(key);
                    },
                );
            }
            reverse
        })
        .reduce_with(|mut left, right| {
            for (left, right) in left.iter_mut().zip(right) {
                *left = (*left).min(right);
            }
            left
        })
        .unwrap_or_default();
    (
        fwd,
        reverse.into_iter().map(|key| key as u32 as usize).collect(),
    )
}

/// Match binary descriptors using brute-force Hamming distance.
///
/// For each descriptor in `descriptors1`, finds the nearest neighbor in `descriptors2`.
/// Optionally filters matches by maximum distance, cross-check, and Lowe's ratio test.
///
/// # Arguments
///
/// * `descriptors1` - First set of N-byte binary descriptors.
/// * `descriptors2` - Second set of N-byte binary descriptors.
/// * `max_distance` - If set, discard matches with Hamming distance above this threshold.
/// * `cross_check` - If true, keep only mutual nearest neighbors.
/// * `max_ratio` - Apply the strict Lowe ratio test (`best / second-best < ratio`)
///   when set to a value below 1.0. `None`, NaN, and values >= 1.0 disable it.
///   Zero-distance ties are rejected when enabled. With only one candidate,
///   the second-best distance is `u32::MAX`. The test applies only forward.
///
/// # Returns
///
/// Vector of `(i, j)` index pairs into `descriptors1` and `descriptors2`.
///
/// # Errors
///
/// This function does not return errors.
///
/// # Example
///
/// ```
/// use kornia_imgproc::features::match_descriptors;
/// let a = [[0u8; 32]];
/// let b = [[0u8; 32], [255u8; 32]];
/// assert_eq!(match_descriptors(&a, &b, Some(50), true, Some(0.8)), vec![(0, 0)]);
/// ```
pub fn match_descriptors<const N: usize>(
    descriptors1: &[[u8; N]],
    descriptors2: &[[u8; N]],
    max_distance: Option<u32>,
    cross_check: bool,
    max_ratio: Option<f32>,
) -> Vec<(usize, usize)> {
    if descriptors1.is_empty() || descriptors2.is_empty() {
        return vec![];
    }

    use rayon::prelude::*;
    let needs_second = max_ratio.is_some_and(|ratio| ratio < 1.0);
    let threads = rayon::current_num_threads();
    // Worker-local column minima cost more than a second scan on small
    // parallel workloads or too few queries to amortize column scratch.
    // Use fusion for ORB widths only, above the measured crossover.
    let work_per_thread = descriptors1.len().saturating_mul(descriptors2.len()) / threads;
    let fusion_threshold = if threads == 1 { 64 * 1024 } else { 256 * 1024 };
    let (fwd, rev_best_i) = if cross_check
        && N == 32
        && descriptors1.len() >= threads.saturating_mul(64)
        && descriptors1.len() <= u32::MAX as usize
        && work_per_thread >= fusion_threshold
    {
        let (fwd, reverse) = if needs_second {
            hamming_rows_mutual::<N, true>(descriptors1, descriptors2)
        } else {
            hamming_rows_mutual::<N, false>(descriptors1, descriptors2)
        };
        (fwd, Some(reverse))
    } else {
        let fwd = descriptors1
            .par_iter()
            .map(|d1| {
                if needs_second {
                    hamming_row::<N, true>(d1, descriptors2)
                } else {
                    hamming_row::<N, false>(d1, descriptors2)
                }
            })
            .collect::<Vec<_>>();
        // The two-pass fallback also retains full usize indices for query
        // slices whose indices do not fit the packed key.
        let reverse = cross_check.then(|| {
            descriptors2
                .par_iter()
                .map(|d2| hamming_row::<N, false>(d2, descriptors1).0)
                .collect::<Vec<_>>()
        });
        (fwd, reverse)
    };

    // Build matches applying all filters in one pass.
    let mut matches = Vec::new();
    for (i, &(j, best_dist, second_dist)) in fwd.iter().enumerate() {
        if let Some(max_dist) = max_distance {
            if best_dist == u32::MAX || best_dist > max_dist {
                continue;
            }
        }

        if let Some(ref rev) = rev_best_i {
            if rev[j] != i {
                continue;
            }
        }

        if let Some(ratio) = max_ratio {
            if ratio < 1.0 && (second_dist == 0 || best_dist as f32 / second_dist as f32 >= ratio) {
                continue;
            }
        }

        matches.push((i, j));
    }

    matches
}

/// Cosine similarity (dot product) between two L2-normalised f32 descriptors.
///
/// Assumes both inputs are unit-norm; the dot product is then the cosine.
/// LLVM autovectorises the loop on aarch64 (NEON) and x86_64 (AVX2) for
/// fixed-size arrays such as `[f32; 64]` (XFeat) or `[f32; 256]` (SuperPoint).
#[inline]
pub fn dot_product<const D: usize>(a: &[f32; D], b: &[f32; D]) -> f32 {
    // Fused multiply-add when available; iterator form lets LLVM unroll +
    // vectorise. Manual NEON intrinsics offered no measurable win in benches.
    let mut acc = 0.0f32;
    for i in 0..D {
        acc += a[i] * b[i];
    }
    acc
}

/// Match f32 descriptors via cosine similarity (assumes L2-normalised inputs).
///
/// Equivalent to `xfeat.match` (XFeat / SuperPoint convention) with optional
/// mutual nearest neighbour check and Lowe's ratio test in cosine space.
///
/// # Arguments
/// * `descriptors1` - First set of D-dim descriptors (length N1, each `&[f32; D]`).
/// * `descriptors2` - Second set (length N2).
/// * `min_cossim`   - Minimum cosine similarity to keep the match (default ~0.82).
/// * `cross_check`  - If true, require mutual NN match.
/// * `max_ratio_cos` - Lowe-ratio in cosine space:
///   `(1 - best_cos) / (1 - second_best_cos) <= max_ratio_cos`
///   (lower = more discriminative; typical 0.8-0.85).
///
/// # Returns
/// Vec<(i, j)> index pairs into descriptors1 / descriptors2.
///
/// Uses rayon for parallelism across descriptors1; cossim computed inline
/// as the dot product (descriptors must be L2-normalised). NEON SIMD over
/// the 64-dim inner loop is autovectorised by LLVM on aarch64.
pub fn match_descriptors_f32<const D: usize>(
    descriptors1: &[[f32; D]],
    descriptors2: &[[f32; D]],
    min_cossim: Option<f32>,
    cross_check: bool,
    max_ratio_cos: Option<f32>,
) -> Vec<(usize, usize)> {
    if descriptors1.is_empty() || descriptors2.is_empty() {
        return vec![];
    }

    // Forward pass: for each desc1[i], find best and second-best cosine in desc2.
    // Each row is independent — parallelize across descriptors1.
    use rayon::prelude::*;
    let fwd: Vec<(usize, f32, f32)> = descriptors1
        .par_iter()
        .map(|d1| {
            let mut best_j = 0usize;
            let mut best_cos = f32::NEG_INFINITY;
            let mut second_cos = f32::NEG_INFINITY;
            for (j, d2) in descriptors2.iter().enumerate() {
                let cos = dot_product(d1, d2);
                if cos > best_cos {
                    second_cos = best_cos;
                    best_cos = cos;
                    best_j = j;
                } else if cos > second_cos {
                    second_cos = cos;
                }
            }
            (best_j, best_cos, second_cos)
        })
        .collect();

    // Reverse pass (only if cross-check): for each desc2[j], find best match in desc1.
    let rev_best_i = if cross_check {
        let rev: Vec<usize> = descriptors2
            .par_iter()
            .map(|d2| {
                let mut best_i = 0usize;
                let mut best_cos = f32::NEG_INFINITY;
                for (i, d1) in descriptors1.iter().enumerate() {
                    let cos = dot_product(d1, d2);
                    if cos > best_cos {
                        best_cos = cos;
                        best_i = i;
                    }
                }
                best_i
            })
            .collect();
        Some(rev)
    } else {
        None
    };

    // Build matches applying all filters in one pass.
    let mut matches = Vec::new();
    for (i, &(j, best_cos, second_cos)) in fwd.iter().enumerate() {
        if let Some(min_cs) = min_cossim {
            if best_cos < min_cs {
                continue;
            }
        }

        if let Some(ref rev) = rev_best_i {
            if rev[j] != i {
                continue;
            }
        }

        if let Some(ratio) = max_ratio_cos {
            if ratio < 1.0 && second_cos > f32::NEG_INFINITY {
                // Cosine-space Lowe ratio: (1 - best) / (1 - second).
                // Smaller ratio = more discriminative (best much closer to 1
                // than second).
                let denom = (1.0 - second_cos).max(1e-6);
                let r = (1.0 - best_cos) / denom;
                if r > ratio {
                    continue;
                }
            }
        }

        matches.push((i, j));
    }

    matches
}

/// Borrowed view over ORB features — descriptors + keypoint positions +
/// per-keypoint octaves, all as parallel slices.
///
/// Used by [`match_orb_by_projection`] on both sides of the match (predicted
/// projections from a map or previous frame, and observed keypoints in the
/// current frame). Collapses what would otherwise be six positional slice
/// arguments on the matcher into two view structs.
///
/// For the "predicted" side, `keypoints_xy` are the projections of map points
/// into the current frame (not the originally-detected keypoints); octaves
/// are the octave the map point was originally detected at, which drives the
/// scale-aware search radius.
#[derive(Debug, Copy, Clone)]
pub struct OrbFeaturesView<'a, const N: usize> {
    /// Binary descriptors (one row per feature, `N` bytes each).
    pub descriptors: &'a [[u8; N]],
    /// Feature positions as `[col, row]` in image pixels.
    pub keypoints_xy: &'a [[f32; 2]],
    /// Pyramid octave per feature (0 = full resolution, higher = coarser).
    pub octaves: &'a [u8],
}

impl<const N: usize> OrbFeaturesView<'_, N> {
    /// Number of features in the view. All three slices must have this
    /// length; this is asserted by the matcher.
    pub fn len(&self) -> usize {
        self.descriptors.len()
    }

    /// `true` when the view contains zero features.
    pub fn is_empty(&self) -> bool {
        self.descriptors.is_empty()
    }
}

/// Configuration for [`match_orb_by_projection`].
///
/// Mirrors the knobs exposed by ORB-SLAM3's `ORBmatcher::SearchByProjection`.
#[derive(Debug, Clone)]
pub struct ByProjectionConfig {
    /// Base search radius in pixels at octave 0. The effective radius for
    /// a candidate predicted at octave `o` is `base_radius * scale_factors[o]`
    /// — higher octaves (coarser pyramid) project back to larger pixel
    /// uncertainty at full resolution.
    pub base_radius: f32,
    /// Per-octave scale multiplier, typically `downscale.powi(o)` for each
    /// octave `o`. Caller provides this precomputed; ORB-SLAM3 stores it as
    /// `mvScaleFactors` on the frame.
    pub scale_factors: Vec<f32>,
    /// Maximum allowed octave difference between predicted and candidate
    /// keypoint (default 1). BRIEF is scale-variant so cross-octave matches
    /// are unreliable.
    pub max_octave_diff: u8,
    /// Upper bound on Hamming distance for an accepted match (default 50 for
    /// 256-bit BRIEF, matching ORB-SLAM3's `TH_HIGH` / `TH_LOW` tuning).
    pub max_distance: u32,
    /// Lowe's ratio test threshold (default 0.75). A best/second-best ratio
    /// above this rejects the match. Set to >=1.0 to disable.
    pub max_ratio: f32,
}

impl Default for ByProjectionConfig {
    fn default() -> Self {
        Self {
            base_radius: 15.0,
            scale_factors: Vec::new(),
            max_octave_diff: 1,
            max_distance: 50,
            max_ratio: 0.75,
        }
    }
}

/// Scale-aware guided Hamming matcher — ORB-SLAM3's `SearchByProjection`.
///
/// For each predicted feature (from projecting a map point or tracked keypoint
/// into the current frame), searches observed keypoints inside a scale-
/// aware radius and with a compatible octave, and returns the best Hamming
/// match subject to Lowe's ratio test and a distance threshold.
///
/// This is the hot path for ORB-SLAM-family tracking — it replaces the O(M·N)
/// brute-force matcher with a spatially-gated one, while still letting the
/// descriptor distance break ties inside each gate.
///
/// # Arguments
///
/// * `predicted` - View over predicted features. `keypoints_xy` here are the
///   projections of map points / prior-frame features into the current
///   frame; `octaves` are the octaves at which those features were
///   originally detected (stored on the map point).
/// * `observed` - View over currently-detected features in the target frame.
/// * `cfg` - [`ByProjectionConfig`] — search radius, octave tolerance, ratio
///   and distance thresholds.
///
/// # Returns
///
/// Vector of `(i, j)` index pairs into `predicted` and `observed`
/// respectively. Order is deterministic (sorted by `i`). Rows with no eligible
/// candidates are omitted, even when `max_distance` is `u32::MAX`. An enabled
/// ratio test rejects zero-distance ties and is skipped for a single eligible
/// candidate. NaN and values >= 1.0 disable the ratio test.
///
/// # Errors
///
/// This function does not return errors.
///
/// # Panics
///
/// Panics if a view's descriptor, position and octave lengths differ, or if
/// both views are nonempty and `cfg.scale_factors` is empty.
///
/// # Example
///
/// ```
/// use kornia_imgproc::features::{match_orb_by_projection, OrbFeaturesView, ByProjectionConfig};
/// let descriptors = [[0u8; 32]];
/// let features = OrbFeaturesView { descriptors: &descriptors,
///     keypoints_xy: &[[10.0, 20.0]], octaves: &[0] };
/// let cfg = ByProjectionConfig { scale_factors: vec![1.0], ..Default::default() };
/// assert_eq!(match_orb_by_projection(features, features, &cfg), vec![(0, 0)]);
/// ```
pub fn match_orb_by_projection<const N: usize>(
    predicted: OrbFeaturesView<'_, N>,
    observed: OrbFeaturesView<'_, N>,
    cfg: &ByProjectionConfig,
) -> Vec<(usize, usize)> {
    use rayon::prelude::*;

    assert_eq!(predicted.descriptors.len(), predicted.keypoints_xy.len());
    assert_eq!(predicted.descriptors.len(), predicted.octaves.len());
    assert_eq!(observed.descriptors.len(), observed.keypoints_xy.len());
    assert_eq!(observed.descriptors.len(), observed.octaves.len());

    if predicted.is_empty() || observed.is_empty() {
        return Vec::new();
    }

    // `Default` leaves scale_factors empty on purpose — the caller must pass
    // the per-octave scale from its pyramid. Silently defaulting to 1.0 would
    // make the matcher non-scale-aware with no signal.
    assert!(
        !cfg.scale_factors.is_empty(),
        "ByProjectionConfig.scale_factors must be populated",
    );

    let base_r = cfg.base_radius;
    let max_oct_diff = cfg.max_octave_diff as i32;
    let max_dist = cfg.max_distance;
    let ratio = cfg.max_ratio;
    let scale_factors = &cfg.scale_factors;

    // Per-predicted-feature forward pass: find best/second-best observed kp
    // inside the scale-aware gate. Parallelize over predictions.
    let matches: Vec<Option<(usize, usize)>> = predicted
        .descriptors
        .par_iter()
        .enumerate()
        .map(|(i, d_pred)| {
            let pred_oct = predicted.octaves[i];
            let scale = scale_factors.get(pred_oct as usize).copied().unwrap_or(1.0);
            let radius = base_r * scale;
            let radius_sq = radius * radius;
            let [px, py] = predicted.keypoints_xy[i];

            let (best_j, best_dist, second_dist) = hamming_row_scan::<N, true, true>(
                d_pred,
                observed.descriptors,
                |j| {
                    let oct_diff = (observed.octaves[j] as i32 - pred_oct as i32).abs();
                    if oct_diff > max_oct_diff {
                        return false;
                    }
                    let [cx, cy] = observed.keypoints_xy[j];
                    let dx = cx - px;
                    let dy = cy - py;
                    // Preserve the original gate's treatment of NaN coordinates.
                    let outside = dx * dx + dy * dy > radius_sq;
                    !outside
                },
                |_, _| {},
            );

            if best_dist == u32::MAX || best_dist > max_dist {
                return None;
            }
            if ratio < 1.0
                && second_dist != u32::MAX
                && (second_dist == 0 || best_dist as f32 / second_dist as f32 >= ratio)
            {
                return None;
            }
            Some((i, best_j))
        })
        .collect();

    matches.into_iter().flatten().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_ratio_rejects_ambiguous_zero_ties() {
        let zero = [0; 32];
        for cross_check in [false, true] {
            assert!(
                match_descriptors(&[zero], &[zero, zero], None, cross_check, Some(0.8)).is_empty()
            );
        }
    }

    #[test]
    fn binary_ratio_uses_strict_f32_boundary() {
        let zero = [0; 32];
        let one = desc_from_bit(1);
        let three = desc_from_bit(7);
        let boundary = 1.0f32 / 3.0;
        assert!(match_descriptors(&[zero], &[one, three], None, false, Some(boundary)).is_empty());
        assert_eq!(
            reference_binary_matches(&[zero], &[one, three], None, false, Some(boundary)),
            Vec::<(usize, usize)>::new()
        );
        assert_eq!(
            match_descriptors(
                &[zero],
                &[one, three],
                None,
                false,
                Some(boundary + f32::EPSILON)
            ),
            vec![(0, 0)]
        );
    }

    #[test]
    fn projection_rejects_rows_without_eligible_candidates() {
        let descriptors = [[0; 32]];
        let positions = [[0.0, 0.0]];
        let octaves = [0];
        let predicted = OrbFeaturesView {
            descriptors: &descriptors,
            keypoints_xy: &positions,
            octaves: &octaves,
        };
        let cfg = ByProjectionConfig {
            scale_factors: vec![1.0],
            max_distance: u32::MAX,
            max_ratio: 1.0,
            ..Default::default()
        };
        for (positions, octaves) in [([[100.0, 0.0]], [0]), ([[0.0, 0.0]], [2])] {
            let observed = OrbFeaturesView {
                descriptors: &descriptors,
                keypoints_xy: &positions,
                octaves: &octaves,
            };
            assert!(match_orb_by_projection(predicted, observed, &cfg).is_empty());
        }
    }

    #[test]
    fn projection_preserves_nan_and_single_candidate_gates() {
        let descriptors = [[0; 32]];
        let octaves = [0];
        let predicted = OrbFeaturesView {
            descriptors: &descriptors,
            keypoints_xy: &[[0.0, 0.0]],
            octaves: &octaves,
        };
        let cfg = ByProjectionConfig {
            scale_factors: vec![1.0],
            max_ratio: -1.0,
            ..Default::default()
        };
        for positions in [[[0.0, 0.0]], [[f32::NAN, 0.0]]] {
            let observed = OrbFeaturesView {
                descriptors: &descriptors,
                keypoints_xy: &positions,
                octaves: &octaves,
            };
            // The ratio test is skipped without a runner-up; NaN coordinates
            // retain the original greater-than gate behavior.
            assert_eq!(
                match_orb_by_projection(predicted, observed, &cfg),
                vec![(0, 0)]
            );
        }
    }

    #[test]
    fn projection_ratio_rejects_ambiguous_zero_ties() {
        let descriptors = [[0; 32], [0; 32]];
        let positions = [[0.0, 0.0]; 2];
        let octaves = [0; 2];
        let features = OrbFeaturesView {
            descriptors: &descriptors,
            keypoints_xy: &positions,
            octaves: &octaves,
        };
        let cfg = ByProjectionConfig {
            scale_factors: vec![1.0],
            ..Default::default()
        };
        assert!(match_orb_by_projection(features, features, &cfg).is_empty());
    }

    fn random_binary_descs<const N: usize>(n: usize, mut seed: u64) -> Vec<[u8; N]> {
        (0..n)
            .map(|_| {
                let mut descriptor = [0; N];
                for byte in &mut descriptor {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    *byte = seed as u8;
                }
                descriptor
            })
            .collect()
    }

    // Sort complete scalar distance rows to independently check nearest-neighbor
    // ordering, ties and the existing forward-only ratio semantics.
    fn reference_binary_matches<const N: usize>(
        queries: &[[u8; N]],
        candidates: &[[u8; N]],
        max_distance: Option<u32>,
        cross_check: bool,
        max_ratio: Option<f32>,
    ) -> Vec<(usize, usize)> {
        let distance = |a: &[u8; N], b: &[u8; N]| -> u32 {
            a.iter().zip(b).map(|(a, b)| (a ^ b).count_ones()).sum()
        };
        queries
            .iter()
            .enumerate()
            .filter_map(|(i, query)| {
                let mut row: Vec<_> = candidates
                    .iter()
                    .enumerate()
                    .map(|(j, candidate)| (distance(query, candidate), j))
                    .collect();
                row.sort_unstable();
                let &(best, j) = row.first()?;
                if max_distance.is_some_and(|limit| best > limit) {
                    return None;
                }
                if cross_check {
                    let reverse = queries
                        .iter()
                        .enumerate()
                        .map(|(k, query)| (distance(query, &candidates[j]), k))
                        .min()?;
                    if reverse.1 != i {
                        return None;
                    }
                }
                let second = row.get(1).map_or(u32::MAX, |&(dist, _)| dist);
                // Contract: disabled for None, NaN or >= 1; otherwise a strict
                // ratio, with an ambiguous zero runner-up always rejected.
                let enabled =
                    max_ratio.filter(|r| r.partial_cmp(&1.0) == Some(std::cmp::Ordering::Less));
                if let Some(ratio) = enabled {
                    let relative_distance = (best as f64 / second as f64) as f32;
                    if second == 0
                        || relative_distance.partial_cmp(&ratio) != Some(std::cmp::Ordering::Less)
                    {
                        return None;
                    }
                }
                Some((i, j))
            })
            .collect()
    }

    fn check_binary_matches<const N: usize>() {
        let mut queries = random_binary_descs::<N>(37, 0x1234_5678);
        let mut candidates = random_binary_descs::<N>(43, 0x9876_5432);
        queries[3] = queries[0];
        candidates[1] = queries[0];
        candidates[5] = queries[0];
        candidates[11] = queries[7];
        for query_count in [0, 1, 37] {
            for candidate_count in [0, 1, 2, 43] {
                for cross_check in [false, true] {
                    for max_distance in [None, Some(0), Some(1), Some(128), Some(u32::MAX)] {
                        for ratio in [
                            None,
                            Some(f32::NEG_INFINITY),
                            Some(-0.5),
                            Some(0.0),
                            Some(0.5),
                            Some(0.99),
                            Some(1.0),
                            Some(1.5),
                            Some(f32::INFINITY),
                            Some(f32::NAN),
                        ] {
                            let queries = &queries[..query_count];
                            let candidates = &candidates[..candidate_count];
                            assert_eq!(
                                match_descriptors(queries, candidates, max_distance, cross_check, ratio),
                                reference_binary_matches(queries, candidates, max_distance, cross_check, ratio),
                                "N={N}, queries={query_count}, candidates={candidate_count}, cross_check={cross_check}, max_distance={max_distance:?}, ratio={ratio:?}",
                            );
                        }
                    }
                }
            }
        }
    }

    fn check_scalar_rows<const N: usize>() {
        let queries = random_binary_descs::<N>(5, 0x1234_5678);
        let mut candidates = random_binary_descs::<N>(71, 0x8765_4321);
        candidates[3] = queries[0];
        candidates[4] = queries[0];
        for query in &queries {
            for count in [0, 1, 2, 3, 4, 5, 7, 8, 9, 31, 32, 33, 71] {
                let candidates = &candidates[..count];
                let mut sorted: Vec<_> = candidates
                    .iter()
                    .enumerate()
                    .map(|(j, d)| {
                        (
                            query
                                .iter()
                                .zip(d)
                                .map(|(a, b)| (a ^ b).count_ones())
                                .sum::<u32>(),
                            j,
                        )
                    })
                    .collect();
                sorted.sort_unstable();
                let (best, index) = sorted.first().copied().unwrap_or((u32::MAX, 0));
                let second = sorted.get(1).map_or(u32::MAX, |entry| entry.0);
                assert_eq!(
                    hamming_row_scalar::<N, true>(query, candidates),
                    (index, best, second)
                );
                assert_eq!(
                    hamming_row_scalar::<N, false>(query, candidates),
                    (index, best, u32::MAX)
                );
                assert_eq!(
                    hamming_row::<N, true>(query, candidates),
                    (index, best, second)
                );
                assert_eq!(
                    hamming_row::<N, false>(query, candidates),
                    (index, best, u32::MAX)
                );
                let mut visited = Vec::new();
                let gated = hamming_row_scan::<N, true, true>(
                    query,
                    candidates,
                    |j| j % 3 == 1,
                    |j, d| visited.push((d, j)),
                );
                let mut expected: Vec<_> = sorted.into_iter().filter(|(_, j)| j % 3 == 1).collect();
                let (best, index) = expected.first().copied().unwrap_or((u32::MAX, 0));
                let second = expected.get(1).map_or(u32::MAX, |entry| entry.0);
                assert_eq!(gated, (index, best, second));
                visited.sort_unstable();
                expected.sort_unstable();
                assert_eq!(visited, expected);
            }
        }
    }

    #[test]
    fn forced_scalar_and_gated_rows_agree_with_sorted_distances() {
        check_scalar_rows::<0>();
        check_scalar_rows::<1>();
        check_scalar_rows::<7>();
        check_scalar_rows::<16>();
        check_scalar_rows::<31>();
        check_scalar_rows::<32>();
        check_scalar_rows::<33>();
        check_scalar_rows::<64>();
    }

    #[test]
    fn fused_rows_preserve_earliest_ties_across_workers() {
        for threads in [1, 4] {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| {
                    for (n1, n2) in [(1, 1), (37, 43), (71, 5), (5, 71)] {
                        let mut queries = random_binary_descs::<32>(n1, 0x1234_5678);
                        let mut candidates = random_binary_descs::<32>(n2, 0x8765_4321);
                        queries[n1 - 1] = queries[0];
                        candidates[n2 - 1] = queries[0];
                        let (fwd, reverse) = hamming_rows_mutual::<32, true>(&queries, &candidates);
                        for (query, row) in queries.iter().zip(fwd) {
                            assert_eq!(row, hamming_row_scalar::<32, true>(query, &candidates));
                        }
                        for (candidate, index) in candidates.iter().zip(reverse) {
                            assert_eq!(
                                index,
                                hamming_row_scalar::<32, false>(candidate, &queries).0
                            );
                        }
                        assert_eq!(
                            hamming_rows_mutual::<32, true>(&vec![[0; 32]; n1], &vec![[0; 32]; n2]),
                            (
                                vec![(0, 0, if n2 > 1 { 0 } else { u32::MAX }); n1],
                                vec![0; n2]
                            )
                        );
                    }
                });
        }
    }

    #[test]
    fn binary_rows_agree_with_sorted_scalar_distances() {
        let queries = random_binary_descs::<32>(11, 0x1122_3344);
        let mut candidates = random_binary_descs::<32>(71, 0x5566_7788);
        candidates[7] = queries[0];
        candidates[9] = queries[0];
        for query in &queries {
            for count in [0, 1, 2, 71] {
                let candidates = &candidates[..count];
                let mut distances: Vec<_> = candidates
                    .iter()
                    .enumerate()
                    .map(|(j, candidate)| {
                        let distance: u32 = query
                            .iter()
                            .zip(candidate)
                            .map(|(a, b)| (a ^ b).count_ones())
                            .sum();
                        (distance, j)
                    })
                    .collect();
                distances.sort_unstable();
                let (distance, index) = distances.first().copied().unwrap_or((u32::MAX, 0));
                let second = distances.get(1).map_or(u32::MAX, |entry| entry.0);
                assert_eq!(
                    hamming_row::<32, true>(query, candidates),
                    (index, distance, second)
                );
                assert_eq!(
                    hamming_row::<32, false>(query, candidates),
                    (index, distance, u32::MAX)
                );
                assert_eq!(
                    hamming_row_scalar::<32, true>(query, candidates),
                    (index, distance, second)
                );
                #[cfg(target_arch = "x86_64")]
                if crate::simd::cpu_features().has_avx2 {
                    // SAFETY: AVX2 detected above and descriptors are 32 bytes.
                    assert_eq!(
                        unsafe {
                            hamming_row_avx2::<true, false>(query, candidates, |_| true, |_, _| {})
                        },
                        (index, distance, second)
                    );
                    // SAFETY: AVX2 detected above and descriptors are 32 bytes.
                    assert_eq!(
                        unsafe {
                            hamming_row_avx2::<false, false>(query, candidates, |_| true, |_, _| {})
                        },
                        (index, distance, u32::MAX)
                    );
                }
            }
        }
    }

    #[test]
    fn binary_matches_agree_with_scalar_reference() {
        for threads in [1, 4] {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| {
                    check_binary_matches::<0>();
                    check_binary_matches::<1>();
                    check_binary_matches::<7>();
                    check_binary_matches::<16>();
                    check_binary_matches::<31>();
                    check_binary_matches::<32>();
                    check_binary_matches::<33>();
                    check_binary_matches::<64>();
                });
        }
    }

    #[test]
    fn binary_matches_include_maximum_distance_at_the_cap() {
        let zero = [0; 32];
        let ones = [u8::MAX; 32];
        assert_eq!(hamming_distance(&zero, &ones), 256);
        assert_eq!(hamming_row::<32, true>(&zero, &[ones, ones]), (0, 256, 256));
        assert_eq!(
            match_descriptors(&[zero], &[ones], Some(256), true, None),
            vec![(0, 0)]
        );
        assert!(match_descriptors(&[zero], &[ones], Some(255), true, None).is_empty());
    }

    #[test]
    fn binary_matches_accept_unaligned_borrowed_rows() {
        let mut query_storage = [0; 34];
        let query_offset = if (query_storage.as_ptr() as usize + 1).is_multiple_of(32) {
            2
        } else {
            1
        };
        query_storage[query_offset..query_offset + 32]
            .copy_from_slice(&random_binary_descs::<32>(1, 0x1234_5678)[0]);
        let query: &[u8; 32] = query_storage[query_offset..query_offset + 32]
            .try_into()
            .unwrap();
        let mut candidate_storage = vec![0; 32 * 71 + 2];
        let candidate_offset = if (candidate_storage.as_ptr() as usize + 1).is_multiple_of(32) {
            2
        } else {
            1
        };
        for (destination, source) in candidate_storage[candidate_offset..]
            .as_chunks_mut::<32>()
            .0
            .iter_mut()
            .zip(random_binary_descs::<32>(71, 0x8765_4321))
        {
            destination.copy_from_slice(&source);
        }
        let (candidates, _) = candidate_storage[candidate_offset..].as_chunks::<32>();
        assert_ne!(query.as_ptr() as usize % 32, 0);
        assert_ne!(candidates.as_ptr() as usize % 32, 0);
        for cross_check in [false, true] {
            for ratio in [None, Some(0.8), Some(1.0)] {
                assert_eq!(
                    match_descriptors(
                        std::slice::from_ref(query),
                        candidates,
                        None,
                        cross_check,
                        ratio
                    ),
                    reference_binary_matches(
                        std::slice::from_ref(query),
                        candidates,
                        None,
                        cross_check,
                        ratio
                    ),
                );
            }
        }
    }

    #[test]
    fn binary_matches_keep_first_tie_and_forward_ratio_behavior() {
        let zero = [0; 32];
        let one = desc_from_bit(1);
        let two = desc_from_bit(3);
        assert_eq!(
            match_descriptors(&[one, one], &[zero, zero], None, true, None),
            vec![(0, 0)]
        );
        assert_eq!(
            match_descriptors(&[zero], &[zero, zero], None, false, Some(0.5)),
            vec![]
        );
        assert!(match_descriptors(&[one], &[zero, zero], None, false, Some(0.99)).is_empty());
        assert_eq!(
            match_descriptors(&[one], &[zero], Some(1), false, Some(0.5)),
            vec![(0, 0)]
        );
        assert!(match_descriptors(&[one], &[zero], Some(0), false, None).is_empty());
        // A reverse tie does not apply a second ratio test.
        assert_eq!(
            match_descriptors(&[zero, zero], &[one, two], None, true, Some(0.75)),
            vec![(0, 0)]
        );
    }

    /// Generate `n` L2-normalised random 64-dim descriptors using a simple
    /// LCG so the test stays deterministic and dependency-light.
    fn random_normalised_descs(n: usize, seed: u64) -> Vec<[f32; 64]> {
        // xorshift64*: cheap, deterministic, no dep on rand.
        let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            // Map to [-1, 1).
            ((state as u32) as f32) / (u32::MAX as f32 / 2.0) - 1.0
        };
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let mut d = [0f32; 64];
            for v in d.iter_mut() {
                *v = next();
            }
            let norm: f32 = d.iter().map(|x| x * x).sum::<f32>().sqrt();
            let inv = if norm > 0.0 { 1.0 / norm } else { 1.0 };
            for v in d.iter_mut() {
                *v *= inv;
            }
            out.push(d);
        }
        out
    }

    fn l2_normalise(d: &mut [f32; 64]) {
        let norm: f32 = d.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            let inv = 1.0 / norm;
            for v in d.iter_mut() {
                *v *= inv;
            }
        }
    }

    #[test]
    fn match_f32_recovers_planted_pairs() {
        // Build two sets of 100 descriptors. First 50 of each are "matched
        // pairs" (small noise added on the second side); remaining 50 are
        // independent random outliers.
        let mut descs1 = random_normalised_descs(100, 0xDEAD_BEEF);
        let mut descs2 = random_normalised_descs(100, 0xCAFE_BABE);

        // Inject planted pairs: descs2[i] = descs1[i] + small noise, then
        // re-normalise. Tiny noise gives cos > 0.99 between planted pairs.
        // We share the same `state` machinery for noise.
        let mut state: u64 = 0xF00D_F00D_F00D_F00D;
        let mut noise = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state as u32) as f32) / (u32::MAX as f32 / 2.0) - 1.0
        };

        for i in 0..50 {
            let mut d = descs1[i];
            for v in d.iter_mut() {
                *v += 0.01 * noise();
            }
            l2_normalise(&mut d);
            descs2[i] = d;
        }
        // Re-normalise descs1 (LCG already gave unit vectors but cheap insurance).
        for d in descs1.iter_mut() {
            l2_normalise(d);
        }

        let matches = match_descriptors_f32::<64>(&descs1, &descs2, Some(0.95), true, Some(0.8));

        // We planted 50 pairs. With cos~0.99 between planted, ~random elsewhere,
        // cross-check + min_cossim=0.95 should recover most planted pairs and
        // reject outliers.
        assert!(
            matches.len() >= 45,
            "expected ~50 matches, got {}",
            matches.len()
        );
        // Inlier rate: planted match is (i, i) for i < 50.
        let correct = matches.iter().filter(|&&(i, j)| i == j && i < 50).count();
        let inlier_rate = correct as f32 / matches.len() as f32;
        assert!(
            inlier_rate >= 0.9,
            "inlier rate {} too low ({} correct of {})",
            inlier_rate,
            correct,
            matches.len()
        );
    }

    #[test]
    fn match_f32_empty_inputs() {
        let empty: Vec<[f32; 64]> = vec![];
        let some = random_normalised_descs(10, 1);
        assert!(match_descriptors_f32::<64>(&empty, &some, None, false, None).is_empty());
        assert!(match_descriptors_f32::<64>(&some, &empty, None, false, None).is_empty());
    }

    fn desc_from_bit(bit: u8) -> [u8; 32] {
        let mut d = [0u8; 32];
        d[0] = bit;
        d
    }

    #[test]
    fn by_projection_respects_radius_and_octave() {
        // Three predicted features at different octaves.
        let d_pred = vec![
            desc_from_bit(0x01),
            desc_from_bit(0x02),
            desc_from_bit(0x04),
        ];
        let pred_xy = vec![[100.0, 100.0], [50.0, 50.0], [200.0, 200.0]];
        let pred_oct = vec![0u8, 1, 2];

        // Current-frame candidates.
        // j=0: identical to pred[0], at (101,100), oct 0 — should match pred[0].
        // j=1: matches pred[1] descriptor at (55,51) oct 1 — within scaled radius.
        // j=2: matches pred[2] descriptor but at oct 0 (cross-octave) — must be rejected.
        // j=3: matches pred[2] descriptor at oct 2 but 100px away — radius reject.
        // j=4: matches pred[2] descriptor at (198,201) oct 2 — should match pred[2].
        let d_curr = vec![
            desc_from_bit(0x01),
            desc_from_bit(0x02),
            desc_from_bit(0x04),
            desc_from_bit(0x04),
            desc_from_bit(0x04),
        ];
        let curr_xy = vec![
            [101.0, 100.0],
            [55.0, 51.0],
            [200.0, 200.0],
            [100.0, 100.0],
            [198.0, 201.0],
        ];
        let curr_oct = vec![0u8, 1, 0, 2, 2];

        let cfg = ByProjectionConfig {
            base_radius: 10.0,
            scale_factors: vec![1.0, 1.2, 1.44],
            max_octave_diff: 1,
            max_distance: 256,
            max_ratio: 1.0, // disable ratio test — we want the spatial/octave gates only
        };

        let predicted = OrbFeaturesView {
            descriptors: &d_pred,
            keypoints_xy: &pred_xy,
            octaves: &pred_oct,
        };
        let observed = OrbFeaturesView {
            descriptors: &d_curr,
            keypoints_xy: &curr_xy,
            octaves: &curr_oct,
        };

        let mut matches = match_orb_by_projection(predicted, observed, &cfg);
        matches.sort();

        assert_eq!(matches, vec![(0, 0), (1, 1), (2, 4)]);
    }

    #[test]
    fn by_projection_empty_inputs() {
        let empty_desc: Vec<[u8; 32]> = vec![];
        let empty_xy: Vec<[f32; 2]> = vec![];
        let empty_oct: Vec<u8> = vec![];
        let cfg = ByProjectionConfig::default();

        let view = OrbFeaturesView {
            descriptors: &empty_desc,
            keypoints_xy: &empty_xy,
            octaves: &empty_oct,
        };
        let m = match_orb_by_projection(view, view, &cfg);
        assert!(m.is_empty());
    }
}
