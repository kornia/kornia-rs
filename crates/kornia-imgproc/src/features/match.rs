/// Hamming distance between two fixed-size byte descriptors.
///
/// Specialized for `N=32` (ORB's 256-bit descriptor) on aarch64 (NEON) and
/// x86_64 (POPCNT). Other widths and CPUs use scalar XOR and popcount.
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
        HammingKernel::Popcnt => {
            if let (Ok(a), Ok(b)) = (a.as_slice().try_into(), b.as_slice().try_into()) {
                // SAFETY: the shared selector confirms POPCNT; the converted
                // array references each contain exactly 32 valid bytes.
                return unsafe { hamming32_popcnt(a, b) };
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
    Popcnt,
    #[cfg(target_arch = "aarch64")]
    Neon,
}

#[inline]
fn hamming_kernel<const N: usize>() -> HammingKernel {
    if N == 32 {
        #[cfg(target_arch = "x86_64")]
        if crate::simd::cpu_features().has_popcnt {
            return HammingKernel::Popcnt;
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
#[inline(always)]
fn descriptor_words(descriptor: &[u8; 32]) -> [u64; 4] {
    let mut words = [0; 4];
    for (word, chunk) in words.iter_mut().zip(descriptor.as_chunks::<8>().0) {
        *word = u64::from_ne_bytes(*chunk);
    }
    words
}

#[cfg(target_arch = "x86_64")]
struct PopcntHamming {
    query: [u64; 4],
}

#[cfg(target_arch = "x86_64")]
impl PopcntHamming {
    // These helpers inherit the caller's target features once inlined. Only
    // the thin target_feature wrappers are called from baseline code.
    #[inline(always)]
    fn new(query: &[u8; 32]) -> Self {
        Self {
            query: descriptor_words(query),
        }
    }

    #[inline(always)]
    fn distance(&self, candidate: &[u8; 32]) -> u32 {
        let candidate = descriptor_words(candidate);
        (self.query[0] ^ candidate[0]).count_ones()
            + (self.query[1] ^ candidate[1]).count_ones()
            + (self.query[2] ^ candidate[2]).count_ones()
            + (self.query[3] ^ candidate[3]).count_ones()
    }
}

/// 32-byte Hamming distance as XOR and POPCNT over four 64-bit words.
///
/// The same word loop as OpenCV's `normHamming` under `CV_POPCNT_U64`
/// (`modules/core/src/stat.simd.hpp`).
///
/// # Safety
///
/// The caller must ensure POPCNT is available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "popcnt")]
unsafe fn hamming32_popcnt(a: &[u8; 32], b: &[u8; 32]) -> u32 {
    PopcntHamming::new(a).distance(b)
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
    // Select once per row, including on CPUs without POPCNT. No pair calls
    // go back through the public runtime dispatcher.
    match hamming_kernel::<N>() {
        #[cfg(target_arch = "x86_64")]
        HammingKernel::Popcnt => {
            if let Ok(query) = query.as_slice().try_into() {
                let (candidates, _) = candidates.as_flattened().as_chunks::<32>();
                if GATED {
                    // Sparse projection scans spend most of their time in
                    // scalar gates. Keep that loop in the baseline context;
                    // call the selected distance kernel only for eligible rows.
                    return scan_hamming::<32, SECOND, true>(
                        candidates,
                        |candidate| {
                            // SAFETY: selector confirms POPCNT; both arrays are 32 bytes.
                            unsafe { hamming32_popcnt(query, candidate) }
                        },
                        |batch| {
                            std::array::from_fn(|i| {
                                // SAFETY: selector confirms POPCNT; each array is 32 bytes.
                                unsafe { hamming32_popcnt(query, &batch[i]) }
                            })
                        },
                        gate,
                        visit,
                    );
                }
                // SAFETY: the selector confirms POPCNT and N == 32.
                return unsafe { hamming_row_popcnt::<SECOND>(query, candidates, gate, visit) };
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
/// The caller must ensure POPCNT is available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "popcnt")]
unsafe fn hamming_row_popcnt<const SECOND: bool>(
    query: &[u8; 32],
    candidates: &[[u8; 32]],
    gate: impl FnMut(usize) -> bool,
    visit: impl FnMut(usize, u32),
) -> (usize, u32, u32) {
    let kernel = PopcntHamming::new(query);
    let distance = |candidate: &[u8; 32]| kernel.distance(candidate);
    let distances4 = |batch: &[[u8; 32]; 4]| std::array::from_fn(|i| kernel.distance(&batch[i]));
    // Four-candidate batches help best-only scans; tracking the runner-up is
    // faster one candidate at a time (see bench_hamming_match).
    if SECOND {
        scan_hamming::<32, SECOND, true>(candidates, distance, distances4, gate, visit)
    } else {
        scan_hamming::<32, SECOND, false>(candidates, distance, distances4, gate, visit)
    }
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

// A task's column keys hold its query offsets in 16 bits.
const MAX_ROWS_PER_SMNN_TASK: usize = 1 << 16;

// Packed (distance, query offset) keys keep the earliest of tied queries. The
// runner-up only needs its distance, and any key with that distance gives it.
#[inline(always)]
fn update_columns_scalar(distances: &[u16], offset: u32, best: &mut [u32], second: &mut [u32]) {
    for ((&distance, best), second) in distances.iter().zip(best).zip(second) {
        let key = ((distance as u32) << 16) | offset;
        *second = (*second).min((*best).max(key));
        *best = (*best).min(key);
    }
}

/// [`update_columns_scalar`], autovectorized for AVX2's unsigned 32-bit min and max.
///
/// # Safety
///
/// The caller must ensure AVX2 is available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn update_columns_avx2(
    distances: &[u16],
    offset: u32,
    best: &mut [u32],
    second: &mut [u32],
) {
    update_columns_scalar(distances, offset, best, second)
}

#[inline]
fn update_columns(distances: &[u16], offset: u32, best: &mut [u32], second: &mut [u32]) {
    #[cfg(target_arch = "x86_64")]
    if crate::simd::cpu_features().has_avx2 {
        // SAFETY: the shared selector confirms AVX2.
        unsafe { update_columns_avx2(distances, offset, best, second) };
        return;
    }
    update_columns_scalar(distances, offset, best, second)
}

// Each column's nearest query as a packed (distance, query) key, and its
// runner-up distance.
struct ColumnNearest {
    best: Vec<u64>,
    second: Vec<u16>,
}

impl ColumnNearest {
    fn merge(mut self, other: Self) -> Self {
        let columns = self.best.iter_mut().zip(&mut self.second);
        for ((best, second), (other_best, other_second)) in
            columns.zip(other.best.into_iter().zip(other.second))
        {
            let larger_best = ((*best).max(other_best) >> 32) as u16;
            *second = (*second).min(other_second).min(larger_best);
            *best = (*best).min(other_best);
        }
        self
    }
}

/// One match of [`match_smnn`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SmnnMatch {
    /// Row in `descriptors1`.
    pub query: usize,
    /// Row in `descriptors2`.
    pub train: usize,
    /// The larger of the two nearest-to-second-nearest distance ratios.
    pub ratio: f32,
}

/// Match mutual nearest neighbors that pass Lowe's ratio test in both directions.
///
/// The contract of `kornia.feature.match_smnn`, on Hamming distances: a pair is kept
/// when the two descriptors are mutual nearest neighbors and
/// `nearest / second_nearest <= th` both from `descriptors1` to `descriptors2` and
/// back. Ratios are `f32`. One pass over the distance table serves both directions,
/// at about half the cost of two [`match_descriptors`] ratio scans.
///
/// Of tied nearest neighbors, the earliest row is the nearest. A tie has ratio 1, or
/// is not a number at distance zero, which `th` never accepts.
///
/// # Arguments
///
/// * `descriptors1` - First set of N-byte binary descriptors. `N` is at most 8,190.
/// * `descriptors2` - Second set of N-byte binary descriptors.
/// * `th` - Distance ratio threshold; `kornia.feature.match_smnn` defaults to 0.95.
///
/// # Returns
///
/// The matches, in the order of the rows of `descriptors1`. Sets with fewer than two
/// descriptors have no second neighbor and return no matches.
///
/// # Errors
///
/// This function does not return errors.
///
/// # Example
///
/// ```
/// use kornia_imgproc::features::match_smnn;
/// let a = [[0u8; 32], [255u8; 32]];
/// let b = [[255u8; 32], [1u8; 32]];
/// let matches = match_smnn(&a, &b, 0.95);
/// assert_eq!((matches[0].query, matches[0].train), (0, 1));
/// assert_eq!((matches[1].query, matches[1].train), (1, 0));
/// ```
pub fn match_smnn<const N: usize>(
    descriptors1: &[[u8; N]],
    descriptors2: &[[u8; N]],
    th: f32,
) -> Vec<SmnnMatch> {
    const {
        assert!(
            N * 8 < u16::MAX as usize,
            "Hamming distances must fit u16 below its sentinel"
        )
    };
    if descriptors1.len() < 2 || descriptors2.len() < 2 {
        return Vec::new();
    }

    use rayon::prelude::*;
    let rows_per_task = descriptors1
        .len()
        .div_ceil(rayon::current_num_threads() * 4)
        .min(MAX_ROWS_PER_SMNN_TASK);
    let mut rows = vec![(0, 0, 0); descriptors1.len()];
    let columns = rows
        .par_chunks_mut(rows_per_task)
        .zip(descriptors1.par_chunks(rows_per_task))
        .enumerate()
        .map(|(task, (results, queries))| {
            let mut distances = vec![0u16; descriptors2.len()];
            let mut best = vec![u32::MAX; descriptors2.len()];
            let mut second = vec![u32::MAX; descriptors2.len()];
            for (offset, (result, query)) in results.iter_mut().zip(queries).enumerate() {
                *result = hamming_row_scan::<N, true, false>(
                    query,
                    descriptors2,
                    |_| true,
                    |j, distance| distances[j] = distance as u16,
                );
                update_columns(&distances, offset as u32, &mut best, &mut second);
            }
            let first_query = (task * rows_per_task) as u64;
            ColumnNearest {
                best: best
                    .into_iter()
                    .map(|key| (((key >> 16) as u64) << 32) | (first_query + (key & 0xffff) as u64))
                    .collect(),
                second: second.into_iter().map(|key| (key >> 16) as u16).collect(),
            }
        })
        .reduce_with(ColumnNearest::merge);
    let Some(columns) = columns else {
        return Vec::new();
    };

    let mut matches = Vec::new();
    for (query, &(train, first, second)) in rows.iter().enumerate() {
        let column = columns.best[train];
        if column as u32 as usize != query {
            continue;
        }
        let forward = first as f32 / second as f32;
        let backward = (column >> 32) as f32 / columns.second[train] as f32;
        if forward <= th && backward <= th {
            matches.push(SmnnMatch {
                query,
                train,
                ratio: forward.max(backward),
            });
        }
    }
    matches
}

// Candidate tiles keep one tile of train rows resident in the L1 data cache while
// every query of a task passes over it.
const FLOAT_TILE_BYTES: usize = 32 * 1024;
const MIN_FLOAT_TILE_ROWS: usize = 8;

/// What a one-pass float scan minimises; cosine similarity is minimised negated.
#[derive(Clone, Copy, PartialEq)]
pub(crate) enum FloatMetric {
    SquaredL2,
    NegativeDot,
}

/// A query's earliest nearest train row, with its distance and the runner-up's.
/// `index` is `usize::MAX` and both distances are infinite if no distance was finite.
#[derive(Clone, Copy)]
pub(crate) struct RowNearest {
    pub(crate) index: usize,
    pub(crate) best: f32,
    pub(crate) second: f32,
}

/// Each train row's earliest nearest query, with its distance and the runner-up's.
#[derive(Default)]
pub(crate) struct ColumnsNearest {
    pub(crate) index: Vec<usize>,
    pub(crate) best: Vec<f32>,
    pub(crate) second: Vec<f32>,
}

impl ColumnsNearest {
    // `self` holds earlier queries than `other`, so a tie keeps its index.
    fn merge(mut self, other: Self) -> Self {
        for (j, (other_best, other_second)) in other.best.into_iter().zip(other.second).enumerate()
        {
            let larger = if other_best < self.best[j] {
                self.best[j]
            } else {
                other_best
            };
            self.second[j] = self.second[j].min(other_second).min(larger);
            if other_best < self.best[j] {
                self.best[j] = other_best;
                self.index[j] = other.index[j];
            }
        }
        self
    }
}

// Comparisons rather than `f32::min`: they vectorize to one instruction each, and a
// NaN distance fails all of them, so it changes nothing.
#[inline(always)]
fn update_float_columns(
    distances: &[f32],
    query: u32,
    best: &mut [f32],
    index: &mut [u32],
    second: &mut [f32],
) {
    let columns = best.iter_mut().zip(index).zip(second);
    for (&distance, ((best, index), second)) in distances.iter().zip(columns) {
        let larger = if distance < *best { *best } else { distance };
        *second = if larger < *second { larger } else { *second };
        *index = if distance < *best { query } else { *index };
        *best = if distance < *best { distance } else { *best };
    }
}

#[inline(always)]
fn merge_tile_into_row(row: &mut RowNearest, distances: &[f32], first_train: usize) {
    let (mut best, mut second, mut index) = (f32::INFINITY, f32::INFINITY, 0);
    for (j, &distance) in distances.iter().enumerate() {
        if distance < best {
            second = best;
            best = distance;
            index = j;
        } else if distance < second {
            second = distance;
        }
    }
    if best < row.best {
        row.second = row.best.min(second);
        row.best = best;
        row.index = first_train + index;
    } else if best < row.second {
        row.second = best;
    }
}

#[inline(always)]
fn scan_float_task(
    queries: &[f32],
    train: &[f32],
    dim: usize,
    columns: bool,
    rows: &mut [RowNearest],
    distance: impl Fn(&[f32], &[f32]) -> f32,
) -> (Vec<f32>, Vec<u32>, Vec<f32>) {
    let n_train = if columns { train.len() / dim } else { 0 };
    let (mut best, mut index, mut second) = (
        vec![f32::INFINITY; n_train],
        vec![0; n_train],
        vec![f32::INFINITY; n_train],
    );
    let tile_rows = (FLOAT_TILE_BYTES / (dim * size_of::<f32>())).max(MIN_FLOAT_TILE_ROWS);
    let mut distances = vec![0.0; tile_rows];
    for (tile, candidates) in train.chunks(tile_rows * dim).enumerate() {
        let first_train = tile * tile_rows;
        let distances = &mut distances[..candidates.len() / dim];
        for (offset, (query, row)) in queries.chunks_exact(dim).zip(rows.iter_mut()).enumerate() {
            for (out, candidate) in distances.iter_mut().zip(candidates.chunks_exact(dim)) {
                *out = distance(query, candidate);
            }
            merge_tile_into_row(row, distances, first_train);
            if columns {
                let range = first_train..first_train + distances.len();
                update_float_columns(
                    distances,
                    offset as u32,
                    &mut best[range.clone()],
                    &mut index[range.clone()],
                    &mut second[range],
                );
            }
        }
    }
    (best, index, second)
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn horizontal_sum_avx(sum: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    // SAFETY: caller guarantees AVX; these intrinsics only use registers.
    unsafe {
        let x = _mm_add_ps(_mm256_castps256_ps128(sum), _mm256_extractf128_ps(sum, 1));
        let x = _mm_add_ps(x, _mm_movehl_ps(x, x));
        _mm_cvtss_f32(_mm_add_ss(x, _mm_shuffle_ps(x, x, 1)))
    }
}

/// Squared L2 over the common prefix of two rows.
///
/// Four accumulators, because one would be a chain of FMAs bound by their latency.
///
/// # Safety
///
/// The caller must ensure AVX2 and FMA are available.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub(crate) unsafe fn squared_l2_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let (a, b) = (&a[..a.len().min(b.len())], &b[..a.len().min(b.len())]);
    let (wide, rest) = (a.len() / 32 * 32, a.len() / 8 * 8);
    // SAFETY: caller guarantees AVX2 and FMA. Each 8-lane load starts at most 8 floats
    // before `wide` or `rest`, which are within both equally long slices.
    unsafe {
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        let difference =
            |k: usize| _mm256_sub_ps(_mm256_loadu_ps(pa.add(k)), _mm256_loadu_ps(pb.add(k)));
        let mut acc = [_mm256_setzero_ps(); 4];
        for k in (0..wide).step_by(32) {
            for (lane, acc) in acc.iter_mut().enumerate() {
                let d = difference(k + lane * 8);
                *acc = _mm256_fmadd_ps(d, d, *acc);
            }
        }
        for k in (wide..rest).step_by(8) {
            let d = difference(k);
            acc[0] = _mm256_fmadd_ps(d, d, acc[0]);
        }
        let sum = _mm256_add_ps(_mm256_add_ps(acc[0], acc[1]), _mm256_add_ps(acc[2], acc[3]));
        let tail = (rest..a.len())
            .map(|k| (a[k] - b[k]) * (a[k] - b[k]))
            .sum::<f32>();
        horizontal_sum_avx(sum) + tail
    }
}

/// Dot product over the common prefix of two rows.
///
/// # Safety
///
/// The caller must ensure AVX2 and FMA are available.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let (a, b) = (&a[..a.len().min(b.len())], &b[..a.len().min(b.len())]);
    let (wide, rest) = (a.len() / 32 * 32, a.len() / 8 * 8);
    // SAFETY: caller guarantees AVX2 and FMA. Each 8-lane load starts at most 8 floats
    // before `wide` or `rest`, which are within both equally long slices.
    unsafe {
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        let mut acc = [_mm256_setzero_ps(); 4];
        for k in (0..wide).step_by(32) {
            for (lane, acc) in acc.iter_mut().enumerate() {
                let at = k + lane * 8;
                *acc = _mm256_fmadd_ps(
                    _mm256_loadu_ps(pa.add(at)),
                    _mm256_loadu_ps(pb.add(at)),
                    *acc,
                );
            }
        }
        for k in (wide..rest).step_by(8) {
            acc[0] = _mm256_fmadd_ps(
                _mm256_loadu_ps(pa.add(k)),
                _mm256_loadu_ps(pb.add(k)),
                acc[0],
            );
        }
        let sum = _mm256_add_ps(_mm256_add_ps(acc[0], acc[1]), _mm256_add_ps(acc[2], acc[3]));
        let tail = (rest..a.len()).map(|k| a[k] * b[k]).sum::<f32>();
        horizontal_sum_avx(sum) + tail
    }
}

/// Fixed-width [`squared_l2_avx2`]: `D` is a multiple of 32, so no tail or bounds remain.
///
/// # Safety
///
/// The caller must ensure AVX2 and FMA are available, and both rows hold `D` floats.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn squared_l2_fixed_avx2<const D: usize>(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    // SAFETY: caller guarantees AVX2, FMA and `D` floats per row; `k + 32 <= D`.
    unsafe {
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        let mut acc = [_mm256_setzero_ps(); 4];
        for k in (0..D).step_by(32) {
            for (lane, acc) in acc.iter_mut().enumerate() {
                let at = k + lane * 8;
                let d = _mm256_sub_ps(_mm256_loadu_ps(pa.add(at)), _mm256_loadu_ps(pb.add(at)));
                *acc = _mm256_fmadd_ps(d, d, *acc);
            }
        }
        horizontal_sum_avx(_mm256_add_ps(
            _mm256_add_ps(acc[0], acc[1]),
            _mm256_add_ps(acc[2], acc[3]),
        ))
    }
}

/// Fixed-width [`dot_avx2`]: `D` is a multiple of 32, so no tail or bounds remain.
///
/// # Safety
///
/// The caller must ensure AVX2 and FMA are available, and both rows hold `D` floats.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn dot_fixed_avx2<const D: usize>(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    // SAFETY: caller guarantees AVX2, FMA and `D` floats per row; `k + 32 <= D`.
    unsafe {
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        let mut acc = [_mm256_setzero_ps(); 4];
        for k in (0..D).step_by(32) {
            for (lane, acc) in acc.iter_mut().enumerate() {
                let at = k + lane * 8;
                *acc = _mm256_fmadd_ps(
                    _mm256_loadu_ps(pa.add(at)),
                    _mm256_loadu_ps(pb.add(at)),
                    *acc,
                );
            }
        }
        horizontal_sum_avx(_mm256_add_ps(
            _mm256_add_ps(acc[0], acc[1]),
            _mm256_add_ps(acc[2], acc[3]),
        ))
    }
}

/// # Safety
///
/// The caller must ensure AVX2 and FMA are available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn scan_float_task_avx2(
    queries: &[f32],
    train: &[f32],
    dim: usize,
    columns: bool,
    rows: &mut [RowNearest],
    metric: FloatMetric,
) -> (Vec<f32>, Vec<u32>, Vec<f32>) {
    // SAFETY, every closure: the caller guarantees AVX2 and FMA, and rows come from
    // `chunks_exact(dim)`, so fixed-width kernels read exactly `dim` floats.
    macro_rules! scan {
        ($kernel:expr) => {
            scan_float_task(queries, train, dim, columns, rows, $kernel)
        };
    }
    match (metric, dim) {
        (FloatMetric::SquaredL2, 64) => scan!(|q, c| unsafe { squared_l2_fixed_avx2::<64>(q, c) }),
        (FloatMetric::SquaredL2, 128) => {
            scan!(|q, c| unsafe { squared_l2_fixed_avx2::<128>(q, c) })
        }
        (FloatMetric::SquaredL2, 256) => {
            scan!(|q, c| unsafe { squared_l2_fixed_avx2::<256>(q, c) })
        }
        (FloatMetric::SquaredL2, _) => scan!(|q, c| unsafe { squared_l2_avx2(q, c) }),
        (FloatMetric::NegativeDot, 64) => scan!(|q, c| -unsafe { dot_fixed_avx2::<64>(q, c) }),
        (FloatMetric::NegativeDot, 128) => scan!(|q, c| -unsafe { dot_fixed_avx2::<128>(q, c) }),
        (FloatMetric::NegativeDot, 256) => scan!(|q, c| -unsafe { dot_fixed_avx2::<256>(q, c) }),
        (FloatMetric::NegativeDot, _) => scan!(|q, c| -unsafe { dot_avx2(q, c) }),
    }
}

fn scan_float_task_portable(
    queries: &[f32],
    train: &[f32],
    dim: usize,
    columns: bool,
    rows: &mut [RowNearest],
    metric: FloatMetric,
    scalar: bool,
) -> (Vec<f32>, Vec<u32>, Vec<f32>) {
    use super::sift::{l2_sq, l2_sq_scalar};
    let dot = |q: &[f32], c: &[f32]| q.iter().zip(c).map(|(x, y)| x * y).sum::<f32>();
    match (metric, scalar) {
        (FloatMetric::SquaredL2, true) => {
            scan_float_task(queries, train, dim, columns, rows, l2_sq_scalar)
        }
        (FloatMetric::SquaredL2, false) => {
            scan_float_task(queries, train, dim, columns, rows, l2_sq)
        }
        (FloatMetric::NegativeDot, _) => {
            scan_float_task(queries, train, dim, columns, rows, |q, c| -dot(q, c))
        }
    }
}

/// One pass over the float distance table: every query's nearest train rows and, when
/// `columns` is set, every train row's nearest queries. `scalar` forces the portable
/// kernels on every CPU. Rows beyond the last whole `dim` are ignored; `dim` is not 0.
pub(crate) fn scan_float_rows(
    queries: &[f32],
    train: &[f32],
    dim: usize,
    metric: FloatMetric,
    columns: bool,
    scalar: bool,
) -> (Vec<RowNearest>, ColumnsNearest) {
    use rayon::prelude::*;
    let n_queries = queries.len() / dim;
    let train = &train[..train.len() / dim * dim];
    let unset = RowNearest {
        index: usize::MAX,
        best: f32::INFINITY,
        second: f32::INFINITY,
    };
    let mut rows = vec![unset; n_queries];
    if n_queries == 0 {
        return (rows, ColumnsNearest::default());
    }
    let rows_per_task = n_queries
        .div_ceil(rayon::current_num_threads() * 4)
        .min(u32::MAX as usize);
    #[cfg(target_arch = "x86_64")]
    let simd = !scalar && {
        let cpu = crate::simd::cpu_features();
        cpu.has_avx2 && cpu.has_fma
    };
    let nearest = rows
        .par_chunks_mut(rows_per_task)
        .zip(queries[..n_queries * dim].par_chunks(rows_per_task * dim))
        .enumerate()
        .map(|(task, (rows, chunk))| {
            let first_query = task * rows_per_task;
            #[cfg(target_arch = "x86_64")]
            let (best, index, second) = if simd {
                // SAFETY: `simd` confirms AVX2 and FMA.
                unsafe { scan_float_task_avx2(chunk, train, dim, columns, rows, metric) }
            } else {
                scan_float_task_portable(chunk, train, dim, columns, rows, metric, scalar)
            };
            #[cfg(not(target_arch = "x86_64"))]
            let (best, index, second) =
                scan_float_task_portable(chunk, train, dim, columns, rows, metric, scalar);
            ColumnsNearest {
                index: index
                    .into_iter()
                    .map(|i| first_query + i as usize)
                    .collect(),
                best,
                second,
            }
        })
        .reduce_with(ColumnsNearest::merge)
        .unwrap_or_default();
    (rows, nearest)
}

/// Match mutual nearest neighbors of float descriptors that pass Lowe's ratio test in
/// both directions, on L2 distances.
///
/// The contract of `kornia.feature.match_smnn`: a pair is kept when the two descriptors
/// are mutual nearest neighbors and `nearest / second_nearest <= th` both from
/// `descriptors1` to `descriptors2` and back, with the ratio taken of the square roots
/// of the two `f32` squared distances. One pass over the distance table serves both
/// directions, in tiles of train rows that stay in the L1 data cache.
///
/// Of tied nearest neighbors, the earliest row is the nearest. A NaN distance is never
/// nearest. Summation order can differ from another library's, so distances of
/// non-integer descriptors may differ by rounding in the last bits.
///
/// # Arguments
///
/// * `descriptors1` - First set, `dim` floats per row, row after row.
/// * `descriptors2` - Second set, in the same layout. A trailing partial row is ignored.
/// * `dim` - Floats per descriptor. `0` returns no matches.
/// * `th` - Distance ratio threshold; `kornia.feature.match_smnn` defaults to 0.95.
///
/// # Returns
///
/// The matches, in the order of the rows of `descriptors1`. Sets with fewer than two
/// descriptors have no second neighbor and return no matches.
///
/// # Errors
///
/// This function does not return errors.
///
/// # Example
///
/// ```
/// use kornia_imgproc::features::match_smnn_f32;
/// let a = [0.0, 0.0, 9.0, 9.0];
/// let b = [9.0, 9.0, 1.0, 0.0];
/// let matches = match_smnn_f32(&a, &b, 2, 0.95);
/// assert_eq!((matches[0].query, matches[0].train), (0, 1));
/// assert_eq!((matches[1].query, matches[1].train), (1, 0));
/// ```
pub fn match_smnn_f32(
    descriptors1: &[f32],
    descriptors2: &[f32],
    dim: usize,
    th: f32,
) -> Vec<SmnnMatch> {
    if dim == 0 || descriptors1.len() / dim < 2 || descriptors2.len() / dim < 2 {
        return Vec::new();
    }
    let (rows, columns) = scan_float_rows(
        descriptors1,
        descriptors2,
        dim,
        FloatMetric::SquaredL2,
        true,
        false,
    );
    let mut matches = Vec::new();
    for (query, row) in rows.iter().enumerate() {
        if row.index == usize::MAX || columns.index[row.index] != query {
            continue;
        }
        let train = row.index;
        let forward = row.best.sqrt() / row.second.sqrt();
        let backward = columns.best[train].sqrt() / columns.second[train].sqrt();
        if forward <= th && backward <= th {
            matches.push(SmnnMatch {
                query,
                train,
                ratio: forward.max(backward),
            });
        }
    }
    matches
}

/// Cosine similarity (dot product) between two L2-normalised f32 descriptors.
///
/// Assumes both inputs are unit-norm; the dot product is then the cosine.
/// LLVM autovectorises the loop for the build's baseline on fixed-size arrays
/// such as `[f32; 64]` (XFeat) or `[f32; 256]` (SuperPoint).
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
/// Uses rayon for parallelism across descriptors1, in one pass over the table
/// that also serves the mutual check. Descriptors must be L2-normalised. x86_64
/// dispatches to AVX2 and FMA; the portable loop is autovectorised by LLVM.
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

    // One pass serves both directions: every cosine of the mutual check is one the
    // forward search computes. Distances are negated cosines.
    let (rows, columns) = scan_float_rows(
        descriptors1.as_flattened(),
        descriptors2.as_flattened(),
        D,
        FloatMetric::NegativeDot,
        cross_check,
        false,
    );
    let fwd: Vec<(usize, f32, f32)> = rows
        .iter()
        .map(|row| {
            let index = if row.index == usize::MAX {
                0
            } else {
                row.index
            };
            (index, -row.best, -row.second)
        })
        .collect();
    let rev_best_i = cross_check.then_some(columns.index);

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

    type KorniaMatch = (usize, usize, f32);

    fn hex_rows(rows: &[&str]) -> Vec<[u8; 32]> {
        rows.iter()
            .map(|row| {
                std::array::from_fn(|i| {
                    u8::from_str_radix(&row[i * 2..i * 2 + 2], 16).expect("test fixture is hex")
                })
            })
            .collect()
    }

    // Values from kornia.feature.match_smnn (kornia 5310f2589), given the Hamming
    // table of these descriptors as `dm`: 12 x 9 random rows, with near copies at
    // 10, 40 and 90 flipped bits, one exact copy, and one target stored twice.
    #[test]
    fn match_smnn_returns_the_values_of_kornia() {
        let a = hex_rows(&[
            "babbb6faf3fe08fa81ace9fcf7815461b43cde2b7cdd59ec8e3eda1a5246fe42",
            "841e87234658b051fa763a891f3a3b1e8751bdcb5265e43d9bf23ac986708b51",
            "dc4b22cbbce5cdf6fb0e55f5b68d7e4330df3c8ef7c6e570023835414d7e209c",
            "5d46a8e5084916dd74a85ea503391fddcd59ba849105c5acad3280f47786eda8",
            "7027190cd49d5abcf2a919bd3c6206394868f9dd8b870c2ca566e9c90084d3de",
            "3e969b7b403f650fcbaf8e29813c06af17eaa30a3c41d6ab24acc523a6ac6b9c",
            "e51e7a8caf28650f6d51b12d4d164ffa660e476741225f7064b00a74b7255888",
            "5c9b6b311047cd005dd64ea5800a53401c635ba88007c6db81ed4f6e4d5ae06c",
            "7ba40d2d33a25ebc1422ebb235060becc997a5c651154a27ee13b1cd6bb304fe",
            "b38d8d5bb44bad2e301d19da393dabf0abcb92f9c8f73d16a3cf03e14ca8dc77",
            "5e30886edea238d4333dbde7850af347a34596c201c87cde90b386ddc56ff6f9",
            "05875d54d78b7ad7b016dd6c5961e972844be6c1f4a3d95eabe6d2c06639907b",
        ]);
        let b = hex_rows(&[
            "5d47a8e51849165d74285ea503390f9dcd59ba849105c5bcad3281f477a6eca8",
            "3e969b7b403f650fcbaf8e29813c06af17eaa30a3c41d6ab24acc523a6ac6b9c",
            "5c8ffb3100cf8d425edf4ead990b5340b8635b80d817f6da81e54f6cc55bc3ec",
            "f0cfbcbbb6ce8cb1b93f6c5a9d9f89323219f36b59efb03726ff10675c99f947",
            "841e8f234650b050da763aa91fba3b1a87599fdb527de43d9bf33ee983688b51",
            "841e8f234650b050da763aa91fba3b1a87599fdb527de43d9bf33ee983688b51",
            "f316389e0ba187338a27906bb89cc241cc981d07c40d3b57d7deb3c15f89910d",
            "23a6e7fbab249553897299be7c9ac732605b2d1ea56d47d435a8418b2e671aa3",
            "d953ea6aad71df15dc19ba05366f5c38c21c5512fb8c61aebb2531591ab36822",
        ]);
        let expected: [(f32, &[KorniaMatch]); 3] = [
            (
                0.5f32,
                &[(3, 0, 0.09090909f32), (5, 1, 0.0f32), (7, 2, 0.3478261f32)],
            ),
            (
                0.8f32,
                &[
                    (3, 0, 0.09090909f32),
                    (5, 1, 0.0f32),
                    (7, 2, 0.3478261f32),
                    (9, 3, 0.79646015f32),
                ],
            ),
            (
                0.95f32,
                &[
                    (3, 0, 0.09090909f32),
                    (5, 1, 0.0f32),
                    (6, 7, 0.92561984f32),
                    (7, 2, 0.3478261f32),
                    (9, 3, 0.79646015f32),
                ],
            ),
        ];
        for (th, pairs) in expected {
            let got: Vec<_> = match_smnn(&a, &b, th)
                .into_iter()
                .map(|m| (m.query, m.train, m.ratio))
                .collect();
            assert_eq!(got, pairs, "th={th}");
        }
    }

    fn reference_smnn<const N: usize>(a: &[[u8; N]], b: &[[u8; N]], th: f32) -> Vec<SmnnMatch> {
        if a.len() < 2 || b.len() < 2 {
            return Vec::new();
        }
        let table: Vec<Vec<u32>> = a
            .iter()
            .map(|x| {
                b.iter()
                    .map(|y| x.iter().zip(y).map(|(p, q)| (p ^ q).count_ones()).sum())
                    .collect()
            })
            .collect();
        let top2 = |values: Vec<u32>| {
            let mut order: Vec<usize> = (0..values.len()).collect();
            order.sort_by_key(|&k| (values[k], k));
            (order[0], values[order[0]], values[order[1]])
        };
        let mut out = Vec::new();
        for (i, row) in table.iter().enumerate() {
            let (j, d1, d2) = top2(row.clone());
            let (back, e1, e2) = top2(table.iter().map(|r| r[j]).collect());
            let (r1, r2) = (d1 as f32 / d2 as f32, e1 as f32 / e2 as f32);
            if back == i && r1 <= th && r2 <= th {
                out.push(SmnnMatch {
                    query: i,
                    train: j,
                    ratio: r1.max(r2),
                });
            }
        }
        out
    }

    fn check_smnn_against_reference<const N: usize>() {
        // A few repeated rows force the ties and zero distances random rows never reach.
        let with_duplicates = |n: usize, seed: u64| {
            let unique = random_binary_descs::<N>(n / 2 + 1, seed);
            (0..n)
                .map(|i| unique[(i * 7) % unique.len()])
                .collect::<Vec<_>>()
        };
        for threads in [1, 4] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("test thread pool");
            for (n1, n2) in [
                (0, 5),
                (1, 5),
                (5, 1),
                (2, 2),
                (3, 17),
                (64, 40),
                (129, 300),
            ] {
                for (a, b) in [
                    (
                        random_binary_descs::<N>(n1, 11),
                        random_binary_descs::<N>(n2, 13),
                    ),
                    (with_duplicates(n1, 17), with_duplicates(n2, 17)),
                ] {
                    for th in [0.0, 0.5, 0.8, 0.95, 1.0, 2.0, f32::NAN] {
                        let got = pool.install(|| match_smnn(&a, &b, th));
                        assert_eq!(got, reference_smnn(&a, &b, th), "{n1}x{n2} th={th}");
                    }
                }
            }
        }
    }

    #[test]
    fn match_smnn_agrees_with_the_reference_across_widths() {
        check_smnn_against_reference::<16>();
        check_smnn_against_reference::<32>();
        check_smnn_against_reference::<64>();
    }

    #[test]
    fn match_smnn_rejects_zero_distance_ties_at_any_threshold() {
        let a = [[0u8; 32], [255u8; 32]];
        let b = [[0u8; 32], [0u8; 32], [255u8; 32]];
        // Row 0 ties at distance zero (0/0 is not <= th); row 1 is unambiguous.
        for th in [0.5, 1.0, 100.0] {
            assert_eq!(
                match_smnn(&a, &b, th),
                vec![SmnnMatch {
                    query: 1,
                    train: 2,
                    ratio: 0.0
                }],
            );
        }
    }

    #[test]
    fn match_smnn_keeps_query_indices_past_one_task() {
        // One thread and this many rows reach the 16-bit limit on a task's rows, so
        // both matches below cross it: the earliest row must win a tie across tasks,
        // and a row of a later task must report its real index.
        let mut a = vec![[0xaau8; 32]; (1 << 18) + 9];
        let near = (1 << 16) + 1;
        a[near] = [0x55u8; 32];
        let mut b = [[0x0fu8; 32], [0x55u8; 32]];
        b[1][0] ^= 1;
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("test thread pool");
        assert_eq!(
            pool.install(|| match_smnn(&a, &b, 1.0)),
            vec![
                SmnnMatch {
                    query: 0,
                    train: 0,
                    ratio: 1.0
                },
                SmnnMatch {
                    query: near,
                    train: 1,
                    ratio: 1.0 / 128.0
                },
            ],
        );
    }

    #[test]
    fn column_updates_agree_across_kernels() {
        let rows: [[u16; 9]; 3] = [
            [5, 1, 9, 256, 0, 7, 7, 3, 2],
            [5, 2, 8, 255, 0, 6, 8, 3, 1],
            [4, 1, 9, 256, 1, 7, 6, 2, 2],
        ];
        let mut scalar = ([u32::MAX; 9], [u32::MAX; 9]);
        let mut selected = scalar;
        for (offset, row) in rows.iter().enumerate() {
            update_columns_scalar(row, offset as u32, &mut scalar.0, &mut scalar.1);
            update_columns(row, offset as u32, &mut selected.0, &mut selected.1);
        }
        assert_eq!(scalar, selected);
        // Column 0 saw 5, 5, 4: nearest is row 2, runner-up distance 5.
        assert_eq!((scalar.0[0], scalar.1[0] >> 16), ((4 << 16) | 2, 5));
    }

    // Values from kornia.feature.match_smnn (kornia 5310f2589) on these descriptors,
    // with its own L2 distances: 12 x 9 integer-valued rows of 16 dimensions, with
    // near copies, one exact copy and one target stored twice.
    #[test]
    fn match_smnn_f32_returns_the_values_of_kornia() {
        let a: [[f32; 16]; 12] = [
            [
                245.0, 129.0, 193.0, 144.0, 45.0, 131.0, 241.0, 248.0, 247.0, 157.0, 80.0, 145.0,
                166.0, 73.0, 246.0, 141.0,
            ],
            [
                90.0, 119.0, 27.0, 156.0, 87.0, 238.0, 76.0, 62.0, 154.0, 79.0, 87.0, 100.0, 176.0,
                69.0, 232.0, 89.0,
            ],
            [
                198.0, 239.0, 177.0, 96.0, 170.0, 198.0, 195.0, 10.0, 23.0, 76.0, 90.0, 179.0,
                23.0, 115.0, 203.0, 227.0,
            ],
            [
                16.0, 111.0, 76.0, 159.0, 212.0, 149.0, 150.0, 153.0, 75.0, 167.0, 43.0, 130.0,
                232.0, 49.0, 22.0, 17.0,
            ],
            [
                7.0, 64.0, 71.0, 24.0, 213.0, 28.0, 72.0, 71.0, 144.0, 151.0, 110.0, 87.0, 170.0,
                199.0, 119.0, 247.0,
            ],
            [
                196.0, 84.0, 245.0, 60.0, 176.0, 176.0, 78.0, 146.0, 88.0, 197.0, 77.0, 157.0,
                130.0, 71.0, 228.0, 233.0,
            ],
            [
                129.0, 146.0, 70.0, 183.0, 213.0, 172.0, 113.0, 197.0, 86.0, 192.0, 44.0, 105.0,
                163.0, 9.0, 190.0, 124.0,
            ],
            [
                129.0, 229.0, 80.0, 37.0, 35.0, 200.0, 92.0, 167.0, 14.0, 180.0, 223.0, 55.0,
                228.0, 19.0, 92.0, 68.0,
            ],
            [
                73.0, 159.0, 194.0, 99.0, 59.0, 65.0, 94.0, 101.0, 251.0, 186.0, 112.0, 65.0, 2.0,
                139.0, 3.0, 104.0,
            ],
            [
                60.0, 254.0, 255.0, 218.0, 128.0, 184.0, 107.0, 140.0, 24.0, 101.0, 131.0, 173.0,
                89.0, 215.0, 81.0, 35.0,
            ],
            [
                220.0, 10.0, 102.0, 117.0, 3.0, 69.0, 212.0, 226.0, 105.0, 67.0, 69.0, 233.0,
                242.0, 128.0, 116.0, 213.0,
            ],
            [
                74.0, 116.0, 114.0, 183.0, 12.0, 250.0, 192.0, 42.0, 188.0, 66.0, 104.0, 131.0,
                176.0, 96.0, 56.0, 205.0,
            ],
        ];
        let b: [[f32; 16]; 9] = [
            [
                17.0, 113.0, 79.0, 162.0, 214.0, 148.0, 153.0, 154.0, 74.0, 166.0, 44.0, 127.0,
                232.0, 47.0, 22.0, 19.0,
            ],
            [
                196.0, 84.0, 245.0, 60.0, 176.0, 176.0, 78.0, 146.0, 88.0, 197.0, 77.0, 157.0,
                130.0, 71.0, 228.0, 233.0,
            ],
            [
                143.0, 226.0, 71.0, 25.0, 9.0, 186.0, 68.0, 155.0, 0.0, 163.0, 248.0, 79.0, 239.0,
                36.0, 53.0, 76.0,
            ],
            [
                106.0, 255.0, 255.0, 148.0, 131.0, 245.0, 24.0, 94.0, 0.0, 31.0, 48.0, 177.0,
                111.0, 190.0, 14.0, 73.0,
            ],
            [
                86.0, 124.0, 24.0, 163.0, 87.0, 232.0, 69.0, 65.0, 162.0, 76.0, 93.0, 103.0, 175.0,
                73.0, 235.0, 95.0,
            ],
            [
                86.0, 124.0, 24.0, 163.0, 87.0, 232.0, 69.0, 65.0, 162.0, 76.0, 93.0, 103.0, 175.0,
                73.0, 235.0, 95.0,
            ],
            [
                220.0, 226.0, 59.0, 245.0, 47.0, 65.0, 209.0, 62.0, 228.0, 14.0, 15.0, 191.0,
                195.0, 95.0, 27.0, 171.0,
            ],
            [
                206.0, 205.0, 226.0, 65.0, 236.0, 254.0, 46.0, 153.0, 119.0, 228.0, 39.0, 131.0,
                104.0, 233.0, 220.0, 124.0,
            ],
            [
                223.0, 207.0, 13.0, 96.0, 197.0, 0.0, 77.0, 185.0, 168.0, 29.0, 59.0, 241.0, 118.0,
                144.0, 149.0, 92.0,
            ],
        ];
        let expected: [(f32, &[KorniaMatch]); 3] = [
            (
                0.5f32,
                &[
                    (3, 0, 0.029885547f32),
                    (5, 1, 0.0f32),
                    (7, 2, 0.21327893f32),
                ],
            ),
            (
                0.8f32,
                &[
                    (3, 0, 0.029885547f32),
                    (5, 1, 0.0f32),
                    (7, 2, 0.21327893f32),
                    (9, 3, 0.5456152f32),
                ],
            ),
            (
                0.95f32,
                &[
                    (3, 0, 0.029885547f32),
                    (5, 1, 0.0f32),
                    (7, 2, 0.21327893f32),
                    (9, 3, 0.5456152f32),
                ],
            ),
        ];
        for (th, pairs) in expected {
            let got: Vec<_> = match_smnn_f32(a.as_flattened(), b.as_flattened(), 16, th)
                .into_iter()
                .map(|m| (m.query, m.train, m.ratio))
                .collect();
            assert_eq!(got, pairs, "th={th}");
        }
    }

    fn integer_rows(n: usize, dim: usize, mut seed: u64, levels: u64) -> Vec<f32> {
        (0..n * dim)
            .map(|_| {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((seed >> 33) % levels) as f32
            })
            .collect()
    }

    fn reference_smnn_f32(a: &[f32], b: &[f32], dim: usize, th: f32) -> Vec<SmnnMatch> {
        let (n1, n2) = (a.len() / dim, b.len() / dim);
        if n1 < 2 || n2 < 2 {
            return Vec::new();
        }
        let table: Vec<Vec<f32>> = a
            .chunks_exact(dim)
            .map(|x| {
                b.chunks_exact(dim)
                    .map(|y| x.iter().zip(y).map(|(p, q)| (p - q) * (p - q)).sum())
                    .collect()
            })
            .collect();
        let top2 = |values: Vec<f32>| {
            let mut order: Vec<usize> = (0..values.len()).collect();
            order.sort_by(|&p, &q| values[p].total_cmp(&values[q]).then(p.cmp(&q)));
            (order[0], values[order[0]], values[order[1]])
        };
        let mut out = Vec::new();
        for (i, row) in table.iter().enumerate() {
            let (j, d1, d2) = top2(row.clone());
            let (back, e1, e2) = top2(table.iter().map(|r| r[j]).collect());
            let (r1, r2) = (d1.sqrt() / d2.sqrt(), e1.sqrt() / e2.sqrt());
            if back == i && r1 <= th && r2 <= th {
                out.push(SmnnMatch {
                    query: i,
                    train: j,
                    ratio: r1.max(r2),
                });
            }
        }
        out
    }

    #[test]
    fn match_smnn_f32_agrees_with_the_reference_across_dimensions() {
        // Integer values keep every sum exact, whatever the summation order. Few
        // levels force ties and zero distances; large sets cross several tiles.
        for dim in [1, 3, 8, 9, 16, 31, 32, 33, 64, 128, 130, 256] {
            for threads in [1, 4] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .expect("test thread pool");
                for (n1, n2, levels) in [
                    (0, 5, 256),
                    (1, 5, 256),
                    (5, 1, 256),
                    (2, 2, 2),
                    (17, 3, 3),
                    (70, 300, 256),
                    (129, 80, 2),
                ] {
                    let (a, b) = (
                        integer_rows(n1, dim, 11, levels),
                        integer_rows(n2, dim, 13, levels),
                    );
                    for th in [0.0, 0.5, 0.8, 0.95, 1.0, 2.0, f32::NAN] {
                        let got = pool.install(|| match_smnn_f32(&a, &b, dim, th));
                        assert_eq!(
                            got,
                            reference_smnn_f32(&a, &b, dim, th),
                            "{dim}D {n1}x{n2} levels={levels} th={th} threads={threads}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn match_smnn_f32_handles_degenerate_dimensions_and_non_finite_rows() {
        let a = [1.0f32, 2.0, 3.0, 4.0];
        assert!(match_smnn_f32(&a, &a, 0, 0.95).is_empty());
        // A trailing partial row is ignored: two whole rows on each side remain.
        assert_eq!(
            match_smnn_f32(&[0.0, 0.0, 9.0, 9.0, 5.0], &[0.0, 0.0, 9.0, 9.0], 2, 0.95).len(),
            2
        );
        // A NaN row has no finite distance, so it can't become anyone's nearest.
        let mut b = vec![0.0f32, 0.0, 5.0, 5.0, 9.0, 9.0];
        b[0] = f32::NAN;
        let got = match_smnn_f32(&[5.0, 5.0, 9.0, 9.0], &b, 2, 0.95);
        assert_eq!(
            got.iter().map(|m| (m.query, m.train)).collect::<Vec<_>>(),
            [(0, 1), (1, 2)]
        );
    }

    #[test]
    fn cosine_one_pass_agrees_with_the_reference() {
        // Small integers keep every dot product exact, whatever the summation order.
        fn rows<const D: usize>(n: usize, seed: u64) -> Vec<[f32; D]> {
            integer_rows(n, D, seed, 5).as_chunks::<D>().0.to_vec()
        }
        fn check<const D: usize>() {
            let (a, b) = (rows::<D>(150, 7), rows::<D>(90, 9));
            let dots: Vec<Vec<f32>> = a
                .iter()
                .map(|x| {
                    b.iter()
                        .map(|y| x.iter().zip(y).map(|(p, q)| p * q).sum())
                        .collect()
                })
                .collect();
            let top2 = |values: Vec<f32>| {
                let mut order: Vec<usize> = (0..values.len()).collect();
                order.sort_by(|&p, &q| values[q].total_cmp(&values[p]).then(p.cmp(&q)));
                (order[0], values[order[0]], values[order[1]])
            };
            for threads in [1, 4] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .expect("test thread pool");
                for (min_cos, cross, ratio) in [
                    (None, false, None),
                    (None, true, None),
                    (Some(5.0), true, None),
                    (None, true, Some(0.95)),
                    (None, false, Some(0.5)),
                ] {
                    let got = pool.install(|| match_descriptors_f32(&a, &b, min_cos, cross, ratio));
                    let mut want = Vec::new();
                    for (i, row) in dots.iter().enumerate() {
                        let (j, best, second) = top2(row.clone());
                        let back = top2(dots.iter().map(|r| r[j]).collect()).0;
                        let cos_ok = min_cos.is_none_or(|m| best >= m);
                        let ratio_ok =
                            ratio.is_none_or(|r| (1.0 - best) / (1.0 - second).max(1e-6) <= r);
                        if cos_ok && (!cross || back == i) && ratio_ok {
                            want.push((i, j));
                        }
                    }
                    assert_eq!(got, want, "{D}D {min_cos:?} {cross} {ratio:?} {threads}");
                }
            }
        }
        check::<16>();
        check::<19>();
        check::<64>();
    }

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
                if crate::simd::cpu_features().has_popcnt {
                    // SAFETY: POPCNT detected above and descriptors are 32 bytes.
                    assert_eq!(
                        unsafe {
                            hamming_row_popcnt::<true>(query, candidates, |_| true, |_, _| {})
                        },
                        (index, distance, second)
                    );
                    // SAFETY: POPCNT detected above and descriptors are 32 bytes.
                    assert_eq!(
                        unsafe {
                            hamming_row_popcnt::<false>(query, candidates, |_| true, |_, _| {})
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
