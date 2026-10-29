use super::downsample::test_access::sorted_words;
use super::*;
use crate::pointcloud::PointCloud;

type TestResult = Result<(), Box<dyn std::error::Error>>;

/// Deterministic xorshift64 points in `[lo, hi)³`.
pub(super) fn random_points(n: usize, lo: f64, hi: f64, seed: u64) -> Vec<[f64; 3]> {
    let mut s = seed.max(1);
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        lo + (s >> 11) as f64 / (1u64 << 53) as f64 * (hi - lo)
    };
    (0..n).map(|_| [next(), next(), next()]).collect()
}

fn unit_layout() -> Result<VoxelLayout, VoxelError> {
    VoxelLayout::new(1.0, [0.0; 3])
}

/// Brute-force reference: per voxel (BTreeMap, so key order), its point
/// indices in ascending order.
fn reference_runs(
    points: &[[f64; 3]],
    layout: &VoxelLayout,
) -> std::collections::BTreeMap<VoxelKey, Vec<usize>> {
    let mut runs = std::collections::BTreeMap::<VoxelKey, Vec<usize>>::new();
    for (i, p) in points.iter().enumerate() {
        if let Some(k) = layout.key_from_point(p) {
            runs.entry(k).or_default().push(i);
        }
    }
    runs
}

#[test]
fn layout_rejects_bad_arguments() {
    for s in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(matches!(
            VoxelLayout::new(s, [0.0; 3]),
            Err(VoxelError::InvalidVoxelSize(_))
        ));
    }
    assert!(matches!(
        VoxelLayout::new(1.0, [0.0, f64::NAN, 0.0]),
        Err(VoxelError::InvalidOrigin(_))
    ));
}

#[test]
fn keys_straddle_zero_and_follow_origin() -> TestResult {
    let l = VoxelLayout::new(0.25, [0.0; 3])?;
    assert_eq!(l.key_from_point(&[0.0, -0.0, 0.2499]), Some([0, 0, 0]));
    assert_eq!(
        l.key_from_point(&[-1e-12, -0.25, -0.2500001]),
        Some([-1, -1, -2])
    );
    assert_eq!(l.key_from_point(&[0.25, 0.5, -0.5]), Some([1, 2, -2]));
    let l = VoxelLayout::new(0.5, [1.0, -1.0, 0.25])?;
    assert_eq!(l.key_from_point(&[1.0, -1.0, 0.25]), Some([0, 0, 0]));
    assert_eq!(l.key_from_point(&[0.9, -1.1, 0.0]), Some([-1, -1, -1]));
    assert_eq!(l.centre_from_key(&[-1, -1, -1]), [0.75, -1.25, 0.0]);
    Ok(())
}

#[test]
fn keys_at_the_i32_limits() -> TestResult {
    let l = unit_layout()?;
    let max = i32::MAX as f64;
    let min = i32::MIN as f64;
    assert_eq!(
        l.key_from_point(&[max + 0.5, min, 0.0]),
        Some([i32::MAX, i32::MIN, 0])
    );
    assert_eq!(l.key_from_point(&[max + 1.0, 0.0, 0.0]), None);
    assert_eq!(l.key_from_point(&[min - 0.5, 0.0, 0.0]), None);
    assert_eq!(l.key_from_point(&[f64::NAN, 0.0, 0.0]), None);
    assert_eq!(l.key_from_point(&[0.0, f64::INFINITY, 0.0]), None);
    assert_eq!(l.key_from_point(&[1e300, 0.0, 0.0]), None);
    Ok(())
}

/// The reference rule the screened `axis_key` must reproduce exactly.
fn division_key(v: f64, o: f64, s: f64) -> Option<i32> {
    let f = ((v - o) / s).floor();
    (i32::MIN as f64..=i32::MAX as f64)
        .contains(&f)
        .then_some(f as i32)
}

/// Voxel boundaries `o + k · s` and their neighbouring doubles, where the
/// multiply estimate and the division can disagree, for voxel sizes whose
/// reciprocal is inexact, subnormal sizes (no reciprocal) and huge ones.
#[test]
fn screened_keys_equal_the_division_floor() -> TestResult {
    let sizes = [
        0.1,
        0.3,
        1.0 / 3.0,
        0.05,
        7.0,
        1e-7,
        1e-300,
        5e-324,
        1e300,
        1.7e308,
    ];
    let origins = [0.0, -0.05, 1.234_567_890_123, -1e6];
    let mut checked = 0usize;
    for &s in &sizes {
        for &o in &origins {
            let l = VoxelLayout::new(s, [o; 3])?;
            let r = l.inv_size();
            let mut probe = |v: f64| {
                assert_eq!(
                    axis_key(v, o, s, r),
                    division_key(v, o, s),
                    "v={v:e} o={o:e} s={s:e}"
                );
                checked += 1;
            };
            for k in [
                -2_000_000_000i64,
                -1_000_003,
                -10,
                -1,
                0,
                1,
                2,
                9,
                10,
                77,
                999_983,
            ] {
                let b = o + k as f64 * s;
                let mut v = b;
                for _ in 0..4 {
                    v = v.next_down();
                }
                for _ in 0..9 {
                    probe(v);
                    v = v.next_up();
                }
            }
            for v in [
                0.0,
                -0.0,
                1e-310,
                -1e-310,
                f64::MIN_POSITIVE,
                f64::NAN,
                f64::INFINITY,
            ] {
                probe(v);
            }
        }
    }
    let pts = random_points(20_000, -50.0, 50.0, 41);
    for &s in &[0.1, 0.07, 0.3] {
        let l = VoxelLayout::new(s, [0.0; 3])?;
        for p in pts.as_flattened() {
            assert_eq!(axis_key(*p, 0.0, s, l.inv_size()), division_key(*p, 0.0, s));
            checked += 1;
        }
    }
    assert!(checked > 60_000);
    Ok(())
}

#[test]
fn connectivity_offsets() {
    assert_eq!(VoxelConnectivity::Six.offsets().len(), 6);
    assert_eq!(VoxelConnectivity::Eighteen.offsets().len(), 18);
    assert_eq!(VoxelConnectivity::TwentySix.offsets().len(), 26);
    assert_eq!(
        VoxelConnectivity::Six.offsets(),
        &[
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, -1],
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0]
        ]
    );
    for c in [
        VoxelConnectivity::Six,
        VoxelConnectivity::Eighteen,
        VoxelConnectivity::TwentySix,
    ] {
        let o = c.offsets();
        assert!(o.windows(2).all(|w| w[0] < w[1]), "{c:?} not sorted");
        assert!(o.iter().all(|d| *d != [0, 0, 0]));
    }
}

#[test]
fn grid_insert_get_remove_neighbours() -> TestResult {
    let mut g = VoxelGrid::<i32>::with_capacity(VoxelLayout::new(0.1, [0.0; 3])?, 8);
    assert!(g.is_empty());
    assert_eq!(g.insert([0, 0, 0], 1), None);
    assert_eq!(g.insert([0, 0, 0], 2), Some(1));
    g.insert([1, 1, 1], 3);
    g.insert([0, 0, 1], 4);
    g.insert([-1, 0, 0], 5);
    *g.get_or_insert_with([5, 5, 5], || 10) += 1;
    assert_eq!(g.get(&[5, 5, 5]), Some(&11));
    if let Some(v) = g.get_mut(&[0, 0, 0]) {
        *v = 7;
    }
    assert_eq!(g.get(&[0, 0, 0]), Some(&7));
    assert_eq!(g.len(), 5);
    assert_eq!(
        g.sorted_keys(),
        vec![[-1, 0, 0], [0, 0, 0], [0, 0, 1], [1, 1, 1], [5, 5, 5]]
    );

    let n6: Vec<_> = g.neighbours([0, 0, 0], VoxelConnectivity::Six).collect();
    assert_eq!(n6, vec![([-1, 0, 0], &5), ([0, 0, 1], &4)]);
    let n18: Vec<_> = g
        .neighbours([0, 0, 0], VoxelConnectivity::Eighteen)
        .collect();
    assert_eq!(n18, n6);
    let n26: Vec<_> = g
        .neighbours([0, 0, 0], VoxelConnectivity::TwentySix)
        .collect();
    assert_eq!(
        n26,
        vec![([-1, 0, 0], &5), ([0, 0, 1], &4), ([1, 1, 1], &3)]
    );

    assert_eq!(g.remove(&[1, 1, 1]), Some(3));
    assert!(!g.contains_key(&[1, 1, 1]));
    assert_eq!(g.iter().count(), 4);
    for (_, v) in g.iter_mut() {
        *v = 0;
    }
    assert!(g.iter().all(|(_, &v)| v == 0));
    g.clear();
    assert!(g.is_empty());
    Ok(())
}

#[test]
fn grid_neighbours_skip_overflowing_keys() -> TestResult {
    let mut g = VoxelGrid::<()>::new(unit_layout()?);
    g.insert([i32::MAX, 0, 0], ());
    g.insert([i32::MAX - 1, 0, 0], ());
    g.insert([i32::MIN, 0, 0], ());
    let n: Vec<_> = g
        .neighbours([i32::MAX, 0, 0], VoxelConnectivity::TwentySix)
        .map(|(k, _)| k)
        .collect();
    assert_eq!(n, vec![[i32::MAX - 1, 0, 0]]);
    Ok(())
}

#[test]
fn downsample_exact_three_voxels() -> TestResult {
    let l = unit_layout()?;
    let points = [
        [1.25, 0.5, 0.5],  // (1, 0, 0)
        [0.25, 0.25, 0.5], // (0, 0, 0)
        [-0.5, 0.5, 0.5],  // (-1, 0, 0)
        [0.75, 0.75, 0.5], // (0, 0, 0)
        [1.75, 0.5, 0.5],  // (1, 0, 0)
        [0.5, 0.5, 0.5],   // (0, 0, 0): exactly at the centre
    ];
    let mut ds = VoxelDownsampler::new();
    let (mut out, mut index) = (Vec::new(), Vec::new());

    ds.downsample_into(
        &points,
        &l,
        DownsampleMode::Centroid,
        &mut out,
        Some(&mut index),
    )?;
    assert_eq!(
        out,
        vec![[-0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [1.5, 0.5, 0.5]]
    );
    assert_eq!(index, vec![2, 1, 0]);

    ds.downsample_into(
        &points,
        &l,
        DownsampleMode::First,
        &mut out,
        Some(&mut index),
    )?;
    assert_eq!(out, vec![points[2], points[1], points[0]]);
    assert_eq!(index, vec![2, 1, 0]);

    ds.downsample_into(
        &points,
        &l,
        DownsampleMode::NearestToCentre,
        &mut out,
        Some(&mut index),
    )?;
    // Voxel (1, 0, 0): both points are 0.25 from the centre; the lower index wins.
    assert_eq!(index, vec![2, 5, 0]);
    assert_eq!(out, vec![points[2], points[5], points[0]]);
    Ok(())
}

#[test]
fn downsample_matches_brute_force() -> TestResult {
    let points = random_points(20_000, -3.0, 3.0, 5);
    let l = VoxelLayout::new(0.37, [0.11, -0.2, 0.05])?;
    let runs = reference_runs(&points, &l);
    let mut ds = VoxelDownsampler::new();
    let (mut out, mut index) = (Vec::new(), Vec::new());

    ds.downsample_into(
        &points,
        &l,
        DownsampleMode::Centroid,
        &mut out,
        Some(&mut index),
    )?;
    assert_eq!(out.len(), runs.len());
    for ((_, run), (p, &i)) in runs.iter().zip(out.iter().zip(&index)) {
        let mut s = [0.0f64; 3];
        for &j in run {
            s[0] += points[j][0];
            s[1] += points[j][1];
            s[2] += points[j][2];
        }
        let c = run.len() as f64;
        assert_eq!(*p, [s[0] / c, s[1] / c, s[2] / c]);
        assert_eq!(i as usize, run[0]);
    }

    ds.downsample_into(
        &points,
        &l,
        DownsampleMode::NearestToCentre,
        &mut out,
        Some(&mut index),
    )?;
    for ((k, run), &i) in runs.iter().zip(&index) {
        let c = l.centre_from_key(k);
        let d = |j: usize| {
            let p = points[j];
            let e = [p[0] - c[0], p[1] - c[1], p[2] - c[2]];
            e[0] * e[0] + e[1] * e[1] + e[2] * e[2]
        };
        let best = run
            .iter()
            .copied()
            .fold(run[0], |b, j| if d(j) < d(b) { j } else { b });
        assert_eq!(i as usize, best);
    }
    Ok(())
}

#[test]
fn downsample_is_deterministic_and_permutation_invariant_for_first() -> TestResult {
    let points = random_points(50_000, -10.0, 10.0, 9);
    let l = VoxelLayout::new(0.5, [0.0; 3])?;
    let mut ds = VoxelDownsampler::new();
    let (mut a, mut b) = (Vec::new(), Vec::new());
    ds.downsample_into(&points, &l, DownsampleMode::Centroid, &mut a, None)?;
    ds.downsample_into(&points, &l, DownsampleMode::Centroid, &mut b, None)?;
    assert_eq!(a, b);
    assert!(a.windows(2).all(|w| {
        let ka = l.key_from_point(&w[0]);
        let kb = l.key_from_point(&w[1]);
        ka < kb
    }));

    // Reversing the input keeps the voxel set and key order: the same voxel
    // count and, per voxel, a point of the same voxel.
    let rev: Vec<_> = points.iter().rev().copied().collect();
    ds.downsample_into(&rev, &l, DownsampleMode::First, &mut b, None)?;
    ds.downsample_into(&points, &l, DownsampleMode::First, &mut a, None)?;
    assert_eq!(a.len(), b.len());
    for (p, q) in a.iter().zip(&b) {
        assert_eq!(l.key_from_point(p), l.key_from_point(q));
    }
    Ok(())
}

#[test]
fn downsample_edge_cases() -> TestResult {
    let l = unit_layout()?;
    let mut ds = VoxelDownsampler::new();
    let (mut out, mut index) = (vec![[9.0; 3]], vec![9]);
    ds.downsample_into(
        &[],
        &l,
        DownsampleMode::Centroid,
        &mut out,
        Some(&mut index),
    )?;
    assert!(out.is_empty() && index.is_empty());

    let single = [[3.5, -2.5, 1e6]];
    ds.downsample_into(&single, &l, DownsampleMode::NearestToCentre, &mut out, None)?;
    assert_eq!(out, vec![[3.5, -2.5, 1e6]]);

    // Huge but representable coordinates: keys near the i32 limits.
    let big = [[2.0e9, -2.0e9, 0.5], [2.0e9 + 0.5, -2.0e9 + 0.25, 0.5]];
    ds.downsample_into(
        &big,
        &l,
        DownsampleMode::Centroid,
        &mut out,
        Some(&mut index),
    )?;
    assert_eq!(out.len(), 1);
    assert_eq!(index, vec![0]);

    // Unrepresentable: the smallest offending index is reported.
    let bad = [[0.0; 3], [1e10, 0.0, 0.0], [f64::NAN, 0.0, 0.0]];
    assert!(matches!(
        ds.downsample_into(&bad, &l, DownsampleMode::First, &mut out, None),
        Err(VoxelError::PointOutOfRange { index: 1 })
    ));
    Ok(())
}

#[test]
fn downsample_carries_attributes() -> TestResult {
    let l = unit_layout()?;
    let cloud = PointCloud::new(
        vec![[0.2, 0.5, 0.5], [0.4, 0.5, 0.5], [3.5, 0.5, 0.5]],
        Some(vec![[10, 0, 255], [11, 1, 254], [7, 7, 7]]),
        Some(vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
    );
    let c = voxel_downsample(&cloud, &l, DownsampleMode::Centroid)?;
    assert_eq!(c.len(), 2);
    // (10 + 11) / 2 = 10.5 rounds half up; (0 + 1) / 2 -> 1; (255 + 254) / 2 -> 255.
    assert_eq!(c.colors(), Some(&vec![[11, 1, 255], [7, 7, 7]]));
    let h = 1.0 / 2f64.sqrt();
    assert_eq!(c.normals(), Some(&vec![[h, h, 0.0], [1.0, 0.0, 0.0]]));

    let f = voxel_downsample(&cloud, &l, DownsampleMode::NearestToCentre)?;
    assert_eq!(f.points(), &vec![[0.4, 0.5, 0.5], [3.5, 0.5, 0.5]]);
    assert_eq!(f.colors(), Some(&vec![[11, 1, 254], [7, 7, 7]]));

    let bad = PointCloud::new(vec![[0.0; 3]; 2], None, Some(vec![[0.0; 3]]));
    assert!(matches!(
        voxel_downsample(&bad, &l, DownsampleMode::First),
        Err(VoxelError::Filter(_))
    ));
    Ok(())
}

/// The parallel centroid colors and normals equal a serial per-voxel pass
/// in ascending point index, bit for bit, on a cloud spanning many rayon
/// tasks.
#[test]
fn centroid_attributes_match_serial_reduction() -> TestResult {
    let points = random_points(60_000, -4.0, 4.0, 17);
    let colors: Vec<[u8; 3]> = (0..points.len())
        .map(|i| [(i % 251) as u8, (i * 7 % 256) as u8, (i * 13 % 239) as u8])
        .collect();
    let normals: Vec<[f64; 3]> = points
        .iter()
        .map(|p| [p[1].sin(), p[2].cos(), p[0] * 0.25])
        .collect();
    let cloud = PointCloud::new(points.clone(), Some(colors.clone()), Some(normals.clone()));
    let l = VoxelLayout::new(0.25, [0.0; 3])?;
    let out = voxel_downsample(&cloud, &l, DownsampleMode::Centroid)?;
    let runs = reference_runs(&points, &l);
    assert!(runs.len() > 4 * 4096, "too few voxels to split into tasks");
    let want_colors: Vec<[u8; 3]> = runs
        .values()
        .map(|run| {
            let n = run.len() as u64;
            let mut sum = [0u64; 3];
            for &i in run {
                for a in 0..3 {
                    sum[a] += u64::from(colors[i][a]);
                }
            }
            sum.map(|s| ((s + n / 2) / n) as u8)
        })
        .collect();
    let want_normals: Vec<[u64; 3]> = runs
        .values()
        .map(|run| {
            let mut sum = [0.0f64; 3];
            for &i in run {
                for a in 0..3 {
                    sum[a] += normals[i][a];
                }
            }
            let norm = (sum[0] * sum[0] + sum[1] * sum[1] + sum[2] * sum[2]).sqrt();
            let unit = if norm > 0.0 {
                sum.map(|s| s / norm)
            } else {
                sum
            };
            unit.map(f64::to_bits)
        })
        .collect();
    assert_eq!(out.colors(), Some(&want_colors));
    let got_normals: Vec<[u64; 3]> = out
        .normals()
        .ok_or("normals")?
        .iter()
        .map(|n| n.map(f64::to_bits))
        .collect();
    assert_eq!(got_normals, want_normals);
    Ok(())
}

#[test]
fn sort_words_are_strictly_ascending() -> TestResult {
    // Many duplicate keys: the index part keeps the words unique.
    let mut points = random_points(10_000, -2.0, 2.0, 3);
    points.extend_from_within(..5000);
    let words = sorted_words(&points, &VoxelLayout::new(0.5, [0.0; 3])?)?;
    assert_eq!(words.len(), points.len());
    assert!(words.windows(2).all(|w| w[0] < w[1]));
    Ok(())
}

/// A plus-shaped 3D pattern: centre voxel, its 6 face neighbours, one edge
/// neighbour and one corner neighbour of the centre.
fn pattern() -> Vec<[f64; 3]> {
    let keys = [
        [0, 0, 0],
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1],
        [1, 1, 0],
        [1, 1, 1],
    ];
    // Two points in the centre voxel to check per-point output.
    let mut p: Vec<[f64; 3]> = keys
        .iter()
        .map(|k| [k[0] as f64 + 0.5, k[1] as f64 + 0.5, k[2] as f64 + 0.5])
        .collect();
    p.push([0.1, 0.9, 0.2]);
    p
}

#[test]
fn neighbour_counts_on_known_pattern() -> TestResult {
    let l = unit_layout()?;
    let p = pattern();
    let c6 = voxel_neighbour_counts(&p, &l, VoxelConnectivity::Six)?;
    let c18 = voxel_neighbour_counts(&p, &l, VoxelConnectivity::Eighteen)?;
    let c26 = voxel_neighbour_counts(&p, &l, VoxelConnectivity::TwentySix)?;
    // centre: 6 faces; + edge (1,1,0) for 18; + corner (1,1,1) for 26.
    assert_eq!((c6[0], c18[0], c26[0]), (6, 7, 8));
    assert_eq!(c26[9], c26[0]);
    // (1,0,0): centre (face), (1,1,0) (face), (0,1,0) and (0,0,1) and (0,-1,0)
    // and (0,0,-1) (edges), (1,1,1) (edge).
    assert_eq!((c6[1], c18[1], c26[1]), (2, 7, 7));
    // (1,1,1): (1,1,0) face; (0,0,1), (1,0,0), (0,1,0) edges; centre corner.
    assert_eq!((c6[8], c18[8], c26[8]), (1, 4, 5));
    // Brute force over all points and connectivities.
    let all = [
        VoxelConnectivity::Six,
        VoxelConnectivity::Eighteen,
        VoxelConnectivity::TwentySix,
    ];
    for conn in all {
        let mut g = VoxelGrid::new(l);
        for q in &p {
            if let Some(k) = l.key_from_point(q) {
                g.insert(k, ());
            }
        }
        let got = voxel_neighbour_counts(&p, &l, conn)?;
        for (q, &c) in p.iter().zip(&got) {
            let k = l.key_from_point(q).ok_or("key")?;
            assert_eq!(
                usize::from(c),
                g.neighbours(k, conn).count(),
                "{conn:?} {k:?}"
            );
        }
    }
    Ok(())
}

#[test]
fn neighbour_counts_match_grid_on_random_cloud() -> TestResult {
    let p = random_points(30_000, -4.0, 4.0, 21);
    let l = VoxelLayout::new(0.6, [0.3, 0.0, -0.1])?;
    let mut g = VoxelGrid::new(l);
    for q in &p {
        if let Some(k) = l.key_from_point(q) {
            g.insert(k, ());
        }
    }
    let got = voxel_neighbour_counts(&p, &l, VoxelConnectivity::TwentySix)?;
    for (q, &c) in p.iter().zip(&got) {
        let k = l.key_from_point(q).ok_or("key")?;
        assert_eq!(
            usize::from(c),
            g.neighbours(k, VoxelConnectivity::TwentySix).count()
        );
    }
    Ok(())
}

#[test]
fn neighbour_counts_at_the_i32_limits() -> TestResult {
    let l = unit_layout()?;
    let max = i32::MAX as f64;
    let min = i32::MIN as f64;
    let p = [
        [max + 0.5, 0.5, 0.5],
        [max - 0.5, 0.5, 0.5],
        [min + 0.5, 0.5, 0.5],
        [0.5, max + 0.5, min],
    ];
    let c = voxel_neighbour_counts(&p, &l, VoxelConnectivity::TwentySix)?;
    assert_eq!(c, vec![1, 1, 0, 0]);
    Ok(())
}

#[test]
fn remove_sparse_voxels_keeps_order_and_attributes() -> TestResult {
    let l = unit_layout()?;
    let mut p = pattern();
    p.push([20.5, 0.5, 0.5]);
    let colors: Vec<[u8; 3]> = (0..p.len() as u8).map(|i| [i, i, i]).collect();
    let cloud = PointCloud::new(p.clone(), Some(colors), None);
    let out = remove_sparse_voxels(&cloud, &l, VoxelConnectivity::Six, 2)?;
    // Six-connectivity counts: centre 6 (two points), (1,0,0) 2, (0,1,0) 2,
    // (1,1,0) 3, the other pattern voxels 1, the isolated point 0.
    assert_eq!(out.points(), &vec![p[0], p[1], p[3], p[7], p[9]]);
    assert_eq!(
        out.colors(),
        Some(&vec![[0; 3], [1; 3], [3; 3], [7; 3], [9; 3]])
    );
    assert_eq!(
        remove_sparse_voxels(&cloud, &l, VoxelConnectivity::Six, 0)?.len(),
        p.len()
    );
    Ok(())
}

fn take8(bytes: &[u8], at: &mut usize) -> Result<[u8; 8], Box<dyn std::error::Error>> {
    let b: [u8; 8] = bytes
        .get(*at..*at + 8)
        .ok_or("truncated fixture")?
        .try_into()?;
    *at += 8;
    Ok(b)
}

/// Parity with Open3D 0.20's `voxel_down_sample` on a fixture written by
/// `benches/bench_voxel_baselines.py --write-fixtures`.
///
/// `tests/data/voxel_downsample_open3d.bin`, little endian: `f64`
/// voxel size, `3 × f64` origin, `u64` input count, the input `x, y, z`
/// `f64` triplets, `u64` output count, the Open3D output triplets. The
/// origin must be Open3D's implicit `min_bound - voxel_size / 2`. Open3D's
/// output is in hash order and sums in another order, so the comparison is
/// on the key-sorted sets with an absolute tolerance.
#[test]
fn open3d_voxel_downsample_parity_fixture() -> TestResult {
    const TOL: f64 = 1e-9;
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/data/voxel_downsample_open3d.bin");
    let bytes = std::fs::read(&path)?;
    let mut at = 0usize;
    let size = f64::from_le_bytes(take8(&bytes, &mut at)?);
    let origin = [
        f64::from_le_bytes(take8(&bytes, &mut at)?),
        f64::from_le_bytes(take8(&bytes, &mut at)?),
        f64::from_le_bytes(take8(&bytes, &mut at)?),
    ];
    let triplets = |at: &mut usize| -> Result<Vec<[f64; 3]>, Box<dyn std::error::Error>> {
        let n = usize::try_from(u64::from_le_bytes(take8(&bytes, at)?))?;
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push([
                f64::from_le_bytes(take8(&bytes, at)?),
                f64::from_le_bytes(take8(&bytes, at)?),
                f64::from_le_bytes(take8(&bytes, at)?),
            ]);
        }
        Ok(out)
    };
    let input = triplets(&mut at)?;
    let mut expected = triplets(&mut at)?;
    let l = VoxelLayout::new(size, origin)?;
    let mut got = Vec::new();
    VoxelDownsampler::new().downsample_into(
        &input,
        &l,
        DownsampleMode::Centroid,
        &mut got,
        None,
    )?;
    expected.sort_by_key(|p| l.key_from_point(p));
    assert_eq!(got.len(), expected.len());
    for (a, b) in got.iter().zip(&expected) {
        for k in 0..3 {
            assert!((a[k] - b[k]).abs() <= TOL, "{a:?} vs {b:?}");
        }
    }
    Ok(())
}
