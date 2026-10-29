use super::*;

type TestResult = Result<(), Box<dyn std::error::Error>>;

/// Deterministic xorshift64 points in a `[0, scale)` cube.
fn random_points(n: usize, scale: f64, seed: u64) -> Vec<[f64; 3]> {
    let mut s = seed;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64 * scale
    };
    (0..n).map(|_| [next(), next(), next()]).collect()
}

/// Brute-force radius rule: other points within `radius` (inclusive).
fn radius_reference(points: &[[f64; 3]], radius: f64, min_neighbours: usize) -> Vec<u32> {
    let r2 = radius * radius;
    (0..points.len())
        .filter(|&i| {
            let others = points
                .iter()
                .enumerate()
                .filter(|&(j, q)| {
                    let p = points[i];
                    let d = (p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2);
                    j != i && d <= r2
                })
                .count();
            others >= min_neighbours
        })
        .map(|i| i as u32)
        .collect()
}

/// Brute-force statistical rule on finite points: each point's `k` nearest
/// (itself first) summed as `mean_of_sorted`, then the sample deviation.
fn statistical_reference(points: &[[f64; 3]], k: usize, std_ratio: f64) -> Vec<u32> {
    let hits = k.min(points.len().saturating_sub(1)) + 1;
    let mean: Vec<f64> = points
        .iter()
        .map(|p| {
            let mut d: Vec<f64> = points
                .iter()
                .map(|q| (p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2))
                .collect();
            if hits < d.len() {
                d.select_nth_unstable_by(hits - 1, f64::total_cmp);
            }
            d.truncate(hits);
            d.sort_by(f64::total_cmp);
            grid::mean_of_sorted(&d)
        })
        .collect();
    let n = mean.len() as f64;
    let mu = mean.iter().sum::<f64>() / n;
    let sd = (mean.iter().map(|d| (d - mu) * (d - mu)).sum::<f64>() / (n - 1.0)).sqrt();
    (0..points.len() as u32)
        .filter(|&i| mean[i as usize] <= mu + std_ratio * sd)
        .collect()
}

/// The finite points of `points` and their input indices.
fn finite_subset(points: &[[f64; 3]]) -> (Vec<[f64; 3]>, Vec<u32>) {
    points
        .iter()
        .enumerate()
        .filter(|(_, p)| p.iter().all(|v| v.is_finite()))
        .map(|(i, p)| (*p, i as u32))
        .unzip()
}

#[test]
fn radius_keeps_dense_and_drops_isolated() -> TestResult {
    let points = [
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [0.0, 0.1, 0.0],
        [3.0, 3.0, 3.0],
        [0.0, 0.0, 0.1],
    ];
    let mut kept = Vec::new();
    radius_inlier_indices(&points, 0.2, 2, &mut kept)?;
    assert_eq!(kept, vec![0, 1, 2, 4]);
    radius_inlier_indices(&points, 0.2, 4, &mut kept)?;
    assert_eq!(kept, Vec::<u32>::new());
    radius_inlier_indices(&points, 0.2, 0, &mut kept)?;
    assert_eq!(kept, vec![0, 1, 2, 3, 4]);
    Ok(())
}

#[test]
fn radius_boundary_is_inclusive() -> TestResult {
    // 0.5 and 0.25 are exact in binary, so the squared distance equals radius².
    let points = [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]];
    let mut kept = Vec::new();
    radius_inlier_indices(&points, 0.5, 1, &mut kept)?;
    assert_eq!(kept, vec![0, 1]);
    radius_inlier_indices(&points, 0.499, 1, &mut kept)?;
    assert!(kept.is_empty());
    Ok(())
}

#[test]
fn radius_duplicates_count_as_neighbours() -> TestResult {
    let points = [[1.0, 2.0, 3.0]; 4];
    let mut kept = Vec::new();
    radius_inlier_indices(&points, 1e-6, 3, &mut kept)?;
    assert_eq!(kept, vec![0, 1, 2, 3]);
    radius_inlier_indices(&points, 1e-6, 4, &mut kept)?;
    assert!(kept.is_empty());
    Ok(())
}

#[test]
fn radius_matches_brute_force() -> TestResult {
    let points = random_points(600, 1.0, 7);
    let mut kept = Vec::new();
    for (radius, min) in [(0.05, 1), (0.1, 3), (0.2, 12)] {
        radius_inlier_indices(&points, radius, min, &mut kept)?;
        let expected = radius_reference(&points, radius, min);
        assert_eq!(kept, expected, "r={radius} min={min}");
    }
    Ok(())
}

#[test]
fn radius_rejects_bad_radius() {
    let mut kept = Vec::new();
    for r in [0.0, -1.0, f64::NAN, f64::INFINITY, 1e200] {
        assert!(matches!(
            radius_inlier_indices(&[[0.0; 3]], r, 1, &mut kept),
            Err(FilterError::InvalidRadius(_))
        ));
    }
}

#[test]
fn filters_on_empty_and_single_point() -> TestResult {
    let mut kept = vec![9];
    radius_inlier_indices(&[], 1.0, 1, &mut kept)?;
    assert!(kept.is_empty());
    statistical_inlier_indices(&[], 3, 1.0, &mut kept)?;
    assert!(kept.is_empty());
    radius_inlier_indices(&[[1.0; 3]], 1.0, 1, &mut kept)?;
    assert!(kept.is_empty());
    statistical_inlier_indices(&[[1.0; 3]], 3, 1.0, &mut kept)?;
    assert_eq!(kept, vec![0]);
    Ok(())
}

#[test]
fn statistical_exact_small_case() -> TestResult {
    // Mean 1-NN distances: 1, 1, 1, 1, 6 (the last point is 6 from its
    // nearest). mean = 2, sample std = sqrt(20 / 4) = sqrt(5) ~ 2.236.
    let points = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [9.0, 0.0, 0.0],
    ];
    let mut kept = Vec::new();
    // threshold 2 + 1.8 * sqrt(5) ~ 6.025 >= 6: the outlier stays.
    statistical_inlier_indices(&points, 1, 1.8, &mut kept)?;
    assert_eq!(kept, vec![0, 1, 2, 3, 4]);
    // threshold 2 + 1.7 * sqrt(5) ~ 5.801 < 6.
    statistical_inlier_indices(&points, 1, 1.7, &mut kept)?;
    assert_eq!(kept, vec![0, 1, 2, 3]);
    // A negative ratio puts the threshold below the mean: 2 - 0.2 * sqrt(5) ~ 1.553.
    statistical_inlier_indices(&points, 1, -0.2, &mut kept)?;
    assert_eq!(kept, vec![0, 1, 2, 3]);
    Ok(())
}

#[test]
fn statistical_k_larger_than_cloud_uses_all_others() -> TestResult {
    let points = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
    let mut a = Vec::new();
    let mut b = Vec::new();
    statistical_inlier_indices(&points, 2, 0.5, &mut a)?;
    statistical_inlier_indices(&points, 50, 0.5, &mut b)?;
    assert_eq!(a, b);
    // Means: 1.5, 1, 1.5 -> mean 4/3, sample std sqrt(1/12) ~ 0.289;
    // threshold 4/3 + 0.144 ~ 1.478 < 1.5.
    assert_eq!(a, vec![1]);
    Ok(())
}

#[test]
fn statistical_is_deterministic() -> TestResult {
    let mut points = random_points(5000, 1.0, 11);
    points.extend(random_points(50, 30.0, 12));
    let mut a = Vec::new();
    let mut b = Vec::new();
    statistical_inlier_indices(&points, 8, 1.0, &mut a)?;
    statistical_inlier_indices(&points, 8, 1.0, &mut b)?;
    assert_eq!(a, b);
    assert!(a.len() < points.len());
    Ok(())
}

#[test]
fn statistical_rejects_bad_arguments() {
    let mut kept = Vec::new();
    assert!(matches!(
        statistical_inlier_indices(&[[0.0; 3]], 0, 1.0, &mut kept),
        Err(FilterError::InvalidNeighbourCount)
    ));
    assert!(matches!(
        statistical_inlier_indices(&[[0.0; 3]], 1, f64::NAN, &mut kept),
        Err(FilterError::InvalidStdRatio(_))
    ));
}

#[test]
fn remove_carries_attributes() -> TestResult {
    let cloud = PointCloud::new(
        vec![[0.0; 3], [0.1, 0.0, 0.0], [9.0, 0.0, 0.0]],
        Some(vec![[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        Some(vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    );
    let out = remove_radius_outliers(&cloud, 0.5, 1)?;
    assert_eq!(out.points(), &vec![[0.0; 3], [0.1, 0.0, 0.0]]);
    assert_eq!(out.colors(), Some(&vec![[1, 2, 3], [4, 5, 6]]));
    assert_eq!(out.normals(), Some(&vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]));
    Ok(())
}

#[test]
fn select_rejects_bad_index_and_attributes() {
    let cloud = PointCloud::new(vec![[0.0; 3]; 2], Some(vec![[0; 3]]), None);
    assert!(matches!(
        select_points(&cloud, &[0]),
        Err(FilterError::AttributeLengthMismatch {
            attribute: "colors",
            len: 1,
            expected: 2
        })
    ));
    let cloud = PointCloud::new(vec![[0.0; 3]; 2], None, None);
    assert!(matches!(
        select_points(&cloud, &[2]),
        Err(FilterError::IndexOutOfRange { index: 2, len: 2 })
    ));
}

/// Clouds where a grid search is easy to get wrong: uniform, planar,
/// clustered with far outliers, duplicates, points on cell boundaries and
/// negative coordinates.
fn grid_clouds() -> Vec<Vec<[f64; 3]>> {
    let uniform = random_points(6000, 4.0, 11);
    let planar: Vec<[f64; 3]> = random_points(5000, 3.0, 12)
        .into_iter()
        .map(|p| [p[0], p[1], 0.25])
        .collect();
    let mut clustered: Vec<[f64; 3]> = random_points(4000, 0.2, 13);
    clustered.extend(
        random_points(3000, 0.1, 14)
            .into_iter()
            .map(|p| [p[0] + 50.0, p[1], p[2]]),
    );
    clustered.extend([[1e4, 0.0, 0.0], [-3e3, 7.0, 1.0], [25.0, 25.0, 25.0]]);
    let mut dup = random_points(3000, 1.0, 15);
    for i in (0..3000).step_by(3) {
        dup[i] = dup[(i * 7) % 3000];
    }
    let lattice: Vec<[f64; 3]> = (0..4096)
        .map(|i| {
            let (x, y, z) = (i % 16, (i / 16) % 16, i / 256);
            [
                x as f64 * 0.05 - 0.4,
                y as f64 * 0.05 - 0.4,
                z as f64 * 0.05 - 0.4,
            ]
        })
        .collect();
    vec![uniform, planar, clustered, dup, lattice]
}

#[test]
fn radius_grid_matches_kd_tree() -> TestResult {
    for (c, points) in grid_clouds().iter().enumerate() {
        for &(radius, min_neighbours) in &[(0.05, 1), (0.1, 4), (0.15, 20), (0.0125, 2), (3.0, 50)]
        {
            let r2 = radius * radius;
            let cap = NonZero::new(min_neighbours + 1).ok_or("cap")?;
            let grid = grid::CellGrid::for_radius(points, radius).ok_or("grid")?;
            let mut got = vec![false; points.len()];
            grid::radius_dense(&grid, r2, cap.get(), &mut got);
            let mut want = vec![false; points.len()];
            radius_dense_kd(points, r2, cap, &mut want);
            assert_eq!(got, want, "cloud {c} radius {radius} min {min_neighbours}");
        }
    }
    Ok(())
}

#[test]
fn knn_grid_matches_kd_tree_bit_for_bit() -> TestResult {
    for (c, points) in grid_clouds().iter().enumerate() {
        let tree = KdTree::new_from_slice(points);
        for k in [1usize, 4, 16, 33] {
            let cap = NonZero::new(k + 1).ok_or("cap")?;
            for divisor in [1, KNN_CELL_LOAD_DIVISOR, 8] {
                let grid =
                    grid::CellGrid::for_knn(points, (k + 1).div_ceil(divisor)).ok_or("grid")?;
                let mut got = vec![0.0; points.len()];
                grid::knn_mean_distances(&grid, k + 1, &|p| knn_mean_kd(&tree, p, cap), &mut got);
                for (i, (g, p)) in got.iter().zip(points).enumerate() {
                    let want = knn_mean_kd(&tree, p, cap);
                    assert_eq!(g.to_bits(), want.to_bits(), "cloud {c} k {k} point {i}");
                }
            }
        }
    }
    Ok(())
}

#[test]
fn statistical_matches_brute_force_on_a_grid_sized_cloud() -> TestResult {
    let mut points = random_points(3000, 2.0, 21);
    points.extend([[9.0, 9.0, 9.0], [-4.0, 1.0, 1.0]]);
    let mut kept = Vec::new();
    statistical_inlier_indices(&points, 8, 0.5, &mut kept)?;
    assert_eq!(kept, statistical_reference(&points, 8, 0.5));
    Ok(())
}

/// Runs both filters on `points` and on its finite subset, and checks that
/// the results agree once the subset's indices are mapped back.
fn assert_same_as_finite_subset(points: &[[f64; 3]]) -> TestResult {
    let (finite, index) = finite_subset(points);
    let map = |kept: &[u32]| -> Vec<u32> { kept.iter().map(|&i| index[i as usize]).collect() };
    let (mut got, mut want) = (Vec::new(), Vec::new());
    for (radius, min_neighbours) in [(0.1, 6), (0.2, 0)] {
        radius_inlier_indices(points, radius, min_neighbours, &mut got)?;
        radius_inlier_indices(&finite, radius, min_neighbours, &mut want)?;
        assert_eq!(got, map(&want), "radius {radius} min {min_neighbours}");
    }
    for (k, std_ratio) in [(8, 1.0), (3, 0.0)] {
        statistical_inlier_indices(points, k, std_ratio, &mut got)?;
        statistical_inlier_indices(&finite, k, std_ratio, &mut want)?;
        assert_eq!(got, map(&want), "k {k} std_ratio {std_ratio}");
        assert!(!got.is_empty() && got.len() < finite.len());
    }
    Ok(())
}

#[test]
fn non_finite_points_are_dropped_as_if_absent() -> TestResult {
    let points = random_points(2000, 1.0, 31);
    let nan = [0.5, f64::NAN, 0.5];
    let inf = [f64::INFINITY, 0.5, 0.5];
    for bad in [
        vec![(500, nan)],
        vec![(1500, inf)],
        vec![(3, nan), (1999, inf)],
    ] {
        let mut cloud = points.clone();
        for &(at, p) in &bad {
            cloud.insert(at, p);
        }
        assert_same_as_finite_subset(&cloud)?;
        // The finite points still get the grid search.
        let subset = FiniteSubset::of(&cloud).ok_or("no non-finite point")?;
        assert!(grid::CellGrid::for_radius(&subset.points, 0.1).is_some());
        assert!(grid::CellGrid::for_knn(&subset.points, 5).is_some());
    }
    assert!(FiniteSubset::of(&points).is_none());
    Ok(())
}

#[test]
fn all_non_finite_cloud_keeps_nothing() -> TestResult {
    let points = [
        [f64::NAN; 3],
        [f64::INFINITY, 0.0, 0.0],
        [0.0, f64::NEG_INFINITY, 0.0],
        [0.0, 0.0, f64::NAN],
    ];
    let mut kept = vec![7];
    for min_neighbours in [0, 1] {
        radius_inlier_indices(&points, 1.0, min_neighbours, &mut kept)?;
        assert!(kept.is_empty());
    }
    for k in [1, 3, 10] {
        statistical_inlier_indices(&points, k, 1.0, &mut kept)?;
        assert!(kept.is_empty());
    }
    Ok(())
}

#[test]
fn statistical_with_k_plus_one_finite_points() -> TestResult {
    let k = 3;
    let points = [
        [f64::NAN, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, f64::INFINITY, 0.0],
        [2.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [0.0, 0.0, f64::NEG_INFINITY],
    ];
    // Mean distances to the other three finite points: 13/3, 11/3, 11/3, 9.
    // mean ~ 5.17, sample std ~ 2.57: a zero ratio keeps the three below
    // the mean, a ratio of 2 keeps all four.
    let mut kept = Vec::new();
    statistical_inlier_indices(&points, k, 0.0, &mut kept)?;
    assert_eq!(kept, vec![1, 2, 4]);
    statistical_inlier_indices(&points, k, 2.0, &mut kept)?;
    assert_eq!(kept, vec![1, 2, 4, 5]);
    // One finite point: kept, whatever `k`.
    statistical_inlier_indices(&points[..2], k, 0.0, &mut kept)?;
    assert_eq!(kept, vec![1]);
    Ok(())
}

/// Grid clouds with non-finite points spliced in: both filters equal brute
/// force on the finite subset, the non-finite indices excluded.
#[test]
fn grid_with_non_finite_points_matches_brute_force() -> TestResult {
    let bad = [
        [f64::NAN, 0.0, 0.0],
        [0.0, f64::INFINITY, 0.0],
        [0.0, 0.0, f64::NEG_INFINITY],
        [f64::NAN; 3],
    ];
    for (c, mut points) in grid_clouds().into_iter().enumerate() {
        for (j, i) in (0..points.len()).step_by(997).enumerate() {
            points[i] = bad[j % bad.len()];
        }
        let (finite, index) = finite_subset(&points);
        assert!(finite.len() < points.len() && finite.len() >= KNN_GRID_MIN_POINTS);
        let map =
            |kept: Vec<u32>| -> Vec<u32> { kept.iter().map(|&i| index[i as usize]).collect() };
        let mut kept = Vec::new();
        radius_inlier_indices(&points, 0.1, 4, &mut kept)?;
        assert_eq!(
            kept,
            map(radius_reference(&finite, 0.1, 4)),
            "cloud {c} radius"
        );
        statistical_inlier_indices(&points, 8, 1.0, &mut kept)?;
        assert_eq!(
            kept,
            map(statistical_reference(&finite, 8, 1.0)),
            "cloud {c} statistical"
        );
    }
    Ok(())
}

/// Little-endian reader over a parity fixture.
struct Fixture<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl Fixture<'_> {
    fn word(&mut self) -> Result<[u8; 8], Box<dyn std::error::Error>> {
        let b = self
            .bytes
            .get(self.at..self.at + 8)
            .ok_or("truncated fixture")?;
        self.at += 8;
        Ok(b.try_into()?)
    }
    fn f64(&mut self) -> Result<f64, Box<dyn std::error::Error>> {
        Ok(f64::from_le_bytes(self.word()?))
    }
    fn usize(&mut self) -> Result<usize, Box<dyn std::error::Error>> {
        Ok(usize::try_from(u64::from_le_bytes(self.word()?))?)
    }
    fn triplets(&mut self) -> Result<Vec<[f64; 3]>, Box<dyn std::error::Error>> {
        let n = self.usize()?;
        (0..n)
            .map(|_| Ok([self.f64()?, self.f64()?, self.f64()?]))
            .collect()
    }
    fn indices(&mut self) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
        let n = self.usize()?;
        (0..n).map(|_| Ok(u32::try_from(self.usize()?)?)).collect()
    }
}

/// Parity with Open3D 0.20's `remove_radius_outlier` and
/// `remove_statistical_outlier` on a fixture written by
/// `benches/bench_voxel_baselines.py --write-fixtures`.
///
/// Open3D's legacy `remove_radius_outlier` keeps a point with more than
/// `nb_points` points in the radius, itself included, so `nb_points` is
/// kornia's `min_neighbours`. Its `nb_neighbors` counts the query point, so
/// it is kornia's `k` plus one; its statistical mean then also averages in
/// the query's zero distance, which scales the mean and the deviation by
/// the same factor and selects the same points. `tests/data/outlier_filters_open3d.bin`, little
/// endian: `f64` radius, `u64` `nb_points`, `u64` `nb_neighbors`, `f64`
/// `std_ratio`, `u64` point count and the `f64` triplets, then the kept
/// indices of each filter (`u64` count and `u64` indices).
#[test]
fn open3d_outlier_filters_parity_fixture() -> TestResult {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/data/outlier_filters_open3d.bin");
    let bytes = std::fs::read(&path)?;
    let mut f = Fixture {
        bytes: &bytes,
        at: 0,
    };
    let radius = f.f64()?;
    let nb_points = f.usize()?;
    let nb_neighbors = f.usize()?;
    let std_ratio = f.f64()?;
    let points = f.triplets()?;
    let radius_kept = f.indices()?;
    let stat_kept = f.indices()?;
    assert_eq!(f.at, bytes.len(), "trailing fixture bytes");

    let mut kept = Vec::new();
    radius_inlier_indices(&points, radius, nb_points, &mut kept)?;
    assert_eq!(kept, radius_kept);
    statistical_inlier_indices(&points, nb_neighbors - 1, std_ratio, &mut kept)?;
    assert_eq!(kept, stat_kept);
    // Both filters must have removed something for the parity to mean anything.
    assert!(radius_kept.len() < points.len() && stat_kept.len() < points.len());
    Ok(())
}
