#![allow(missing_docs)]

pub fn match_descriptors(
    descriptors1: &[Vec<u8>],
    descriptors2: &[Vec<u8>],
    max_distance: Option<u32>,
    cross_check: bool,
    max_ratio: Option<f32>,
) -> Vec<(usize, usize)> {
    let m = descriptors1.len();
    let n = descriptors2.len();
    if m == 0 || n == 0 {
        return vec![];
    }

    let desc_len = descriptors1[0].len();
    assert!(
        descriptors2.iter().all(|d| d.len() == desc_len),
        "Descriptor length mismatch"
    );

    let mut distance = vec![vec![0u32; n]; m];
    for i in 0..m {
        for j in 0..n {
            distance[i][j] = descriptors1[i]
                .iter()
                .zip(&descriptors2[j])
                .map(|(&a, &b)| (a ^ b).count_ones())
                .sum();
        }
    }

    let mut indices1: Vec<usize> = (0..m).collect();
    let mut indices2: Vec<usize> = (0..m)
        .map(|i| {
            let (min_j, _) = distance[i]
                .iter()
                .enumerate()
                .min_by_key(|&(_, &d)| d)
                .unwrap();
            min_j
        })
        .collect();

    if cross_check {
        let mut matches1: Vec<usize> = (0..n)
            .map(|j| {
                let (min_i, _) = (0..m)
                    .map(|i| (i, distance[i][j]))
                    .min_by_key(|&(_, d)| d)
                    .unwrap();
                min_i
            })
            .collect();

        let mask: Vec<bool> = indices1
            .iter()
            .zip(indices2.iter())
            .map(|(&i1, &i2)| matches1[i2] == i1)
            .collect();

        indices1 = indices1
            .into_iter()
            .zip(mask.iter())
            .filter_map(|(i, &m)| if m { Some(i) } else { None })
            .collect();

        indices2 = indices2
            .into_iter()
            .zip(mask.iter())
            .filter_map(|(i, &m)| if m { Some(i) } else { None })
            .collect();
    }

    if let Some(max_dist) = max_distance {
        let mask: Vec<bool> = indices1
            .iter()
            .zip(indices2.iter())
            .map(|(&i1, &i2)| distance[i1][i2] <= max_dist)
            .collect();

        indices1 = indices1
            .into_iter()
            .zip(mask.iter())
            .filter_map(|(i, &m)| if m { Some(i) } else { None })
            .collect();

        indices2 = indices2
            .into_iter()
            .zip(mask.iter())
            .filter_map(|(i, &m)| if m { Some(i) } else { None })
            .collect();
    }

    if let Some(max_ratio) = max_ratio {
        if max_ratio < 1.0 {
            let mut keep = vec![false; indices1.len()];
            for (k, (&i1, &i2)) in indices1.iter().zip(indices2.iter()).enumerate() {
                let best_dist = distance[i1][i2];

                // Temp set best to max to find the second best
                let mut row = distance[i1].clone();
                row[i2] = u32::MAX;

                let &second_best_dist = row.iter().min().unwrap();
                let denom = if second_best_dist == 0 {
                    std::f32::EPSILON
                } else {
                    second_best_dist as f32
                };

                let ratio = best_dist as f32 / denom;
                if ratio < max_ratio {
                    keep[k] = true;
                }
            }

            indices1 = indices1
                .into_iter()
                .zip(&keep)
                .filter_map(|(i, &k)| if k { Some(i) } else { None })
                .collect();

            indices2 = indices2
                .into_iter()
                .zip(&keep)
                .filter_map(|(i, &k)| if k { Some(i) } else { None })
                .collect();
        }
    }

    indices1.into_iter().zip(indices2.into_iter()).collect()
}
