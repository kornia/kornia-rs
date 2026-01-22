use crate::{
    metric::DistanceMetric, BlockCluster, BlockContent, BowError, BowResult, InternalMeta,
    LeafData, Vocabulary,
};
use rand::Rng;
use rayon::prelude::*;
use std::collections::VecDeque;
use std::sync::Mutex;

/// A recursive node used internally during vocabulary construction.
struct BuilderNode<M: DistanceMetric> {
    children: Vec<BuilderNode<M>>,
    centroid: M::Data,
    weight: f32,
}

/// Computes centroids using the KMeans++ algorithm.
fn kmeans_plusplus<M: DistanceMetric>(data: &[M::Data], k_clusters: usize) -> Vec<M::Data> {
    let mut centroids = Vec::with_capacity(k_clusters);
    let mut rng = rand::rng();

    let first_centroid_idx = rng.random_range(0..data.len());
    let first_centroid = data[first_centroid_idx];
    centroids.push(first_centroid);

    let mut distances = vec![M::max_distance(); data.len()];

    distances
        .par_iter_mut()
        .zip(data.par_iter())
        .for_each(|(d, p)| {
            *d = M::distance(p, &first_centroid);
        });

    for _iter in 1..k_clusters {
        let last_centroid = centroids[centroids.len() - 1];

        distances
            .par_iter_mut()
            .zip(data.par_iter())
            .for_each(|(min_dist, point)| {
                let dist = M::distance(point, &last_centroid);
                if dist < *min_dist {
                    *min_dist = dist;
                }
            });

        let total_dist: f32 = distances.iter().map(|&d| M::to_f32(d) * M::to_f32(d)).sum();

        if total_dist == 0.0 {
            centroids.push(data[rng.random_range(0..data.len())]);
            continue;
        }

        let mut rand_val = rng.random_range(0.0..1.0) * total_dist;
        let mut chosen_idx = 0;

        for (i, &dist) in distances.iter().enumerate() {
            let d_f32 = M::to_f32(dist);
            rand_val -= d_f32 * d_f32;
            if rand_val <= 0.0 {
                chosen_idx = i;
                break;
            }
        }
        centroids.push(data[chosen_idx]);
    }

    centroids
}

/// Internal recursive training step.
fn train_recursive<const B: usize, M: DistanceMetric>(
    data: &mut [M::Data],
    depth: usize,
    max_depth: usize,
) -> BuilderNode<M> {
    if depth == max_depth || data.len() < B {
        return BuilderNode {
            children: Vec::new(),
            centroid: M::Data::default(),
            weight: data.len() as f32,
        };
    }

    let centroids = kmeans_plusplus::<M>(data, B);

    let assignments: Vec<usize> = data
        .par_iter()
        .map(|point| {
            let mut best_idx = 0;
            let mut min_dist = M::max_distance();
            for (i, centroid) in centroids.iter().enumerate() {
                let dist = M::distance(point, centroid);
                if dist < min_dist {
                    min_dist = dist;
                    best_idx = i;
                }
            }
            best_idx
        })
        .collect();

    let mut partitioned = vec![M::Data::default(); data.len()];
    let mut counts = [0usize; B];
    for &a in &assignments {
        counts[a] += 1;
    }

    let mut offsets = [0usize; B];
    for i in 1..B {
        offsets[i] = offsets[i - 1] + counts[i - 1];
    }

    let mut current_offsets = offsets;
    for (i, &point) in data.iter().enumerate() {
        let cluster_idx = assignments[i];
        partitioned[current_offsets[cluster_idx]] = point;
        current_offsets[cluster_idx] += 1;
    }
    data.copy_from_slice(&partitioned);

    let results: Vec<Mutex<Option<BuilderNode<M>>>> = (0..B).map(|_| Mutex::new(None)).collect();

    rayon::scope(|s| {
        let mut remaining_data = &mut *data;

        for i in 0..B {
            let count = counts[i];
            if count == 0 {
                continue;
            }

            let (child_data, tail) = remaining_data.split_at_mut(count);
            remaining_data = tail;

            let centroid = centroids[i];
            let res_slot = &results[i];

            s.spawn(move |_| {
                let mut node = train_recursive::<B, M>(child_data, depth + 1, max_depth);
                node.centroid = centroid;
                let mut guard = match res_slot.lock() {
                    Ok(g) => g,
                    Err(e) => e.into_inner(),
                };
                *guard = Some(node);
            });
        }
    });

    let children = results
        .into_iter()
        .filter_map(|m| match m.into_inner() {
            Ok(g) => g,
            Err(e) => e.into_inner(),
        })
        .collect();

    BuilderNode {
        children,
        centroid: M::Data::default(),
        weight: 0.0,
    }
}

/// Trains a hierarchical vocabulary tree.
pub fn train<const B: usize, M: DistanceMetric>(
    data: &[M::Data],
    max_depth: usize,
) -> BowResult<Vocabulary<B, M>> {
    if data.is_empty() {
        return Err(BowError::NoFeatures);
    }

    let total_features = data.len();
    let mut workspace = data.to_vec();
    let root = train_recursive::<B, M>(&mut workspace, 0, max_depth);
    Ok(convert_to_vocabulary::<B, M>(root, total_features))
}

/// Flattens a recursive tree into a cache-friendly `Vocabulary`.
fn convert_to_vocabulary<const B: usize, M: DistanceMetric>(
    root: BuilderNode<M>,
    total_features: usize,
) -> Vocabulary<B, M> {
    let mut flat_blocks: Vec<BlockCluster<B, M>> = Vec::new();
    flat_blocks.push(BlockCluster::default());

    let mut queue = VecDeque::new();
    queue.push_back((root, 0u32));

    let mut next_free_idx = 1u32;

    while let Some((node, block_idx)) = queue.pop_front() {
        let mut block = BlockCluster::default();
        for d in block.descriptors.iter_mut() {
            *d = M::padding();
        }

        let is_leaf_layer = node.children.iter().all(|c| c.children.is_empty());

        if is_leaf_layer {
            let mut weights = [0.0; B];

            for (i, child) in node.children.into_iter().enumerate() {
                if i >= B {
                    break;
                }
                block.descriptors[i] = child.centroid;
                // IDF Weighting: ln(N / n_i)
                let n_i = child.weight.max(1.0);
                weights[i] = (total_features as f32 / n_i).ln();
            }

            block.content = BlockContent::Leaf(LeafData { weights });
        } else {
            let children_base = next_free_idx;
            block.content = BlockContent::Internal(InternalMeta {
                children_base_idx: children_base,
            });

            next_free_idx += B as u32;
            if flat_blocks.len() < next_free_idx as usize {
                flat_blocks.resize(next_free_idx as usize, BlockCluster::default());
            }

            for (i, child) in node.children.into_iter().enumerate() {
                if i >= B {
                    break;
                }

                block.descriptors[i] = child.centroid;
                let child_block_idx = children_base + i as u32;

                if child.children.is_empty() {
                    let wrapper = BuilderNode {
                        centroid: child.centroid,
                        weight: child.weight,
                        children: vec![child],
                    };
                    queue.push_back((wrapper, child_block_idx));
                } else {
                    queue.push_back((child, child_block_idx));
                }
            }
        }

        if block_idx as usize >= flat_blocks.len() {
            flat_blocks.resize(block_idx as usize + 1, BlockCluster::default());
        }
        flat_blocks[block_idx as usize] = block;
    }

    flat_blocks.shrink_to_fit();

    Vocabulary {
        blocks: flat_blocks,
        root_idx: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metric::{Feature, Hamming, L2};
    use rand::{rngs::StdRng, SeedableRng};

    const B: usize = 10;
    const D: usize = 4;

    fn generate_clustered_descriptors(count: usize, num_clusters: usize) -> Vec<Feature<u64, D>> {
        let mut rng = StdRng::from_seed([42; 32]);
        let mut data = Vec::with_capacity(count);

        let centers: Vec<[u64; D]> = (0..num_clusters)
            .map(|_| {
                let mut desc = [0u64; D];
                for val in desc.iter_mut() {
                    *val = rng.random();
                }
                desc
            })
            .collect();

        for _ in 0..count {
            let center_idx = rng.random_range(0..num_clusters);
            let mut point = centers[center_idx];
            for _ in 0..3 {
                let word_idx = rng.random_range(0..D);
                let bit_idx = rng.random_range(0..64);
                point[word_idx] ^= 1 << bit_idx;
            }
            data.push(Feature(point));
        }
        data
    }

    #[test]
    fn test_train_empty_features() {
        let empty_data: Vec<Feature<u64, D>> = Vec::new();
        let result = train::<B, Hamming<D>>(&empty_data, 3);
        assert!(matches!(result, Err(BowError::NoFeatures)));
    }

    #[test]
    fn test_correctness() {
        let training_data = generate_clustered_descriptors(1000, 10);
        let vocab = train::<B, Hamming<D>>(&training_data, 3).unwrap();
        let test_descriptor = &training_data[0];
        let (_leaf_id, weight) = vocab.transform_one(test_descriptor);
        assert!(weight > 0.0);
    }

    #[test]
    fn test_l2_vocabulary() {
        let mut rng = StdRng::from_seed([42; 32]);
        let training_data: Vec<Feature<f32, 128>> = (0..100)
            .map(|_| {
                let mut desc = [0.0f32; 128];
                for val in desc.iter_mut() {
                    *val = rng.random();
                }
                Feature(desc)
            })
            .collect();

        let vocab = train::<B, L2<128>>(&training_data, 2).unwrap();
        let (_leaf_id, weight) = vocab.transform_one(&training_data[0]);
        assert!(weight >= 0.0);
    }

    #[test]
    fn test_train_small_data() {
        let training_data = generate_clustered_descriptors(5, 2);
        // B=10, data=5. Should still train successfully.
        let vocab = train::<10, Hamming<D>>(&training_data, 2).unwrap();
        assert!(!vocab.blocks.is_empty());
    }

    #[test]
    fn test_train_identical_features() {
        let feature = Feature([123u64; D]);
        let training_data = vec![feature; 100];
        let vocab = train::<B, Hamming<D>>(&training_data, 3).unwrap();
        assert!(!vocab.blocks.is_empty());
        let (_leaf_id, weight) = vocab.transform_one(&feature);
        // With IDF, if a word contains all features, ln(N/N) = 0.0
        assert!(weight >= 0.0);
    }

    #[test]
    fn test_unbalanced_tree() {
        // Create data that forces an unbalanced tree.
        let mut data = Vec::new();

        // Cluster 1 (Low count)
        for _ in 0..5 {
            data.push(Feature([0, 0, 0, 0]));
        }

        // Cluster 2 (High count)
        for i in 0..20 {
            let mut desc = [u64::MAX; D];
            desc[0] ^= i as u64;
            data.push(Feature(desc));
        }

        let vocab = train::<10, Hamming<D>>(&data, 3).unwrap();
        let f1 = Feature([0, 0, 0, 0]);
        let (id1, _) = vocab.transform_one(&f1);

        let f2 = Feature([u64::MAX; D]);
        let (id2, _) = vocab.transform_one(&f2);

        assert_ne!(id1, id2);
    }
}
