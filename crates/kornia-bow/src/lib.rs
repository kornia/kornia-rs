use bincode::{Decode, Encode};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_big_array::BigArray;

pub mod bow;
pub mod constructor;
pub mod io;
pub mod metric;

pub use bow::{BoW, DirectIndex};
use metric::{DistanceMetric, MetricType};
use thiserror::Error;

/// Errors related to Bag of Words operations.
#[derive(Debug, Error)]
pub enum BowError {
    #[error("No features provided")]
    NoFeatures,
    #[error("Io error")]
    Io(#[from] std::io::Error),
    #[error("Bincode error: {0}")]
    Bincode(String),
    #[error("Vocabulary mismatch: expected B={expected_b}, but found B={found_b}")]
    VocabularyMismatch { expected_b: usize, found_b: usize },
    #[error("Metric mismatch: expected {expected:?}, but found {found:?}")]
    MetricMismatch {
        expected: MetricType,
        found: MetricType,
    },
    #[error("Corrupted vocabulary: block index out of bounds")]
    CorruptedVocabulary,
}

pub type BowResult<T> = Result<T, BowError>;

/// A block representing a set of children in the vocabulary tree.
#[derive(Clone, Serialize, Deserialize)]
pub struct BlockCluster<const B: usize, M: DistanceMetric> {
    #[serde(with = "BigArray")]
    pub descriptors: [M::Data; B],
    pub content: BlockContent<B>,
}

impl<const B: usize, M: DistanceMetric> Encode for BlockCluster<B, M> {
    fn encode<E: bincode::enc::Encoder>(
        &self,
        encoder: &mut E,
    ) -> Result<(), bincode::error::EncodeError> {
        Encode::encode(&self.descriptors, encoder)?;
        Encode::encode(&self.content, encoder)?;
        Ok(())
    }
}

impl<const B: usize, M: DistanceMetric> Decode<()> for BlockCluster<B, M> {
    fn decode<D: bincode::de::Decoder<Context = ()>>(
        decoder: &mut D,
    ) -> Result<Self, bincode::error::DecodeError> {
        Ok(Self {
            descriptors: Decode::decode(decoder)?,
            content: Decode::decode(decoder)?,
        })
    }
}

impl<const B: usize, M: DistanceMetric> Copy for BlockCluster<B, M> where M::Data: Copy {}

impl<const B: usize, M: DistanceMetric> Default for BlockCluster<B, M> {
    fn default() -> Self {
        Self {
            descriptors: [M::Data::default(); B],
            content: BlockContent::Internal(InternalMeta {
                children_base_idx: 0,
            }),
        }
    }
}

/// The content of a block, either metadata for internal nodes or weights for leaves.
#[derive(Clone, Copy, Serialize, Deserialize, Encode, Decode)]
pub enum BlockContent<const B: usize> {
    Internal(InternalMeta),
    Leaf(LeafData<B>),
}

/// Metadata for internal nodes pointing to the base index of their children.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, Encode, Decode)]
pub struct InternalMeta {
    pub children_base_idx: u32,
}

/// Data for leaf nodes containing weights for each word in the block.
#[repr(C)]
#[derive(Clone, Copy, Debug, Serialize, Deserialize, Encode, Decode)]
pub struct LeafData<const B: usize> {
    #[serde(with = "BigArray")]
    pub weights: [f32; B],
}

impl<const B: usize> Default for LeafData<B> {
    fn default() -> Self {
        Self { weights: [0.0; B] }
    }
}

/// A highly optimized hierarchical vocabulary for fast BoW computation.
#[derive(Serialize, Deserialize)]
#[serde(bound = "M: DistanceMetric")]
pub struct Vocabulary<const B: usize, M: DistanceMetric> {
    pub blocks: Vec<BlockCluster<B, M>>,
    pub root_idx: u32,
}

impl<const B: usize, M: DistanceMetric> Encode for Vocabulary<B, M> {
    fn encode<E: bincode::enc::Encoder>(
        &self,
        encoder: &mut E,
    ) -> Result<(), bincode::error::EncodeError> {
        Encode::encode(&self.blocks, encoder)?;
        Encode::encode(&self.root_idx, encoder)?;
        Ok(())
    }
}

impl<const B: usize, M: DistanceMetric> Decode<()> for Vocabulary<B, M> {
    fn decode<D: bincode::de::Decoder<Context = ()>>(
        decoder: &mut D,
    ) -> Result<Self, bincode::error::DecodeError> {
        Ok(Self {
            blocks: Decode::decode(decoder)?,
            root_idx: Decode::decode(decoder)?,
        })
    }
}

impl<const B: usize, M: DistanceMetric> Vocabulary<B, M> {
    /// Trains a hierarchical vocabulary tree.
    pub fn train(data: &[M::Data], max_depth: usize) -> BowResult<Self> {
        constructor::train::<B, M>(data, max_depth)
    }

    /// Maps a single descriptor to a word index and its IDF weight.
    #[inline(always)]
    pub fn transform_one(&self, feature: &M::Data) -> (u32, f32) {
        let (word_id, weight, _) = self.traverse(feature, false);
        (word_id, weight)
    }

    /// Traverses the tree for a single descriptor.
    pub fn traverse(&self, feature: &M::Data, collect_path: bool) -> (u32, f32, Vec<u32>) {
        let mut curr_idx = self.root_idx;
        let mut path = if collect_path {
            Vec::with_capacity(8)
        } else {
            Vec::new()
        };

        if collect_path {
            path.push(curr_idx);
        }

        loop {
            let block = unsafe { self.blocks.get_unchecked(curr_idx as usize) };

            let mut best_dist = M::max_distance();
            let mut best_child_index = 0;

            for i in 0..B {
                let dist = M::distance(feature, &block.descriptors[i]);
                if dist < best_dist {
                    best_dist = dist;
                    best_child_index = i;
                }
            }

            let node_id = (curr_idx * B as u32) + best_child_index as u32;
            if collect_path {
                path.push(node_id);
            }

            match block.content {
                BlockContent::Internal(meta) => {
                    curr_idx = meta.children_base_idx + best_child_index as u32;
                }
                BlockContent::Leaf(leaf) => {
                    return (node_id, leaf.weights[best_child_index], path);
                }
            }
        }
    }

    /// Transforms a set of descriptors into a sparse BoW vector.
    pub fn transform(&self, features: &[M::Data]) -> BowResult<BoW> {
        if features.is_empty() {
            return Err(BowError::NoFeatures);
        }

        let tf_factor = 1.0 / (features.len() as f32);
        let mut raw_entries: Vec<(u32, f32)> = features
            .par_iter()
            .map(|feature| {
                let (word_id, idf_weight, _) = self.traverse(feature, false);
                (word_id, idf_weight * tf_factor)
            })
            .collect();

        raw_entries.sort_unstable_by_key(|k| k.0);

        let mut compacted_bow: Vec<(u32, f32)> = Vec::with_capacity(raw_entries.len());

        if !raw_entries.is_empty() {
            let mut current_id = raw_entries[0].0;
            let mut current_val = raw_entries[0].1;

            for &(next_id, next_val) in &raw_entries[1..] {
                if next_id == current_id {
                    current_val += next_val;
                } else {
                    compacted_bow.push((current_id, current_val));
                    current_id = next_id;
                    current_val = next_val;
                }
            }
            compacted_bow.push((current_id, current_val));
        }

        let sum: f32 = compacted_bow.iter().map(|&(_, v)| v).sum();
        if sum > 0.0 {
            for (_, v) in compacted_bow.iter_mut() {
                *v /= sum;
            }
        }

        Ok(BoW(compacted_bow))
    }

    /// Transforms descriptors into a sparse BoW vector and a Direct Index.
    pub fn transform_with_direct_index(
        &self,
        features: &[M::Data],
        level: usize,
    ) -> BowResult<(BoW, DirectIndex)> {
        if features.is_empty() {
            return Err(BowError::NoFeatures);
        }

        let tf_factor = 1.0 / (features.len() as f32);

        let results: Vec<(u32, f32, Option<u32>)> = features
            .par_iter()
            .map(|feature| {
                let (word_id, idf_weight, path) = self.traverse(feature, true);
                let node_at_level = path.get(level).copied();
                (word_id, idf_weight * tf_factor, node_at_level)
            })
            .collect();

        let mut raw_entries: Vec<(u32, f32)> = Vec::with_capacity(features.len());
        let mut di_entries: Vec<(u32, u32)> = Vec::with_capacity(features.len());

        for (feat_idx, (word_id, weight, node_at_level)) in results.into_iter().enumerate() {
            raw_entries.push((word_id, weight));
            if let Some(node_id) = node_at_level {
                di_entries.push((node_id, feat_idx as u32));
            }
        }

        raw_entries.sort_unstable_by_key(|k| k.0);
        let mut compacted_bow: Vec<(u32, f32)> = Vec::with_capacity(raw_entries.len());

        if !raw_entries.is_empty() {
            let mut current_id = raw_entries[0].0;
            let mut current_val = raw_entries[0].1;

            for &(next_id, next_val) in &raw_entries[1..] {
                if next_id == current_id {
                    current_val += next_val;
                } else {
                    compacted_bow.push((current_id, current_val));
                    current_id = next_id;
                    current_val = next_val;
                }
            }
            compacted_bow.push((current_id, current_val));
        }

        let sum: f32 = compacted_bow.iter().map(|&(_, v)| v).sum();
        if sum > 0.0 {
            for (_, v) in compacted_bow.iter_mut() {
                *v /= sum;
            }
        }

        di_entries.sort_unstable_by_key(|k| k.0);
        let mut compacted_di: Vec<(u32, Vec<u32>)> = Vec::with_capacity(di_entries.len().min(100));

        if !di_entries.is_empty() {
            let mut current_node = di_entries[0].0;
            let mut current_feats = vec![di_entries[0].1];

            for &(next_node, next_feat) in &di_entries[1..] {
                if next_node == current_node {
                    current_feats.push(next_feat);
                } else {
                    compacted_di.push((current_node, current_feats));
                    current_node = next_node;
                    current_feats = vec![next_feat];
                }
            }
            compacted_di.push((current_node, current_feats));
        }

        Ok((BoW(compacted_bow), DirectIndex(compacted_di)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metric::{Feature, Hamming};
    use rand::{rngs::StdRng, Rng, SeedableRng};

    const B: usize = 10;
    const D: usize = 4;

    fn generate_random_descriptors(count: usize) -> Vec<Feature<u64, D>> {
        let mut rng = StdRng::from_seed([42; 32]);
        (0..count).map(|_| Feature(rng.random())).collect()
    }

    #[test]
    fn test_transform() {
        let data = generate_random_descriptors(100);
        let vocab = Vocabulary::<B, Hamming<D>>::train(&data, 3).unwrap();

        let features = generate_random_descriptors(10);
        let bow = vocab.transform(&features).unwrap();

        assert!(!bow.0.is_empty());
        let sum_weights: f32 = bow.0.iter().map(|&(_, w)| w).sum();
        assert!((sum_weights - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_transform_with_direct_index() {
        let data = generate_random_descriptors(100);
        let vocab = Vocabulary::<B, Hamming<D>>::train(&data, 3).unwrap();

        let features = generate_random_descriptors(10);
        let (bow, di) = vocab.transform_with_direct_index(&features, 1).unwrap();

        assert!(!bow.0.is_empty());
        assert!(!di.0.is_empty());

        let total_features: usize = di.0.iter().map(|(_, f)| f.len()).sum();
        assert_eq!(total_features, 10);
    }
}
