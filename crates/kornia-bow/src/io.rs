use std::fs::File;
use std::io::{BufReader, BufWriter};

use crate::{
    metric::{DistanceMetric, MetricType},
    BlockContent, BowError, BowResult, Vocabulary,
};

impl<const B: usize, M: DistanceMetric> Vocabulary<B, M> {
    /// Persists the vocabulary to a file.
    pub fn save(&self, path: &str) -> BowResult<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        let config = bincode::config::standard();

        bincode::encode_into_std_write(B as u64, &mut writer, config)
            .map_err(|e| BowError::Bincode(e.to_string()))?;
        bincode::encode_into_std_write(M::metric_type(), &mut writer, config)
            .map_err(|e| BowError::Bincode(e.to_string()))?;
        bincode::encode_into_std_write(self, &mut writer, config)
            .map_err(|e| BowError::Bincode(e.to_string()))?;

        Ok(())
    }

    /// Loads the vocabulary from a file and verifies compatibility.
    pub fn load(path: &str) -> BowResult<Self> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let config = bincode::config::standard();

        let loaded_b: u64 = bincode::decode_from_std_read(&mut reader, config)
            .map_err(|e| BowError::Bincode(e.to_string()))?;
        let loaded_metric: MetricType = bincode::decode_from_std_read(&mut reader, config)
            .map_err(|e| BowError::Bincode(e.to_string()))?;

        if loaded_b != B as u64 {
            return Err(BowError::VocabularyMismatch {
                expected_b: B,
                found_b: loaded_b as usize,
            });
        }

        if loaded_metric != M::metric_type() {
            return Err(BowError::MetricMismatch {
                expected: M::metric_type(),
                found: loaded_metric,
            });
        }

        let vocab: Vocabulary<B, M> = bincode::decode_from_std_read(&mut reader, config)
            .map_err(|e| BowError::Bincode(e.to_string()))?;

        // Validate structure to ensure unsafe traversal is safe
        for block in &vocab.blocks {
            if let BlockContent::Internal(meta) = block.content {
                // Ensure children_base_idx + B is within bounds
                // We use u64 for calculation to avoid overflow before comparison
                let end_idx = meta.children_base_idx as u64 + B as u64;
                if end_idx > vocab.blocks.len() as u64 {
                    return Err(BowError::CorruptedVocabulary);
                }
            }
        }

        Ok(vocab)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metric::{Feature, Hamming, L2};
    use crate::{BlockCluster, InternalMeta};
    use rand::{rngs::StdRng, Rng, SeedableRng};

    const B: usize = 10;
    const D: usize = 4;

    #[test]
    fn test_load_corrupted_vocabulary() {
        // Create a vocabulary with an invalid child index
        let mut blocks = Vec::new();
        let block = BlockCluster {
            content: BlockContent::Internal(InternalMeta {
                children_base_idx: 100,
            }),
            ..Default::default()
        };
        blocks.push(block);

        let vocab: Vocabulary<B, Hamming<D>> = Vocabulary {
            blocks,
            root_idx: 0,
        };

        let path = "test_corrupted.bin";
        vocab.save(path).unwrap();

        let result = Vocabulary::<B, Hamming<D>>::load(path);
        std::fs::remove_file(path).unwrap();

        assert!(matches!(result, Err(BowError::CorruptedVocabulary)));
    }

    #[test]
    fn test_load_non_existent_file() {
        let result = Vocabulary::<B, Hamming<D>>::load("non_existent_file.bow");
        assert!(matches!(result, Err(BowError::Io(_))));
    }

    #[test]
    fn test_save_invalid_path() {
        let vocab: Vocabulary<B, Hamming<D>> = Vocabulary {
            blocks: Vec::new(),
            root_idx: 0,
        };
        let result = vocab.save("/path/to/non/existent/directory/vocab.bow");
        assert!(matches!(result, Err(BowError::Io(_))));
    }

    #[test]
    fn test_save_and_load() {
        let mut rng = StdRng::from_seed([42; 32]);
        let data: Vec<Feature<u64, D>> = (0..100).map(|_| Feature(rng.random())).collect();
        let vocab = Vocabulary::<B, Hamming<D>>::train(&data, 3).unwrap();

        let path = "test_vocab.bin";
        vocab.save(path).unwrap();
        let loaded_vocab = Vocabulary::<B, Hamming<D>>::load(path).unwrap();
        std::fs::remove_file(path).unwrap();

        assert_eq!(vocab.root_idx, loaded_vocab.root_idx);
        assert_eq!(vocab.blocks.len(), loaded_vocab.blocks.len());
    }

    #[test]
    fn test_l2_save_load() {
        let mut rng = StdRng::from_seed([42; 32]);
        let data: Vec<Feature<f32, 16>> = (0..50)
            .map(|_| {
                let mut desc = [0.0f32; 16];
                for val in desc.iter_mut() {
                    *val = rng.random();
                }
                Feature(desc)
            })
            .collect();

        let vocab = Vocabulary::<B, L2<16>>::train(&data, 2).unwrap();

        let path = "test_l2_vocab.bin";
        vocab.save(path).unwrap();
        let loaded_vocab = Vocabulary::<B, L2<16>>::load(path).unwrap();
        std::fs::remove_file(path).unwrap();

        assert_eq!(vocab.blocks.len(), loaded_vocab.blocks.len());
    }

    #[test]
    fn test_vocabulary_mismatch_b() {
        let mut rng = StdRng::from_seed([42; 32]);
        let data: Vec<Feature<u64, D>> = (0..100).map(|_| Feature(rng.random())).collect();
        let vocab = Vocabulary::<10, Hamming<D>>::train(&data, 3).unwrap();

        let path = "test_b_mismatch.bin";
        vocab.save(path).unwrap();

        // Try to load with B=8
        let result = Vocabulary::<8, Hamming<D>>::load(path);
        std::fs::remove_file(path).unwrap();

        assert!(matches!(
            result,
            Err(BowError::VocabularyMismatch {
                expected_b: 8,
                found_b: 10
            })
        ));
    }

    #[test]
    fn test_vocabulary_mismatch_metric() {
        let mut rng = StdRng::from_seed([42; 32]);
        let data: Vec<Feature<u64, 16>> = (0..100)
            .map(|_| {
                let mut d = [0u64; 16];
                for v in d.iter_mut() {
                    *v = rng.random();
                }
                Feature(d)
            })
            .collect();
        let vocab = Vocabulary::<B, Hamming<16>>::train(&data, 3).unwrap();

        let path = "test_metric_mismatch.bin";
        vocab.save(path).unwrap();

        // Try to load with L2 metric (assuming data layout compatibility for the sake of the error check)
        let result = Vocabulary::<B, L2<16>>::load(path);
        std::fs::remove_file(path).unwrap();

        assert!(matches!(
            result,
            Err(BowError::MetricMismatch {
                expected: MetricType::L2,
                found: MetricType::Hamming
            })
        ));
    }
}
