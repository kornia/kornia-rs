use std::fs::File;
use std::io::{BufReader, BufWriter};

use crate::{
    metric::{DistanceMetric, MetricType},
    BlockCluster, BlockContent, BowError, BowResult, Vocabulary,
};

/// Upper bound on the bytes a single decoded value (header field or block) may claim.
const MAX_DECODE_BYTES: usize = 64 * 1024 * 1024;

/// Blocks reserved up front while loading; the rest grow as data is actually read.
const PREALLOC_BLOCKS: usize = 1 << 16;

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
    ///
    /// The file is treated as untrusted: declared lengths are checked against the
    /// file size before allocating, every decode is byte-limited, and the block
    /// graph indices (root and children) are validated to be in bounds.
    ///
    /// # Arguments
    ///
    /// * `path` - Path of a file written by [`Vocabulary::save`].
    ///
    /// # Returns
    ///
    /// The decoded vocabulary.
    ///
    /// # Errors
    ///
    /// Returns [`BowError::Io`] / [`BowError::Bincode`] on read or decode failure,
    /// [`BowError::VocabularyMismatch`] / [`BowError::MetricMismatch`] if the file
    /// was written for a different `B` or metric, [`BowError::VocabularyTooLarge`]
    /// if the declared block count cannot fit in the file, and
    /// [`BowError::CorruptedVocabulary`] if the vocabulary is empty or a root or
    /// child index is out of bounds.
    pub fn load(path: &str) -> BowResult<Self> {
        let file = File::open(path)?;
        let file_len = file.metadata()?.len();
        let mut reader = BufReader::new(file);
        // Every value below is decoded with its own call, so this caps the bytes a
        // single header field or block may claim (bincode checks container lengths
        // against it before allocating).
        let config = bincode::config::standard().with_limit::<MAX_DECODE_BYTES>();

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

        // Decode `Vocabulary` field by field (same wire format as its `Encode` impl:
        // `Vec` length, blocks, root index) so the attacker-controlled block count is
        // validated against the file size before anything is allocated for it.
        let n_blocks: u64 = bincode::decode_from_std_read(&mut reader, config)
            .map_err(|e| BowError::Bincode(e.to_string()))?;
        // Every encoded block takes at least one byte (its `BlockContent` tag).
        if n_blocks > file_len {
            return Err(BowError::VocabularyTooLarge {
                declared: n_blocks,
                file_len,
            });
        }
        let n_blocks = n_blocks as usize;
        let mut blocks = Vec::with_capacity(n_blocks.min(PREALLOC_BLOCKS));
        for _ in 0..n_blocks {
            let block: BlockCluster<B, M> = bincode::decode_from_std_read(&mut reader, config)
                .map_err(|e| BowError::Bincode(e.to_string()))?;
            blocks.push(block);
        }
        let root_idx: u32 = bincode::decode_from_std_read(&mut reader, config)
            .map_err(|e| BowError::Bincode(e.to_string()))?;
        let vocab = Vocabulary { blocks, root_idx };

        // The root must name an existing block (this also rejects empty vocabularies).
        if vocab.root_idx as usize >= vocab.blocks.len() {
            return Err(BowError::CorruptedVocabulary);
        }

        // Validate child indices so traversal stays in bounds.
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
    use rand::{rngs::StdRng, RngExt, SeedableRng};

    const B: usize = 10;
    const D: usize = 4;

    /// Writes a hand-crafted vocabulary file: header, block count, blocks, root index.
    fn write_raw_vocab(name: &str, n_blocks: u64, blocks: &[u8], root_idx: u32) -> String {
        let cfg = bincode::config::standard();
        let mut buf = Vec::new();
        buf.extend(bincode::encode_to_vec(B as u64, cfg).unwrap());
        buf.extend(bincode::encode_to_vec(MetricType::Hamming, cfg).unwrap());
        buf.extend(bincode::encode_to_vec(n_blocks, cfg).unwrap());
        buf.extend_from_slice(blocks);
        buf.extend(bincode::encode_to_vec(root_idx, cfg).unwrap());
        let path = std::env::temp_dir().join(format!("kornia_bow_{}_{name}", std::process::id()));
        std::fs::write(&path, buf).unwrap();
        path.to_string_lossy().into_owned()
    }

    #[test]
    fn test_load_rejects_out_of_bounds_root() {
        // Regression: root_idx was never validated and traversal used get_unchecked.
        let path = write_raw_vocab("root.bow", 0, &[], 0x4000_0000);
        let result = Vocabulary::<B, Hamming<D>>::load(&path);
        std::fs::remove_file(&path).unwrap();
        assert!(matches!(result, Err(BowError::CorruptedVocabulary)));

        // Same with a non-empty vocabulary.
        let vocab: Vocabulary<B, Hamming<D>> = Vocabulary {
            blocks: vec![BlockCluster {
                content: BlockContent::Leaf(Default::default()),
                ..Default::default()
            }],
            root_idx: 1,
        };
        let path =
            std::env::temp_dir().join(format!("kornia_bow_{}_root2.bow", std::process::id()));
        let path = path.to_string_lossy().into_owned();
        vocab.save(&path).unwrap();
        let result = Vocabulary::<B, Hamming<D>>::load(&path);
        std::fs::remove_file(&path).unwrap();
        assert!(matches!(result, Err(BowError::CorruptedVocabulary)));
    }

    #[test]
    fn test_load_rejects_huge_block_count() {
        // Regression: an attacker-declared Vec length used to be allocated up front
        // (capacity overflow / allocation abort).
        let path = write_raw_vocab("len.bow", u64::MAX / 1024, &[], 0);
        let result = Vocabulary::<B, Hamming<D>>::load(&path);
        std::fs::remove_file(&path).unwrap();
        assert!(matches!(result, Err(BowError::VocabularyTooLarge { .. })));
    }

    #[test]
    fn test_load_cyclic_vocabulary_traversal_terminates() {
        // Every block is internal and points back at blocks 0..B: in bounds, but
        // cyclic. Traversal used to loop forever.
        let mut blocks = vec![BlockCluster::<B, Hamming<D>>::default(); B];
        for b in blocks.iter_mut() {
            b.content = BlockContent::Internal(InternalMeta {
                children_base_idx: 0,
            });
        }
        let vocab = Vocabulary::<B, Hamming<D>> {
            blocks,
            root_idx: 0,
        };
        let path =
            std::env::temp_dir().join(format!("kornia_bow_{}_cycle.bow", std::process::id()));
        let path = path.to_string_lossy().into_owned();
        vocab.save(&path).unwrap();
        let loaded = Vocabulary::<B, Hamming<D>>::load(&path);
        std::fs::remove_file(&path).unwrap();
        let loaded = loaded.unwrap();

        let (_, weight, path) = loaded.traverse(&Feature([0u64; D]), true);
        assert_eq!(weight, 0.0);
        assert!(path.len() <= B + 1);
    }

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
