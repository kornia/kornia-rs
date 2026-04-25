//! Loader for ORB-SLAM3 / DBoW2 vocabulary text files.
//!
//! Parses the plain-text `ORBvoc.txt` emitted by ORB-SLAM3's DBoW2 fork and
//! reshapes it into the flat [`Vocabulary`] layout used by this crate.
//!
//! File format:
//! ```text
//! k L weighting scoring          // header
//! parent_id is_leaf b0 b1 ... b31 weight   // one node per line, node id = line number (1-based)
//! ```
//!
//! The root (id 0) is implicit — its children are the nodes with `parent_id == 0`.

use std::collections::VecDeque;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use crate::metric::{Feature, Hamming};
use crate::{BlockCluster, BlockContent, BowError, BowResult, InternalMeta, LeafData, Vocabulary};

/// Branching factor expected in ORB-SLAM3's vocabulary.
pub const ORB_SLAM3_B: usize = 10;
/// Number of u64 words per packed descriptor (256 bits = 4 × u64).
pub const ORB_SLAM3_D: usize = 4;

/// Packs a 32-byte ORB descriptor into 4 little-endian u64 words.
///
/// Use this for any query features transformed against a vocabulary loaded
/// via [`load_orb_slam3_vocabulary`] so that byte ordering matches.
#[inline]
pub fn pack_orb_descriptor(bytes: &[u8; 32]) -> Feature<u64, ORB_SLAM3_D> {
    let mut words = [0u64; ORB_SLAM3_D];
    for (i, w) in words.iter_mut().enumerate() {
        let mut chunk = [0u8; 8];
        chunk.copy_from_slice(&bytes[i * 8..(i + 1) * 8]);
        *w = u64::from_le_bytes(chunk);
    }
    Feature(words)
}

struct RawNode {
    parent: u32,
    is_leaf: bool,
    desc: Feature<u64, ORB_SLAM3_D>,
    weight: f32,
}

/// Loads an ORB-SLAM3 `ORBvoc.txt` file into a [`Vocabulary`].
pub fn load_orb_slam3_vocabulary<P: AsRef<Path>>(
    path: P,
) -> BowResult<Vocabulary<ORB_SLAM3_B, Hamming<ORB_SLAM3_D>>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    // Header: "k L weighting scoring"
    let mut header = String::new();
    reader.read_line(&mut header)?;
    let mut hp = header.split_whitespace();
    let k: usize = hp
        .next()
        .and_then(|s| s.parse().ok())
        .ok_or(BowError::CorruptedVocabulary)?;
    let _l: usize = hp
        .next()
        .and_then(|s| s.parse().ok())
        .ok_or(BowError::CorruptedVocabulary)?;
    // weighting and scoring are parsed but not used (we use the per-leaf weights as-is).
    let _ = hp.next();
    let _ = hp.next();

    if k != ORB_SLAM3_B {
        return Err(BowError::VocabularyMismatch {
            expected_b: ORB_SLAM3_B,
            found_b: k,
        });
    }

    // Parse all nodes. Node id = line index (1-based).
    let mut raw: Vec<RawNode> = Vec::with_capacity(1_200_000);
    let mut line = String::new();
    loop {
        line.clear();
        let n = reader.read_line(&mut line)?;
        if n == 0 {
            break;
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let mut it = trimmed.split_ascii_whitespace();
        let parent: u32 = it
            .next()
            .and_then(|s| s.parse().ok())
            .ok_or(BowError::CorruptedVocabulary)?;
        let is_leaf_flag: u32 = it
            .next()
            .and_then(|s| s.parse().ok())
            .ok_or(BowError::CorruptedVocabulary)?;

        let mut bytes = [0u8; 32];
        for b in bytes.iter_mut() {
            *b = it
                .next()
                .and_then(|s| s.parse().ok())
                .ok_or(BowError::CorruptedVocabulary)?;
        }
        let weight: f32 = it
            .next()
            .and_then(|s| s.parse().ok())
            .ok_or(BowError::CorruptedVocabulary)?;

        raw.push(RawNode {
            parent,
            is_leaf: is_leaf_flag != 0,
            desc: pack_orb_descriptor(&bytes),
            weight,
        });
    }

    if raw.is_empty() {
        return Err(BowError::NoFeatures);
    }

    // Group children by parent node id. Root has id 0.
    let n_nodes = raw.len();
    let mut children_of: Vec<Vec<u32>> = vec![Vec::new(); n_nodes + 1];
    for (i, r) in raw.iter().enumerate() {
        let own_id = (i as u32) + 1;
        let pidx = r.parent as usize;
        if pidx > n_nodes {
            return Err(BowError::CorruptedVocabulary);
        }
        children_of[pidx].push(own_id);
    }

    // BFS flatten into the crate's block layout.
    // Block 0 holds the root's children.
    let mut blocks: Vec<BlockCluster<ORB_SLAM3_B, Hamming<ORB_SLAM3_D>>> =
        vec![BlockCluster::default()];
    let mut queue: VecDeque<(u32, u32)> = VecDeque::new();
    queue.push_back((0, 0));
    let mut next_free: u32 = 1;

    while let Some((parent_node, block_idx)) = queue.pop_front() {
        let kids = &children_of[parent_node as usize];
        if kids.is_empty() {
            // Unexpected: internal node with no children. Write a dummy leaf to keep bounds valid.
            let mut b = BlockCluster::default();
            for d in b.descriptors.iter_mut() {
                *d = Feature([u64::MAX; ORB_SLAM3_D]);
            }
            b.content = BlockContent::Leaf(LeafData {
                weights: [0.0; ORB_SLAM3_B],
            });
            if block_idx as usize >= blocks.len() {
                blocks.resize(block_idx as usize + 1, BlockCluster::default());
            }
            blocks[block_idx as usize] = b;
            continue;
        }

        if kids.len() > ORB_SLAM3_B {
            return Err(BowError::CorruptedVocabulary);
        }

        let mut block: BlockCluster<ORB_SLAM3_B, Hamming<ORB_SLAM3_D>> = BlockCluster::default();
        for d in block.descriptors.iter_mut() {
            *d = Feature([u64::MAX; ORB_SLAM3_D]);
        }

        let any_internal = kids.iter().any(|&c| !raw[(c - 1) as usize].is_leaf);

        if !any_internal {
            // Leaf block: every child is a leaf — fold them directly.
            let mut weights = [0.0f32; ORB_SLAM3_B];
            for (i, &c) in kids.iter().enumerate() {
                let r = &raw[(c - 1) as usize];
                block.descriptors[i] = r.desc;
                weights[i] = r.weight;
            }
            // Pad unused slots with the last real descriptor so traversal's argmin
            // can never prefer a padded slot over a real one.
            if let Some(&last) = kids.last() {
                let r = &raw[(last - 1) as usize];
                for i in kids.len()..ORB_SLAM3_B {
                    block.descriptors[i] = r.desc;
                    weights[i] = r.weight;
                }
            }
            block.content = BlockContent::Leaf(LeafData { weights });
        } else {
            // Internal block: allocate B child slots even if some children are leaves.
            let children_base = next_free;
            next_free += ORB_SLAM3_B as u32;
            if blocks.len() < next_free as usize {
                blocks.resize(next_free as usize, BlockCluster::default());
            }
            block.content = BlockContent::Internal(InternalMeta {
                children_base_idx: children_base,
            });

            for (i, &c) in kids.iter().enumerate() {
                let r = &raw[(c - 1) as usize];
                block.descriptors[i] = r.desc;
                let child_block_idx = children_base + i as u32;
                if r.is_leaf {
                    // Wrapper leaf block: slot 0 is the real leaf, others replicate
                    // the same descriptor+weight so any argmin tie still returns the
                    // correct (node_id, weight).
                    let mut wb: BlockCluster<ORB_SLAM3_B, Hamming<ORB_SLAM3_D>> =
                        BlockCluster::default();
                    for d in wb.descriptors.iter_mut() {
                        *d = r.desc;
                    }
                    let weights = [r.weight; ORB_SLAM3_B];
                    wb.content = BlockContent::Leaf(LeafData { weights });
                    blocks[child_block_idx as usize] = wb;
                } else {
                    queue.push_back((c, child_block_idx));
                }
            }

            // Pad unused slots of this internal block the same way as above.
            if let Some(&last) = kids.last() {
                let r = &raw[(last - 1) as usize];
                for i in kids.len()..ORB_SLAM3_B {
                    block.descriptors[i] = r.desc;
                    // Also dead-end these slots with a dummy leaf wrapper to keep
                    // children_base_idx + i in bounds for traversal.
                    let dummy_idx = children_base + i as u32;
                    let mut db: BlockCluster<ORB_SLAM3_B, Hamming<ORB_SLAM3_D>> =
                        BlockCluster::default();
                    for d in db.descriptors.iter_mut() {
                        *d = r.desc;
                    }
                    db.content = BlockContent::Leaf(LeafData {
                        weights: [r.weight; ORB_SLAM3_B],
                    });
                    blocks[dummy_idx as usize] = db;
                }
            }
        }

        if block_idx as usize >= blocks.len() {
            blocks.resize(block_idx as usize + 1, BlockCluster::default());
        }
        blocks[block_idx as usize] = block;
    }

    // Safety check: every Internal block must point to a valid child range.
    for b in &blocks {
        if let BlockContent::Internal(meta) = b.content {
            let end = meta.children_base_idx as u64 + ORB_SLAM3_B as u64;
            if end > blocks.len() as u64 {
                return Err(BowError::CorruptedVocabulary);
            }
        }
    }

    blocks.shrink_to_fit();

    Ok(Vocabulary {
        blocks,
        root_idx: 0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_descriptor_le() {
        let mut bytes = [0u8; 32];
        bytes[0] = 0x01;
        bytes[8] = 0x02;
        let f = pack_orb_descriptor(&bytes);
        assert_eq!(f.0[0], 0x0000_0000_0000_0001);
        assert_eq!(f.0[1], 0x0000_0000_0000_0002);
    }
}
