//! Loader for ORB-SLAM3 / DBoW2 text vocabularies (`ORBvoc.txt`).
//!
//! DBoW2 stores its vocabulary as a flat list of tree nodes. The first line is
//! the header `k L scoring weighting` (ORBvoc uses `10 6 0 0`: branching factor
//! 10, depth 6, L1 scoring, TF-IDF weighting). Every following line describes
//! one node:
//!
//! ```text
//!   <parent_id> <is_leaf> <b0> <b1> ... <b31> <weight>
//! ```
//!
//! where `b0..b31` are the 32 bytes of the node's ORB descriptor and `weight`
//! is the precomputed IDF weight (`ln(N / n_i)`, nonzero only on leaves). Node
//! ids are implicit: the root is node 0 and each subsequent line is the next
//! id in order, so a line's `parent_id` always references an earlier node.
//!
//! This module reshapes that node list into [`Vocabulary`]'s cache-friendly
//! block layout (one [`BlockCluster`] per internal node, holding its up-to-`B`
//! children) so traversal and [`Vocabulary::transform`] work unchanged. Because
//! the file already carries final IDF weights, leaf weights are copied verbatim
//! rather than recomputed from feature counts.

use std::collections::VecDeque;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use crate::metric::{DistanceMetric, Feature, Hamming};
use crate::{BlockCluster, BlockContent, BowError, BowResult, InternalMeta, LeafData, Vocabulary};

/// Branching factor of the ORB-SLAM3 vocabulary (`k` in the DBoW2 header).
pub const ORB_BRANCHING: usize = 10;
/// ORB descriptor width in `u64` words (256 bits = 4 × 64).
pub const ORB_WORDS: usize = 4;

/// The concrete vocabulary type for ORB descriptors: branching 10, Hamming over
/// 256-bit (`4 × u64`) descriptors.
pub type OrbVocabulary = Vocabulary<ORB_BRANCHING, Hamming<ORB_WORDS>>;

/// Packs a 32-byte ORB descriptor into the `Feature<u64, 4>` used by the
/// Hamming metric. The byte order is fixed (little-endian per 8-byte group) and
/// shared by the vocabulary loader and every query, so the absolute layout is
/// irrelevant as long as both sides agree.
#[inline]
pub fn pack_orb_descriptor(bytes: &[u8; 32]) -> Feature<u64, ORB_WORDS> {
    let mut words = [0u64; ORB_WORDS];
    for (w, chunk) in words.iter_mut().zip(bytes.chunks_exact(8)) {
        let mut arr = [0u8; 8];
        arr.copy_from_slice(chunk);
        *w = u64::from_le_bytes(arr);
    }
    Feature(words)
}

/// A parsed DBoW2 node prior to block layout.
struct DbowNode {
    descriptor: Feature<u64, ORB_WORDS>,
    weight: f32,
    children: Vec<usize>,
    is_leaf: bool,
}

/// Loads an ORB-SLAM3 text vocabulary (`ORBvoc.txt`) into an [`OrbVocabulary`].
///
/// The parse is single-pass and tolerant of blank trailing lines. Loading the
/// full 1M-word ORBvoc takes a few seconds; convert once and persist with
/// [`Vocabulary::save`] for fast subsequent loads.
pub fn load_orb_slam3_vocabulary<P: AsRef<Path>>(path: P) -> BowResult<OrbVocabulary> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    let mut header = String::new();
    reader.read_line(&mut header)?;
    let mut hdr = header.split_whitespace();
    let k: usize = hdr
        .next()
        .and_then(|s| s.parse().ok())
        .ok_or(BowError::CorruptedVocabulary)?;
    if k != ORB_BRANCHING {
        return Err(BowError::VocabularyMismatch {
            expected_b: ORB_BRANCHING,
            found_b: k,
        });
    }

    // Node 0 is the implicit root; data lines fill in ids 1, 2, ... in order.
    let mut nodes: Vec<DbowNode> = vec![DbowNode {
        descriptor: Feature([0u64; ORB_WORDS]),
        weight: 0.0,
        children: Vec::new(),
        is_leaf: false,
    }];

    let mut line = String::new();
    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            break;
        }
        if line.trim().is_empty() {
            continue;
        }

        let mut it = line.split_whitespace();
        let parent: usize = it
            .next()
            .and_then(|s| s.parse().ok())
            .ok_or(BowError::CorruptedVocabulary)?;
        let is_leaf: u32 = it
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

        if parent >= nodes.len() {
            return Err(BowError::CorruptedVocabulary);
        }

        let id = nodes.len();
        nodes.push(DbowNode {
            descriptor: pack_orb_descriptor(&bytes),
            weight,
            children: Vec::new(),
            is_leaf: is_leaf > 0,
        });
        nodes[parent].children.push(id);
    }

    build_vocabulary(&nodes)
}

/// Reshapes the parsed DBoW2 node tree into [`Vocabulary`]'s block layout.
///
/// Mirrors `constructor::convert_to_vocabulary`: a BFS assigns one block per
/// internal node, lays its children's blocks out contiguously, and collapses a
/// node whose children are all leaves into a single `Leaf` block. Leaf children
/// of an otherwise-internal node are wrapped into their own one-entry leaf
/// block (the same trick the trainer uses for unbalanced trees). Leaf weights
/// come straight from the file.
fn build_vocabulary(nodes: &[DbowNode]) -> BowResult<OrbVocabulary> {
    let padding = Hamming::<ORB_WORDS>::padding();

    // Fill for block slots that are never assigned a real node — the padding
    // child slots of an internal node with fewer than `B` children. A default
    // `BlockCluster` is an `Internal` node pointing back to the root, so a query
    // that descends into one (a query nearer the `u64::MAX` padding descriptor
    // than any real child) would loop forever. A self-terminating leaf block
    // ends traversal harmlessly with weight 0 instead. Real ORBvoc nodes are
    // full so this never fires, but it keeps the loader safe for unbalanced
    // DBoW2 trees.
    let terminator = BlockCluster {
        descriptors: [padding; ORB_BRANCHING],
        content: BlockContent::Leaf(LeafData {
            weights: [0.0; ORB_BRANCHING],
        }),
    };

    let mut blocks: Vec<BlockCluster<ORB_BRANCHING, Hamming<ORB_WORDS>>> = vec![terminator];

    // Each queue entry is (children node ids of the block, block index).
    let mut queue: VecDeque<(Vec<usize>, u32)> = VecDeque::new();
    queue.push_back((nodes[0].children.clone(), 0));

    let mut next_free_idx = 1u32;

    while let Some((child_ids, block_idx)) = queue.pop_front() {
        let mut block = BlockCluster::default();

        let n_children = child_ids.len();
        if n_children == 0 || n_children > ORB_BRANCHING {
            return Err(BowError::CorruptedVocabulary);
        }

        let is_leaf_layer = child_ids.iter().all(|&c| nodes[c].is_leaf);

        if is_leaf_layer {
            let mut weights = [0.0f32; ORB_BRANCHING];
            for (i, &cid) in child_ids.iter().enumerate() {
                block.descriptors[i] = nodes[cid].descriptor;
                weights[i] = nodes[cid].weight;
            }
            block.content = BlockContent::Leaf(LeafData { weights });
        } else {
            let children_base = next_free_idx;
            block.content = BlockContent::Internal(InternalMeta {
                children_base_idx: children_base,
            });

            next_free_idx += ORB_BRANCHING as u32;
            if blocks.len() < next_free_idx as usize {
                blocks.resize(next_free_idx as usize, terminator);
            }

            for (i, &cid) in child_ids.iter().enumerate() {
                block.descriptors[i] = nodes[cid].descriptor;
                let child_block_idx = children_base + i as u32;

                if nodes[cid].is_leaf {
                    // A leaf sitting at an internal layer becomes its own
                    // single-entry leaf block so the parent stays uniform.
                    queue.push_back((vec![cid], child_block_idx));
                } else {
                    queue.push_back((nodes[cid].children.clone(), child_block_idx));
                }
            }
        }

        // Pad the unused descriptor slots of an under-full node with a copy of
        // the first real child's descriptor. Traversal picks the argmin with a
        // strict `<`, so a padded slot can only ever tie — never beat — the real
        // child at index 0, and the lower index wins. This keeps a high-bit-count
        // query from being lured into a `u64::MAX` padding slot (and thus a dead
        // terminator block). Real ORBvoc nodes are full, so this only matters for
        // unbalanced DBoW2 trees.
        for i in n_children..ORB_BRANCHING {
            block.descriptors[i] = block.descriptors[0];
        }

        if block_idx as usize >= blocks.len() {
            blocks.resize(block_idx as usize + 1, terminator);
        }
        blocks[block_idx as usize] = block;
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
    fn pack_roundtrip_bit_count() {
        let mut bytes = [0u8; 32];
        bytes[0] = 0b1010_1010;
        bytes[8] = 0b0000_1111;
        let f = pack_orb_descriptor(&bytes);
        let total: u32 = f.0.iter().map(|w| w.count_ones()).sum();
        assert_eq!(total, 4 + 4);
    }

    /// A tiny hand-built two-level vocabulary: root with 2 internal children,
    /// each with 2 leaf children. Exercises both the internal and leaf paths.
    #[test]
    fn build_small_tree() {
        let leaf = |b: u8, w: f32| DbowNode {
            descriptor: pack_orb_descriptor(&[b; 32]),
            weight: w,
            children: Vec::new(),
            is_leaf: true,
        };
        let mut nodes = vec![DbowNode {
            descriptor: Feature([0; ORB_WORDS]),
            weight: 0.0,
            children: vec![1, 2],
            is_leaf: false,
        }];
        nodes.push(DbowNode {
            descriptor: pack_orb_descriptor(&[0x10; 32]),
            weight: 0.0,
            children: vec![3, 4],
            is_leaf: false,
        });
        nodes.push(DbowNode {
            descriptor: pack_orb_descriptor(&[0xF0; 32]),
            weight: 0.0,
            children: vec![5, 6],
            is_leaf: false,
        });
        nodes.push(leaf(0x00, 1.5));
        nodes.push(leaf(0x1F, 2.0));
        nodes.push(leaf(0xF0, 0.5));
        nodes.push(leaf(0xFF, 3.0));

        let vocab = build_vocabulary(&nodes).unwrap();

        // A near-0x00 descriptor should land on the first leaf (weight 1.5).
        let (_w, weight) = vocab.transform_one(&pack_orb_descriptor(&[0x01; 32]));
        assert!((weight - 1.5).abs() < 1e-6, "got weight {weight}");

        // A near-0xFF descriptor should land on the last leaf (weight 3.0).
        let (_w, weight) = vocab.transform_one(&pack_orb_descriptor(&[0xFE; 32]));
        assert!((weight - 3.0).abs() < 1e-6, "got weight {weight}");
    }
}
