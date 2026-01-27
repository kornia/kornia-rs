# kornia-bow

High-performance, hierarchical Bag of Words (BoW) library for computer vision in Rust.

`kornia-bow` provides a highly optimized implementation of vocabulary trees, designed for real-time visual place recognition, loop closure detection, and image retrieval.

## Features

- **Blazing Fast Lookups**: Achieves ~80ns per descriptor lookup.
- **TF-IDF Weighting**: Automatic IDF calculation during training.
- **Parallel Processing**: Uses `Rayon` for parallel training and transforming.
- **Universal Metrics**: Supports `Hamming` (binary) and `L2` (floating-point).
- **Const Generics**: Branching factor and descriptor size are compile-time constants.
- **Multiple Scoring Methods**: L1, L2, Chi-Square, KL Divergence, Bhattacharyya, Dot Product all for comparing Bow vectors.
- **Safe I/O**: Validated serialization and deserialization.

## Usage

### 1. Training

```rust
use kornia_bow::Vocabulary;
use kornia_bow::metric::{Hamming, Feature};

// Define branching factor (B) and descriptor size (D in u64s)
const B: usize = 10;
const D: usize = 4; // 256 bits

// Prepare data
let data: Vec<Feature<u64, D>> = load_descriptors();

// Train (max_depth=3)
let vocab = Vocabulary::<B, Hamming<D>>::train(&data, 3).expect("Training failed");
```

### 2. Lookup and Scoring

```rust
// Transform descriptor to word
let feature = Feature([0u64; D]);
let (word_id, weight) = vocab.transform_one(&feature);

// Transform image features to BoW vector
let features = vec![feature; 100];
let bow1 = vocab.transform(&features).unwrap();
let bow2 = vocab.transform(&features).unwrap();

// Compute similarity
let score_l1 = bow1.l1_similarity(&bow2);
let score_chi = bow1.chi_square(&bow2);
let score_kl = bow1.kl_divergence(&bow2);
```

### 3. Geometric Verification (Direct Index)

The direct index maps nodes in the vocabulary tree back to the original feature indices, which is crucial for geometric verification steps (like RANSAC) after retrieving candidate images.

```rust
// Get BoW vector AND Direct Index (at depth level 2)
let (bow, direct_index) = vocab.transform_with_direct_index(&features, 2).expect("Transform failed");

// Access features assigned to a specific node
// DirectIndex is a sorted list of (NodeID, Vec<FeatureIndex>)
if let Ok(idx) = direct_index.0.binary_search_by_key(&node_id, |&(id, _)| id) {
    let feature_indices = &direct_index.0[idx].1;
    for &idx in feature_indices {
        println!("Feature {} is assigned to node {}", idx, node_id);
    }
}
```

### 4. Save/Load

The library performs strict validation when loading to ensure the vocabulary matches your compile-time configuration (`B` and `D`) and is structurally sound.

```rust
vocab.save("vocab.bin").expect("Save failed");

// Will return an error if B or MetricType doesn't match the file
let loaded = Vocabulary::<B, Hamming<D>>::load("vocab.bin").expect("Load failed");
```

### 5. Using L2 Metric (Floating Point)

For floating-point descriptors (like SIFT or standard vectors), use `L2`.

```rust
use kornia_bow::metric::L2;

// 128-dimensional float descriptor
const D_FLOAT: usize = 128;
let float_vocab = Vocabulary::<B, L2<D_FLOAT>>::train(&float_data, 3)?;
```

## Architecture

The library uses a flat memory layout (`Vocabulary`) with `BlockCluster` structs. Each cluster contains descriptors and children indices packed contiguously.

## References
kornia-bow is inspired by the Rust crate [abow](https://github.com/donkeyteethUX/abow) and the C++ visual BoW implementations [DBoW2](https://github.com/dorian3d/DBoW2/) and [fbow](https://github.com/rmsalinas/fbow)
