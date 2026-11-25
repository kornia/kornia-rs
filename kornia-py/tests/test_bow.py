"""Tests for the Python-exposed Bag of Words (kornia_rs.bow) submodule.

Covers:
- Vocabulary training and introspection
- Transform and BoW representation
- Ground-truth numerical parity across all similarity metrics
- Disjoint vs identical vs partially overlapping distributions
- End-to-end integration with Kornia ORB feature detector
- DirectIndex multi-level grouping and boundaries
- Binary serialization (.bow) and DBoW2 text loading
- Pickle / multiprocessing serialization
- Negative test cases: invalid shapes, wrong dtypes, non-contiguous layouts, corrupted files
"""

import pickle
import numpy as np
import pytest

import kornia_rs as K


def _generate_synthetic_descriptors(n: int = 100, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, size=(n, 32), dtype=np.uint8)


# -------------------------------------------------------------------------
# Vocabulary Training & Metadata
# -------------------------------------------------------------------------


def test_train_and_vocabulary_properties():
    data = _generate_synthetic_descriptors(100)
    vocab = K.bow.OrbVocabulary.train(data, max_depth=3)

    assert vocab.num_blocks > 0
    assert len(vocab) == vocab.num_blocks
    assert vocab.branching_factor == 10
    assert vocab.descriptor_size == 32
    assert "OrbVocabulary" in repr(vocab)


def test_train_uniform_descriptors():
    # Extreme descriptors: all zeros and all 0xFF
    all_zeros = np.zeros((50, 32), dtype=np.uint8)
    all_ones = np.full((50, 32), 255, dtype=np.uint8)
    combined = np.vstack([all_zeros, all_ones])

    vocab = K.bow.OrbVocabulary.train(combined, max_depth=2)
    assert vocab.num_blocks > 0

    bow_zeros = vocab.transform(all_zeros[:5])
    bow_ones = vocab.transform(all_ones[:5])

    assert len(bow_zeros) > 0
    assert len(bow_ones) > 0
    # Extreme descriptors should be separated in the tree
    assert bow_zeros != bow_ones


# -------------------------------------------------------------------------
# BoW Vector Properties & Protocol Tests
# -------------------------------------------------------------------------


def test_transform_and_bow_properties():
    data = _generate_synthetic_descriptors(100)
    vocab = K.bow.OrbVocabulary.train(data, max_depth=3)

    query = data[:15]
    bow = vocab.transform(query)

    assert len(bow) > 0
    assert "BoW" in repr(bow)
    assert bow == bow

    words = bow.words()
    weights = bow.weights()

    assert words.dtype == np.uint32
    assert weights.dtype == np.float32
    assert len(words) == len(weights) == len(bow)

    # Transform outputs L1-normalized weights (sum == 1.0)
    assert np.isclose(np.sum(weights), 1.0, atol=1e-5)

    # Dictionary conversion
    d = bow.to_dict()
    assert isinstance(d, dict)
    assert len(d) == len(bow)
    for w, wt in zip(words, weights):
        assert w in d
        assert np.isclose(d[w], wt, atol=1e-6)

    # List conversion
    lst = bow.to_list()
    assert isinstance(lst, list)
    assert len(lst) == len(bow)

    # Container protocol (__iter__, items)
    iter_words = [w for w in bow]
    assert iter_words == list(words)
    assert bow.items() == lst

    # Indexing: present word vs absent word
    first_word = int(words[0])
    assert np.isclose(bow[first_word], weights[0], atol=1e-6)
    assert bow[999_999_999] == 0.0

    # __contains__ operator
    assert first_word in bow
    assert 999_999_999 not in bow


# -------------------------------------------------------------------------
# Ground-Truth Mathematical Parity for Similarity Metrics
# -------------------------------------------------------------------------


def test_bow_similarity_ground_truth_reference():
    """Pin down exact numerical values against independent manual calculations.

    A = {(1, 0.4), (2, 0.6)}
    B = {(1, 0.5), (3, 0.5)}

    Expected metrics:
    - L1 distance: |0.4 - 0.5| + 0.6 + 0.5 = 1.2
      L1 score: 1.0 - 0.5 * 1.2 = 0.400000
    - Dot product: 0.4 * 0.5 = 0.200000
    - Chi-Square: 2 * (0.4 * 0.5) / (0.4 + 0.5) = 0.4 / 0.9 = 0.444444
    - Bhattacharyya: sqrt(0.4 * 0.5) = sqrt(0.2) = 0.4472136
    """
    bow1 = K.bow.BoW()
    bow2 = K.bow.BoW()
    # Populate via pickle state restoration to test exact sparse distributions
    bow1.__setstate__([(1, 0.4), (2, 0.6)])
    bow2.__setstate__([(1, 0.5), (3, 0.5)])

    assert np.isclose(bow1.l1_similarity(bow2), 0.4, atol=1e-6)
    assert np.isclose(bow1.score(bow2, "l1"), 0.4, atol=1e-6)

    assert np.isclose(bow1.dot_product(bow2), 0.2, atol=1e-6)
    assert np.isclose(bow1.score(bow2, "dot"), 0.2, atol=1e-6)
    assert np.isclose(bow1 @ bow2, 0.2, atol=1e-6)

    assert np.isclose(bow1.chi_square(bow2), 0.4444444, atol=1e-5)
    assert np.isclose(bow1.score(bow2, "chi_square"), 0.4444444, atol=1e-5)

    assert np.isclose(bow1.bhattacharyya(bow2), 0.4472136, atol=1e-5)
    assert np.isclose(bow1.score(bow2, "bhattacharyya"), 0.4472136, atol=1e-5)


def test_bow_similarity_disjoint_and_identical_boundaries():
    bow_a = K.bow.BoW()
    bow_b = K.bow.BoW()
    bow_a.__setstate__([(10, 1.0)])
    bow_b.__setstate__([(20, 1.0)])

    # Completely disjoint distributions
    assert np.isclose(bow_a.l1_similarity(bow_b), 0.0, atol=1e-6)
    assert np.isclose(bow_a.dot_product(bow_b), 0.0, atol=1e-6)
    assert np.isclose(bow_a.chi_square(bow_b), 0.0, atol=1e-6)
    assert np.isclose(bow_a.bhattacharyya(bow_b), 0.0, atol=1e-6)

    # Identical distributions
    assert np.isclose(bow_a.l1_similarity(bow_a), 1.0, atol=1e-6)
    assert np.isclose(bow_a.score(bow_a, "l1"), 1.0, atol=1e-6)
    assert np.isclose(bow_a.score(bow_a, "l2"), 1.0, atol=1e-5)

    with pytest.raises(ValueError, match="unsupported metric"):
        bow_a.score(bow_b, "unsupported_metric")


def test_bow_normalizations():
    data = _generate_synthetic_descriptors(50)
    vocab = K.bow.OrbVocabulary.train(data, max_depth=3)

    bow = vocab.transform(data[:10])
    l2_bow = bow.normalized_l2()
    l2_weights = l2_bow.weights()
    assert np.isclose(np.sum(l2_weights * l2_weights), 1.0, atol=1e-5)

    l1_bow = l2_bow.normalized_l1()
    l1_weights = l1_bow.weights()
    assert np.isclose(np.sum(l1_weights), 1.0, atol=1e-5)


# -------------------------------------------------------------------------
# Direct Index Tests (Levels, Buckets, Bounds)
# -------------------------------------------------------------------------


def test_transform_with_direct_index():
    data = _generate_synthetic_descriptors(100)
    vocab = K.bow.OrbVocabulary.train(data, max_depth=3)

    query = data[:12]
    # Default level (4)
    bow_def, di_def = vocab.transform_with_direct_index(query)
    assert len(bow_def) > 0

    # Intermediate level 1
    bow1, di1 = vocab.transform_with_direct_index(query, level=1)
    assert len(di1) > 0
    assert "DirectIndex" in repr(di1)
    assert di1 == di1

    di_dict = di1.to_dict()
    assert isinstance(di_dict, dict)
    assert len(di_dict) == len(di1)

    # Total feature count across all node buckets must equal query length (12)
    total_features = sum(len(f_idxs) for f_idxs in di_dict.values())
    assert total_features == 12

    # Python container protocol verification (keys, values, items, __iter__)
    iter_nodes = [node for node in di1]
    assert iter_nodes == list(di_dict.keys())
    assert di1.keys() == list(di_dict.keys())
    assert di1.values() == list(di_dict.values())
    assert di1.items() == list(di_dict.items())

    # Direct index indexing & lookup
    first_node = list(di_dict.keys())[0]
    assert list(di1.get(first_node)) == di_dict[first_node]
    assert list(di1[first_node]) == di_dict[first_node]
    assert first_node in di1
    assert 999_999_999 not in di1

    # Non-existent node
    assert di1.get(999_999_999) is None
    with pytest.raises(KeyError):
        _ = di1[999_999_999]


def test_direct_index_level_boundaries():
    data = _generate_synthetic_descriptors(50)
    vocab = K.bow.OrbVocabulary.train(data, max_depth=3)
    query = data[:8]

    # Level 0 (root): all 8 features must be grouped under root node
    _, di_root = vocab.transform_with_direct_index(query, level=0)
    assert len(di_root) == 1
    root_node = list(di_root.to_dict().keys())[0]
    assert len(di_root.get(root_node)) == 8

    # Level 99 (beyond tree depth): DirectIndex is safe and returns empty
    _, di_deep = vocab.transform_with_direct_index(query, level=99)
    assert len(di_deep) == 0


# -------------------------------------------------------------------------
# Single Feature Transformation
# -------------------------------------------------------------------------


def test_transform_one():
    data = _generate_synthetic_descriptors(50)
    vocab = K.bow.OrbVocabulary.train(data, max_depth=3)

    # 1D array (32,)
    w1, wt1 = vocab.transform_one(data[0])
    assert isinstance(w1, int)
    assert isinstance(wt1, float)
    assert wt1 >= 0.0

    # 2D array (1, 32)
    w2, wt2 = vocab.transform_one(data[0:1])
    assert w1 == w2
    assert np.isclose(wt1, wt2, atol=1e-6)

    # Invalid descriptor shapes
    with pytest.raises(ValueError, match=r"expected 32-element"):
        vocab.transform_one(data[0][:16])

    with pytest.raises(ValueError, match=r"expected \(1, 32\)"):
        vocab.transform_one(data[:2])


# -------------------------------------------------------------------------
# End-to-End Integration with Kornia ORB Detector
# -------------------------------------------------------------------------


def test_end_to_end_orb_detector_integration():
    """Verify that descriptors from kornia_rs.features.orb_detect_and_compute flow
    seamlessly into kornia_rs.bow without layout, stride, or type mismatches."""
    # Synthetic image with high-contrast corners (checkerboard pattern)
    img = np.zeros((200, 200), dtype=np.uint8)
    img[::20, :] = 255
    img[:, ::20] = 255

    features = K.features.orb_detect_and_compute(img, n_keypoints=100)
    descriptors = features.descriptors

    assert descriptors.dtype == np.uint8
    assert descriptors.shape[1] == 32
    assert len(features) > 0

    # Train vocabulary on a set of descriptors
    vocab = K.bow.OrbVocabulary.train(descriptors, max_depth=3)
    assert vocab.num_blocks > 0

    # Transform ORB features into BoW and DirectIndex
    bow, di = vocab.transform_with_direct_index(descriptors, level=1)
    assert len(bow) > 0
    assert len(di) > 0

    # Verify self-similarity
    sim = bow.l1_similarity(bow)
    assert np.isclose(sim, 1.0, atol=1e-6)


# -------------------------------------------------------------------------
# Serialization & Persistence (Binary, Text, Pickle)
# -------------------------------------------------------------------------


def test_save_and_load_binary_roundtrip(tmp_path):
    data = _generate_synthetic_descriptors(80)
    vocab = K.bow.OrbVocabulary.train(data, max_depth=3)

    save_path = tmp_path / "vocab.bow"
    vocab.save(str(save_path))
    assert save_path.exists()

    loaded_vocab = K.bow.OrbVocabulary.from_file(str(save_path))
    assert loaded_vocab.num_blocks == vocab.num_blocks

    loaded_shortcut = K.bow.load_vocabulary(str(save_path))
    assert loaded_shortcut.num_blocks == vocab.num_blocks

    # Verify identical transformation on test features
    test_desc = data[:10]
    bow_orig = vocab.transform(test_desc)
    bow_loaded = loaded_vocab.transform(test_desc)

    np.testing.assert_array_equal(bow_orig.words(), bow_loaded.words())
    np.testing.assert_allclose(bow_orig.weights(), bow_loaded.weights(), rtol=1e-5)


def test_pickle_multiprocessing_roundtrip():
    data = _generate_synthetic_descriptors(60)
    vocab = K.bow.OrbVocabulary.train(data, max_depth=3)
    bow, di = vocab.transform_with_direct_index(data[:10], level=1)

    # 1. BoW pickle
    bow_pickled = pickle.dumps(bow)
    bow_restored = pickle.loads(bow_pickled)
    assert bow_restored == bow

    # 2. DirectIndex pickle
    di_pickled = pickle.dumps(di)
    di_restored = pickle.loads(di_pickled)
    assert di_restored == di

    # 3. OrbVocabulary pickle
    vocab_pickled = pickle.dumps(vocab)
    vocab_restored = pickle.loads(vocab_pickled)
    assert vocab_restored.num_blocks == vocab.num_blocks
    assert vocab_restored.transform(data[:5]) == vocab.transform(data[:5])


def test_load_orb_slam3_text_vocabulary(tmp_path):
    # Minimal synthetic DBoW2 text vocabulary: branching 10, depth 1
    # Line 1: k L scoring weighting
    lines = ["10 1 0 0\n"]
    for i in range(10):
        bytes_str = " ".join([str(i)] * 32)
        lines.append(f"0 1 {bytes_str} 1.0\n")

    voc_path = tmp_path / "ORBvoc_test.txt"
    voc_path.write_text("".join(lines))

    vocab = K.bow.OrbVocabulary.from_orb_slam3(str(voc_path))
    assert vocab.num_blocks > 0

    vocab_shortcut = K.bow.load_orb_slam3_vocabulary(str(voc_path))
    assert vocab_shortcut.num_blocks == vocab.num_blocks

    test_desc = np.zeros((5, 32), dtype=np.uint8)
    bow = vocab.transform(test_desc)
    assert len(bow) > 0


def test_load_orb_slam3_text_vocabulary_mismatch(tmp_path):
    # Header with branching factor k=6 (expected k=10)
    voc_path = tmp_path / "ORBvoc_mismatch.txt"
    voc_path.write_text("6 1 0 0\n0 1 " + " ".join(["0"] * 32) + " 1.0\n")

    with pytest.raises(ValueError, match="Vocabulary mismatch"):
        K.bow.OrbVocabulary.from_orb_slam3(str(voc_path))


# -------------------------------------------------------------------------
# Negative / Error Handling Tests
# -------------------------------------------------------------------------


def test_error_handling_shapes_and_layouts():
    data = _generate_synthetic_descriptors(50)
    vocab = K.bow.OrbVocabulary.train(data, max_depth=3)

    # Empty descriptors
    with pytest.raises(ValueError, match="cannot be empty"):
        vocab.transform(np.zeros((0, 32), dtype=np.uint8))

    with pytest.raises(ValueError, match="cannot be empty"):
        vocab.transform_with_direct_index(np.zeros((0, 32), dtype=np.uint8), level=1)

    # Wrong second dimension
    with pytest.raises(ValueError, match=r"expected \(N, 32\)"):
        vocab.transform(np.zeros((10, 64), dtype=np.uint8))

    # Wrong dtype (float32 instead of uint8)
    with pytest.raises(TypeError):
        vocab.transform(np.zeros((10, 32), dtype=np.float32))

    # Non-contiguous arrays: Fortran order
    f_desc = np.asfortranarray(data[:10])
    with pytest.raises(ValueError, match="must be a C-contiguous array"):
        vocab.transform(f_desc)

    # Non-contiguous arrays: reversed step stride
    stride_desc = data[::2]
    # Slicing along row dimension in C order is contiguous if rows are intact,
    # but negative or column slicing is strided:
    col_strided = data[:10, ::-1]
    with pytest.raises(ValueError, match="must be a C-contiguous array"):
        vocab.transform(col_strided)

    # Non-existent files
    with pytest.raises(IOError):
        K.bow.OrbVocabulary.from_file("non_existent_file.bow")

    with pytest.raises(IOError):
        K.bow.OrbVocabulary.from_orb_slam3("non_existent_file.txt")
