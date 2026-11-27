use numpy::{PyArray1, PyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

use kornia_bow::{
    metric::Feature,
    orb_slam3::{
        load_orb_slam3_vocabulary as load_orb_voc, pack_orb_descriptor, OrbVocabulary,
        ORB_BRANCHING, ORB_WORDS,
    },
    BoW as CoreBoW, BowError, DirectIndex as CoreDirectIndex,
};

use crate::pyutils::{c_slice, require_c_contig_aligned};

/// Map any `BowError` to an appropriate Python exception.
pub(crate) fn bow_to_pyerr(err: BowError) -> PyErr {
    match err {
        BowError::NoFeatures => pyo3::exceptions::PyValueError::new_err(
            "No features provided: descriptor array is empty",
        ),
        BowError::Io(e) => pyo3::exceptions::PyIOError::new_err(e.to_string()),
        BowError::Bincode(e) => {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Serialization error: {e}"))
        }
        BowError::VocabularyMismatch {
            expected_b,
            found_b,
        } => pyo3::exceptions::PyValueError::new_err(format!(
            "Vocabulary mismatch: expected B={expected_b}, found B={found_b}"
        )),
        BowError::MetricMismatch { expected, found } => pyo3::exceptions::PyValueError::new_err(
            format!("Metric mismatch: expected {expected:?}, found {found:?}"),
        ),
        BowError::CorruptedVocabulary => {
            pyo3::exceptions::PyValueError::new_err("Corrupted vocabulary file")
        }
        BowError::VocabularyTooLarge { declared, file_len } => {
            pyo3::exceptions::PyValueError::new_err(format!(
                "Corrupted vocabulary: declares {declared} blocks but file is only {file_len} bytes"
            ))
        }
    }
}

/// Validate shape and alignment of an (N, 32) uint8 numpy array, returning its contiguous byte slice.
fn validate_and_get_descriptor_slice<'a>(arr: &'a Bound<'_, PyArray2<u8>>) -> PyResult<&'a [u8]> {
    require_c_contig_aligned(arr, "descriptors")?;
    let shape = arr.shape();
    if shape[1] != 32 {
        return Err(PyValueError::new_err(format!(
            "expected (N, 32) uint8 descriptors, got ({}, {})",
            shape[0], shape[1]
        )));
    }

    if shape[0] == 0 {
        return Err(PyValueError::new_err(
            "descriptor array cannot be empty (0 features)",
        ));
    }

    c_slice(arr, "descriptors")
}

/// Bit-pack a contiguous slice of 32-byte descriptors using Rayon adaptive parallelism.
///
/// Can be safely executed without the Python GIL.
fn pack_orb_descriptors_slice(slice: &[u8]) -> Vec<Feature<u64, ORB_WORDS>> {
    let to_feat = |chunk: &[u8]| {
        let arr32: &[u8; 32] = chunk.try_into().unwrap_or(&[0u8; 32]);
        pack_orb_descriptor(arr32)
    };

    if slice.len() >= 32 * 2048 {
        slice.par_chunks_exact(32).map(to_feat).collect()
    } else {
        slice.chunks_exact(32).map(to_feat).collect()
    }
}

/// Extract and bit-pack a single 32-byte ORB descriptor.
fn extract_single_orb_descriptor(obj: &Bound<'_, PyAny>) -> PyResult<Feature<u64, ORB_WORDS>> {
    if let Ok(arr1) = obj.cast::<PyArray1<u8>>() {
        require_c_contig_aligned(arr1, "descriptor")?;
        let shape = arr1.shape();
        if shape[0] != 32 {
            return Err(PyValueError::new_err(format!(
                "expected 32-element uint8 descriptor, got shape ({})",
                shape[0]
            )));
        }
        let slice = c_slice(arr1, "descriptor")?;
        let arr32: &[u8; 32] = slice
            .try_into()
            .map_err(|_| PyValueError::new_err("expected 32-element slice"))?;
        return Ok(pack_orb_descriptor(arr32));
    }

    if let Ok(arr2) = obj.cast::<PyArray2<u8>>() {
        require_c_contig_aligned(arr2, "descriptor")?;
        let shape = arr2.shape();
        if shape != [1, 32] {
            return Err(PyValueError::new_err(format!(
                "expected (1, 32) uint8 descriptor, got ({}, {})",
                shape[0], shape[1]
            )));
        }
        let slice = c_slice(arr2, "descriptor")?;
        let arr32: &[u8; 32] = slice
            .try_into()
            .map_err(|_| PyValueError::new_err("expected 32-element slice"))?;
        return Ok(pack_orb_descriptor(arr32));
    }

    Err(PyValueError::new_err(
        "expected a 1D (32,) or 2D (1, 32) uint8 numpy array for descriptor",
    ))
}

/// Sparse Bag of Words vector representing an image: sorted pairs of `(word_id, weight)`.
///
/// Returned by [`OrbVocabulary::transform`].
#[pyclass(name = "BoW", module = "kornia_rs.bow")]
#[derive(Clone, Debug, PartialEq)]
pub struct PyBoW {
    pub(crate) inner: CoreBoW,
}

#[pymethods]
impl PyBoW {
    /// Creates an empty BoW vector.
    ///
    /// # Returns
    ///
    /// A new, empty [`PyBoW`] instance with zero words.
    #[new]
    pub fn new() -> Self {
        Self {
            inner: CoreBoW(Vec::new()),
        }
    }

    /// Equality comparison between two BoW vectors.
    ///
    /// # Arguments
    ///
    /// * `other` - The other [`PyBoW`] to compare against.
    ///
    /// # Returns
    ///
    /// `true` if both sparse vectors have identical words and weights, `false` otherwise.
    pub fn __eq__(&self, other: &PyBoW) -> bool {
        self.inner == other.inner
    }

    /// Checks if a word ID is present in the sparse vector.
    ///
    /// # Arguments
    ///
    /// * `word_id` - The word identifier to check.
    ///
    /// # Returns
    ///
    /// `true` if the word ID has a non-zero weight in this vector, `false` otherwise.
    pub fn __contains__(&self, word_id: u32) -> bool {
        self.inner
            .0
            .binary_search_by_key(&word_id, |&(w, _)| w)
            .is_ok()
    }

    /// Serializes sparse BoW state for pickle support.
    ///
    /// # Returns
    ///
    /// A vector of `(word_id, weight)` tuples.
    pub fn __getstate__(&self) -> Vec<(u32, f32)> {
        self.inner.0.clone()
    }

    /// Deserializes and validates sparse BoW state from pickle.
    ///
    /// # Arguments
    ///
    /// * `state` - Vector of `(word_id, weight)` tuples. Word IDs must be strictly increasing.
    ///
    /// # Returns
    ///
    /// `Ok(())` on success.
    ///
    /// # Errors
    ///
    /// Returns [`PyValueError`] if word IDs are not strictly increasing or weights are non-positive or non-finite.
    pub fn __setstate__(&mut self, state: Vec<(u32, f32)>) -> PyResult<()> {
        for window in state.windows(2) {
            if window[0].0 >= window[1].0 {
                return Err(PyValueError::new_err(
                    "BoW state must have strictly increasing, unique word IDs",
                ));
            }
        }
        for &(w, wt) in &state {
            if !wt.is_finite() || wt <= 0.0 {
                return Err(PyValueError::new_err(format!(
                    "BoW state weights must be positive and finite, got weight {wt} for word {w}"
                )));
            }
        }
        self.inner = CoreBoW(state);
        Ok(())
    }

    /// Returns an iterator over non-zero word IDs in the sparse vector.
    ///
    /// # Arguments
    ///
    /// * `py` - Python runtime marker.
    ///
    /// # Returns
    ///
    /// A Python iterator yielding non-zero `word_id` integers.
    ///
    /// # Errors
    ///
    /// Returns a [`PyErr`] if creating the Python list fails.
    pub fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyIterator>> {
        let words: Vec<u32> = self.inner.0.iter().map(|&(w, _)| w).collect();
        pyo3::types::PyList::new(py, words)?.as_any().try_iter()
    }

    /// Non-zero word IDs in the sparse vector as a newly allocated 1D uint32 numpy array.
    ///
    /// Note: BoW is stored internally as pairs of (word_id, weight). Calling this allocates a new array.
    ///
    /// # Arguments
    ///
    /// * `py` - Python runtime marker.
    ///
    /// # Returns
    ///
    /// 1D NumPy array of `uint32` word IDs.
    pub fn words<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        PyArray1::from_iter(py, self.inner.0.iter().map(|&(w, _)| w))
    }

    /// Weights of non-zero words as a newly allocated 1D float32 numpy array.
    ///
    /// Note: BoW is stored internally as pairs of (word_id, weight). Calling this allocates a new array.
    ///
    /// # Arguments
    ///
    /// * `py` - Python runtime marker.
    ///
    /// # Returns
    ///
    /// 1D NumPy array of `float32` word weights.
    pub fn weights<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_iter(py, self.inner.0.iter().map(|&(_, wt)| wt))
    }

    /// Returns a list of `(word_id, weight)` tuples for non-zero words.
    ///
    /// # Returns
    ///
    /// Vector of `(u32, f32)` word ID and weight pairs.
    pub fn items(&self) -> Vec<(u32, f32)> {
        self.inner.0.clone()
    }

    /// Converts the sparse BoW vector into a Python dictionary `{word_id: weight}`.
    ///
    /// # Arguments
    ///
    /// * `py` - Python runtime marker.
    ///
    /// # Returns
    ///
    /// A Python dictionary mapping integer word IDs to float weights.
    ///
    /// # Errors
    ///
    /// Returns a [`PyErr`] if inserting items into the dictionary fails.
    pub fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for &(w, wt) in &self.inner.0 {
            dict.set_item(w, wt)?;
        }
        Ok(dict)
    }

    /// Converts the sparse BoW vector into a list of `(word_id, weight)` tuples.
    ///
    /// # Returns
    ///
    /// Vector of `(u32, f32)` tuples.
    pub fn to_list(&self) -> Vec<(u32, f32)> {
        self.inner.0.clone()
    }

    /// Returns the number of non-zero words in the sparse BoW vector.
    ///
    /// # Returns
    ///
    /// Word count as `usize`.
    pub fn __len__(&self) -> usize {
        self.inner.0.len()
    }

    /// String representation of the BoW vector.
    ///
    /// # Returns
    ///
    /// Formatted string like `BoW(words=N)`.
    pub fn __repr__(&self) -> String {
        format!("BoW(words={})", self.inner.0.len())
    }

    /// Gets the weight of a specific word ID, returning 0.0 if not present.
    ///
    /// # Arguments
    ///
    /// * `word_id` - Word ID to query.
    ///
    /// # Returns
    ///
    /// Float weight of the word, or 0.0 if absent.
    pub fn __getitem__(&self, word_id: u32) -> f32 {
        match self.inner.0.binary_search_by_key(&word_id, |&(w, _)| w) {
            Ok(idx) => self.inner.0[idx].1,
            Err(_) => 0.0,
        }
    }

    /// Computes similarity score with another BoW vector using the specified metric.
    ///
    /// Metric choices:
    /// - `"l1"` (default): L1 similarity in range [0, 1] (standard DBoW2 scoring for L1-normalized vectors).
    /// - `"l2"`: L2 similarity in range [0, 1] (automatically L2-normalizes vectors).
    /// - `"dot"`: Dot product similarity.
    /// - `"chi_square"`: Chi-square similarity.
    /// - `"bhattacharyya"`: Bhattacharyya coefficient similarity.
    ///
    /// Note: To compute Kullback-Leibler (KL) divergence distance, use [`PyBoW::kl_divergence`].
    ///
    /// # Arguments
    ///
    /// * `other` - The target [`PyBoW`] to score against.
    /// * `metric` - Metric name string (default `"l1"`).
    ///
    /// # Returns
    ///
    /// Calculated similarity score as `f32`.
    ///
    /// # Errors
    ///
    /// Returns [`PyValueError`] if `metric` is unsupported or `"kl"`.
    #[pyo3(signature = (other, metric="l1"))]
    pub fn score(&self, other: &PyBoW, metric: &str) -> PyResult<f32> {
        match metric {
            "l1" => Ok(self.inner.l1_similarity(&other.inner)),
            "l2" => {
                let mut s_norm = self.inner.clone();
                s_norm.normalize_l2();
                let mut o_norm = other.inner.clone();
                o_norm.normalize_l2();
                Ok(s_norm.l2(&o_norm))
            }
            "dot" => Ok(self.inner.dot_product(&other.inner)),
            "chi_square" => Ok(self.inner.chi_square(&other.inner)),
            "bhattacharyya" => Ok(self.inner.bhattacharyya(&other.inner)),
            _ => Err(PyValueError::new_err(format!(
                "unsupported metric '{metric}'; expected one of: 'l1', 'l2', 'dot', 'chi_square', 'bhattacharyya'. Note: KL divergence is a distance, use `kl_divergence()` method directly."
            ))),
        }
    }

    /// Computes L1 similarity score (in range [0.0, 1.0] for L1-normalized non-negative vectors).
    ///
    /// # Arguments
    ///
    /// * `other` - The other [`PyBoW`] to compare against.
    ///
    /// # Returns
    ///
    /// L1 similarity score as `f32`.
    pub fn l1_similarity(&self, other: &PyBoW) -> f32 {
        self.inner.l1_similarity(&other.inner)
    }

    /// Computes L2 similarity score in range [0.0, 1.0].
    ///
    /// Automatically normalizes both vectors with L2 norm before computing similarity.
    ///
    /// # Arguments
    ///
    /// * `other` - The other [`PyBoW`] to compare against.
    ///
    /// # Returns
    ///
    /// L2 similarity score as `f32`.
    pub fn l2_similarity(&self, other: &PyBoW) -> f32 {
        let mut s_norm = self.inner.clone();
        s_norm.normalize_l2();
        let mut o_norm = other.inner.clone();
        o_norm.normalize_l2();
        s_norm.l2(&o_norm)
    }

    /// Computes dot product similarity.
    ///
    /// # Arguments
    ///
    /// * `other` - The other [`PyBoW`] to compute dot product with.
    ///
    /// # Returns
    ///
    /// Dot product as `f32`.
    pub fn dot_product(&self, other: &PyBoW) -> f32 {
        self.inner.dot_product(&other.inner)
    }

    /// Computes Chi-square similarity score.
    ///
    /// # Arguments
    ///
    /// * `other` - The other [`PyBoW`] to compare against.
    ///
    /// # Returns
    ///
    /// Chi-square similarity score as `f32`.
    pub fn chi_square(&self, other: &PyBoW) -> f32 {
        self.inner.chi_square(&other.inner)
    }

    /// Computes Kullback-Leibler (KL) divergence distance (non-symmetric).
    ///
    /// Note: This is a distance/divergence metric where 0.0 indicates identical
    /// distributions and larger values indicate greater dissimilarity.
    ///
    /// # Arguments
    ///
    /// * `other` - The target distribution [`PyBoW`].
    ///
    /// # Returns
    ///
    /// KL divergence distance as `f32`.
    pub fn kl_divergence(&self, other: &PyBoW) -> f32 {
        self.inner.kl_divergence(&other.inner)
    }

    /// Computes Bhattacharyya coefficient similarity score.
    ///
    /// # Arguments
    ///
    /// * `other` - The other [`PyBoW`] to compare against.
    ///
    /// # Returns
    ///
    /// Bhattacharyya coefficient similarity in range [0.0, 1.0] as `f32`.
    pub fn bhattacharyya(&self, other: &PyBoW) -> f32 {
        self.inner.bhattacharyya(&other.inner)
    }

    /// Python matrix multiplication operator (`@`) computes dot product similarity.
    ///
    /// # Arguments
    ///
    /// * `other` - The other [`PyBoW`] operand.
    ///
    /// # Returns
    ///
    /// Dot product similarity as `f32`.
    pub fn __matmul__(&self, other: &PyBoW) -> f32 {
        self.inner.dot_product(&other.inner)
    }

    /// Returns a new BoW vector normalized by L2 norm.
    ///
    /// # Returns
    ///
    /// A new [`PyBoW`] instance with unit L2 norm.
    pub fn normalized_l2(&self) -> PyBoW {
        let mut copy = self.inner.clone();
        copy.normalize_l2();
        PyBoW { inner: copy }
    }

    /// Returns a new BoW vector normalized by L1 norm.
    ///
    /// # Returns
    ///
    /// A new [`PyBoW`] instance with unit L1 norm.
    pub fn normalized_l1(&self) -> PyBoW {
        let mut copy = self.inner.clone();
        copy.normalize_l1();
        PyBoW { inner: copy }
    }
}

/// Direct Index mapping vocabulary tree node IDs to feature indices in the source image.
///
/// Used in Visual SLAM for fast geometric verification and candidate feature matching.
#[pyclass(name = "DirectIndex", module = "kornia_rs.bow")]
#[derive(Clone, Debug, PartialEq)]
pub struct PyDirectIndex {
    pub(crate) inner: CoreDirectIndex,
}

#[pymethods]
impl PyDirectIndex {
    /// Creates an empty DirectIndex.
    ///
    /// # Returns
    ///
    /// A new, empty [`PyDirectIndex`] instance.
    #[new]
    pub fn new() -> Self {
        Self {
            inner: CoreDirectIndex(Vec::new()),
        }
    }

    /// Number of distinct tree nodes present in the direct index.
    ///
    /// # Returns
    ///
    /// Node count as `usize`.
    pub fn __len__(&self) -> usize {
        self.inner.0.len()
    }

    /// String representation of the direct index.
    ///
    /// # Returns
    ///
    /// Formatted string like `DirectIndex(nodes=N)`.
    pub fn __repr__(&self) -> String {
        format!("DirectIndex(nodes={})", self.inner.0.len())
    }

    /// Equality comparison between two direct indices.
    ///
    /// # Arguments
    ///
    /// * `other` - The other [`PyDirectIndex`] to compare against.
    ///
    /// # Returns
    ///
    /// `true` if identical node mapping, `false` otherwise.
    pub fn __eq__(&self, other: &PyDirectIndex) -> bool {
        self.inner == other.inner
    }

    /// Checks if a node ID is present in the direct index.
    ///
    /// # Arguments
    ///
    /// * `node_id` - Tree node ID to query.
    ///
    /// # Returns
    ///
    /// `true` if node ID exists, `false` otherwise.
    pub fn __contains__(&self, node_id: u32) -> bool {
        self.inner
            .0
            .binary_search_by_key(&node_id, |&(n, _)| n)
            .is_ok()
    }

    /// Returns an iterator over node IDs present in the direct index.
    ///
    /// # Arguments
    ///
    /// * `py` - Python runtime marker.
    ///
    /// # Returns
    ///
    /// A Python iterator yielding `node_id` integers.
    ///
    /// # Errors
    ///
    /// Returns a [`PyErr`] if creating the Python list fails.
    pub fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyIterator>> {
        let keys: Vec<u32> = self.inner.0.iter().map(|&(k, _)| k).collect();
        pyo3::types::PyList::new(py, keys)?.as_any().try_iter()
    }

    /// Returns a list of node IDs present in the direct index.
    ///
    /// # Returns
    ///
    /// Vector of `u32` node IDs.
    pub fn keys(&self) -> Vec<u32> {
        self.inner.0.iter().map(|&(k, _)| k).collect()
    }

    /// Returns a list of feature index vectors present in the direct index.
    ///
    /// # Returns
    ///
    /// Vector of vectors containing feature index integers.
    pub fn values(&self) -> Vec<Vec<u32>> {
        self.inner.0.iter().map(|(_, v)| v.clone()).collect()
    }

    /// Returns a list of `(node_id, feature_indices)` pairs in the direct index.
    ///
    /// # Returns
    ///
    /// Vector of `(u32, Vec<u32>)` pairs.
    pub fn items(&self) -> Vec<(u32, Vec<u32>)> {
        self.inner.0.clone()
    }

    /// Serializes DirectIndex state for pickle support.
    ///
    /// # Returns
    ///
    /// Vector of `(node_id, Vec<feature_indices>)` pairs.
    pub fn __getstate__(&self) -> Vec<(u32, Vec<u32>)> {
        self.inner.0.clone()
    }

    /// Deserializes and validates DirectIndex state from pickle.
    ///
    /// # Arguments
    ///
    /// * `state` - Vector of `(node_id, Vec<feature_indices>)` pairs. Node IDs must be strictly increasing.
    ///
    /// # Returns
    ///
    /// `Ok(())` on success.
    ///
    /// # Errors
    ///
    /// Returns [`PyValueError`] if node IDs are not strictly increasing.
    pub fn __setstate__(&mut self, state: Vec<(u32, Vec<u32>)>) -> PyResult<()> {
        for window in state.windows(2) {
            if window[0].0 >= window[1].0 {
                return Err(PyValueError::new_err(
                    "DirectIndex state must have strictly increasing, unique node IDs",
                ));
            }
        }
        self.inner = CoreDirectIndex(state);
        Ok(())
    }

    /// Look up feature indices associated with a node ID. Returns `None` if node is not present.
    ///
    /// # Arguments
    ///
    /// * `node_id` - Tree node identifier to query.
    ///
    /// # Returns
    ///
    /// Slice of feature indices, or `None` if absent.
    pub fn get(&self, node_id: u32) -> Option<&[u32]> {
        match self.inner.0.binary_search_by_key(&node_id, |&(n, _)| n) {
            Ok(idx) => Some(&self.inner.0[idx].1),
            Err(_) => None,
        }
    }

    /// Look up feature indices associated with a node ID, raising KeyError if not found.
    ///
    /// # Arguments
    ///
    /// * `node_id` - Tree node identifier to query.
    ///
    /// # Returns
    ///
    /// Slice of feature indices.
    ///
    /// # Errors
    ///
    /// Returns [`PyKeyError`] if `node_id` is not in the direct index.
    pub fn __getitem__(&self, node_id: u32) -> PyResult<&[u32]> {
        self.get(node_id)
            .ok_or_else(|| PyKeyError::new_err(format!("node {node_id} not in direct index")))
    }

    /// Converts the direct index into a dictionary `{node_id: [feature_idx, ...]}`.
    ///
    /// # Arguments
    ///
    /// * `py` - Python runtime marker.
    ///
    /// # Returns
    ///
    /// Python dictionary mapping node IDs to lists of feature indices.
    ///
    /// # Errors
    ///
    /// Returns a [`PyErr`] if setting dictionary items fails.
    pub fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for &(node_id, ref feats) in &self.inner.0 {
            dict.set_item(node_id, feats.clone())?;
        }
        Ok(dict)
    }

    /// Converts the direct index into a list of `(node_id, [feature_idx, ...])` tuples.
    ///
    /// # Returns
    ///
    /// Vector of `(u32, Vec<u32>)` tuples.
    pub fn to_list(&self) -> Vec<(u32, Vec<u32>)> {
        self.inner.0.clone()
    }
}

/// Hierarchical Bag of Words vocabulary for ORB descriptors.
///
/// Branching factor 10, Hamming distance over 256-bit (32-byte) descriptors.
#[pyclass(name = "OrbVocabulary", module = "kornia_rs.bow")]
pub struct PyOrbVocabulary {
    pub(crate) inner: OrbVocabulary,
}

#[pymethods]
impl PyOrbVocabulary {
    /// Creates an empty OrbVocabulary.
    ///
    /// # Returns
    ///
    /// A new, uninitialized [`PyOrbVocabulary`].
    #[new]
    pub fn new() -> Self {
        Self {
            inner: OrbVocabulary {
                blocks: Vec::new(),
                root_idx: 0,
            },
        }
    }

    /// Pickle serialization reusing streaming binary format (releases GIL).
    ///
    /// # Arguments
    ///
    /// * `py` - Python runtime marker.
    ///
    /// # Returns
    ///
    /// Serialized binary byte vector.
    ///
    /// # Errors
    ///
    /// Returns [`PyValueError`] if the vocabulary is empty, or a [`PyErr`] if encoding fails.
    pub fn __getstate__(&self, py: Python<'_>) -> PyResult<Vec<u8>> {
        if self.inner.blocks.is_empty() {
            return Err(PyValueError::new_err(
                "cannot serialize an empty vocabulary",
            ));
        }
        py.detach(|| {
            let mut buf = Vec::new();
            self.inner.save_to_writer(&mut buf).map_err(bow_to_pyerr)?;
            Ok(buf)
        })
    }

    /// Pickle deserialization reusing streaming reader and graph validation (releases GIL).
    ///
    /// # Arguments
    ///
    /// * `py` - Python runtime marker.
    /// * `state` - Serialized byte slice.
    ///
    /// # Returns
    ///
    /// `Ok(())` on success.
    ///
    /// # Errors
    ///
    /// Returns [`PyValueError`] or [`PyIOError`] if bytes are invalid or corrupted.
    pub fn __setstate__(&mut self, py: Python<'_>, state: &[u8]) -> PyResult<()> {
        let vocab = py
            .detach(|| {
                let cursor = std::io::Cursor::new(state);
                OrbVocabulary::load_from_reader(cursor, state.len() as u64)
            })
            .map_err(bow_to_pyerr)?;
        self.inner = vocab;
        Ok(())
    }

    /// Loads an ORB-SLAM3 / DBoW2 text vocabulary file (`ORBvoc.txt`).
    ///
    /// # Arguments
    ///
    /// * `py` - Python runtime marker.
    /// * `path` - Path to the text vocabulary file.
    ///
    /// # Returns
    ///
    /// Loaded [`PyOrbVocabulary`].
    ///
    /// # Errors
    ///
    /// Returns [`PyIOError`] if reading fails, or [`PyValueError`] on invalid format.
    #[staticmethod]
    pub fn from_orb_slam3(py: Python<'_>, path: &str) -> PyResult<Self> {
        let inner = py.detach(|| load_orb_voc(path)).map_err(bow_to_pyerr)?;
        Ok(Self { inner })
    }

    /// Loads a fast binary vocabulary file previously saved with `save`.
    ///
    /// # Arguments
    ///
    /// * `py` - Python runtime marker.
    /// * `path` - Path to the binary vocabulary file.
    ///
    /// # Returns
    ///
    /// Loaded [`PyOrbVocabulary`].
    ///
    /// # Errors
    ///
    /// Returns [`PyIOError`] if reading fails, or [`PyValueError`] on invalid header or structure.
    #[staticmethod]
    pub fn from_file(py: Python<'_>, path: &str) -> PyResult<Self> {
        let inner = py
            .detach(|| OrbVocabulary::load(path))
            .map_err(bow_to_pyerr)?;
        Ok(Self { inner })
    }

    /// Trains a new hierarchical ORB vocabulary from a NumPy descriptor array.
    ///
    /// # Arguments
    ///
    /// * `py` - Python runtime marker.
    /// * `descriptors` - `(N, 32)` uint8 ORB descriptors (minimum 10 required).
    /// * `max_depth` - Maximum depth of the vocabulary tree (between 1 and 16, default 6).
    ///
    /// # Returns
    ///
    /// Trained [`PyOrbVocabulary`].
    ///
    /// # Errors
    ///
    /// Returns [`PyValueError`] if descriptors shape or count is invalid or `max_depth` is outside [1, 16].
    #[staticmethod]
    #[pyo3(signature = (descriptors, max_depth=6))]
    pub fn train(
        py: Python<'_>,
        descriptors: Bound<'_, PyArray2<u8>>,
        max_depth: usize,
    ) -> PyResult<Self> {
        if max_depth == 0 || max_depth > 16 {
            return Err(PyValueError::new_err(format!(
                "max_depth must be between 1 and 16, got {max_depth}"
            )));
        }
        let shape = descriptors.shape();
        if shape[0] < ORB_BRANCHING {
            return Err(PyValueError::new_err(format!(
                "training requires at least {ORB_BRANCHING} descriptors for branching factor {ORB_BRANCHING}, got {}",
                shape[0]
            )));
        }
        let slice = validate_and_get_descriptor_slice(&descriptors)?;
        let features = pack_orb_descriptors_slice(slice);
        let inner = py
            .detach(|| OrbVocabulary::train(&features, max_depth))
            .map_err(bow_to_pyerr)?;
        Ok(Self { inner })
    }

    /// Saves the vocabulary to a fast binary format (`.bow`).
    ///
    /// # Arguments
    ///
    /// * `py` - Python runtime marker.
    /// * `path` - Destination file path.
    ///
    /// # Returns
    ///
    /// `Ok(())` on success.
    ///
    /// # Errors
    ///
    /// Returns [`PyValueError`] if vocabulary is empty, or [`PyIOError`] on write/flush failure.
    pub fn save(&self, py: Python<'_>, path: &str) -> PyResult<()> {
        if self.inner.blocks.is_empty() {
            return Err(PyValueError::new_err(
                "cannot save an empty vocabulary to a file",
            ));
        }
        py.detach(|| self.inner.save(path)).map_err(bow_to_pyerr)
    }

    /// Transforms `(N, 32)` ORB descriptors into a sparse BoW vector.
    ///
    /// # Arguments
    ///
    /// * `py` - Python runtime marker.
    /// * `descriptors` - `(N, 32)` uint8 ORB descriptors.
    ///
    /// # Returns
    ///
    /// A normalized sparse [`PyBoW`] vector.
    ///
    /// # Errors
    ///
    /// Returns [`PyValueError`] if the vocabulary is empty or descriptor array is invalid.
    pub fn transform(
        &self,
        py: Python<'_>,
        descriptors: Bound<'_, PyArray2<u8>>,
    ) -> PyResult<PyBoW> {
        if self.inner.blocks.is_empty() {
            return Err(PyValueError::new_err(
                "cannot transform descriptors with an empty vocabulary",
            ));
        }
        let slice = validate_and_get_descriptor_slice(&descriptors)?;
        let features = pack_orb_descriptors_slice(slice);
        let bow = py
            .detach(|| self.inner.transform(&features))
            .map_err(bow_to_pyerr)?;
        Ok(PyBoW { inner: bow })
    }

    /// Transforms descriptors into a sparse BoW vector and a Direct Index.
    ///
    /// # Arguments
    ///
    /// * `descriptors` - `(N, 32)` uint8 ORB descriptors.
    /// * `level` - Vocabulary tree depth level for direct indexing (default 2, matching ORB-SLAM3 `levelsup=4` on depth-6 trees). Level 0 is the root block.
    ///
    /// # Returns
    ///
    /// `(bow, direct_index)` tuple.
    ///
    /// # Errors
    ///
    /// Returns [`PyValueError`] if the vocabulary is empty or descriptor array is invalid.
    #[pyo3(signature = (descriptors, level=2))]
    pub fn transform_with_direct_index(
        &self,
        py: Python<'_>,
        descriptors: Bound<'_, PyArray2<u8>>,
        level: usize,
    ) -> PyResult<(PyBoW, PyDirectIndex)> {
        if self.inner.blocks.is_empty() {
            return Err(PyValueError::new_err(
                "cannot transform descriptors with an empty vocabulary",
            ));
        }
        let slice = validate_and_get_descriptor_slice(&descriptors)?;
        let features = pack_orb_descriptors_slice(slice);
        let (bow, di) = py
            .detach(|| self.inner.transform_with_direct_index(&features, level))
            .map_err(bow_to_pyerr)?;
        Ok((PyBoW { inner: bow }, PyDirectIndex { inner: di }))
    }

    /// Maps a single 32-byte descriptor to its word ID and IDF weight.
    ///
    /// # Arguments
    ///
    /// * `descriptor` - A 1D `(32,)` or 2D `(1, 32)` uint8 NumPy array.
    ///
    /// # Returns
    ///
    /// `(word_id, idf_weight)` tuple.
    ///
    /// # Errors
    ///
    /// Returns [`PyValueError`] if the vocabulary is empty or descriptor format is invalid.
    pub fn transform_one(&self, descriptor: Bound<'_, PyAny>) -> PyResult<(u32, f32)> {
        if self.inner.blocks.is_empty() {
            return Err(PyValueError::new_err(
                "cannot transform descriptor with an empty vocabulary",
            ));
        }
        let feature = extract_single_orb_descriptor(&descriptor)?;
        Ok(self.inner.transform_one(&feature))
    }

    /// Number of blocks in the vocabulary tree.
    ///
    /// # Returns
    ///
    /// Total block count as `usize`.
    #[getter]
    pub fn num_blocks(&self) -> usize {
        self.inner.blocks.len()
    }

    /// Branching factor of the vocabulary tree (k=10 for ORB-SLAM3).
    ///
    /// # Returns
    ///
    /// Branching factor integer (10).
    #[getter]
    pub fn branching_factor(&self) -> usize {
        ORB_BRANCHING
    }

    /// Size of each descriptor in bytes (32 for 256-bit ORB).
    ///
    /// # Returns
    ///
    /// Descriptor size in bytes (32).
    #[getter]
    pub fn descriptor_size(&self) -> usize {
        32
    }

    /// Number of blocks in the vocabulary tree.
    ///
    /// # Returns
    ///
    /// Total block count as `usize`.
    pub fn __len__(&self) -> usize {
        self.inner.blocks.len()
    }

    /// String representation of the vocabulary.
    ///
    /// # Returns
    ///
    /// Formatted string with block count and root index.
    pub fn __repr__(&self) -> String {
        format!(
            "OrbVocabulary(blocks={}, root={})",
            self.inner.blocks.len(),
            self.inner.root_idx
        )
    }
}

/// Convenience function to load an ORB-SLAM3 / DBoW2 text vocabulary file (`ORBvoc.txt`).
///
/// # Arguments
///
/// * `py` - Python runtime marker.
/// * `path` - Path to the text vocabulary file.
///
/// # Returns
///
/// Loaded [`PyOrbVocabulary`].
///
/// # Errors
///
/// Returns [`PyIOError`] if the file cannot be opened or read.
/// Returns [`PyValueError`] if the vocabulary format is invalid or corrupted.
#[pyfunction]
pub fn load_orb_slam3_vocabulary(py: Python<'_>, path: &str) -> PyResult<PyOrbVocabulary> {
    PyOrbVocabulary::from_orb_slam3(py, path)
}

/// Convenience function to load a binary vocabulary file (`.bow`) previously saved with `save`.
///
/// # Arguments
///
/// * `py` - Python runtime marker.
/// * `path` - Path to the binary vocabulary file.
///
/// # Returns
///
/// Loaded [`PyOrbVocabulary`].
///
/// # Errors
///
/// Returns [`PyIOError`] if the file cannot be opened or read.
/// Returns [`PyValueError`] if the binary header or structure is invalid.
#[pyfunction]
pub fn load_vocabulary(py: Python<'_>, path: &str) -> PyResult<PyOrbVocabulary> {
    PyOrbVocabulary::from_file(py, path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bow_to_pyerr_mappings() {
        let _ = bow_to_pyerr(BowError::NoFeatures);
        let _ = bow_to_pyerr(BowError::CorruptedVocabulary);
        let _ = bow_to_pyerr(BowError::Bincode("err".into()));
    }

    #[test]
    fn test_py_bow_properties_and_similarity() {
        let b1 = PyBoW {
            inner: CoreBoW(vec![(1, 0.4), (2, 0.6)]),
        };
        let b2 = PyBoW {
            inner: CoreBoW(vec![(1, 0.5), (3, 0.5)]),
        };

        assert_eq!(b1.__len__(), 2);
        assert_eq!(b1.__getitem__(1), 0.4);
        assert_eq!(b1.__getitem__(2), 0.6);
        assert_eq!(b1.__getitem__(99), 0.0);
        assert!(b1.__contains__(1));
        assert!(!b1.__contains__(99));
        assert_eq!(b1.items(), vec![(1, 0.4), (2, 0.6)]);
        assert_eq!(b1, b1);
        assert_ne!(b1, b2);

        let sim = b1.l1_similarity(&b2);
        assert!((sim - 0.4).abs() < 1e-6);

        let dot = b1.dot_product(&b2);
        assert!((dot - 0.2).abs() < 1e-6);
        assert_eq!(b1.__matmul__(&b2), dot);

        assert_eq!(b1.score(&b2, "l1").unwrap(), sim);
        assert_eq!(b1.score(&b2, "dot").unwrap(), dot);
        assert!(b1.score(&b2, "invalid").is_err());
        // "kl" is a divergence, so score() rejects it with ValueError
        assert!(b1.score(&b2, "kl").is_err());

        // Self-similarity under L2 must be 1.0
        let l2_self = b1.score(&b1, "l2").unwrap();
        assert!((l2_self - 1.0).abs() < 1e-6);
        assert!((b1.l2_similarity(&b1) - 1.0).abs() < 1e-6);

        // KL divergence distance on self must be 0.0
        assert!((b1.kl_divergence(&b1) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_empty_vocab_raises_error() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            let empty_vocab = PyOrbVocabulary::new();
            let desc = numpy::PyArray2::<u8>::zeros(py, [1, 32], false);
            assert!(empty_vocab.transform(py, desc.clone()).is_err());
            assert!(empty_vocab
                .transform_with_direct_index(py, desc.clone(), 2)
                .is_err());
            assert!(empty_vocab.transform_one(desc.as_any().clone()).is_err());
            assert!(empty_vocab.__getstate__(py).is_err());
        });
    }

    #[test]
    fn test_py_bow_normalization() {
        let b = PyBoW {
            inner: CoreBoW(vec![(1, 3.0), (2, 4.0)]),
        };
        let l2_norm = b.normalized_l2();
        let sum_sq: f32 = l2_norm.inner.0.iter().map(|&(_, v)| v * v).sum();
        assert!((sum_sq - 1.0).abs() < 1e-6);

        let l1_norm = b.normalized_l1();
        let sum: f32 = l1_norm.inner.0.iter().map(|&(_, v)| v).sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_py_direct_index() {
        let di = PyDirectIndex {
            inner: CoreDirectIndex(vec![(10, vec![0, 1, 2]), (20, vec![3, 4])]),
        };

        assert_eq!(di.__len__(), 2);
        assert_eq!(di.get(10), Some(&[0, 1, 2][..]));
        assert_eq!(di.get(20), Some(&[3, 4][..]));
        assert_eq!(di.get(99), None);
        assert_eq!(di.__getitem__(10).unwrap(), &[0, 1, 2][..]);
        assert!(di.__getitem__(99).is_err());
        assert!(di.__contains__(10));
        assert!(!di.__contains__(99));
        assert_eq!(di.keys(), vec![10, 20]);
        assert_eq!(di.values(), vec![vec![0, 1, 2], vec![3, 4]]);
        assert_eq!(di.items(), vec![(10, vec![0, 1, 2]), (20, vec![3, 4])]);
        assert_eq!(di, di);
    }

    #[test]
    fn test_py_bow_pickle_state() {
        let b = PyBoW {
            inner: CoreBoW(vec![(5, 1.2), (10, 3.4)]),
        };
        let state = b.__getstate__();
        let mut restored = PyBoW::new();
        restored.__setstate__(state).unwrap();
        assert_eq!(b, restored);
    }

    #[test]
    fn test_py_direct_index_pickle_state() {
        let di = PyDirectIndex {
            inner: CoreDirectIndex(vec![(5, vec![0, 1]), (10, vec![2])]),
        };
        let state = di.__getstate__();
        let mut restored = PyDirectIndex::new();
        restored.__setstate__(state).unwrap();
        assert_eq!(di, restored);
    }

    #[test]
    fn test_py_vocab_pickle_state() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            let dummy_feat = Feature([0u64; ORB_WORDS]);
            let vocab = OrbVocabulary::train(&[dummy_feat; 10], 2).unwrap();
            let py_vocab = PyOrbVocabulary { inner: vocab };
            let state = py_vocab.__getstate__(py).unwrap();

            let mut restored = PyOrbVocabulary {
                inner: OrbVocabulary {
                    blocks: Vec::new(),
                    root_idx: 0,
                },
            };
            restored.__setstate__(py, &state).unwrap();
            assert_eq!(py_vocab.num_blocks(), restored.num_blocks());
        });
    }
}
