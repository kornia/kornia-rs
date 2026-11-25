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
    #[new]
    pub fn new() -> Self {
        Self {
            inner: CoreBoW(Vec::new()),
        }
    }

    /// Equality comparison between two BoW vectors.
    pub fn __eq__(&self, other: &PyBoW) -> bool {
        self.inner == other.inner
    }

    /// Checks if a word ID is present in the sparse vector.
    pub fn __contains__(&self, word_id: u32) -> bool {
        self.inner
            .0
            .binary_search_by_key(&word_id, |&(w, _)| w)
            .is_ok()
    }

    /// Pickle support serialization.
    pub fn __getstate__(&self) -> Vec<(u32, f32)> {
        self.inner.0.clone()
    }

    /// Pickle support deserialization with canonical validation.
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
    pub fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyIterator>> {
        let words: Vec<u32> = self.inner.0.iter().map(|&(w, _)| w).collect();
        pyo3::types::PyList::new(py, words)?.as_any().try_iter()
    }

    /// Non-zero word IDs in the sparse vector as a newly allocated 1D uint32 numpy array.
    ///
    /// Note: BoW is stored internally as pairs of (word_id, weight). Calling this allocates a new array.
    pub fn words<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        PyArray1::from_iter(py, self.inner.0.iter().map(|&(w, _)| w))
    }

    /// Weights of non-zero words as a newly allocated 1D float32 numpy array.
    ///
    /// Note: BoW is stored internally as pairs of (word_id, weight). Calling this allocates a new array.
    pub fn weights<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_iter(py, self.inner.0.iter().map(|&(_, wt)| wt))
    }

    /// Returns a list of `(word_id, weight)` tuples for non-zero words.
    pub fn items(&self) -> Vec<(u32, f32)> {
        self.inner.0.clone()
    }

    /// Converts the sparse BoW vector into a Python dictionary `{word_id: weight}`.
    pub fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for &(w, wt) in &self.inner.0 {
            dict.set_item(w, wt)?;
        }
        Ok(dict)
    }

    /// Converts the sparse BoW vector into a list of `(word_id, weight)` tuples.
    pub fn to_list(&self) -> Vec<(u32, f32)> {
        self.inner.0.clone()
    }

    /// Returns the number of non-zero words in the sparse BoW vector.
    pub fn __len__(&self) -> usize {
        self.inner.0.len()
    }

    /// String representation of the BoW vector.
    pub fn __repr__(&self) -> String {
        format!("BoW(words={})", self.inner.0.len())
    }

    /// Gets the weight of a specific word ID, returning 0.0 if not present.
    pub fn __getitem__(&self, word_id: u32) -> f32 {
        match self.inner.0.binary_search_by_key(&word_id, |&(w, _)| w) {
            Ok(idx) => self.inner.0[idx].1,
            Err(_) => 0.0,
        }
    }

    /// Computes similarity score with another BoW vector using the specified metric.
    ///
    /// Metric choices:
    /// - `"l1"` (default): L1 similarity in range [0, 1] (standard DBoW2 scoring, for L1-normalized non-negative vectors).
    /// - `"l2"`: L2 similarity.
    /// - `"dot"`: Dot product similarity.
    /// - `"chi_square"`: Chi-square similarity.
    /// - `"kl"`: Kullback-Leibler divergence score.
    /// - `"bhattacharyya"`: Bhattacharyya coefficient similarity.
    #[pyo3(signature = (other, metric="l1"))]
    pub fn score(&self, other: &PyBoW, metric: &str) -> PyResult<f32> {
        match metric {
            "l1" => Ok(self.inner.l1_similarity(&other.inner)),
            "l2" => Ok(self.inner.l2(&other.inner)),
            "dot" => Ok(self.inner.dot_product(&other.inner)),
            "chi_square" => Ok(self.inner.chi_square(&other.inner)),
            "kl" => Ok(self.inner.kl_divergence(&other.inner)),
            "bhattacharyya" => Ok(self.inner.bhattacharyya(&other.inner)),
            _ => Err(PyValueError::new_err(format!(
                "unsupported metric '{metric}'; expected one of: 'l1', 'l2', 'dot', 'chi_square', 'kl', 'bhattacharyya'"
            ))),
        }
    }

    /// Computes L1 similarity score (in range [0.0, 1.0] for L1-normalized non-negative vectors).
    pub fn l1_similarity(&self, other: &PyBoW) -> f32 {
        self.inner.l1_similarity(&other.inner)
    }

    /// Computes L2 similarity score.
    pub fn l2_similarity(&self, other: &PyBoW) -> f32 {
        self.inner.l2(&other.inner)
    }

    /// Computes dot product similarity.
    pub fn dot_product(&self, other: &PyBoW) -> f32 {
        self.inner.dot_product(&other.inner)
    }

    /// Computes Chi-square similarity score.
    pub fn chi_square(&self, other: &PyBoW) -> f32 {
        self.inner.chi_square(&other.inner)
    }

    /// Computes Kullback-Leibler divergence score.
    pub fn kl_divergence(&self, other: &PyBoW) -> f32 {
        self.inner.kl_divergence(&other.inner)
    }

    /// Computes Bhattacharyya coefficient similarity score.
    pub fn bhattacharyya(&self, other: &PyBoW) -> f32 {
        self.inner.bhattacharyya(&other.inner)
    }

    /// Python matrix multiplication operator (`@`) computes dot product similarity.
    pub fn __matmul__(&self, other: &PyBoW) -> f32 {
        self.inner.dot_product(&other.inner)
    }

    /// Returns a new BoW vector normalized by L2 norm.
    pub fn normalized_l2(&self) -> PyBoW {
        let mut copy = self.inner.clone();
        copy.normalize_l2();
        PyBoW { inner: copy }
    }

    /// Returns a new BoW vector normalized by L1 norm.
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
    #[new]
    pub fn new() -> Self {
        Self {
            inner: CoreDirectIndex(Vec::new()),
        }
    }

    /// Number of distinct tree nodes present in the direct index.
    pub fn __len__(&self) -> usize {
        self.inner.0.len()
    }

    /// String representation of the direct index.
    pub fn __repr__(&self) -> String {
        format!("DirectIndex(nodes={})", self.inner.0.len())
    }

    /// Equality comparison between two direct indices.
    pub fn __eq__(&self, other: &PyDirectIndex) -> bool {
        self.inner == other.inner
    }

    /// Checks if a node ID is present in the direct index.
    pub fn __contains__(&self, node_id: u32) -> bool {
        self.inner
            .0
            .binary_search_by_key(&node_id, |&(n, _)| n)
            .is_ok()
    }

    /// Returns an iterator over node IDs present in the direct index.
    pub fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyIterator>> {
        let keys: Vec<u32> = self.inner.0.iter().map(|&(k, _)| k).collect();
        pyo3::types::PyList::new(py, keys)?.as_any().try_iter()
    }

    /// Returns a list of node IDs present in the direct index.
    pub fn keys(&self) -> Vec<u32> {
        self.inner.0.iter().map(|&(k, _)| k).collect()
    }

    /// Returns a list of feature index vectors present in the direct index.
    pub fn values(&self) -> Vec<Vec<u32>> {
        self.inner.0.iter().map(|(_, v)| v.clone()).collect()
    }

    /// Returns a list of (node_id, feature_indices) pairs in the direct index.
    pub fn items(&self) -> Vec<(u32, Vec<u32>)> {
        self.inner.0.clone()
    }

    /// Pickle support serialization.
    pub fn __getstate__(&self) -> Vec<(u32, Vec<u32>)> {
        self.inner.0.clone()
    }

    /// Pickle support deserialization with canonical validation.
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

    /// Look up feature indices associated with a node ID. Returns None if node is not present.
    pub fn get(&self, node_id: u32) -> Option<&[u32]> {
        match self.inner.0.binary_search_by_key(&node_id, |&(n, _)| n) {
            Ok(idx) => Some(&self.inner.0[idx].1),
            Err(_) => None,
        }
    }

    /// Look up feature indices associated with a node ID, raising KeyError if not found.
    pub fn __getitem__(&self, node_id: u32) -> PyResult<&[u32]> {
        self.get(node_id)
            .ok_or_else(|| PyKeyError::new_err(format!("node {node_id} not in direct index")))
    }

    /// Converts the direct index into a dictionary `{node_id: [feature_idx, ...]}`.
    pub fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for &(node_id, ref feats) in &self.inner.0 {
            dict.set_item(node_id, feats.clone())?;
        }
        Ok(dict)
    }

    /// Converts the direct index into a list of `(node_id, [feature_idx, ...])` tuples.
    pub fn to_list(&self) -> Vec<(u32, Vec<u32>)> {
        self.inner.0.clone()
    }
}

/// Validate that a vocabulary tree is non-empty, non-cyclic, and all internal indices are within bounds.
fn validate_vocab_tree(vocab: &OrbVocabulary) -> PyResult<()> {
    if vocab.blocks.is_empty() {
        return Err(PyValueError::new_err("Vocabulary contains no blocks"));
    }
    let num_blocks = vocab.blocks.len();
    if (vocab.root_idx as usize) >= num_blocks {
        return Err(PyValueError::new_err(format!(
            "Root block index {} out of bounds (total blocks: {})",
            vocab.root_idx, num_blocks
        )));
    }
    for (i, block) in vocab.blocks.iter().enumerate() {
        if let kornia_bow::BlockContent::Internal(ref meta) = block.content {
            let base = meta.children_base_idx as usize;
            if base >= num_blocks || base + ORB_BRANCHING > num_blocks {
                return Err(PyValueError::new_err(format!(
                    "Internal block {i} has children base index {base} out of bounds (total blocks: {})",
                    num_blocks
                )));
            }
        }
    }
    Ok(())
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
    #[new]
    pub fn new() -> Self {
        Self {
            inner: OrbVocabulary {
                blocks: Vec::new(),
                root_idx: 0,
            },
        }
    }

    /// Pickle support serialization (releases GIL during serialization).
    pub fn __getstate__(&self, py: Python<'_>) -> PyResult<Vec<u8>> {
        py.detach(|| {
            let mut buf = Vec::new();
            let config = bincode::config::standard();
            bincode::encode_into_std_write(ORB_BRANCHING as u64, &mut buf, config)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            bincode::encode_into_std_write(
                kornia_bow::metric::MetricType::Hamming,
                &mut buf,
                config,
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
            bincode::encode_into_std_write(&self.inner, &mut buf, config)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            Ok(buf)
        })
    }

    /// Pickle support deserialization (releases GIL and validates tree structure).
    pub fn __setstate__(&mut self, py: Python<'_>, state: &[u8]) -> PyResult<()> {
        let vocab: OrbVocabulary = py.detach(|| {
            let mut reader = std::io::Cursor::new(state);
            let config = bincode::config::standard();
            let loaded_b: u64 = bincode::decode_from_std_read(&mut reader, config)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let loaded_metric: kornia_bow::metric::MetricType =
                bincode::decode_from_std_read(&mut reader, config)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
            if loaded_b != ORB_BRANCHING as u64
                || loaded_metric != kornia_bow::metric::MetricType::Hamming
            {
                return Err(PyValueError::new_err("Invalid vocabulary binary state"));
            }
            let v: OrbVocabulary = bincode::decode_from_std_read(&mut reader, config)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            // If vocabulary has blocks, validate its structural integrity
            if !v.blocks.is_empty() {
                validate_vocab_tree(&v)?;
            }
            Ok(v)
        })?;
        self.inner = vocab;
        Ok(())
    }
    /// Loads an ORB-SLAM3 / DBoW2 text vocabulary file (`ORBvoc.txt`).
    ///
    /// Args:
    ///     path: path to the text vocabulary file.
    ///
    /// Returns:
    ///     Loaded `OrbVocabulary`.
    #[staticmethod]
    pub fn from_orb_slam3(py: Python<'_>, path: &str) -> PyResult<Self> {
        let inner = py.detach(|| load_orb_voc(path)).map_err(bow_to_pyerr)?;
        Ok(Self { inner })
    }

    /// Loads a fast binary vocabulary file previously saved with `save`.
    ///
    /// Args:
    ///     path: path to the binary vocabulary file.
    ///
    /// Returns:
    ///     Loaded `OrbVocabulary`.
    #[staticmethod]
    pub fn from_file(py: Python<'_>, path: &str) -> PyResult<Self> {
        let inner = py
            .detach(|| OrbVocabulary::load(path))
            .map_err(bow_to_pyerr)?;
        Ok(Self { inner })
    }

    /// Trains a new hierarchical ORB vocabulary from a NumPy descriptor array.
    ///
    /// Args:
    ///     descriptors: `(N, 32)` uint8 ORB descriptors (minimum 10 required).
    ///     max_depth: maximum depth of the vocabulary tree (between 1 and 16, default 6).
    ///
    /// Returns:
    ///     Trained `OrbVocabulary`.
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
        let raw_bytes: Vec<u8> = validate_and_get_descriptor_slice(&descriptors)?.to_vec();
        let inner = py
            .detach(|| {
                let features = pack_orb_descriptors_slice(&raw_bytes);
                OrbVocabulary::train(&features, max_depth)
            })
            .map_err(bow_to_pyerr)?;
        Ok(Self { inner })
    }

    /// Saves the vocabulary to a fast binary format (`.bow`).
    ///
    /// Args:
    ///     path: destination file path.
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
    /// Args:
    ///     descriptors: `(N, 32)` uint8 ORB descriptors.
    ///
    /// Returns:
    ///     A normalized sparse `BoW` vector.
    pub fn transform(
        &self,
        py: Python<'_>,
        descriptors: Bound<'_, PyArray2<u8>>,
    ) -> PyResult<PyBoW> {
        let raw_bytes: Vec<u8> = validate_and_get_descriptor_slice(&descriptors)?.to_vec();
        let bow = py
            .detach(|| {
                let features = pack_orb_descriptors_slice(&raw_bytes);
                self.inner.transform(&features)
            })
            .map_err(bow_to_pyerr)?;
        Ok(PyBoW { inner: bow })
    }

    /// Transforms descriptors into a sparse BoW vector and a Direct Index.
    ///
    /// Args:
    ///     descriptors: `(N, 32)` uint8 ORB descriptors.
    ///     level: vocabulary tree level for direct indexing (default 4 for depth-6 tree).
    ///
    /// Returns:
    ///     `(bow, direct_index)` tuple.
    #[pyo3(signature = (descriptors, level=4))]
    pub fn transform_with_direct_index(
        &self,
        py: Python<'_>,
        descriptors: Bound<'_, PyArray2<u8>>,
        level: usize,
    ) -> PyResult<(PyBoW, PyDirectIndex)> {
        let raw_bytes: Vec<u8> = validate_and_get_descriptor_slice(&descriptors)?.to_vec();
        let (bow, di) = py
            .detach(|| {
                let features = pack_orb_descriptors_slice(&raw_bytes);
                self.inner.transform_with_direct_index(&features, level)
            })
            .map_err(bow_to_pyerr)?;
        Ok((PyBoW { inner: bow }, PyDirectIndex { inner: di }))
    }

    /// Maps a single 32-byte descriptor to its word ID and IDF weight.
    ///
    /// Args:
    ///     descriptor: a 1D `(32,)` or 2D `(1, 32)` uint8 numpy array.
    ///
    /// Returns:
    ///     `(word_id, idf_weight)` tuple.
    pub fn transform_one(&self, descriptor: Bound<'_, PyAny>) -> PyResult<(u32, f32)> {
        let feature = extract_single_orb_descriptor(&descriptor)?;
        Ok(self.inner.transform_one(&feature))
    }

    /// Number of blocks in the vocabulary tree.
    #[getter]
    pub fn num_blocks(&self) -> usize {
        self.inner.blocks.len()
    }

    /// Branching factor of the vocabulary tree (k=10 for ORB-SLAM3).
    #[getter]
    pub fn branching_factor(&self) -> usize {
        ORB_BRANCHING
    }

    /// Size of each descriptor in bytes (32 for 256-bit ORB).
    #[getter]
    pub fn descriptor_size(&self) -> usize {
        32
    }

    /// Number of blocks in the vocabulary tree.
    pub fn __len__(&self) -> usize {
        self.inner.blocks.len()
    }

    /// String representation of the vocabulary.
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
/// Args:
///     path: path to the text vocabulary file.
///
/// Returns:
///     Loaded `OrbVocabulary`.
///
/// Raises:
///     IOError: if the file cannot be opened or read.
///     ValueError: if the vocabulary format is invalid or corrupted.
#[pyfunction]
pub fn load_orb_slam3_vocabulary(py: Python<'_>, path: &str) -> PyResult<PyOrbVocabulary> {
    PyOrbVocabulary::from_orb_slam3(py, path)
}

/// Convenience function to load a binary vocabulary file (`.bow`) previously saved with `save`.
///
/// Args:
///     path: path to the binary vocabulary file.
///
/// Returns:
///     Loaded `OrbVocabulary`.
///
/// Raises:
///     IOError: if the file cannot be opened or read.
///     ValueError: if the binary header or structure is invalid.
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
