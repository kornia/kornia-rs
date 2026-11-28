"""Type stubs for the ``kornia_rs.bow`` submodule (Bag of Words / Visual SLAM Place Recognition)."""

from typing import Any, Dict, Iterator, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt

def __getattr__(name: str) -> Any:
    """Module attribute fallback."""
    ...

class BoW:
    """Sparse Bag of Words vector representing an image: sorted pairs of ``(word_id, weight)``."""

    def __init__(self) -> None:
        """Initializes an empty BoW vector."""
        ...
    def __iter__(self) -> Iterator[int]:
        """Returns an iterator over non-zero word IDs in the sparse vector."""
        ...
    def items(self) -> List[Tuple[int, float]]:
        """Returns a list of ``(word_id, weight)`` tuples for non-zero words."""
        ...
    def words(self) -> npt.NDArray[np.uint32]:
        """Non-zero word IDs in the sparse vector as a newly allocated 1D uint32 numpy array."""
        ...
    def weights(self) -> npt.NDArray[np.float32]:
        """Weights of non-zero words as a newly allocated 1D float32 numpy array."""
        ...
    def to_dict(self) -> Dict[int, float]:
        """Converts the sparse BoW vector into a Python dictionary ``{word_id: weight}``."""
        ...
    def to_list(self) -> List[Tuple[int, float]]:
        """Converts the sparse BoW vector into a list of ``(word_id, weight)`` tuples."""
        ...
    def __len__(self) -> int:
        """Returns the number of non-zero words in the sparse BoW vector."""
        ...
    def __repr__(self) -> str:
        """Returns string representation of the BoW vector."""
        ...
    def __eq__(self, other: object) -> bool:
        """Checks equality with another BoW vector."""
        ...
    def __contains__(self, word_id: int) -> bool:
        """Checks if a word ID is present in the sparse vector."""
        ...
    def __getitem__(self, word_id: int) -> float:
        """Gets the weight of a specific word ID, returning 0.0 if not present."""
        ...
    def score(
        self,
        other: "BoW",
        metric: Literal["l1", "l2", "dot", "chi_square", "bhattacharyya"] = "l1",
    ) -> float:
        """Computes similarity score with another BoW vector using the specified metric.

        Metric choices:
        - ``"l1"``: L1 similarity in range [0.0, 1.0] for L1-normalized vectors.
        - ``"l2"``: L2 similarity in range [0.0, 1.0] (automatically normalizes inputs with L2 norm).
        - ``"dot"``: Dot product similarity.
        - ``"chi_square"``: Chi-square similarity.
        - ``"bhattacharyya"``: Bhattacharyya coefficient similarity.

        Note: To compute Kullback-Leibler divergence distance, use :meth:`kl_divergence`.
        """
        ...
    def l1_similarity(self, other: "BoW") -> float:
        """Computes L1 similarity score (in range [0.0, 1.0] for non-negative, L1-normalized vectors)."""
        ...
    def l2_similarity(self, other: "BoW") -> float:
        """Computes L2 similarity score in range [0.0, 1.0] (automatically L2-normalizes vectors)."""
        ...
    def dot_product(self, other: "BoW") -> float:
        """Computes dot product similarity."""
        ...
    def chi_square(self, other: "BoW") -> float:
        """Computes Chi-square similarity score."""
        ...
    def kl_divergence(self, other: "BoW") -> float:
        """Computes Kullback-Leibler (KL) divergence distance (non-symmetric).

        Note: This is a divergence/distance metric where 0.0 indicates identical
        distributions and larger values indicate greater dissimilarity.
        """
        ...
    def bhattacharyya(self, other: "BoW") -> float:
        """Computes Bhattacharyya coefficient similarity score."""
        ...
    def __matmul__(self, other: "BoW") -> float:
        """Matrix multiplication operator (``@``) computes dot product similarity."""
        ...
    def normalized_l2(self) -> "BoW":
        """Returns a new BoW vector normalized by L2 norm."""
        ...
    def normalized_l1(self) -> "BoW":
        """Returns a new BoW vector normalized by L1 norm."""
        ...
    def __getstate__(self) -> List[Tuple[int, float]]:
        """Serializes BoW sparse vector state for pickling."""
        ...
    def __setstate__(self, state: List[Tuple[int, float]]) -> None:
        """Deserializes BoW state from pickling, validating strictly increasing word IDs."""
        ...

class DirectIndex:
    """Direct Index mapping vocabulary tree node IDs to feature indices in the source image."""

    def __init__(self) -> None:
        """Initializes an empty DirectIndex mapping."""
        ...
    def __len__(self) -> int:
        """Number of distinct tree nodes present in the direct index."""
        ...
    def __repr__(self) -> str:
        """Returns string representation of the direct index."""
        ...
    def __eq__(self, other: object) -> bool:
        """Checks equality with another direct index."""
        ...
    def __contains__(self, node_id: int) -> bool:
        """Checks if a node ID is present in the direct index."""
        ...
    def __iter__(self) -> Iterator[int]:
        """Returns an iterator over node IDs present in the direct index."""
        ...
    def keys(self) -> List[int]:
        """Returns a list of node IDs present in the direct index."""
        ...
    def values(self) -> List[List[int]]:
        """Returns a list of feature index vectors present in the direct index."""
        ...
    def items(self) -> List[Tuple[int, List[int]]]:
        """Returns a list of ``(node_id, [feature_idx, ...])`` pairs in the direct index."""
        ...
    def get(self, node_id: int) -> Optional[Sequence[int]]:
        """Look up feature indices associated with a node ID. Returns None if node is not present."""
        ...
    def __getitem__(self, node_id: int) -> Sequence[int]:
        """Look up feature indices associated with a node ID, raising KeyError if not found."""
        ...
    def to_dict(self) -> Dict[int, List[int]]:
        """Converts the direct index into a dictionary ``{node_id: [feature_idx, ...]}``."""
        ...
    def to_list(self) -> List[Tuple[int, List[int]]]:
        """Converts the direct index into a list of ``(node_id, [feature_idx, ...])`` tuples."""
        ...
    def __getstate__(self) -> List[Tuple[int, List[int]]]:
        """Serializes DirectIndex mapping state for pickling."""
        ...
    def __setstate__(self, state: List[Tuple[int, List[int]]]) -> None:
        """Deserializes DirectIndex state from pickling, validating strictly increasing node IDs."""
        ...

class OrbVocabulary:
    """Hierarchical Bag of Words vocabulary for ORB descriptors."""

    def __init__(self) -> None:
        """Initializes an empty ORB vocabulary."""
        ...
    @staticmethod
    def from_orb_slam3(path: str) -> "OrbVocabulary":
        """Loads an ORB-SLAM3 / DBoW2 text vocabulary file (``ORBvoc.txt``)."""
        ...
    @staticmethod
    def from_file(path: str) -> "OrbVocabulary":
        """Loads a fast binary vocabulary file previously saved with :meth:`save`."""
        ...
    @staticmethod
    def train(descriptors: npt.NDArray[np.uint8], max_depth: int = 6) -> "OrbVocabulary":
        """Trains a new hierarchical ORB vocabulary from an ``(N, 32)`` uint8 descriptor array."""
        ...
    def save(self, path: str) -> None:
        """Saves the vocabulary to a fast binary format (``.bow``)."""
        ...
    def transform(self, descriptors: npt.NDArray[np.uint8]) -> BoW:
        """Transforms ``(N, 32)`` ORB descriptors into a sparse BoW vector."""
        ...
    def transform_with_direct_index(
        self, descriptors: npt.NDArray[np.uint8], level: int = 2
    ) -> Tuple[BoW, DirectIndex]:
        """Transforms descriptors into a sparse BoW vector and a Direct Index at tree ``level``.

        Args:
            descriptors: (N, 32) uint8 ORB descriptors.
            level: Vocabulary tree depth level for direct indexing (default 2,
                matching ORB-SLAM3's ``levelsup=4`` on depth-6 trees). Level 0 is the root block.
        """
        ...
    def transform_one(
        self, descriptor: npt.NDArray[np.uint8]
    ) -> Tuple[int, float]:
        """Maps a single 32-byte descriptor to its ``(word_id, idf_weight)``."""
        ...
    @property
    def num_blocks(self) -> int:
        """Number of blocks in the vocabulary tree."""
        ...
    @property
    def branching_factor(self) -> int:
        """Branching factor of the vocabulary tree (k=10 for ORB-SLAM3)."""
        ...
    @property
    def descriptor_size(self) -> int:
        """Size of each descriptor in bytes (32 for 256-bit ORB)."""
        ...
    def __len__(self) -> int:
        """Number of blocks in the vocabulary tree."""
        ...
    def __repr__(self) -> str:
        """Returns string representation of the vocabulary."""
        ...
    def __getstate__(self) -> bytes:
        """Serializes vocabulary state into bytes for pickling."""
        ...
    def __setstate__(self, state: bytes) -> None:
        """Deserializes vocabulary state from bytes, validating binary format and tree bounds."""
        ...

def load_orb_slam3_vocabulary(path: str) -> OrbVocabulary:
    """Convenience function to load an ORB-SLAM3 / DBoW2 text vocabulary file."""
    ...

def load_vocabulary(path: str) -> OrbVocabulary:
    """Convenience function to load a binary vocabulary file."""
    ...
