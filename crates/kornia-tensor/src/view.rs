use crate::{
    get_strides_from_shape,
    storage::TensorStorage,
    tensor::{checked_numel, checked_offset, validate_layout},
    Tensor, TensorError,
};
use rayon::prelude::*;

/// A non-owning view into tensor data.
///
/// `TensorView` provides a lightweight, non-owning reference to tensor data with its own
/// shape and stride information. Views enable zero-copy operations like reshaping and
/// dimension permutation without duplicating the underlying data.
///
/// # Lifetime
///
/// The view borrows the storage for its lifetime `'a`, ensuring the underlying data
/// remains valid while the view exists.
///
/// # Memory Layout
///
/// Views can have different strides than the original tensor, allowing for operations
/// like transposition and dimension permutation without copying data. However, some
/// operations may require converting to a contiguous layout.
///
/// # Examples
///
/// Creating a view through reshaping:
///
/// ```rust
/// use kornia_tensor::Tensor;
///
/// let data = vec![1, 2, 3, 4, 5, 6];
/// let tensor = Tensor::<i32, 1>::from_shape_vec([6], data).unwrap();
///
/// // Create a 2x3 view of the 1D tensor
/// let view = tensor.reshape([2, 3]).unwrap();
/// assert_eq!(view.shape, [2, 3]);
/// assert_eq!(view.get([0, 0]), Some(&1));
/// assert_eq!(view.get([1, 2]), Some(&6));
/// ```
///
/// Converting a view to a contiguous tensor:
///
/// ```rust
/// use kornia_tensor::Tensor;
///
/// let data = vec![1, 2, 3, 4];
/// let tensor = Tensor::<i32, 2>::from_shape_vec([2, 2], data).unwrap();
///
/// // Permute creates a non-contiguous view
/// let view = tensor.permute_axes([1, 0]).unwrap();
///
/// // Convert to an owned contiguous tensor
/// let contiguous = view.as_contiguous().unwrap();
/// assert_eq!(contiguous.as_slice(), &[1, 3, 2, 4]);
/// ```
pub struct TensorView<'a, T, const N: usize> {
    /// Reference to the storage held by another tensor.
    pub storage: &'a TensorStorage<T>,

    /// The shape of the tensor view.
    pub shape: [usize; N],

    /// The strides for accessing elements in the view.
    pub strides: [usize; N],
}

impl<T: Send, const N: usize> TensorView<'_, T, N> {
    /// Returns a slice view of the underlying storage.
    ///
    /// Note: This returns the entire underlying storage slice, not just the elements
    /// visible through this view's shape and strides. For element-wise access respecting
    /// the view's layout, use [`get`](Self::get).
    ///
    /// # Returns
    ///
    /// A slice containing all elements in the underlying storage.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        self.storage.as_slice()
    }

    /// Returns a raw pointer to the underlying storage.
    ///
    /// # Returns
    ///
    /// A const pointer to the first element of the storage.
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.storage.as_ptr()
    }

    /// Returns the total number of elements in the view.
    ///
    /// This is computed from the view's shape, not the underlying storage size.
    ///
    /// # Returns
    ///
    /// The total number of elements (product of all dimensions in the shape).
    /// Saturates at `usize::MAX` for a malformed view whose (public) `shape`
    /// overflows; use [`checked_numel`] to detect that case.
    #[inline]
    pub fn numel(&self) -> usize {
        checked_numel(&self.shape).unwrap_or(usize::MAX)
    }

    /// Gets the element at the given index, checking bounds.
    ///
    /// # Arguments
    ///
    /// * `index` - The multi-dimensional index to access
    ///
    /// # Returns
    ///
    /// `Some(&T)` if `index[i] < shape[i]` for every dimension and the resulting
    /// offset lies inside the storage, `None` otherwise.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kornia_tensor::Tensor;
    ///
    /// let data = vec![1, 2, 3, 4, 5, 6];
    /// let tensor = Tensor::<i32, 1>::from_shape_vec([6], data).unwrap();
    /// let view = tensor.reshape([2, 3]).unwrap();
    ///
    /// assert_eq!(view.get([1, 2]), Some(&6));
    /// assert_eq!(view.get([2, 0]), None);
    /// ```
    pub fn get(&self, index: [usize; N]) -> Option<&T> {
        let offset = checked_offset(&index, &self.shape, &self.strides)?;
        self.storage.as_slice().get(offset)
    }

    /// Gets the element at the given index without bounds checking.
    ///
    /// This method uses the view's strides to compute the offset into the storage,
    /// allowing efficient access to elements in non-contiguous views. For a checked
    /// alternative use [`get`](Self::get).
    ///
    /// # Arguments
    ///
    /// * `index` - The multi-dimensional index to access
    ///
    /// # Returns
    ///
    /// A reference to the element at the given index.
    ///
    /// # Safety
    ///
    /// The offset `sum(index[i] * strides[i])` must be smaller than the number of
    /// elements in the underlying storage. This holds if `index[i] < shape[i]` for
    /// every dimension and the view layout is in bounds (the `shape`/`strides` fields
    /// are public, so the caller is responsible for both). Out-of-bounds access
    /// results in undefined behavior.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kornia_tensor::Tensor;
    ///
    /// let data = vec![1, 2, 3, 4, 5, 6];
    /// let tensor = Tensor::<i32, 1>::from_shape_vec([6], data).unwrap();
    /// let view = tensor.reshape([2, 3]).unwrap();
    ///
    /// // SAFETY: the indices are within the [2, 3] shape of a freshly reshaped view.
    /// unsafe {
    ///     assert_eq!(*view.get_unchecked([0, 0]), 1);
    ///     assert_eq!(*view.get_unchecked([0, 1]), 2);
    ///     assert_eq!(*view.get_unchecked([1, 2]), 6);
    /// }
    /// ```
    #[inline]
    pub unsafe fn get_unchecked(&self, index: [usize; N]) -> &T {
        let offset = index
            .iter()
            .zip(self.strides.iter())
            .fold(0usize, |acc, (i, s)| acc.wrapping_add(i.wrapping_mul(*s)));
        // SAFETY: the caller guarantees `offset` is within the storage (see `# Safety`).
        unsafe { self.storage.as_slice().get_unchecked(offset) }
    }

    /// Converts the view to an owned tensor with contiguous memory layout.
    ///
    /// This method is essential when working with non-contiguous views (e.g., after
    /// permutation or transposition). It iterates through all elements according to
    /// the view's shape and strides, creating a new tensor with standard row-major layout.
    ///
    /// # Returns
    ///
    /// A new [`Tensor`] instance with contiguous memory containing the same logical
    /// data as this view, using the allocator handle of the source storage.
    ///
    /// # Errors
    ///
    /// * [`TensorError::ShapeOverflow`] if the product of the view's shape overflows.
    /// * [`TensorError::InvalidLayout`] if the view's shape/strides address memory
    ///   outside of its storage.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kornia_tensor::Tensor;
    ///
    /// let data = vec![1, 2, 3, 4, 5, 6];
    /// let tensor = Tensor::<i32, 2>::from_shape_vec([2, 3], data).unwrap();
    ///
    /// // Transpose by permuting axes
    /// let transposed = tensor.permute_axes([1, 0]).unwrap();
    ///
    /// // Convert to contiguous layout: [[1, 4], [2, 5], [3, 6]]
    /// let contiguous = transposed.as_contiguous().unwrap();
    /// assert_eq!(contiguous.as_slice(), &[1, 4, 2, 5, 3, 6]);
    /// ```
    pub fn as_contiguous(&self) -> Result<Tensor<T, N>, TensorError>
    where
        T: Clone + Sync,
    {
        // Iterate the view's *logical* shape, not the storage size: a view may address
        // fewer (or, if malformed, more) elements than the storage holds.
        let numel = checked_numel(&self.shape)?;
        let storage_len = self.storage.num_elements();
        validate_layout(&self.shape, &self.strides, storage_len)?;

        let data: Vec<T> = (0..numel)
            .into_par_iter()
            .map(|flat_idx| {
                let index = self.flat_index_to_multi_index(flat_idx);
                // SAFETY: `flat_idx < numel = product(shape)`, so every `index[i] <
                // shape[i]`; `validate_layout` proved that every such index maps to an
                // offset `< storage_len` without overflow.
                unsafe { self.get_unchecked(index) }.clone()
            })
            .collect();

        let strides = get_strides_from_shape(self.shape);
        Ok(Tensor {
            storage: TensorStorage::from_vec(data, self.storage.alloc().clone()),
            shape: self.shape,
            strides,
        })
    }

    /// Convert 1D to N-dimensional index for retrieving elements.
    pub(super) fn flat_index_to_multi_index(&self, mut flat_idx: usize) -> [usize; N] {
        let mut index = [0; N];
        for i in (0..N).rev() {
            index[i] = flat_idx % self.shape[i];
            flat_idx /= self.shape[i];
        }
        index
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::allocator::{host_alloc, TensorAllocatorError};

    #[test]
    fn test_tensor_view_from_vec() -> Result<(), TensorAllocatorError> {
        let vec = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let storage = TensorStorage::from_vec(vec, host_alloc());

        let view = TensorView::<u8, 1> {
            storage: &storage,
            shape: [8],
            strides: [1],
        };

        assert_eq!(view.numel(), 8);
        assert!(!view.as_ptr().is_null());

        // check slice
        let data = view.as_slice();
        assert_eq!(data.len(), 8);
        assert_eq!(data[0], 1);
        assert_eq!(data[1], 2);
        assert_eq!(data[2], 3);
        assert_eq!(data[3], 4);
        assert_eq!(data[4], 5);
        assert_eq!(data[5], 6);
        assert_eq!(data[6], 7);
        assert_eq!(data[7], 8);

        // check get / get_unchecked
        for i in 0..8 {
            assert_eq!(view.get([i]), Some(&(i as u8 + 1)));
            // SAFETY: `i < 8 == shape[0]` and the storage holds 8 contiguous elements.
            assert_eq!(unsafe { view.get_unchecked([i]) }, &(i as u8 + 1));
        }
        assert_eq!(view.get([8]), None);

        Ok(())
    }

    #[test]
    fn test_view_get_out_of_bounds_layout() {
        // A malformed view whose strides reach past the storage: `get` must refuse
        // instead of reading out of bounds.
        let storage = TensorStorage::from_vec(vec![1u8, 2, 3, 4], host_alloc());
        let view = TensorView::<u8, 2> {
            storage: &storage,
            shape: [2, 2],
            strides: [4, 1],
        };
        assert_eq!(view.get([0, 1]), Some(&2));
        assert_eq!(view.get([1, 0]), None);
        assert!(view.as_contiguous().is_err());
    }

    #[test]
    fn test_as_contiguous_uses_logical_shape() -> Result<(), crate::TensorError> {
        // Regression: `as_contiguous` iterated `storage.numel()` instead of the view's
        // shape product. A sub-view with fewer elements than the storage must produce
        // exactly `product(shape)` elements.
        let storage = TensorStorage::from_vec((0u32..12).collect(), host_alloc());
        let view = TensorView::<u32, 2> {
            storage: &storage,
            shape: [2, 3],
            strides: [4, 1],
        };
        let t = view.as_contiguous()?;
        assert_eq!(t.shape, [2, 3]);
        assert_eq!(t.as_slice(), &[0, 1, 2, 4, 5, 6]);
        Ok(())
    }

    #[test]
    fn test_flat_index_to_multi_index_2d() {
        let vec: Vec<u8> = (0..12).collect();
        let storage = TensorStorage::from_vec(vec, host_alloc());

        let view = TensorView::<u8, 2> {
            storage: &storage,
            shape: [3, 4],
            strides: [4, 1],
        };

        // Test each flat index
        assert_eq!(view.flat_index_to_multi_index(0), [0, 0]);
        assert_eq!(view.flat_index_to_multi_index(1), [0, 1]);
        assert_eq!(view.flat_index_to_multi_index(2), [0, 2]);
        assert_eq!(view.flat_index_to_multi_index(3), [0, 3]);
        assert_eq!(view.flat_index_to_multi_index(4), [1, 0]);
        assert_eq!(view.flat_index_to_multi_index(5), [1, 1]);
        assert_eq!(view.flat_index_to_multi_index(9), [2, 1]);
        assert_eq!(view.flat_index_to_multi_index(11), [2, 3]);
    }
}
