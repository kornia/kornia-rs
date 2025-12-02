use crate::{
    get_strides_from_shape, storage::TensorStorage, CpuAllocator, Tensor, TensorAllocator,
};

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
/// use kornia_tensor::{Tensor, CpuAllocator};
///
/// let data = vec![1, 2, 3, 4, 5, 6];
/// let tensor = Tensor::<i32, 1, _>::from_shape_vec([6], data, CpuAllocator).unwrap();
///
/// // Create a 2x3 view of the 1D tensor
/// let view = tensor.reshape([2, 3]).unwrap();
/// assert_eq!(view.shape, [2, 3]);
/// assert_eq!(*view.get_unchecked([0, 0]), 1);
/// assert_eq!(*view.get_unchecked([1, 2]), 6);
/// ```
///
/// Converting a view to a contiguous tensor:
///
/// ```rust
/// use kornia_tensor::{Tensor, CpuAllocator};
///
/// let data = vec![1, 2, 3, 4];
/// let tensor = Tensor::<i32, 2, _>::from_shape_vec([2, 2], data, CpuAllocator).unwrap();
///
/// // Permute creates a non-contiguous view
/// let view = tensor.permute_axes([1, 0]);
///
/// // Convert to an owned contiguous tensor
/// let contiguous = view.as_contiguous();
/// assert_eq!(contiguous.as_slice(), &[1, 3, 2, 4]);
/// ```
pub struct TensorView<'a, T, const N: usize, A: TensorAllocator> {
    /// Reference to the storage held by another tensor.
    pub storage: &'a TensorStorage<T, A>,

    /// The shape of the tensor view.
    pub shape: [usize; N],

    /// The strides for accessing elements in the view.
    pub strides: [usize; N],
}

impl<T, const N: usize, A: TensorAllocator> TensorView<'_, T, N, A> {
    /// Returns a slice view of the underlying storage.
    ///
    /// Note: This returns the entire underlying storage slice, not just the elements
    /// visible through this view's shape and strides. For element-wise access respecting
    /// the view's layout, use [`get_unchecked`](Self::get_unchecked).
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
    #[inline]
    pub fn numel(&self) -> usize {
        self.storage.len() / std::mem::size_of::<T>()
    }

    /// Gets the element at the given index without bounds checking.
    ///
    /// This method uses the view's strides to compute the offset into the storage,
    /// allowing efficient access to elements in non-contiguous views.
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
    /// The caller must ensure that the index is within the bounds defined by the
    /// view's shape. Out-of-bounds access results in undefined behavior.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kornia_tensor::{Tensor, CpuAllocator};
    ///
    /// let data = vec![1, 2, 3, 4, 5, 6];
    /// let tensor = Tensor::<i32, 1, _>::from_shape_vec([6], data, CpuAllocator).unwrap();
    /// let view = tensor.reshape([2, 3]).unwrap();
    ///
    /// assert_eq!(*view.get_unchecked([0, 0]), 1);
    /// assert_eq!(*view.get_unchecked([0, 1]), 2);
    /// assert_eq!(*view.get_unchecked([1, 2]), 6);
    /// ```
    pub fn get_unchecked(&self, index: [usize; N]) -> &T {
        let offset = index
            .iter()
            .zip(self.strides.iter())
            .fold(0, |acc, (i, s)| acc + i * s);
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
    /// data as this view, allocated using [`CpuAllocator`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use kornia_tensor::{Tensor, CpuAllocator};
    ///
    /// let data = vec![1, 2, 3, 4, 5, 6];
    /// let tensor = Tensor::<i32, 2, _>::from_shape_vec([2, 3], data, CpuAllocator).unwrap();
    ///
    /// // Transpose by permuting axes
    /// let transposed = tensor.permute_axes([1, 0]);
    ///
    /// // Convert to contiguous layout: [[1, 4], [2, 5], [3, 6]]
    /// let contiguous = transposed.as_contiguous();
    /// assert_eq!(contiguous.as_slice(), &[1, 4, 2, 5, 3, 6]);
    /// ```
    pub fn as_contiguous(&self) -> Tensor<T, N, CpuAllocator>
    where
        T: Clone,
    {
        let mut data = Vec::<T>::with_capacity(self.numel());
        let mut index = [0; N];

        loop {
            let val = self.get_unchecked(index);
            data.push(val.clone());

            // Increment index
            let mut i = N - 1;
            while i > 0 && index[i] == self.shape[i] - 1 {
                index[i] = 0;
                i -= 1;
            }
            if i == 0 && index[0] == self.shape[0] - 1 {
                break;
            }
            index[i] += 1;
        }

        let strides = get_strides_from_shape(self.shape);

        Tensor {
            storage: TensorStorage::from_vec(data, CpuAllocator),
            shape: self.shape,
            strides,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::allocator::{CpuAllocator, TensorAllocatorError};

    #[test]
    fn test_tensor_view_from_vec() -> Result<(), TensorAllocatorError> {
        let vec = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let storage = TensorStorage::from_vec(vec, CpuAllocator);

        let view = TensorView::<u8, 1, _> {
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

        // check get_unchecked
        assert_eq!(view.get_unchecked([0]), &1);
        assert_eq!(view.get_unchecked([1]), &2);
        assert_eq!(view.get_unchecked([2]), &3);
        assert_eq!(view.get_unchecked([3]), &4);
        assert_eq!(view.get_unchecked([4]), &5);
        assert_eq!(view.get_unchecked([5]), &6);
        assert_eq!(view.get_unchecked([6]), &7);
        assert_eq!(view.get_unchecked([7]), &8);

        Ok(())
    }
}
