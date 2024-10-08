use crate::{
    get_strides_from_shape, storage::TensorStorage, SafeTensorType, Tensor, TensorAllocator,
};

/// A view into a tensor.
pub struct TensorView<'a, T: SafeTensorType, const N: usize, A: TensorAllocator> {
    /// Reference to the storage held by the another tensor.
    pub storage: &'a TensorStorage<T, A>,

    /// The shape of the tensor.
    pub shape: [usize; N],

    /// The strides of the tensor.
    pub strides: [usize; N],
}

impl<'a, T: SafeTensorType, const N: usize, A: TensorAllocator + 'static> TensorView<'a, T, N, A> {
    /// Returns the data slice of the tensor.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        self.storage.as_slice()
    }

    /// Returns the data pointer of the tensor.
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.storage.as_ptr()
    }

    /// Returns the length of the tensor.
    #[inline]
    pub fn numel(&self) -> usize {
        self.storage.len()
    }

    /// Get the element at the given index.
    ///
    /// # Returns
    ///
    /// A reference to the element at the given index.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the index is within the bounds of the tensor.
    pub fn get_unchecked(&self, index: [usize; N]) -> &T {
        let offset = index
            .iter()
            .zip(self.strides.iter())
            .fold(0, |acc, (i, s)| acc + i * s);
        self.storage.get_unchecked(offset)
    }

    /// Convert the view an owned tensor with contiguous memory.
    ///
    /// # Returns
    ///
    /// A new `Tensor` instance with contiguous memory.
    pub fn as_contiguous(&self) -> Tensor<T, N, A> {
        let mut data = Vec::with_capacity(self.numel());
        let mut index = [0; N];

        loop {
            data.push(*self.get_unchecked(index));

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
            storage: TensorStorage::from_vec(data, self.storage.alloc().clone()),
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
    fn test_tensor_view_storage() -> Result<(), TensorAllocatorError> {
        let allocator = CpuAllocator;
        let storage = TensorStorage::<u8, _>::new(1024, allocator)?;
        let view = TensorView::<u8, 1, _> {
            storage: &storage,
            shape: [1024],
            strides: [1],
        };

        assert!(view.shape == [1024]);
        assert!(view.strides == [1]);
        assert_eq!(view.numel(), 1024);
        assert!(!view.as_ptr().is_null());

        Ok(())
    }

    #[test]
    fn test_tensor_view_from_vec() -> Result<(), TensorAllocatorError> {
        let vec = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let allocator = CpuAllocator;
        let storage = TensorStorage::<u8, _>::from_vec(vec, allocator);

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
