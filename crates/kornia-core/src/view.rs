use crate::{storage::TensorStorage, SafeTensorType, TensorAllocator};

/// A view into a tensor.
pub struct TensorView<'a, T: SafeTensorType, const N: usize, A: TensorAllocator> {
    /// Reference to the storage held by the another tensor.
    pub storage: &'a TensorStorage<T, A>,

    /// The shape of the tensor.
    pub shape: [usize; N],

    /// The strides of the tensor.
    pub strides: [usize; N],
}

impl<'a, T: SafeTensorType, const N: usize, A: TensorAllocator> TensorView<'a, T, N, A> {
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
        &self.as_slice()[offset]
    }
}
