use thiserror::Error;

use super::{
    allocator::{CpuAllocator, TensorAllocator, TensorAllocatorError},
    storage::TensorStorage,
    view::TensorView,
};

/// An error type for tensor operations.
#[derive(Error, Debug, PartialEq)]
pub enum TensorError {
    /// Error when the cast operation fails.
    #[error("Failed to cast data")]
    CastError,

    /// The number of elements in the data does not match the shape of the tensor.
    #[error("The number of elements in the data does not match the shape of the tensor: {0}")]
    InvalidShape(usize),

    /// Index out of bounds.
    #[error("Index out of bounds. The index {0} is out of bounds.")]
    IndexOutOfBounds(usize),

    /// Error with the tensor storage.
    #[error("Error with the tensor storage: {0}")]
    StorageError(#[from] TensorAllocatorError),

    /// Dimension mismatch for operations requiring compatible shapes.
    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),

    /// Unsupported operation for the given data type or tensor configuration.
    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),
}

/// Compute the strides from the shape of a tensor.
///
/// # Arguments
///
/// * `shape` - The shape of the tensor.
///
/// # Returns
///
/// * `strides` - The strides of the tensor.
pub fn get_strides_from_shape<const N: usize>(shape: [usize; N]) -> [usize; N] {
    let mut strides: [usize; N] = [0; N];
    let mut stride = 1;
    for i in (0..shape.len()).rev() {
        strides[i] = stride;
        stride *= shape[i];
    }
    strides
}

/// A data structure to represent a multi-dimensional tensor.
///
/// NOTE: Internally, the data is stored as an `arrow::ScalarBuffer` which represents a contiguous memory
/// region that can be shared with other buffers and across thread boundaries.
///
/// # Attributes
///
/// * `storage` - The storage of the tensor.
/// * `shape` - The shape of the tensor.
/// * `strides` - The strides of the tensor data in memory.
///
/// # Example
///
/// ```
/// use kornia_tensor::{Tensor, CpuAllocator};
///
/// let data: Vec<u8> = vec![1, 2, 3, 4];
/// let t = Tensor::<u8, 2, CpuAllocator>::from_shape_vec([2, 2], data, CpuAllocator).unwrap();
/// assert_eq!(t.shape, [2, 2]);
/// ```
pub struct Tensor<T, const N: usize, A: TensorAllocator> {
    /// The storage of the tensor.
    pub storage: TensorStorage<T, A>,
    /// The shape of the tensor.
    pub shape: [usize; N],
    /// The strides of the tensor data in memory.
    pub strides: [usize; N],
}

impl<T, const N: usize, A: TensorAllocator> Tensor<T, N, A>
where
    A: 'static,
{
    /// Get the data of the tensor as a slice.
    ///
    /// # Returns
    ///
    /// A slice containing the data of the tensor.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        self.storage.as_slice()
    }

    /// Get the data of the tensor as a mutable slice.
    ///
    /// # Returns
    ///
    /// A mutable slice containing the data of the tensor.
    #[inline]
    pub fn as_slice_mut(&mut self) -> &mut [T] {
        self.storage.as_mut_slice()
    }

    /// Get the data of the tensor as a pointer.
    ///
    /// # Returns
    ///
    /// A pointer to the data of the tensor.
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.storage.as_ptr()
    }

    /// Get the data of the tensor as a mutable pointer.
    ///
    /// # Returns
    ///
    /// A mutable pointer to the data of the tensor.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.storage.as_mut_ptr()
    }

    /// Consumes the tensor and returns the underlying vector.
    ///
    /// This method destroys the tensor and returns ownership of the underlying data.
    /// The returned vector will have a length equal to the total number of elements in the tensor.
    ///
    #[inline]
    pub fn into_vec(self) -> Vec<T> {
        self.storage.into_vec()
    }

    /// Creates a new `Tensor` with the given shape and data.
    ///
    /// # Arguments
    ///
    /// * `shape` - An array containing the shape of the tensor.
    /// * `data` - A vector containing the data of the tensor.
    /// * `alloc` - The allocator to use.
    ///
    /// # Returns
    ///
    /// A new `Tensor` instance.
    ///
    /// # Errors
    ///
    /// If the number of elements in the data does not match the shape of the tensor, an error is returned.
    ///
    /// # Example
    ///
    /// ```
    /// use kornia_tensor::{Tensor, CpuAllocator};
    ///
    /// let data: Vec<u8> = vec![1, 2, 3, 4];
    /// let t = Tensor::<u8, 2, CpuAllocator>::from_shape_vec([2, 2], data, CpuAllocator).unwrap();
    /// assert_eq!(t.shape, [2, 2]);
    /// ```
    pub fn from_shape_vec(shape: [usize; N], data: Vec<T>, alloc: A) -> Result<Self, TensorError> {
        let numel = shape.iter().product::<usize>();
        if numel != data.len() {
            return Err(TensorError::InvalidShape(numel));
        }
        let storage = TensorStorage::from_vec(data, alloc);
        let strides = get_strides_from_shape(shape);
        Ok(Self {
            storage,
            shape,
            strides,
        })
    }

    /// Creates a new `Tensor` with the given shape and slice of data.
    ///
    /// # Arguments
    ///
    /// * `shape` - An array containing the shape of the tensor.
    /// * `data` - A slice containing the data of the tensor.
    /// * `alloc` - The allocator to use.
    ///
    /// # Returns
    ///
    /// A new `Tensor` instance.
    ///
    /// # Errors
    ///
    /// If the number of elements in the data does not match the shape of the tensor, an error is returned.
    pub fn from_shape_slice(shape: [usize; N], data: &[T], alloc: A) -> Result<Self, TensorError>
    where
        T: Clone,
    {
        let numel = shape.iter().product::<usize>();
        if numel != data.len() {
            return Err(TensorError::InvalidShape(numel));
        }
        let storage = TensorStorage::from_vec(data.to_vec(), alloc);
        let strides = get_strides_from_shape(shape);
        Ok(Self {
            storage,
            shape,
            strides,
        })
    }

    /// Creates a new `Tensor` with the given shape and raw parts.
    ///
    /// # Arguments
    ///
    /// * `shape` - An array containing the shape of the tensor.
    /// * `data` - A pointer to the data of the tensor.
    /// * `len` - The length of the data.
    /// * `alloc` - The allocator to use.
    ///
    /// # Safety
    ///
    /// The pointer must be non-null and the length must be valid.
    pub unsafe fn from_raw_parts(
        shape: [usize; N],
        data: *const T,
        len: usize,
        alloc: A,
    ) -> Result<Self, TensorError>
    where
        T: Clone,
    {
        let storage = TensorStorage::from_raw_parts(data, len, alloc);
        let strides = get_strides_from_shape(shape);
        Ok(Self {
            storage,
            shape,
            strides,
        })
    }

    /// Creates a new `Tensor` with the given shape and a default value.
    /// Creates a new `Tensor` with the given shape and a default value.
    ///
    /// # Arguments
    ///
    /// * `shape` - An array containing the shape of the tensor.
    /// * `value` - The default value to fill the tensor with.
    ///
    /// # Returns
    ///
    /// A new `Tensor` instance.
    ///
    /// # Example
    ///
    /// ```
    /// use kornia_tensor::{Tensor, CpuAllocator};
    ///
    /// let t = Tensor::<u8, 1, CpuAllocator>::from_shape_val([4], 0, CpuAllocator);
    /// assert_eq!(t.as_slice(), vec![0, 0, 0, 0]);
    ///
    /// let t = Tensor::<u8, 2, CpuAllocator>::from_shape_val([2, 2], 1, CpuAllocator);
    /// assert_eq!(t.as_slice(), vec![1, 1, 1, 1]);
    ///
    /// let t = Tensor::<u8, 3, CpuAllocator>::from_shape_val([2, 1, 3], 2, CpuAllocator);
    /// assert_eq!(t.as_slice(), vec![2, 2, 2, 2, 2, 2]);
    /// ```
    pub fn from_shape_val(shape: [usize; N], value: T, alloc: A) -> Self
    where
        T: Clone,
    {
        let numel = shape.iter().product::<usize>();
        let data = vec![value; numel];
        let storage = TensorStorage::from_vec(data, alloc);
        let strides = get_strides_from_shape(shape);
        Self {
            storage,
            shape,
            strides,
        }
    }

    /// Create a new `Tensor` with the given shape and a function to generate the data.
    ///
    /// The function `f` is called with the index of the element to generate.
    ///
    /// # Arguments
    ///
    /// * `shape` - An array containing the shape of the tensor.
    /// * `f` - The function to generate the data.
    ///
    /// # Returns
    ///
    /// A new `Tensor` instance.
    ///
    /// # Example
    ///
    /// ```
    /// use kornia_tensor::{Tensor, CpuAllocator};
    ///
    /// let t = Tensor::<u8, 1, CpuAllocator>::from_shape_fn([4], CpuAllocator, |[i]| i as u8);
    /// assert_eq!(t.as_slice(), vec![0, 1, 2, 3]);
    ///
    /// let t = Tensor::<u8, 2, CpuAllocator>::from_shape_fn([2, 2], CpuAllocator, |[i, j]| (i * 2 + j) as u8);
    /// assert_eq!(t.as_slice(), vec![0, 1, 2, 3]);
    /// ```
    pub fn from_shape_fn<F>(shape: [usize; N], alloc: A, f: F) -> Self
    where
        F: Fn([usize; N]) -> T,
    {
        let numel = shape.iter().product::<usize>();
        let data: Vec<T> = (0..numel)
            .map(|i| {
                let mut index = [0; N];
                let mut j = i;
                for k in (0..N).rev() {
                    index[k] = j % shape[k];
                    j /= shape[k];
                }
                f(index)
            })
            .collect();
        let storage = TensorStorage::from_vec(data, alloc);
        let strides = get_strides_from_shape(shape);
        Self {
            storage,
            shape,
            strides,
        }
    }

    /// Returns the number of elements in the tensor.
    ///
    /// # Returns
    ///
    /// The number of elements in the tensor.
    #[inline]
    pub fn numel(&self) -> usize {
        self.storage.len() / std::mem::size_of::<T>()
    }

    /// Get the offset of the element at the given index.
    ///
    /// # Arguments
    ///
    /// * `index` - The list of indices to get the element from.
    ///
    /// # Returns
    ///
    /// The offset of the element at the given index.
    pub fn get_iter_offset(&self, index: [usize; N]) -> Option<usize> {
        let mut offset = 0;
        for ((&idx, dim_size), stride) in index.iter().zip(self.shape).zip(self.strides) {
            if idx >= dim_size {
                return None;
            }
            offset += idx * stride;
        }
        Some(offset)
    }

    /// Get the offset of the element at the given index without checking dim sizes.
    ///
    /// # Arguments
    ///
    /// * `index` - The list of indices to get the element from.
    ///
    /// # Returns
    ///
    /// The offset of the element at the given index.
    pub fn get_iter_offset_unchecked(&self, index: [usize; N]) -> usize {
        let mut offset = 0;
        for (&idx, stride) in index.iter().zip(self.strides) {
            offset += idx * stride;
        }
        offset
    }

    /// Get the index of the element at the given offset without checking dim sizes. The reverse of `Self::get_iter_offset_unchecked`.
    ///
    /// # Arguments
    ///
    /// * `offset` - The offset of the element at the given index.
    ///
    /// # Returns
    ///
    /// The array of indices to get the element from.
    pub fn get_index_unchecked(&self, offset: usize) -> [usize; N] {
        let mut idx = [0; N];
        let mut rem = offset;
        for (dim_i, s) in self.strides.iter().enumerate() {
            idx[dim_i] = rem / s;
            rem = offset % s;
        }

        idx
    }

    /// Get the index of the element at the given offset. The reverse of `Self::get_iter_offset`.
    ///
    /// # Arguments
    ///
    /// * `offset` - The offset of the element at the given index.
    ///
    /// # Returns
    ///
    /// The array of indices to get the element from.
    ///
    /// # Errors
    ///
    /// If the offset is out of bounds (>= numel), an error is returned.
    pub fn get_index(&self, offset: usize) -> Result<[usize; N], TensorError> {
        if offset >= self.numel() {
            return Err(TensorError::IndexOutOfBounds(offset));
        }
        let idx = self.get_index_unchecked(offset);

        Ok(idx)
    }

    /// Get the element at the given index without checking if the index is out of bounds.
    ///
    /// # Arguments
    ///
    /// * `index` - The list of indices to get the element from.
    ///
    /// # Returns
    ///
    /// A reference to the element at the given index.
    ///
    /// # Example
    ///
    /// ```
    /// use kornia_tensor::{Tensor, CpuAllocator};
    ///
    /// let data: Vec<u8> = vec![1, 2, 3, 4];
    ///
    /// let t = Tensor::<u8, 2, CpuAllocator>::from_shape_vec([2, 2], data, CpuAllocator).unwrap();
    /// assert_eq!(*t.get_unchecked([0, 0]), 1);
    /// assert_eq!(*t.get_unchecked([0, 1]), 2);
    /// assert_eq!(*t.get_unchecked([1, 0]), 3);
    /// assert_eq!(*t.get_unchecked([1, 1]), 4);
    /// ```
    pub fn get_unchecked(&self, index: [usize; N]) -> &T {
        let offset = self.get_iter_offset_unchecked(index);
        unsafe { self.storage.as_slice().get_unchecked(offset) }
    }

    /// Get the element at the given index, checking if the index is out of bounds.
    ///
    /// # Arguments
    ///
    /// * `index` - The list of indices to get the element from.
    ///
    /// # Returns
    ///
    /// A reference to the element at the given index.
    ///
    /// # Errors
    ///
    /// If the index is out of bounds, an error is returned.
    ///
    /// # Example
    ///
    /// ```
    /// use kornia_tensor::{Tensor, CpuAllocator};
    ///
    /// let data: Vec<u8> = vec![1, 2, 3, 4];
    ///
    /// let t = Tensor::<u8, 2, CpuAllocator>::from_shape_vec([2, 2], data, CpuAllocator).unwrap();
    ///
    /// assert_eq!(t.get([0, 0]), Some(&1));
    /// assert_eq!(t.get([0, 1]), Some(&2));
    /// assert_eq!(t.get([1, 0]), Some(&3));
    /// assert_eq!(t.get([1, 1]), Some(&4));
    ///
    /// assert!(t.get([2, 0]).is_none());
    /// ```
    pub fn get(&self, index: [usize; N]) -> Option<&T> {
        self.get_iter_offset(index)
            .and_then(|i| self.storage.as_slice().get(i))
    }

    /// Reshape the tensor to a new shape.
    ///
    /// # Arguments
    ///
    /// * `shape` - The new shape of the tensor.
    ///
    /// # Returns
    ///
    /// A new `TensorView` instance.
    ///
    /// # Errors
    ///
    /// If the number of elements in the new shape does not match the number of elements in the tensor, an error is returned.
    ///
    /// # Example
    ///
    /// ```
    /// use kornia_tensor::{Tensor, CpuAllocator};
    ///
    /// let data: Vec<u8> = vec![1, 2, 3, 4];
    ///
    /// let t = Tensor::<u8, 1, CpuAllocator>::from_shape_vec([4], data, CpuAllocator).unwrap();
    /// let t2 = t.reshape([2, 2]).unwrap();
    /// assert_eq!(t2.shape, [2, 2]);
    /// assert_eq!(t2.as_slice(), vec![1, 2, 3, 4]);
    /// assert_eq!(t2.strides, [2, 1]);
    /// assert_eq!(t2.numel(), 4);
    /// ```
    pub fn reshape<const M: usize>(
        &self,
        shape: [usize; M],
    ) -> Result<TensorView<T, M, A>, TensorError> {
        let numel = shape.iter().product::<usize>();
        if numel != self.storage.len() {
            return Err(TensorError::DimensionMismatch(format!(
                "Cannot reshape tensor of shape {:?} with {} elements to shape {:?} with {} elements",
                self.shape, self.storage.len(), shape, numel
            )));
        }

        let strides = get_strides_from_shape(shape);

        Ok(TensorView {
            storage: &self.storage,
            shape,
            strides,
        })
    }

    /// Permute the dimensions of the tensor.
    ///
    /// The permutation is given as an array of indices, where the value at each index is the new index of the dimension.
    /// The data is not moved, only the order of the dimensions is changed.
    ///
    /// # Arguments
    ///
    /// * `axes` - The new order of the dimensions.
    ///
    /// # Returns
    ///
    /// A view of the tensor with the dimensions permuted.
    pub fn permute_axes(&self, axes: [usize; N]) -> TensorView<T, N, A> {
        let mut new_shape = [0; N];
        let mut new_strides = [0; N];
        for (i, &axis) in axes.iter().enumerate() {
            new_shape[i] = self.shape[axis];
            new_strides[i] = self.strides[axis];
        }

        TensorView {
            storage: &self.storage,
            shape: new_shape,
            strides: new_strides,
        }
    }

    /// Return a view of the tensor.
    ///
    /// The view is a reference to the tensor storage with a different shape and strides.
    ///
    /// # Returns
    ///
    /// A `TensorView` instance.
    pub fn view(&self) -> TensorView<T, N, A> {
        TensorView {
            storage: &self.storage,
            shape: self.shape,
            strides: self.strides,
        }
    }

    /// Create a new tensor with all elements set to zero.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the tensor.
    /// * `alloc` - The allocator to use.
    ///
    /// # Returns
    pub fn zeros(shape: [usize; N], alloc: A) -> Tensor<T, N, A>
    where
        T: Clone + num_traits::Zero,
    {
        // TODO: add allocator parameter
        Self::from_shape_val(shape, T::zero(), alloc)
    }

    /// Apply a function to each element of the tensor.
    ///
    /// # Arguments
    ///
    /// * `f` - The function to apply to each element.
    ///
    /// # Returns
    ///
    /// A new `Tensor` instance.
    ///
    /// # Example
    ///
    /// ```
    /// use kornia_tensor::{Tensor, CpuAllocator};
    ///
    /// let data: Vec<u8> = vec![1, 2, 3, 4];
    /// let t = Tensor::<u8, 1, CpuAllocator>::from_shape_vec([4], data, CpuAllocator).unwrap();
    ///
    /// let t2 = t.map(|x| *x + 1);
    /// assert_eq!(t2.as_slice(), vec![2, 3, 4, 5]);
    /// ```
    pub fn map<U, F>(&self, f: F) -> Tensor<U, N, A>
    where
        F: Fn(&T) -> U,
    {
        let data: Vec<U> = self.as_slice().iter().map(f).collect();
        let storage = TensorStorage::from_vec(data, self.storage.alloc().clone());

        Tensor {
            storage,
            shape: self.shape,
            strides: self.strides,
        }
    }

    /// Cast the tensor to a new type.
    ///
    /// # Returns
    ///
    /// A new `Tensor` instance.
    ///
    /// # Example
    ///
    /// ```
    /// use kornia_tensor::{Tensor, CpuAllocator};
    ///
    /// let data: Vec<u8> = vec![1, 2, 3, 4];
    /// let t = Tensor::<u8, 1, CpuAllocator>::from_shape_vec([4], data, CpuAllocator).unwrap();
    ///
    /// let t2 = t.cast::<f32>();
    /// assert_eq!(t2.as_slice(), vec![1.0, 2.0, 3.0, 4.0]);
    /// ```
    pub fn cast<U>(&self) -> Tensor<U, N, CpuAllocator>
    where
        U: From<T>,
        T: Clone,
    {
        let mut data: Vec<U> = Vec::with_capacity(self.storage.len());
        self.as_slice().iter().for_each(|x| {
            data.push(U::from(x.clone()));
        });
        let storage = TensorStorage::from_vec(data, CpuAllocator);
        Tensor {
            storage,
            shape: self.shape,
            strides: self.strides,
        }
    }

    /// Perform an element-wise operation on two tensors.
    ///
    /// # Arguments
    ///
    /// * `other` - The other tensor to perform the operation with.
    /// * `op` - The operation to perform.
    ///
    /// # Returns
    ///
    /// A new `Tensor` instance.
    ///
    /// # Example
    ///
    /// ```
    /// use kornia_tensor::{Tensor, CpuAllocator};
    ///
    /// let data1: Vec<u8> = vec![1, 2, 3, 4];
    /// let t1 = Tensor::<u8, 1, CpuAllocator>::from_shape_vec([4], data1, CpuAllocator).unwrap();
    ///
    /// let data2: Vec<u8> = vec![1, 2, 3, 4];
    /// let t2 = Tensor::<u8, 1, CpuAllocator>::from_shape_vec([4], data2, CpuAllocator).unwrap();
    ///
    /// let t3 = t1.element_wise_op(&t2, |a, b| *a + *b).unwrap();
    /// assert_eq!(t3.as_slice(), vec![2, 4, 6, 8]);
    ///
    /// let t4 = t1.element_wise_op(&t2, |a, b| *a - *b).unwrap();
    /// assert_eq!(t4.as_slice(), vec![0, 0, 0, 0]);
    ///
    /// let t5 = t1.element_wise_op(&t2, |a, b| *a * *b).unwrap();
    /// assert_eq!(t5.as_slice(), vec![1, 4, 9, 16]);
    ///
    /// let t6 = t1.element_wise_op(&t2, |a, b| *a / *b).unwrap();
    /// assert_eq!(t6.as_slice(), vec![1, 1, 1, 1]);
    /// ```
    pub fn element_wise_op<F>(
        &self,
        other: &Tensor<T, N, CpuAllocator>,
        op: F,
    ) -> Result<Tensor<T, N, CpuAllocator>, TensorError>
    where
        F: Fn(&T, &T) -> T,
    {
        if self.shape != other.shape {
            return Err(TensorError::DimensionMismatch(format!(
                "Shapes {:?} and {:?} are not compatible for element-wise operations",
                self.shape, other.shape
            )));
        }

        let data = self
            .as_slice()
            .iter()
            .zip(other.as_slice().iter())
            .map(|(a, b)| op(a, b))
            .collect();

        let storage = TensorStorage::from_vec(data, CpuAllocator);

        Ok(Tensor {
            storage,
            shape: self.shape,
            strides: self.strides,
        })
    }
}

impl<T, const N: usize, A> Clone for Tensor<T, N, A>
where
    T: Clone,
    A: TensorAllocator + Clone + 'static,
{
    fn clone(&self) -> Self {
        Self {
            storage: self.storage.clone(),
            shape: self.shape,
            strides: self.strides,
        }
    }
}

impl<T, const N: usize, A> std::fmt::Display for Tensor<T, N, A>
where
    T: std::fmt::Display + std::fmt::LowerExp,
    A: TensorAllocator + 'static,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let width = self
            .storage
            .as_slice()
            .iter()
            .map(|v| format!("{v:.4}").len())
            .max()
            .unwrap();

        let scientific = width > 8;

        let should_mask: [bool; N] = self.shape.map(|s| s > 8);
        let mut skip_until = 0;

        for (i, v) in self.storage.as_slice().iter().enumerate() {
            if i < skip_until {
                continue;
            }
            let mut value = String::new();
            let mut prefix = String::new();
            let mut suffix = String::new();
            let mut separator = ",".to_string();
            let mut last_size = 1;
            for (dim, (&size, maskable)) in self.shape.iter().zip(should_mask).enumerate().rev() {
                let prod = size * last_size;
                if i % prod == (3 * last_size) && maskable {
                    let pad = if dim == (N - 1) { 0 } else { dim + 1 };
                    value = format!("{}...", " ".repeat(pad));
                    skip_until = i + (size - 4) * last_size;
                    prefix = "".to_string();
                    if dim != (N - 1) {
                        separator = "\n".repeat(N - 1 - dim);
                    }
                    break;
                } else if i % prod == 0 {
                    prefix.push('[');
                } else if (i + 1) % prod == 0 {
                    suffix.push(']');
                    separator.push('\n');
                    if dim == 0 {
                        separator = "".to_string();
                    }
                } else {
                    break;
                }
                last_size = prod;
            }
            if !prefix.is_empty() {
                prefix = format!("{prefix:>N$}");
            }

            if value.is_empty() {
                value = if scientific {
                    let num = format!("{v:.4e}");
                    let (before, after) = num.split_once('e').unwrap();
                    let after = if let Some(stripped) = after.strip_prefix('-') {
                        format!("-{:0>2}", &stripped)
                    } else {
                        format!("+{:0>2}", &after)
                    };
                    format!("{before}e{after}")
                } else {
                    let rounded = format!("{v:.4}");
                    format!("{rounded:>width$}")
                }
            };
            write!(f, "{prefix}{value}{suffix}{separator}",)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::allocator::CpuAllocator;
    use crate::tensor::{Tensor, TensorError};

    #[test]
    fn constructor_1d() -> Result<(), TensorError> {
        let data: Vec<u8> = vec![1];
        let t = Tensor::<u8, 1, _>::from_shape_vec([1], data, CpuAllocator)?;
        assert_eq!(t.shape, [1]);
        assert_eq!(t.as_slice(), vec![1]);
        assert_eq!(t.strides, [1]);
        assert_eq!(t.numel(), 1);
        Ok(())
    }

    #[test]
    fn constructor_2d() -> Result<(), TensorError> {
        let data: Vec<u8> = vec![1, 2];
        let t = Tensor::<u8, 2, _>::from_shape_vec([1, 2], data, CpuAllocator)?;
        assert_eq!(t.shape, [1, 2]);
        assert_eq!(t.as_slice(), vec![1, 2]);
        assert_eq!(t.strides, [2, 1]);
        assert_eq!(t.numel(), 2);
        Ok(())
    }

    #[test]
    fn get_1d() -> Result<(), TensorError> {
        let data: Vec<u8> = vec![1, 2, 3, 4];
        let t = Tensor::<u8, 1, _>::from_shape_vec([4], data, CpuAllocator)?;
        assert_eq!(t.get([0]), Some(&1));
        assert_eq!(t.get([1]), Some(&2));
        assert_eq!(t.get([2]), Some(&3));
        assert_eq!(t.get([3]), Some(&4));
        assert!(t.get([4]).is_none());
        Ok(())
    }

    #[test]
    fn get_2d() -> Result<(), TensorError> {
        let data: Vec<u8> = vec![1, 2, 3, 4];
        let t = Tensor::<u8, 2, _>::from_shape_vec([2, 2], data, CpuAllocator)?;
        assert_eq!(t.get([0, 0]), Some(&1));
        assert_eq!(t.get([0, 1]), Some(&2));
        assert_eq!(t.get([1, 0]), Some(&3));
        assert_eq!(t.get([1, 1]), Some(&4));
        assert!(t.get([2, 0]).is_none());
        assert!(t.get([0, 2]).is_none());
        Ok(())
    }

    #[test]
    fn get_3d() -> Result<(), TensorError> {
        let data: Vec<u8> = vec![1, 2, 3, 4, 5, 6];
        let t = Tensor::<u8, 3, _>::from_shape_vec([2, 1, 3], data, CpuAllocator)?;
        assert_eq!(t.get([0, 0, 0]), Some(&1));
        assert_eq!(t.get([0, 0, 1]), Some(&2));
        assert_eq!(t.get([0, 0, 2]), Some(&3));
        assert_eq!(t.get([1, 0, 0]), Some(&4));
        assert_eq!(t.get([1, 0, 1]), Some(&5));
        assert_eq!(t.get([1, 0, 2]), Some(&6));
        assert!(t.get([2, 0, 0]).is_none());
        assert!(t.get([0, 1, 0]).is_none());
        assert!(t.get([0, 0, 3]).is_none());
        Ok(())
    }

    #[test]
    fn get_checked_1d() -> Result<(), TensorError> {
        let data: Vec<u8> = vec![1, 2, 3, 4];
        let t = Tensor::<u8, 1, _>::from_shape_vec([4], data, CpuAllocator)?;
        assert_eq!(*t.get_unchecked([0]), 1);
        assert_eq!(*t.get_unchecked([1]), 2);
        assert_eq!(*t.get_unchecked([2]), 3);
        assert_eq!(*t.get_unchecked([3]), 4);
        Ok(())
    }

    #[test]
    fn get_checked_2d() -> Result<(), TensorError> {
        let data: Vec<u8> = vec![1, 2, 3, 4];
        let t = Tensor::<u8, 2, _>::from_shape_vec([2, 2], data, CpuAllocator)?;
        assert_eq!(*t.get_unchecked([0, 0]), 1);
        assert_eq!(*t.get_unchecked([0, 1]), 2);
        assert_eq!(*t.get_unchecked([1, 0]), 3);
        assert_eq!(*t.get_unchecked([1, 1]), 4);
        Ok(())
    }
    #[test]
    fn reshape_1d() -> Result<(), TensorError> {
        let data: Vec<u8> = vec![1, 2, 3, 4];
        let t = Tensor::<u8, 1, _>::from_shape_vec([4], data, CpuAllocator)?;

        let view = t.reshape([2, 2])?;

        assert_eq!(view.shape, [2, 2]);
        assert_eq!(view.as_slice(), vec![1, 2, 3, 4]);
        assert_eq!(view.strides, [2, 1]);
        assert_eq!(view.numel(), 4);
        assert_eq!(view.as_contiguous().as_slice(), vec![1, 2, 3, 4]);
        Ok(())
    }

    #[test]
    fn reshape_2d() -> Result<(), TensorError> {
        let data: Vec<u8> = vec![1, 2, 3, 4];
        let t = Tensor::<u8, 2, _>::from_shape_vec([2, 2], data, CpuAllocator)?;
        let t2 = t.reshape([4])?;

        assert_eq!(t2.shape, [4]);
        assert_eq!(t2.as_slice(), vec![1, 2, 3, 4]);
        assert_eq!(t2.strides, [1]);
        assert_eq!(t2.numel(), 4);
        assert_eq!(t2.as_contiguous().as_slice(), vec![1, 2, 3, 4]);
        Ok(())
    }

    #[test]
    fn reshape_get_1d() -> Result<(), TensorError> {
        let data: Vec<u8> = vec![1, 2, 3, 4];
        let t = Tensor::<u8, 1, _>::from_shape_vec([4], data, CpuAllocator)?;
        let view = t.reshape([2, 2])?;
        assert_eq!(*view.get_unchecked([0, 0]), 1);
        assert_eq!(*view.get_unchecked([0, 1]), 2);
        assert_eq!(*view.get_unchecked([1, 0]), 3);
        assert_eq!(*view.get_unchecked([1, 1]), 4);
        assert_eq!(view.numel(), 4);
        assert_eq!(view.as_contiguous().as_slice(), vec![1, 2, 3, 4]);
        Ok(())
    }

    #[test]
    fn permute_axes_1d() -> Result<(), TensorError> {
        let data: Vec<u8> = vec![1, 2, 3, 4];
        let t = Tensor::<u8, 1, _>::from_shape_vec([4], data, CpuAllocator)?;
        let t2 = t.permute_axes([0]);
        assert_eq!(t2.shape, [4]);
        assert_eq!(t2.as_slice(), vec![1, 2, 3, 4]);
        assert_eq!(t2.strides, [1]);
        assert_eq!(t2.as_contiguous().as_slice(), vec![1, 2, 3, 4]);
        Ok(())
    }

    #[test]
    fn permute_axes_2d() -> Result<(), TensorError> {
        let data: Vec<u8> = vec![1, 2, 3, 4];
        let t = Tensor::<u8, 2, _>::from_shape_vec([2, 2], data, CpuAllocator)?;
        let view = t.permute_axes([1, 0]);
        assert_eq!(view.shape, [2, 2]);
        assert_eq!(*view.get_unchecked([0, 0]), 1u8);
        assert_eq!(*view.get_unchecked([1, 0]), 2u8);
        assert_eq!(*view.get_unchecked([0, 1]), 3u8);
        assert_eq!(*view.get_unchecked([1, 1]), 4u8);
        assert_eq!(view.strides, [1, 2]);
        assert_eq!(view.as_contiguous().as_slice(), vec![1, 3, 2, 4]);
        Ok(())
    }

    #[test]
    fn contiguous_2d() -> Result<(), TensorError> {
        let data: Vec<u8> = vec![1, 2, 3, 4, 5, 6];
        let t = Tensor::<u8, 2, _>::from_shape_vec([2, 3], data, CpuAllocator)?;

        let view = t.permute_axes([1, 0]);

        let contiguous = view.as_contiguous();

        assert_eq!(contiguous.shape, [3, 2]);
        assert_eq!(contiguous.strides, [2, 1]);
        assert_eq!(contiguous.as_slice(), vec![1, 4, 2, 5, 3, 6]);

        Ok(())
    }

    #[test]
    fn zeros_1d() -> Result<(), TensorError> {
        let t = Tensor::<u8, 1, _>::zeros([4], CpuAllocator);
        assert_eq!(t.as_slice(), vec![0, 0, 0, 0]);
        Ok(())
    }

    #[test]
    fn zeros_2d() -> Result<(), TensorError> {
        let t = Tensor::<u8, 2, _>::zeros([2, 2], CpuAllocator);
        assert_eq!(t.as_slice(), vec![0, 0, 0, 0]);
        Ok(())
    }

    #[test]
    fn map_1d() -> Result<(), TensorError> {
        let data: Vec<u8> = vec![1, 2, 3, 4];
        let t = Tensor::<u8, 1, _>::from_shape_vec([4], data, CpuAllocator)?;
        let t2 = t.map(|x| *x + 1);
        assert_eq!(t2.as_slice(), vec![2, 3, 4, 5]);
        Ok(())
    }

    #[test]
    fn map_2d() -> Result<(), TensorError> {
        let data: Vec<u8> = vec![1, 2, 3, 4];
        let t = Tensor::<u8, 2, _>::from_shape_vec([2, 2], data, CpuAllocator)?;
        let t2 = t.map(|x| *x + 1);
        assert_eq!(t2.as_slice(), vec![2, 3, 4, 5]);
        Ok(())
    }

    #[test]
    fn from_shape_val_1d() -> Result<(), TensorError> {
        let t = Tensor::<u8, 1, _>::from_shape_val([4], 0, CpuAllocator);
        assert_eq!(t.as_slice(), vec![0, 0, 0, 0]);
        Ok(())
    }

    #[test]
    fn from_shape_val_2d() -> Result<(), TensorError> {
        let t = Tensor::<u8, 2, _>::from_shape_val([2, 2], 1, CpuAllocator);
        assert_eq!(t.as_slice(), vec![1, 1, 1, 1]);
        Ok(())
    }

    #[test]
    fn from_shape_val_3d() -> Result<(), TensorError> {
        let t = Tensor::<u8, 3, _>::from_shape_val([2, 1, 3], 2, CpuAllocator);
        assert_eq!(t.as_slice(), vec![2, 2, 2, 2, 2, 2]);
        Ok(())
    }

    #[test]
    fn cast_1d() -> Result<(), TensorError> {
        let data: Vec<u8> = vec![1, 2, 3, 4];
        let t = Tensor::<u8, 1, _>::from_shape_vec([4], data, CpuAllocator)?;
        let t2 = t.cast::<u16>();
        assert_eq!(t2.as_slice(), vec![1, 2, 3, 4]);
        Ok(())
    }

    #[test]
    fn cast_2d() -> Result<(), TensorError> {
        let data: Vec<u8> = vec![1, 2, 3, 4];
        let t = Tensor::<u8, 2, _>::from_shape_vec([2, 2], data, CpuAllocator)?;
        let t2 = t.cast::<u16>();
        assert_eq!(t2.as_slice(), vec![1, 2, 3, 4]);
        Ok(())
    }

    #[test]
    fn from_shape_fn_1d() -> Result<(), TensorError> {
        let alloc = CpuAllocator;
        let t = Tensor::from_shape_fn([3, 3], alloc, |[i, j]| ((1 + i) * (1 + j)) as u8);
        assert_eq!(t.as_slice(), vec![1, 2, 3, 2, 4, 6, 3, 6, 9]);
        Ok(())
    }

    #[test]
    fn from_shape_fn_2d() -> Result<(), TensorError> {
        let alloc = CpuAllocator;
        let t = Tensor::from_shape_fn([3, 3], alloc, |[i, j]| ((1 + i) * (1 + j)) as f32);
        assert_eq!(
            t.as_slice(),
            vec![1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0]
        );
        Ok(())
    }

    #[test]
    fn from_shape_fn_3d() -> Result<(), TensorError> {
        let alloc = CpuAllocator;
        let t = Tensor::from_shape_fn([2, 3, 3], alloc, |[x, y, c]| {
            ((1 + x) * (1 + y) * (1 + c)) as i16
        });
        assert_eq!(
            t.as_slice(),
            vec![1, 2, 3, 2, 4, 6, 3, 6, 9, 2, 4, 6, 4, 8, 12, 6, 12, 18]
        );
        Ok(())
    }

    #[test]
    fn view_1d() -> Result<(), TensorError> {
        let alloc = CpuAllocator;
        let data: Vec<u8> = vec![1, 2, 3, 4];
        let t = Tensor::<u8, 1, _>::from_shape_vec([4], data, alloc)?;
        let view = t.view();

        // check that the view has the same data
        assert_eq!(view.as_slice(), t.as_slice());

        // check that the data pointer is the same
        assert!(std::ptr::eq(view.as_ptr(), t.as_ptr()));

        Ok(())
    }

    #[test]
    fn from_slice() -> Result<(), TensorError> {
        let data: [u8; 4] = [1, 2, 3, 4];
        let t = Tensor::<u8, 2, _>::from_shape_slice([2, 2], &data, CpuAllocator)?;

        assert_eq!(t.shape, [2, 2]);
        assert_eq!(t.as_slice(), &[1, 2, 3, 4]);

        Ok(())
    }

    #[test]
    fn display_2d() -> Result<(), TensorError> {
        let data: [u8; 4] = [1, 2, 3, 4];
        let t = Tensor::<u8, 2, _>::from_shape_slice([2, 2], &data, CpuAllocator)?;
        let disp = t.to_string();
        let lines = disp.lines().collect::<Vec<_>>();

        #[rustfmt::skip]
        assert_eq!(lines.as_slice(),
        ["[[1,2],",
         " [3,4]]"]);
        Ok(())
    }

    #[test]
    fn display_3d() -> Result<(), TensorError> {
        let data: [u8; 12] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let t = Tensor::<u8, 3, _>::from_shape_slice([2, 3, 2], &data, CpuAllocator)?;
        let disp = t.to_string();
        let lines = disp.lines().collect::<Vec<_>>();

        #[rustfmt::skip]
        assert_eq!(lines.as_slice(),
        ["[[[ 1, 2],",
         "  [ 3, 4],",
         "  [ 5, 6]],",
         "",
         " [[ 7, 8],",
         "  [ 9,10],",
         "  [11,12]]]"]);
        Ok(())
    }

    #[test]
    fn display_float() -> Result<(), TensorError> {
        let data: [f32; 4] = [1.00001, 1.00009, 0.99991, 0.99999];
        let t = Tensor::<f32, 2, _>::from_shape_slice([2, 2], &data, CpuAllocator)?;
        let disp = t.to_string();
        let lines = disp.lines().collect::<Vec<_>>();

        #[rustfmt::skip]
        assert_eq!(lines.as_slice(),
        ["[[1.0000,1.0001],",
         " [0.9999,1.0000]]"]);
        Ok(())
    }

    #[test]
    fn display_big_float() -> Result<(), TensorError> {
        let data: [f32; 4] = [1000.00001, 1.00009, 0.99991, 0.99999];
        let t = Tensor::<f32, 2, _>::from_shape_slice([2, 2], &data, CpuAllocator)?;
        let disp = t.to_string();
        let lines = disp.lines().collect::<Vec<_>>();

        #[rustfmt::skip]
        assert_eq!(lines.as_slice(),
        ["[[1.0000e+03,1.0001e+00],",
         " [9.9991e-01,9.9999e-01]]"]);
        Ok(())
    }

    #[test]
    fn display_big_tensor() -> Result<(), TensorError> {
        let data: [u8; 1000] = [0; 1000];
        let t = Tensor::<u8, 3, _>::from_shape_slice([10, 10, 10], &data, CpuAllocator)?;
        let disp = t.to_string();
        let lines = disp.lines().collect::<Vec<_>>();

        #[rustfmt::skip]
        assert_eq!(lines.as_slice(),
        ["[[[0,0,0,...,0],",
         "  [0,0,0,...,0],",
         "  [0,0,0,...,0],",
         "  ...",
         "  [0,0,0,...,0]],",
         "",
         " [[0,0,0,...,0],",
         "  [0,0,0,...,0],",
         "  [0,0,0,...,0],",
         "  ...",
         "  [0,0,0,...,0]],",
         "",
         " [[0,0,0,...,0],",
         "  [0,0,0,...,0],",
         "  [0,0,0,...,0],",
         "  ...",
         "  [0,0,0,...,0]],",
         "",
         " ...",
         "",
         " [[0,0,0,...,0],",
         "  [0,0,0,...,0],",
         "  [0,0,0,...,0],",
         "  ...",
         "  [0,0,0,...,0]]]"]);
        Ok(())
    }

    #[test]
    fn get_index_unchecked_1d() -> Result<(), TensorError> {
        let data: Vec<u8> = vec![1, 2, 3, 4];
        let t = Tensor::<u8, 1, CpuAllocator>::from_shape_vec([4], data, CpuAllocator)?;
        assert_eq!(t.get_index_unchecked(0), [0]);
        assert_eq!(t.get_index_unchecked(1), [1]);
        assert_eq!(t.get_index_unchecked(2), [2]);
        assert_eq!(t.get_index_unchecked(3), [3]);
        Ok(())
    }

    #[test]
    fn get_index_unchecked_2d() -> Result<(), TensorError> {
        let data: Vec<u8> = vec![1, 2, 3, 4];
        let t = Tensor::<u8, 2, CpuAllocator>::from_shape_vec([2, 2], data, CpuAllocator)?;
        assert_eq!(t.get_index_unchecked(0), [0, 0]);
        assert_eq!(t.get_index_unchecked(1), [0, 1]);
        assert_eq!(t.get_index_unchecked(2), [1, 0]);
        assert_eq!(t.get_index_unchecked(3), [1, 1]);
        Ok(())
    }

    #[test]
    fn get_index_unchecked_3d() -> Result<(), TensorError> {
        let data: Vec<u8> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let t = Tensor::<u8, 3, CpuAllocator>::from_shape_vec([2, 2, 3], data, CpuAllocator)?;
        assert_eq!(t.get_index_unchecked(0), [0, 0, 0]);
        assert_eq!(t.get_index_unchecked(1), [0, 0, 1]);
        assert_eq!(t.get_index_unchecked(2), [0, 0, 2]);
        assert_eq!(t.get_index_unchecked(3), [0, 1, 0]);
        assert_eq!(t.get_index_unchecked(4), [0, 1, 1]);
        assert_eq!(t.get_index_unchecked(5), [0, 1, 2]);
        assert_eq!(t.get_index_unchecked(6), [1, 0, 0]);
        assert_eq!(t.get_index_unchecked(7), [1, 0, 1]);
        assert_eq!(t.get_index_unchecked(8), [1, 0, 2]);
        assert_eq!(t.get_index_unchecked(9), [1, 1, 0]);
        assert_eq!(t.get_index_unchecked(10), [1, 1, 1]);
        assert_eq!(t.get_index_unchecked(11), [1, 1, 2]);
        Ok(())
    }

    #[test]
    fn get_index_to_offset_and_back() -> Result<(), TensorError> {
        let data: Vec<u8> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let t = Tensor::<u8, 3, CpuAllocator>::from_shape_vec([2, 2, 3], data, CpuAllocator)?;
        for offset in 0..12 {
            assert_eq!(
                t.get_iter_offset_unchecked(t.get_index_unchecked(offset)),
                offset
            );
        }
        Ok(())
    }

    #[test]
    fn get_offset_to_index_and_back() -> Result<(), TensorError> {
        let data: Vec<u8> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let t = Tensor::<u8, 3, CpuAllocator>::from_shape_vec([2, 2, 3], data, CpuAllocator)?;
        for ind in [
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, 2],
            [0, 1, 0],
            [0, 1, 1],
            [0, 1, 2],
            [1, 0, 0],
            [1, 0, 1],
            [1, 0, 2],
            [1, 1, 0],
            [1, 1, 1],
            [1, 1, 2],
        ] {
            assert_eq!(t.get_index_unchecked(t.get_iter_offset_unchecked(ind)), ind);
        }
        Ok(())
    }

    #[test]
    fn get_index_1d() -> Result<(), TensorError> {
        let data: Vec<u8> = vec![1, 2, 3, 4];
        let t = Tensor::<u8, 1, CpuAllocator>::from_shape_vec([4], data, CpuAllocator)?;
        assert_eq!(t.get_index(3), Ok([3]));
        assert!(t
            .get_index(4)
            .is_err_and(|x| x == TensorError::IndexOutOfBounds(4)));
        Ok(())
    }

    #[test]
    fn get_index_2d() -> Result<(), TensorError> {
        let data: Vec<u8> = vec![1, 2, 3, 4];
        let t = Tensor::<u8, 2, CpuAllocator>::from_shape_vec([2, 2], data, CpuAllocator)?;
        assert_eq!(t.get_index_unchecked(3), [1, 1]);
        assert!(t
            .get_index(4)
            .is_err_and(|x| x == TensorError::IndexOutOfBounds(4)));
        Ok(())
    }

    #[test]
    fn get_index_3d() -> Result<(), TensorError> {
        let data: Vec<u8> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let t = Tensor::<u8, 3, CpuAllocator>::from_shape_vec([2, 2, 3], data, CpuAllocator)?;
        assert_eq!(t.get_index_unchecked(11), [1, 1, 2]);
        assert!(t
            .get_index(12)
            .is_err_and(|x| x == TensorError::IndexOutOfBounds(12)));
        Ok(())
    }

    #[test]
    fn from_raw_parts() -> Result<(), TensorError> {
        let data: Vec<u8> = vec![1, 2, 3, 4];
        let t = unsafe { Tensor::from_raw_parts([2, 2], data.as_ptr(), data.len(), CpuAllocator)? };
        std::mem::forget(data);
        assert_eq!(t.shape, [2, 2]);
        assert_eq!(t.as_slice(), &[1, 2, 3, 4]);
        Ok(())
    }
}
