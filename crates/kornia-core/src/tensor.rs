use num_traits::Float;
use thiserror::Error;

use super::{
    allocator::{CpuAllocator, TensorAllocator, TensorAllocatorError},
    storage::{SafeTensorType, TensorStorage},
    view::TensorView,
};

/// An error type for tensor operations.
#[derive(Error, Debug)]
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
pub(crate) fn get_strides_from_shape<const N: usize>(shape: [usize; N]) -> [usize; N] {
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
/// use kornia_core::{Tensor, CpuAllocator};
///
/// let data: Vec<u8> = vec![1, 2, 3, 4];
/// let t = Tensor::<u8, 2>::new_uninitialized([2, 2], CpuAllocator).unwrap();
/// assert_eq!(t.shape, [2, 2]);
/// ```
pub struct Tensor<T, const N: usize, A: TensorAllocator = CpuAllocator>
where
    T: SafeTensorType,
{
    /// The storage of the tensor.
    pub storage: TensorStorage<T, A>,
    /// The shape of the tensor.
    pub shape: [usize; N],
    /// The strides of the tensor data in memory.
    pub strides: [usize; N],
}

/// Implementation of the Tensor struct.
impl<T, const N: usize, A> Tensor<T, N, A>
where
    T: SafeTensorType,
    A: TensorAllocator,
{
    /// Create a new `Tensor` with uninitialized data.
    ///
    /// # Arguments
    ///
    /// * `shape` - An array containing the shape of the tensor.
    /// * `alloc` - The allocator to use.
    ///
    /// # Returns
    ///
    /// A new `Tensor` instance with uninitialized data.
    ///
    /// # Errors
    ///
    /// If the allocation fails, an error is returned.
    pub fn new_uninitialized(shape: [usize; N], alloc: A) -> Result<Self, TensorError> {
        let numel = shape.iter().product::<usize>();
        let strides = get_strides_from_shape(shape);
        let storage = TensorStorage::new(numel, alloc)?;
        Ok(Self {
            storage,
            shape,
            strides,
        })
    }

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
    /// use kornia_core::{Tensor, CpuAllocator};
    ///
    /// let data: Vec<u8> = vec![1, 2, 3, 4];
    /// let t = Tensor::<u8, 2>::from_shape_vec([2, 2], data, CpuAllocator).unwrap();
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

    /// Creates a new `Tensor` with the given shape and data from a slice.
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
    ///
    /// # Example
    ///
    /// ```
    /// use kornia_core::{Tensor, CpuAllocator};
    ///
    /// let data: [u8; 4] = [1, 2, 3, 4];
    /// let t = Tensor::<u8, 2>::from_shape_slice([2, 2], &data, CpuAllocator).unwrap();
    /// assert_eq!(t.shape, [2, 2]);
    /// assert_eq!(t.as_slice(), &[1, 2, 3, 4]);
    /// ```
    pub fn from_shape_slice(shape: [usize; N], data: &[T], alloc: A) -> Result<Self, TensorError> {
        let numel = shape.iter().product::<usize>();
        if numel != data.len() {
            return Err(TensorError::InvalidShape(numel));
        }
        let storage = TensorStorage::from_slice(data, alloc)?;
        let strides = get_strides_from_shape(shape);
        Ok(Self {
            storage,
            shape,
            strides,
        })
    }

    /// Creates a new `Tensor` with the given shape and a default value.
    ///
    /// # Arguments
    ///
    /// * `shape` - An array containing the shape of the tensor.
    /// * `value` - The default value to fill the tensor with.
    /// * `alloc` - The allocator to use.
    ///
    /// # Returns
    ///
    /// A new `Tensor` instance.
    ///
    /// # Example
    ///
    /// ```
    /// use kornia_core::{Tensor, CpuAllocator};
    ///
    /// let t = Tensor::<u8, 1>::from_shape_val([4], 0, CpuAllocator);
    /// assert_eq!(t.as_slice(), vec![0, 0, 0, 0]);
    ///
    /// let t = Tensor::<u8, 2>::from_shape_val([2, 2], 1, CpuAllocator);
    /// assert_eq!(t.as_slice(), vec![1, 1, 1, 1]);
    ///
    /// let t = Tensor::<u8, 3>::from_shape_val([2, 1, 3], 2, CpuAllocator);
    /// assert_eq!(t.as_slice(), vec![2, 2, 2, 2, 2, 2]);
    /// ```
    pub fn from_shape_val(shape: [usize; N], value: T, alloc: A) -> Self {
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
    /// * `alloc` - The allocator to use.
    ///
    /// # Returns
    ///
    /// A new `Tensor` instance.
    ///
    /// # Example
    ///
    /// ```
    /// use kornia_core::{Tensor, CpuAllocator};
    ///
    /// let t = Tensor::<u8, 1>::from_shape_fn([4], |[i]| i as u8, CpuAllocator);
    /// assert_eq!(t.as_slice(), vec![0, 1, 2, 3]);
    ///
    /// let t = Tensor::<u8, 2>::from_shape_fn([2, 2], |[i, j]| (i * 2 + j) as u8, CpuAllocator);
    /// assert_eq!(t.as_slice(), vec![0, 1, 2, 3]);
    /// ```
    pub fn from_shape_fn<F>(shape: [usize; N], f: F, alloc: A) -> Self
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
        self.storage.len()
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
    /// use kornia_core::{Tensor, CpuAllocator};
    ///
    /// let data: Vec<u8> = vec![1, 2, 3, 4];
    ///
    /// let t = Tensor::<u8, 2>::from_shape_vec([2, 2], data, CpuAllocator).unwrap();
    /// assert_eq!(*t.get_unchecked([0, 0]), 1);
    /// assert_eq!(*t.get_unchecked([0, 1]), 2);
    /// assert_eq!(*t.get_unchecked([1, 0]), 3);
    /// assert_eq!(*t.get_unchecked([1, 1]), 4);
    /// ```
    pub fn get_unchecked(&self, index: [usize; N]) -> &T {
        let offset = self.get_iter_offset(index);
        self.storage.get_unchecked(offset)
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
    /// use kornia_core::{Tensor, CpuAllocator};
    ///
    /// let data: Vec<u8> = vec![1, 2, 3, 4];
    ///
    /// let t = Tensor::<u8, 2>::from_shape_vec([2, 2], data, CpuAllocator).unwrap();
    ///
    /// assert_eq!(t.get([0, 0]), Some(&1));
    /// assert_eq!(t.get([0, 1]), Some(&2));
    /// assert_eq!(t.get([1, 0]), Some(&3));
    /// assert_eq!(t.get([1, 1]), Some(&4));
    ///
    /// assert!(t.get([2, 0]).is_none());
    /// ```
    pub fn get(&self, index: [usize; N]) -> Option<&T> {
        let offset = self.get_iter_offset(index);
        self.storage.get(offset)
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
    /// use kornia_core::{Tensor, CpuAllocator};
    ///
    /// let data: Vec<u8> = vec![1, 2, 3, 4];
    ///
    /// let t = Tensor::<u8, 1>::from_shape_vec([4], data, CpuAllocator).unwrap();
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
        T: Default + Copy,
    {
        Self::from_shape_val(shape, T::default(), alloc)
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
    /// use kornia_core::{Tensor, CpuAllocator};
    ///
    /// let data: Vec<u8> = vec![1, 2, 3, 4];
    /// let t = Tensor::<u8, 1>::from_shape_vec([4], data, CpuAllocator).unwrap();
    ///
    /// let t2 = t.map(|x| *x + 1);
    /// assert_eq!(t2.as_slice(), vec![2, 3, 4, 5]);
    /// ```
    pub fn map<F, U>(&self, f: F) -> Tensor<U, N, A>
    where
        F: Fn(&T) -> U,
        U: SafeTensorType,
        A: Clone,
    {
        let mut new_storage =
            TensorStorage::<U, A>::new(self.numel(), self.storage.alloc().clone())
                .expect("Failed to allocate new storage");

        new_storage
            .as_mut_slice()
            .iter_mut()
            .zip(self.as_slice())
            .for_each(|(new, old)| *new = f(old));

        Tensor {
            storage: new_storage,
            shape: self.shape,
            strides: self.strides,
        }
    }

    /// Apply the power function to the pixel data.
    ///
    /// # Arguments
    ///
    /// * `n` - The power to raise the pixel data to.
    ///
    /// # Returns
    ///
    /// A new image with the pixel data raised to the power.
    pub fn powi(&self, n: i32) -> Tensor<T, N, A>
    where
        T: Float,
        A: Clone,
    {
        self.map(|x| x.powi(n))
    }

    /// Compute absolute value of the pixel data.
    ///
    /// # Returns
    ///
    /// A new image with the pixel data absolute value.
    pub fn abs(&self) -> Tensor<T, N, A>
    where
        T: Float,
        A: Clone,
    {
        self.map(|x| x.abs())
    }

    /// Compute the mean of the pixel data.
    ///
    /// # Returns
    ///
    /// The mean of the pixel data.
    pub fn mean(&self) -> Result<T, TensorError>
    where
        T: Float,
    {
        let data_acc = self.as_slice().iter().fold(T::zero(), |acc, &x| acc + x);
        let mean = data_acc / T::from(self.as_slice().len()).ok_or(TensorError::CastError)?;

        Ok(mean)
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
    /// use kornia_core::{Tensor, CpuAllocator};
    ///
    /// let data: Vec<u8> = vec![1, 2, 3, 4];
    /// let t = Tensor::<u8, 1>::from_shape_vec([4], data, CpuAllocator).unwrap();
    ///
    /// let t2 = t.cast::<f32>();
    /// assert_eq!(t2.as_slice(), vec![1.0, 2.0, 3.0, 4.0]);
    /// ```
    pub fn cast<U>(&self) -> Tensor<U, N>
    where
        T: SafeTensorType,
        U: SafeTensorType + From<T>,
    {
        let mut data: Vec<U> = Vec::with_capacity(self.storage.len());
        self.as_slice().iter().for_each(|&x| {
            data.push(U::from(x));
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
    /// use kornia_core::{Tensor, CpuAllocator};
    ///
    /// let data1: Vec<u8> = vec![1, 2, 3, 4];
    /// let t1 = Tensor::<u8, 1>::from_shape_vec([4], data1, CpuAllocator).unwrap();
    ///
    /// let data2: Vec<u8> = vec![1, 2, 3, 4];
    /// let t2 = Tensor::<u8, 1>::from_shape_vec([4], data2, CpuAllocator).unwrap();
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
        other: &Tensor<T, N, A>,
        op: F,
    ) -> Result<Tensor<T, N, A>, TensorError>
    where
        F: Fn(&T, &T) -> T,
        A: TensorAllocator + Clone,
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

        let storage = TensorStorage::from_vec(data, self.storage.alloc().clone());

        Ok(Tensor {
            storage,
            shape: self.shape,
            strides: self.strides,
        })
    }

    /// Perform an element-wise addition on two tensors.
    ///
    /// # Arguments
    ///
    /// * `other` - The other tensor to add.
    ///
    /// # Returns
    ///
    /// A new `Tensor` instance.
    ///
    /// # Example
    ///
    /// ```
    /// use kornia_core::{Tensor, CpuAllocator};
    ///
    /// let data1: Vec<u8> = vec![1, 2, 3, 4];
    /// let t1 = Tensor::<u8, 1>::from_shape_vec([4], data1, CpuAllocator).unwrap();
    ///
    /// let data2: Vec<u8> = vec![1, 2, 3, 4];
    /// let t2 = Tensor::<u8, 1>::from_shape_vec([4], data2, CpuAllocator).unwrap();
    ///
    /// let t3 = t1.add(&t2);
    /// assert_eq!(t3.as_slice(), vec![2, 4, 6, 8]);
    /// ```
    pub fn add(&self, other: &Tensor<T, N, A>) -> Tensor<T, N, A>
    where
        T: std::ops::Add<Output = T>,
        A: TensorAllocator,
    {
        self.element_wise_op(other, |a, b| *a + *b)
            .expect("Tensor dimension mismatch")
    }

    /// Perform an element-wise subtraction on two tensors.
    ///
    /// # Arguments
    ///
    /// * `other` - The other tensor to subtract.
    ///
    /// # Returns
    ///
    /// A new `Tensor` instance.
    ///
    /// # Example
    ///
    /// ```
    /// use kornia_core::{Tensor, CpuAllocator};
    ///
    /// let data1: Vec<u8> = vec![1, 2, 3, 4];
    /// let t1 = Tensor::<u8, 1>::from_shape_vec([4], data1, CpuAllocator).unwrap();
    ///
    /// let data2: Vec<u8> = vec![1, 2, 3, 4];
    /// let t2 = Tensor::<u8, 1>::from_shape_vec([4], data2, CpuAllocator).unwrap();
    ///
    /// let t3 = t1.sub(&t2);
    /// assert_eq!(t3.as_slice(), vec![0, 0, 0, 0]);
    /// ```
    pub fn sub(&self, other: &Tensor<T, N, A>) -> Tensor<T, N, A>
    where
        T: std::ops::Sub<Output = T>,
        A: TensorAllocator,
    {
        self.element_wise_op(other, |a, b| *a - *b)
            .expect("Tensor dimension mismatch")
    }

    /// Perform an element-wise multiplication on two tensors.
    ///
    /// # Arguments
    ///
    /// * `other` - The other tensor to multiply.
    ///
    /// # Returns
    ///
    /// A new `Tensor` instance.
    ///
    /// # Example
    ///
    /// ```
    /// use kornia_core::{Tensor, CpuAllocator};
    ///
    /// let data1: Vec<u8> = vec![1, 2, 3, 4];
    /// let t1 = Tensor::<u8, 1>::from_shape_vec([4], data1, CpuAllocator).unwrap();
    ///
    /// let data2: Vec<u8> = vec![1, 2, 3, 4];
    /// let t2 = Tensor::<u8, 1>::from_shape_vec([4], data2, CpuAllocator).unwrap();
    ///
    /// let t3 = t1.mul(&t2);
    /// assert_eq!(t3.as_slice(), vec![1, 4, 9, 16]);
    /// ```
    pub fn mul(&self, other: &Tensor<T, N, A>) -> Tensor<T, N, A>
    where
        T: std::ops::Mul<Output = T>,
        A: TensorAllocator,
    {
        self.element_wise_op(other, |a, b| *a * *b)
            .expect("Tensor dimension mismatch")
    }

    /// Perform an element-wise division on two tensors.
    ///
    /// # Arguments
    ///
    /// * `other` - The other tensor to divide.
    ///
    /// # Returns
    ///
    /// A new `Tensor` instance.
    ///
    /// # Example
    ///
    /// ```
    /// use kornia_core::{Tensor, CpuAllocator};
    ///
    /// let data1: Vec<u8> = vec![1, 2, 3, 4];
    /// let t1 = Tensor::<u8, 1>::from_shape_vec([4], data1, CpuAllocator).unwrap();
    ///
    /// let data2: Vec<u8> = vec![1, 2, 3, 4];
    /// let t2 = Tensor::<u8, 1>::from_shape_vec([4], data2, CpuAllocator).unwrap();
    ///
    /// let t3 = t1.div(&t2);
    /// assert_eq!(t3.as_slice(), vec![1, 1, 1, 1]);
    /// ```
    pub fn div(&self, other: &Tensor<T, N, A>) -> Tensor<T, N, A>
    where
        T: std::ops::Div<Output = T>,
        A: TensorAllocator,
    {
        self.element_wise_op(other, |a, b| *a / *b)
            .expect("Tensor dimension mismatch")
    }
}

impl<T, const N: usize, A> Clone for Tensor<T, N, A>
where
    T: SafeTensorType + Clone,
    A: TensorAllocator + Clone,
{
    fn clone(&self) -> Self {
        Self {
            storage: self.storage.clone(),
            shape: self.shape,
            strides: self.strides,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::allocator::CpuAllocator;
    use crate::{Tensor, TensorError};

    #[test]
    fn constructor_1d() -> Result<(), TensorError> {
        let data: Vec<u8> = vec![1];
        let t = Tensor::<u8, 1>::from_shape_vec([1], data, CpuAllocator)?;
        assert_eq!(t.shape, [1]);
        assert_eq!(t.as_slice(), vec![1]);
        assert_eq!(t.strides, [1]);
        assert_eq!(t.numel(), 1);
        Ok(())
    }

    #[test]
    fn constructor_2d() -> Result<(), TensorError> {
        let data: Vec<u8> = vec![1, 2];
        let t = Tensor::<u8, 2>::from_shape_vec([1, 2], data, CpuAllocator)?;
        assert_eq!(t.shape, [1, 2]);
        assert_eq!(t.as_slice(), vec![1, 2]);
        assert_eq!(t.strides, [2, 1]);
        assert_eq!(t.numel(), 2);
        Ok(())
    }

    #[test]
    fn get_1d() -> Result<(), TensorError> {
        let data: Vec<u8> = vec![1, 2, 3, 4];
        let t = Tensor::<u8, 1>::from_shape_vec([4], data, CpuAllocator)?;
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
        let t = Tensor::<u8, 2>::from_shape_vec([2, 2], data, CpuAllocator)?;
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
        let t = Tensor::<u8, 3>::from_shape_vec([2, 1, 3], data, CpuAllocator)?;
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
        let t = Tensor::<u8, 1>::from_shape_vec([4], data, CpuAllocator)?;
        assert_eq!(*t.get_unchecked([0]), 1);
        assert_eq!(*t.get_unchecked([1]), 2);
        assert_eq!(*t.get_unchecked([2]), 3);
        assert_eq!(*t.get_unchecked([3]), 4);
        Ok(())
    }

    #[test]
    fn get_checked_2d() -> Result<(), TensorError> {
        let data: Vec<u8> = vec![1, 2, 3, 4];
        let t = Tensor::<u8, 2>::from_shape_vec([2, 2], data, CpuAllocator)?;
        assert_eq!(*t.get_unchecked([0, 0]), 1);
        assert_eq!(*t.get_unchecked([0, 1]), 2);
        assert_eq!(*t.get_unchecked([1, 0]), 3);
        assert_eq!(*t.get_unchecked([1, 1]), 4);
        Ok(())
    }

    #[test]
    fn add_1d() -> Result<(), TensorError> {
        let data1: Vec<u8> = vec![1, 2, 3, 4];
        let t1 = Tensor::<u8, 1>::from_shape_vec([4], data1, CpuAllocator)?;
        let data2: Vec<u8> = vec![1, 2, 3, 4];
        let t2 = Tensor::<u8, 1>::from_shape_vec([4], data2, CpuAllocator)?;
        let t3 = t1.add(&t2);
        assert_eq!(t3.as_slice(), vec![2, 4, 6, 8]);
        Ok(())
    }

    #[test]
    fn add_2d() -> Result<(), TensorError> {
        let data1: Vec<u8> = vec![1, 2, 3, 4];
        let t1 = Tensor::<u8, 2>::from_shape_vec([2, 2], data1, CpuAllocator)?;
        let data2: Vec<u8> = vec![1, 2, 3, 4];
        let t2 = Tensor::<u8, 2>::from_shape_vec([2, 2], data2, CpuAllocator)?;
        let t3 = t1.add(&t2);
        assert_eq!(t3.as_slice(), vec![2, 4, 6, 8]);
        Ok(())
    }

    #[test]
    fn add_3d() -> Result<(), TensorError> {
        let data1: Vec<u8> = vec![1, 2, 3, 4, 5, 6];
        let t1 = Tensor::<u8, 3>::from_shape_vec([2, 1, 3], data1, CpuAllocator)?;
        let data2: Vec<u8> = vec![1, 2, 3, 4, 5, 6];
        let t2 = Tensor::<u8, 3>::from_shape_vec([2, 1, 3], data2, CpuAllocator)?;
        let t3 = t1.add(&t2);
        assert_eq!(t3.as_slice(), vec![2, 4, 6, 8, 10, 12]);
        Ok(())
    }

    #[test]
    fn sub_1d() -> Result<(), TensorError> {
        let data1: Vec<u8> = vec![1, 2, 3, 4];
        let t1 = Tensor::<u8, 1>::from_shape_vec([4], data1, CpuAllocator)?;
        let data2: Vec<u8> = vec![1, 2, 3, 4];
        let t2 = Tensor::<u8, 1>::from_shape_vec([4], data2, CpuAllocator)?;
        let t3 = t1.sub(&t2);
        assert_eq!(t3.as_slice(), vec![0, 0, 0, 0]);
        Ok(())
    }

    #[test]
    fn sub_2d() -> Result<(), TensorError> {
        let data1: Vec<u8> = vec![1, 2, 3, 4];
        let t1 = Tensor::<u8, 2>::from_shape_vec([2, 2], data1, CpuAllocator)?;
        let data2: Vec<u8> = vec![1, 2, 3, 4];
        let t2 = Tensor::<u8, 2>::from_shape_vec([2, 2], data2, CpuAllocator)?;
        let t3 = t1.sub(&t2);
        assert_eq!(t3.as_slice(), vec![0, 0, 0, 0]);
        Ok(())
    }

    #[test]
    fn div_1d() -> Result<(), TensorError> {
        let data1: Vec<u8> = vec![1, 2, 3, 4];
        let t1 = Tensor::<u8, 1>::from_shape_vec([4], data1, CpuAllocator)?;
        let data2: Vec<u8> = vec![1, 2, 3, 4];
        let t2 = Tensor::<u8, 1>::from_shape_vec([4], data2, CpuAllocator)?;
        let t3 = t1.div(&t2);
        assert_eq!(t3.as_slice(), vec![1, 1, 1, 1]);
        Ok(())
    }

    #[test]
    fn div_2d() -> Result<(), TensorError> {
        let data1: Vec<u8> = vec![1, 2, 3, 4];
        let t1 = Tensor::<u8, 2>::from_shape_vec([2, 2], data1, CpuAllocator)?;
        let data2: Vec<u8> = vec![1, 2, 3, 4];
        let t2 = Tensor::<u8, 2>::from_shape_vec([2, 2], data2, CpuAllocator)?;
        let t3 = t1.div(&t2);
        assert_eq!(t3.as_slice(), vec![1, 1, 1, 1]);
        Ok(())
    }

    #[test]
    fn mul_1d() -> Result<(), TensorError> {
        let data1: Vec<u8> = vec![1, 2, 3, 4];
        let t1 = Tensor::<u8, 1>::from_shape_vec([4], data1, CpuAllocator)?;
        let data2: Vec<u8> = vec![1, 2, 3, 4];
        let t2 = Tensor::<u8, 1>::from_shape_vec([4], data2, CpuAllocator)?;
        let t3 = t1.mul(&t2);
        assert_eq!(t3.as_slice(), vec![1, 4, 9, 16]);
        Ok(())
    }

    #[test]
    fn mul_2d() -> Result<(), TensorError> {
        let data1: Vec<u8> = vec![1, 2, 3, 4];
        let t1 = Tensor::<u8, 2>::from_shape_vec([2, 2], data1, CpuAllocator)?;
        let data2: Vec<u8> = vec![1, 2, 3, 4];
        let t2 = Tensor::<u8, 2>::from_shape_vec([2, 2], data2, CpuAllocator)?;
        let t3 = t1.mul(&t2);
        assert_eq!(t3.as_slice(), vec![1, 4, 9, 16]);
        Ok(())
    }

    #[test]
    fn reshape_1d() -> Result<(), TensorError> {
        let data: Vec<u8> = vec![1, 2, 3, 4];
        let t = Tensor::<u8, 1>::from_shape_vec([4], data, CpuAllocator)?;

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
        let t = Tensor::<u8, 2>::from_shape_vec([2, 2], data, CpuAllocator)?;
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
        let t = Tensor::<u8, 1>::from_shape_vec([4], data, CpuAllocator)?;
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
        let t = Tensor::<u8, 1>::from_shape_vec([4], data, CpuAllocator)?;
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
        let t = Tensor::<u8, 2>::from_shape_vec([2, 2], data, CpuAllocator)?;
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
        let t = Tensor::<u8, 2>::from_shape_vec([2, 3], data, CpuAllocator)?;

        let view = t.permute_axes([1, 0]);

        let contiguous = view.as_contiguous();

        assert_eq!(contiguous.shape, [3, 2]);
        assert_eq!(contiguous.strides, [2, 1]);
        assert_eq!(contiguous.as_slice(), vec![1, 4, 2, 5, 3, 6]);

        Ok(())
    }

    #[test]
    fn zeros_1d() -> Result<(), TensorError> {
        let t = Tensor::<u8, 1>::zeros([4], CpuAllocator);
        assert_eq!(t.as_slice(), vec![0, 0, 0, 0]);
        Ok(())
    }

    #[test]
    fn zeros_2d() -> Result<(), TensorError> {
        let t = Tensor::<u8, 2>::zeros([2, 2], CpuAllocator);
        assert_eq!(t.as_slice(), vec![0, 0, 0, 0]);
        Ok(())
    }

    #[test]
    fn map_1d() -> Result<(), TensorError> {
        let data: Vec<u8> = vec![1, 2, 3, 4];
        let t = Tensor::<u8, 1>::from_shape_vec([4], data, CpuAllocator)?;
        let t2 = t.map(|x| *x + 1);
        assert_eq!(t2.as_slice(), vec![2, 3, 4, 5]);
        Ok(())
    }

    #[test]
    fn map_2d() -> Result<(), TensorError> {
        let data: Vec<u8> = vec![1, 2, 3, 4];
        let t = Tensor::<u8, 2>::from_shape_vec([2, 2], data, CpuAllocator)?;
        let t2 = t.map(|x| *x + 1);
        assert_eq!(t2.as_slice(), vec![2, 3, 4, 5]);
        Ok(())
    }

    #[test]
    fn from_shape_val_1d() -> Result<(), TensorError> {
        let t = Tensor::<u8, 1>::from_shape_val([4], 0, CpuAllocator);
        assert_eq!(t.as_slice(), vec![0, 0, 0, 0]);
        Ok(())
    }

    #[test]
    fn from_shape_val_2d() -> Result<(), TensorError> {
        let t = Tensor::<u8, 2>::from_shape_val([2, 2], 1, CpuAllocator);
        assert_eq!(t.as_slice(), vec![1, 1, 1, 1]);
        Ok(())
    }

    #[test]
    fn from_shape_val_3d() -> Result<(), TensorError> {
        let t = Tensor::<u8, 3>::from_shape_val([2, 1, 3], 2, CpuAllocator);
        assert_eq!(t.as_slice(), vec![2, 2, 2, 2, 2, 2]);
        Ok(())
    }

    #[test]
    fn cast_1d() -> Result<(), TensorError> {
        let data: Vec<u8> = vec![1, 2, 3, 4];
        let t = Tensor::<u8, 1>::from_shape_vec([4], data, CpuAllocator)?;
        let t2 = t.cast::<u16>();
        assert_eq!(t2.as_slice(), vec![1, 2, 3, 4]);
        Ok(())
    }

    #[test]
    fn cast_2d() -> Result<(), TensorError> {
        let data: Vec<u8> = vec![1, 2, 3, 4];
        let t = Tensor::<u8, 2>::from_shape_vec([2, 2], data, CpuAllocator)?;
        let t2 = t.cast::<u16>();
        assert_eq!(t2.as_slice(), vec![1, 2, 3, 4]);
        Ok(())
    }

    #[test]
    fn from_shape_fn_1d() -> Result<(), TensorError> {
        let t = Tensor::from_shape_fn([3, 3], |[i, j]| ((1 + i) * (1 + j)) as u8, CpuAllocator);
        assert_eq!(t.as_slice(), vec![1, 2, 3, 2, 4, 6, 3, 6, 9]);
        Ok(())
    }

    #[test]
    fn from_shape_fn_2d() -> Result<(), TensorError> {
        let t = Tensor::from_shape_fn([3, 3], |[i, j]| ((1 + i) * (1 + j)) as f32, CpuAllocator);
        assert_eq!(
            t.as_slice(),
            vec![1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0]
        );
        Ok(())
    }

    #[test]
    fn from_shape_fn_3d() -> Result<(), TensorError> {
        let t = Tensor::from_shape_fn(
            [2, 3, 3],
            |[x, y, c]| ((1 + x) * (1 + y) * (1 + c)) as i16,
            CpuAllocator,
        );
        assert_eq!(
            t.as_slice(),
            vec![1, 2, 3, 2, 4, 6, 3, 6, 9, 2, 4, 6, 4, 8, 12, 6, 12, 18]
        );
        Ok(())
    }

    #[test]
    fn view_1d() -> Result<(), TensorError> {
        let data: Vec<u8> = vec![1, 2, 3, 4];
        let t = Tensor::<u8, 1>::from_shape_vec([4], data, CpuAllocator)?;
        let view = t.view();

        // check that the view has the same data
        assert_eq!(view.as_slice(), t.as_slice());

        // check that the data pointer is the same
        assert!(std::ptr::eq(view.as_ptr(), t.as_ptr()));

        Ok(())
    }

    // New tests for added functionality

    #[test]
    fn powi_and_abs() -> Result<(), TensorError> {
        let data: Vec<f32> = vec![-1.0, 2.0, -3.0, 4.0];
        let t = Tensor::<f32, 1>::from_shape_vec([4], data, CpuAllocator)?;

        let t_powi = t.powi(2);
        assert_eq!(t_powi.as_slice(), &[1.0, 4.0, 9.0, 16.0]);

        let t_abs = t.abs();
        assert_eq!(t_abs.as_slice(), &[1.0, 2.0, 3.0, 4.0]);

        Ok(())
    }

    #[test]
    fn from_slice() -> Result<(), TensorError> {
        let data: [u8; 4] = [1, 2, 3, 4];
        let t = Tensor::<u8, 2>::from_shape_slice([2, 2], &data, CpuAllocator)?;

        assert_eq!(t.shape, [2, 2]);
        assert_eq!(t.as_slice(), &[1, 2, 3, 4]);

        Ok(())
    }
}
