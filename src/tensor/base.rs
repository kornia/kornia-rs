use thiserror::Error;

use super::{
    allocator::{CpuAllocator, TensorAllocator, TensorAllocatorError},
    storage::TensorStorage,
};

#[derive(Error, Debug)]
pub enum TensorError {
    #[error("The number of elements in the data does not match the shape of the tensor: {0}")]
    InvalidShape(usize),

    #[error("Index out of bounds. The index {0} is out of bounds.")]
    IndexOutOfBounds(usize),

    #[error("Error with the tensor storage: {0}")]
    StorageError(#[from] TensorAllocatorError),
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
fn get_strides_from_shape<const N: usize>(shape: [usize; N]) -> [usize; N] {
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
/// NOTE: internally the data is stored as an arrow::Buffer which represents a contiguous memory
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
/// use kornia_rs::tensor::{Tensor, CpuAllocator};
///
/// let data: Vec<u8> = vec![1, 2, 3, 4];
/// let t = Tensor::<u8, 2>::new_uninitialized([2, 2], CpuAllocator).unwrap();
/// assert_eq!(t.shape, [2, 2]);
pub struct Tensor<T, const N: usize, A: TensorAllocator = CpuAllocator>
where
    T: arrow_buffer::ArrowNativeType,
{
    pub storage: TensorStorage<T, A>,
    pub shape: [usize; N],
    pub strides: [usize; N],
}

/// Implementation of the Tensor struct.
impl<T, const N: usize, A> Tensor<T, N, A>
where
    T: arrow_buffer::ArrowNativeType + std::panic::RefUnwindSafe,
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
    pub fn new_uninitialized(shape: [usize; N], alloc: A) -> Result<Self, TensorError> {
        let numel = shape.iter().product::<usize>();
        let strides = get_strides_from_shape(shape);
        let storage = TensorStorage::new(numel, alloc)?;
        Ok(Tensor {
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
    pub fn as_slice(&self) -> &[T] {
        let slice = self.storage.data.typed_data::<T>();
        slice
    }

    /// Get the data of the tensor as a mutable slice.
    ///
    /// # Returns
    ///
    /// A mutable slice containing the data of the tensor.
    pub fn as_slice_mut(&mut self) -> &mut [T] {
        // convert the data to a typed slice
        let slice = self.storage.data.typed_data::<T>();

        // TODO: verify if there is a better way to do this
        unsafe { std::slice::from_raw_parts_mut(slice.as_ptr() as *mut T, slice.len()) }
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
    /// use kornia_rs::tensor::{Tensor, CpuAllocator};
    ///
    /// let data: Vec<u8> = vec![1, 2, 3, 4];
    /// let t = Tensor::<u8, 2>::from_shape_vec([2, 2], data, CpuAllocator).unwrap();
    /// assert_eq!(t.shape, [2, 2]);
    ///```
    pub fn from_shape_vec(shape: [usize; N], data: Vec<T>, alloc: A) -> Result<Self, TensorError> {
        let numel = shape.iter().product::<usize>();
        if numel != data.len() {
            Err(TensorError::InvalidShape(numel))?;
        }
        let storage = TensorStorage::from_vec(data, alloc)?;
        let strides = get_strides_from_shape(shape);
        Ok(Tensor {
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
    ///
    /// # Returns
    ///
    /// A new `Tensor` instance.
    ///
    /// # Example
    ///
    /// ```
    /// use kornia_rs::tensor::{Tensor, CpuAllocator};
    ///
    /// let t = Tensor::<u8, 1>::from_shape_val([4], 0, CpuAllocator).unwrap();
    /// assert_eq!(t.as_slice(), vec![0, 0, 0, 0]);
    ///
    /// let t = Tensor::<u8, 2>::from_shape_val([2, 2], 1, CpuAllocator).unwrap();
    /// assert_eq!(t.as_slice(), vec![1, 1, 1, 1]);
    ///
    /// let t = Tensor::<u8, 3>::from_shape_val([2, 1, 3], 2, CpuAllocator).unwrap();
    /// assert_eq!(t.as_slice(), vec![2, 2, 2, 2, 2, 2]);
    /// ```
    pub fn from_shape_val(shape: [usize; N], value: T, alloc: A) -> Result<Self, TensorError>
    where
        T: Copy,
    {
        let numel = shape.iter().product::<usize>();
        let mut a = Self::new_uninitialized(shape, alloc)?;

        for i in a.as_slice_mut().iter_mut().take(numel) {
            *i = value;
        }

        Ok(a)
    }

    /// Create a new `Tensor` with the given shape and a function to generate the data.
    ///
    /// F is called with the index of the element to generate.
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
    /// use kornia_rs::tensor::{Tensor, CpuAllocator};
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
        let storage = TensorStorage::from_vec(data, alloc).unwrap();
        let strides = get_strides_from_shape(shape);
        Tensor {
            storage,
            shape,
            strides,
        }
    }

    /// Returns the number of elements in the tensor.
    pub fn numel(&self) -> usize {
        self.storage.data.len()
    }

    // TODO: find a better name
    pub fn get_iter_offset(&self, index: [usize; N]) -> usize {
        let mut offset = 0;
        for (i, &idx) in index.iter().enumerate() {
            offset += idx * self.strides[i];
        }
        offset
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
    /// use kornia_rs::tensor::{Tensor, CpuAllocator};
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
        &self.as_slice()[offset]
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
    /// use kornia_rs::tensor::{Tensor, CpuAllocator};
    ///
    /// let data: Vec<u8> = vec![1, 2, 3, 4];
    ///
    /// let t = Tensor::<u8, 2>::from_shape_vec([2, 2], data, CpuAllocator).unwrap();
    ///
    /// assert_eq!(*t.get([0, 0]).unwrap(), 1);
    /// assert_eq!(*t.get([0, 1]).unwrap(), 2);
    /// assert_eq!(*t.get([1, 0]).unwrap(), 3);
    /// assert_eq!(*t.get([1, 1]).unwrap(), 4);
    ///
    /// assert!(t.get([0, 2]).is_err());
    /// ```
    pub fn get(&self, index: [usize; N]) -> Result<&T, TensorError> {
        let mut offset = 0;
        for (i, &idx) in index.iter().enumerate() {
            if idx >= self.shape[i] {
                Err(TensorError::IndexOutOfBounds(idx))?;
            }
            offset += idx * self.strides[i];
        }
        Ok(&self.as_slice()[offset])
    }

    /// Reshape the tensor to a new shape.
    ///
    /// # Arguments
    ///
    /// * `shape` - The new shape of the tensor.
    ///
    /// # Returns
    ///
    /// A new `Tensor` instance.
    ///
    /// # Errors
    ///
    /// If the number of elements in the new shape does not match the number of elements in the tensor, an error is returned.
    ///
    /// # Example
    ///
    /// ```
    /// use kornia_rs::tensor::{Tensor, CpuAllocator};
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
        self,
        shape: [usize; M],
    ) -> Result<Tensor<T, M, A>, TensorError> {
        let numel = shape.iter().product::<usize>();
        if numel != self.storage.data.len() {
            Err(TensorError::InvalidShape(numel))?;
        }

        let strides = get_strides_from_shape(shape);

        Ok(Tensor {
            storage: self.storage,
            shape,
            strides,
        })
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
    /// use kornia_rs::tensor::{Tensor, CpuAllocator};
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
        other: &Tensor<T, N>,
        op: F,
    ) -> Result<Tensor<T, N>, TensorError>
    where
        F: Fn(&T, &T) -> T,
    {
        let data = self
            .as_slice()
            .iter()
            .zip(other.as_slice().iter())
            .map(|(a, b)| op(a, b))
            .collect();

        let storage = TensorStorage::from_vec(data, CpuAllocator)?;

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
    /// use kornia_rs::tensor::{Tensor, CpuAllocator};
    ///
    /// let data1: Vec<u8> = vec![1, 2, 3, 4];
    /// let t1 = Tensor::<u8, 1>::from_shape_vec([4], data1, CpuAllocator).unwrap();
    ///
    /// let data2: Vec<u8> = vec![1, 2, 3, 4];
    /// let t2 = Tensor::<u8, 1>::from_shape_vec([4], data2, CpuAllocator).unwrap();
    ///
    /// let t3 = t1.add(&t2).unwrap();
    /// assert_eq!(t3.as_slice(), vec![2, 4, 6, 8]);
    /// ```
    pub fn add(&self, other: &Tensor<T, N>) -> Result<Tensor<T, N>, TensorError>
    where
        T: std::ops::Add<Output = T> + Copy,
    {
        self.element_wise_op(other, |a, b| *a + *b)
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
    /// use kornia_rs::tensor::{Tensor, CpuAllocator};
    ///
    /// let data1: Vec<u8> = vec![1, 2, 3, 4];
    /// let t1 = Tensor::<u8, 1>::from_shape_vec([4], data1, CpuAllocator).unwrap();
    ///
    /// let data2: Vec<u8> = vec![1, 2, 3, 4];
    /// let t2 = Tensor::<u8, 1>::from_shape_vec([4], data2, CpuAllocator).unwrap();
    ///
    /// let t3 = t1.sub(&t2).unwrap();
    /// assert_eq!(t3.as_slice(), vec![0, 0, 0, 0]);
    /// ```
    pub fn sub(&self, other: &Tensor<T, N>) -> Result<Tensor<T, N>, TensorError>
    where
        T: std::ops::Sub<Output = T> + Copy,
    {
        self.element_wise_op(other, |a, b| *a - *b)
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
    /// use kornia_rs::tensor::{Tensor, CpuAllocator};
    ///
    /// let data1: Vec<u8> = vec![1, 2, 3, 4];
    /// let t1 = Tensor::<u8, 1>::from_shape_vec([4], data1, CpuAllocator).unwrap();
    ///
    /// let data2: Vec<u8> = vec![1, 2, 3, 4];
    /// let t2 = Tensor::<u8, 1>::from_shape_vec([4], data2, CpuAllocator).unwrap();
    ///
    /// let t3 = t1.mul(&t2).unwrap();
    /// assert_eq!(t3.as_slice(), vec![1, 4, 9, 16]);
    /// ```
    pub fn mul(&self, other: &Tensor<T, N>) -> Result<Tensor<T, N>, TensorError>
    where
        T: std::ops::Mul<Output = T> + Copy,
    {
        self.element_wise_op(other, |a, b| *a * *b)
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
    /// use kornia_rs::tensor::{Tensor, CpuAllocator};
    ///
    /// let data1: Vec<u8> = vec![1, 2, 3, 4];
    /// let t1 = Tensor::<u8, 1>::from_shape_vec([4], data1, CpuAllocator).unwrap();
    ///
    /// let data2: Vec<u8> = vec![1, 2, 3, 4];
    /// let t2 = Tensor::<u8, 1>::from_shape_vec([4], data2, CpuAllocator).unwrap();
    ///
    /// let t3 = t1.div(&t2).unwrap();
    /// assert_eq!(t3.as_slice(), vec![1, 1, 1, 1]);
    /// ```
    pub fn div(&self, other: &Tensor<T, N>) -> Result<Tensor<T, N>, TensorError>
    where
        T: std::ops::Div<Output = T> + Copy,
    {
        self.element_wise_op(other, |a, b| *a / *b)
    }

    /// Create a new `Tensor` filled with zeros.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the tensor.
    ///
    /// # Returns
    ///
    /// A new `Tensor` instance.
    ///
    /// # Example
    ///
    /// ```
    /// use kornia_rs::tensor::{Tensor, CpuAllocator};
    ///
    /// let t = Tensor::<u8, 1>::zeros([4], CpuAllocator);
    /// assert_eq!(t.as_slice(), vec![0, 0, 0, 0]);
    ///
    /// let t = Tensor::<u8, 2>::zeros([2, 2], CpuAllocator);
    /// assert_eq!(t.as_slice(), vec![0, 0, 0, 0]);
    /// ```
    pub fn zeros(shape: [usize; N], alloc: A) -> Tensor<T, N, A>
    where
        T: Default + Copy,
    {
        Self::from_shape_val(shape, T::default(), alloc).unwrap()
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
    /// use kornia_rs::tensor::{Tensor, CpuAllocator};
    ///
    /// let data: Vec<u8> = vec![1, 2, 3, 4];
    /// let t = Tensor::<u8, 1>::from_shape_vec([4], data, CpuAllocator).unwrap();
    ///
    /// let t2 = t.map(|x| *x + 1).unwrap();
    /// assert_eq!(t2.as_slice(), vec![2, 3, 4, 5]);
    /// ```
    pub fn map<F>(&self, f: F) -> Result<Tensor<T, N>, TensorError>
    where
        F: Fn(&T) -> T,
    {
        let data: Vec<T> = self.as_slice().iter().map(f).collect();
        let storage = TensorStorage::from_vec(data, CpuAllocator)?;
        Ok(Tensor {
            storage,
            shape: self.shape,
            strides: self.strides,
        })
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
    /// use kornia_rs::tensor::{Tensor, CpuAllocator};
    ///
    /// let data: Vec<u8> = vec![1, 2, 3, 4];
    /// let t = Tensor::<u8, 1>::from_shape_vec([4], data, CpuAllocator).unwrap();
    ///
    /// let t2 = t.cast::<f32>().unwrap();
    /// assert_eq!(t2.as_slice(), vec![1.0, 2.0, 3.0, 4.0]);
    /// ```
    pub fn cast<U>(&self) -> Result<Tensor<U, N>, TensorError>
    where
        T: Copy + Into<U>,
        U: arrow_buffer::ArrowNativeType + std::panic::RefUnwindSafe,
    {
        let data: Vec<U> = self.as_slice().iter().map(|x| (*x).into()).collect();
        let storage = TensorStorage::from_vec(data, CpuAllocator)?;
        Ok(Tensor {
            storage,
            shape: self.shape,
            strides: self.strides,
        })
    }
}

impl<T, const N: usize, A> Clone for Tensor<T, N, A>
where
    T: arrow_buffer::ArrowNativeType + std::panic::RefUnwindSafe,
    A: TensorAllocator,
{
    fn clone(&self) -> Self {
        // create a new tensor with uninitialized data
        let mut cloned_tensor =
            Self::new_uninitialized(self.shape, self.storage.alloc().clone()).unwrap();

        // copy the data
        for (a, b) in cloned_tensor
            .as_slice_mut()
            .iter_mut()
            .zip(self.as_slice().iter())
        {
            *a = *b;
        }

        cloned_tensor
    }
}

#[cfg(test)]
mod tests {
    use crate::tensor::allocator::CpuAllocator;
    use crate::tensor::{Tensor, TensorError};

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
        assert_eq!(*t.get([0])?, 1);
        assert_eq!(*t.get([1])?, 2);
        assert_eq!(*t.get([2])?, 3);
        assert_eq!(*t.get([3])?, 4);
        assert!(t.get([4]).is_err());
        Ok(())
    }

    #[test]
    fn get_2d() -> Result<(), TensorError> {
        let data: Vec<u8> = vec![1, 2, 3, 4];
        let t = Tensor::<u8, 2>::from_shape_vec([2, 2], data, CpuAllocator)?;
        assert_eq!(*t.get([0, 0])?, 1);
        assert_eq!(*t.get([0, 1])?, 2);
        assert_eq!(*t.get([1, 0])?, 3);
        assert_eq!(*t.get([1, 1])?, 4);
        assert!(t.get([2, 0]).is_err());
        Ok(())
    }

    #[test]
    fn get_3d() -> Result<(), TensorError> {
        let data: Vec<u8> = vec![1, 2, 3, 4, 5, 6];
        let t = Tensor::<u8, 3>::from_shape_vec([2, 1, 3], data, CpuAllocator)?;
        assert_eq!(*t.get([0, 0, 0])?, 1);
        assert_eq!(*t.get([0, 0, 1])?, 2);
        assert_eq!(*t.get([0, 0, 2])?, 3);
        assert_eq!(*t.get([1, 0, 0])?, 4);
        assert_eq!(*t.get([1, 0, 1])?, 5);
        assert_eq!(*t.get([1, 0, 2])?, 6);
        assert!(t.get([2, 0, 0]).is_err());
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
        let t3 = t1.add(&t2)?;
        assert_eq!(t3.as_slice(), vec![2, 4, 6, 8]);
        Ok(())
    }

    #[test]
    fn add_2d() -> Result<(), TensorError> {
        let data1: Vec<u8> = vec![1, 2, 3, 4];
        let t1 = Tensor::<u8, 2>::from_shape_vec([2, 2], data1, CpuAllocator)?;
        let data2: Vec<u8> = vec![1, 2, 3, 4];
        let t2 = Tensor::<u8, 2>::from_shape_vec([2, 2], data2, CpuAllocator)?;
        let t3 = t1.add(&t2)?;
        assert_eq!(t3.as_slice(), vec![2, 4, 6, 8]);
        Ok(())
    }

    #[test]
    fn add_3d() -> Result<(), TensorError> {
        let data1: Vec<u8> = vec![1, 2, 3, 4, 5, 6];
        let t1 = Tensor::<u8, 3>::from_shape_vec([2, 1, 3], data1, CpuAllocator)?;
        let data2: Vec<u8> = vec![1, 2, 3, 4, 5, 6];
        let t2 = Tensor::<u8, 3>::from_shape_vec([2, 1, 3], data2, CpuAllocator)?;
        let t3 = t1.add(&t2)?;
        assert_eq!(t3.as_slice(), vec![2, 4, 6, 8, 10, 12]);
        Ok(())
    }

    #[test]
    fn sub_1d() -> Result<(), TensorError> {
        let data1: Vec<u8> = vec![1, 2, 3, 4];
        let t1 = Tensor::<u8, 1>::from_shape_vec([4], data1, CpuAllocator)?;
        let data2: Vec<u8> = vec![1, 2, 3, 4];
        let t2 = Tensor::<u8, 1>::from_shape_vec([4], data2, CpuAllocator)?;
        let t3 = t1.sub(&t2)?;
        assert_eq!(t3.as_slice(), vec![0, 0, 0, 0]);
        Ok(())
    }

    #[test]
    fn sub_2d() -> Result<(), TensorError> {
        let data1: Vec<u8> = vec![1, 2, 3, 4];
        let t1 = Tensor::<u8, 2>::from_shape_vec([2, 2], data1, CpuAllocator)?;
        let data2: Vec<u8> = vec![1, 2, 3, 4];
        let t2 = Tensor::<u8, 2>::from_shape_vec([2, 2], data2, CpuAllocator)?;
        let t3 = t1.sub(&t2)?;
        assert_eq!(t3.as_slice(), vec![0, 0, 0, 0]);
        Ok(())
    }

    #[test]
    fn div_1d() -> Result<(), TensorError> {
        let data1: Vec<u8> = vec![1, 2, 3, 4];
        let t1 = Tensor::<u8, 1>::from_shape_vec([4], data1, CpuAllocator)?;
        let data2: Vec<u8> = vec![1, 2, 3, 4];
        let t2 = Tensor::<u8, 1>::from_shape_vec([4], data2, CpuAllocator)?;
        let t3 = t1.div(&t2)?;
        assert_eq!(t3.as_slice(), vec![1, 1, 1, 1]);
        Ok(())
    }

    #[test]
    fn div_2d() -> Result<(), TensorError> {
        let data1: Vec<u8> = vec![1, 2, 3, 4];
        let t1 = Tensor::<u8, 2>::from_shape_vec([2, 2], data1, CpuAllocator)?;
        let data2: Vec<u8> = vec![1, 2, 3, 4];
        let t2 = Tensor::<u8, 2>::from_shape_vec([2, 2], data2, CpuAllocator)?;
        let t3 = t1.div(&t2)?;
        assert_eq!(t3.as_slice(), vec![1, 1, 1, 1]);
        Ok(())
    }

    #[test]
    fn mul_1d() -> Result<(), TensorError> {
        let data1: Vec<u8> = vec![1, 2, 3, 4];
        let t1 = Tensor::<u8, 1>::from_shape_vec([4], data1, CpuAllocator)?;
        let data2: Vec<u8> = vec![1, 2, 3, 4];
        let t2 = Tensor::<u8, 1>::from_shape_vec([4], data2, CpuAllocator)?;
        let t3 = t1.mul(&t2)?;
        assert_eq!(t3.as_slice(), vec![1, 4, 9, 16]);
        Ok(())
    }

    #[test]
    fn mul_2d() -> Result<(), TensorError> {
        let data1: Vec<u8> = vec![1, 2, 3, 4];
        let t1 = Tensor::<u8, 2>::from_shape_vec([2, 2], data1, CpuAllocator)?;
        let data2: Vec<u8> = vec![1, 2, 3, 4];
        let t2 = Tensor::<u8, 2>::from_shape_vec([2, 2], data2, CpuAllocator)?;
        let t3 = t1.mul(&t2)?;
        assert_eq!(t3.as_slice(), vec![1, 4, 9, 16]);
        Ok(())
    }

    #[test]
    fn reshape_1d() -> Result<(), TensorError> {
        let data: Vec<u8> = vec![1, 2, 3, 4];
        let t = Tensor::<u8, 1>::from_shape_vec([4], data, CpuAllocator)?;
        let t2 = t.reshape([2, 2])?;
        assert_eq!(t2.shape, [2, 2]);
        assert_eq!(t2.as_slice(), vec![1, 2, 3, 4]);
        assert_eq!(t2.strides, [2, 1]);
        assert_eq!(t2.numel(), 4);
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
        Ok(())
    }

    #[test]
    fn reshape_get_1d() -> Result<(), TensorError> {
        let data: Vec<u8> = vec![1, 2, 3, 4];
        let t = Tensor::<u8, 1>::from_shape_vec([4], data, CpuAllocator)?;
        let t2 = t.reshape([2, 2])?;
        assert_eq!(*t2.get([0, 0])?, 1);
        assert_eq!(*t2.get([0, 1])?, 2);
        assert_eq!(*t2.get([1, 0])?, 3);
        assert_eq!(*t2.get([1, 1])?, 4);
        assert_eq!(t2.numel(), 4);
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
        let t2 = t.map(|x| *x + 1)?;
        assert_eq!(t2.as_slice(), vec![2, 3, 4, 5]);
        Ok(())
    }

    #[test]
    fn map_2d() -> Result<(), TensorError> {
        let data: Vec<u8> = vec![1, 2, 3, 4];
        let t = Tensor::<u8, 2>::from_shape_vec([2, 2], data, CpuAllocator)?;
        let t2 = t.map(|x| *x + 1)?;
        assert_eq!(t2.as_slice(), vec![2, 3, 4, 5]);
        Ok(())
    }

    #[test]
    fn from_shape_val_1d() -> Result<(), TensorError> {
        let t = Tensor::<u8, 1>::from_shape_val([4], 0, CpuAllocator)?;
        assert_eq!(t.as_slice(), vec![0, 0, 0, 0]);
        Ok(())
    }

    #[test]
    fn from_shape_val_2d() -> Result<(), TensorError> {
        let t = Tensor::<u8, 2>::from_shape_val([2, 2], 1, CpuAllocator)?;
        assert_eq!(t.as_slice(), vec![1, 1, 1, 1]);
        Ok(())
    }

    #[test]
    fn from_shape_val_3d() -> Result<(), TensorError> {
        let t = Tensor::<u8, 3>::from_shape_val([2, 1, 3], 2, CpuAllocator)?;
        assert_eq!(t.as_slice(), vec![2, 2, 2, 2, 2, 2]);
        Ok(())
    }

    #[test]
    fn cast_1d() -> Result<(), TensorError> {
        let data: Vec<u8> = vec![1, 2, 3, 4];
        let t = Tensor::<u8, 1>::from_shape_vec([4], data, CpuAllocator)?;
        let t2 = t.cast::<u16>()?;
        assert_eq!(t2.as_slice(), vec![1, 2, 3, 4]);
        Ok(())
    }

    #[test]
    fn cast_2d() -> Result<(), TensorError> {
        let data: Vec<u8> = vec![1, 2, 3, 4];
        let t = Tensor::<u8, 2>::from_shape_vec([2, 2], data, CpuAllocator)?;
        let t2 = t.cast::<u16>()?;
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
}
