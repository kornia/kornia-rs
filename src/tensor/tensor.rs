use thiserror::Error;

#[derive(Error, Debug)]
pub enum TensorError {
    #[error("The number of elements in the data does not match the shape of the tensor: {0}")]
    InvalidShape(usize),

    #[error("Index out of bounds. The index {0} is out of bounds.")]
    IndexOutOfBounds(usize),
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
/// # Attributes
///
/// * `shape` - The shape of the tensor.
/// * `data` - The data of the tensor.
/// * `strides` - The strides of the tensor data in memory.
///
/// # Example
///
/// ```
/// use kornia_rs::tensor::Tensor;
///
/// let data: Vec<u8> = vec![1, 2, 3, 4];
/// let t = Tensor::<u8, 2>::new([2, 2], data).unwrap();
/// assert_eq!(t.shape, [2, 2]);
pub struct Tensor<T, const N: usize> {
    pub data: Vec<T>,
    pub shape: [usize; N],
    pub strides: [usize; N],
}

/// Implementation of the Tensor struct.
impl<T, const N: usize> Tensor<T, N> {
    /// Creates a new `Tensor` with the given shape and data.
    ///
    /// # Arguments
    ///
    /// * `shape` - An array containing the shape of the tensor.
    /// * `data` - A vector containing the data of the tensor.
    ///
    /// # Returns
    ///
    /// A new `Tensor` instance.
    pub fn new(shape: [usize; N], data: Vec<T>) -> Result<Self, TensorError> {
        let numel = shape.iter().product::<usize>();
        if numel != data.len() {
            Err(TensorError::InvalidShape(numel))?;
        }
        let strides = get_strides_from_shape(shape);
        Ok(Tensor {
            shape,
            data,
            strides,
        })
    }

    /// Returns the number of elements in the tensor.
    pub fn numel(&self) -> usize {
        self.data.len()
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
    /// use kornia_rs::tensor::Tensor;
    ///
    /// let data: Vec<u8> = vec![1, 2, 3, 4];
    ///
    /// let t = Tensor::<u8, 2>::new([2, 2], data).unwrap();
    /// assert_eq!(*t.get_unchecked([0, 0]), 1);
    /// assert_eq!(*t.get_unchecked([0, 1]), 2);
    /// assert_eq!(*t.get_unchecked([1, 0]), 3);
    /// assert_eq!(*t.get_unchecked([1, 1]), 4);
    /// ```
    pub fn get_unchecked(&self, index: [usize; N]) -> &T {
        let offset = self.get_iter_offset(index);
        &self.data[offset]
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
    /// use kornia_rs::tensor::Tensor;
    ///
    /// let data: Vec<u8> = vec![1, 2, 3, 4];
    ///
    /// let t = Tensor::<u8, 2>::new([2, 2], data).unwrap();
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
        Ok(&self.data[offset])
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
    /// use kornia_rs::tensor::Tensor;
    ///
    /// let data: Vec<u8> = vec![1, 2, 3, 4];
    ///
    /// let t = Tensor::<u8, 1>::new([4], data).unwrap();
    /// let t2 = t.reshape([2, 2]).unwrap();
    /// assert_eq!(t2.shape, [2, 2]);
    /// assert_eq!(t2.data, vec![1, 2, 3, 4]);
    /// assert_eq!(t2.strides, [2, 1]);
    /// assert_eq!(t2.numel(), 4);
    /// ```
    pub fn reshape<const M: usize>(self, shape: [usize; M]) -> Result<Tensor<T, M>, TensorError> {
        let numel = shape.iter().product::<usize>();
        if numel != self.data.len() {
            Err(TensorError::InvalidShape(numel))?;
        }

        let strides = get_strides_from_shape(shape);

        Ok(Tensor {
            shape,
            data: self.data,
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
    /// use kornia_rs::tensor::Tensor;
    ///
    /// let data1: Vec<u8> = vec![1, 2, 3, 4];
    /// let t1 = Tensor::<u8, 1>::new([4], data1).unwrap();
    ///
    /// let data2: Vec<u8> = vec![1, 2, 3, 4];
    /// let t2 = Tensor::<u8, 1>::new([4], data2).unwrap();
    ///
    /// let t3 = t1.element_wise_op(&t2, |a, b| *a + *b);
    /// assert_eq!(t3.data, vec![2, 4, 6, 8]);
    ///
    /// let t4 = t1.element_wise_op(&t2, |a, b| *a - *b);
    /// assert_eq!(t4.data, vec![0, 0, 0, 0]);
    ///
    /// let t5 = t1.element_wise_op(&t2, |a, b| *a * *b);
    /// assert_eq!(t5.data, vec![1, 4, 9, 16]);
    ///
    /// let t6 = t1.element_wise_op(&t2, |a, b| *a / *b);
    /// assert_eq!(t6.data, vec![1, 1, 1, 1]);
    /// ```
    pub fn element_wise_op<F>(&self, other: &Tensor<T, N>, op: F) -> Tensor<T, N>
    where
        F: Fn(&T, &T) -> T,
    {
        let data: Vec<T> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| op(a, b))
            .collect();

        Tensor {
            shape: self.shape,
            data,
            strides: self.strides,
        }
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
    /// use kornia_rs::tensor::Tensor;
    ///
    /// let data1: Vec<u8> = vec![1, 2, 3, 4];
    /// let t1 = Tensor::<u8, 1>::new([4], data1).unwrap();
    ///
    /// let data2: Vec<u8> = vec![1, 2, 3, 4];
    /// let t2 = Tensor::<u8, 1>::new([4], data2).unwrap();
    ///
    /// let t3 = t1.add(&t2);
    /// assert_eq!(t3.data, vec![2, 4, 6, 8]);
    /// ```
    pub fn add(&self, other: &Tensor<T, N>) -> Tensor<T, N>
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
    /// use kornia_rs::tensor::Tensor;
    ///
    /// let data1: Vec<u8> = vec![1, 2, 3, 4];
    /// let t1 = Tensor::<u8, 1>::new([4], data1).unwrap();
    ///
    /// let data2: Vec<u8> = vec![1, 2, 3, 4];
    /// let t2 = Tensor::<u8, 1>::new([4], data2).unwrap();
    ///
    /// let t3 = t1.sub(&t2);
    /// assert_eq!(t3.data, vec![0, 0, 0, 0]);
    /// ```
    pub fn sub(&self, other: &Tensor<T, N>) -> Tensor<T, N>
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
    /// use kornia_rs::tensor::Tensor;
    ///
    /// let data1: Vec<u8> = vec![1, 2, 3, 4];
    /// let t1 = Tensor::<u8, 1>::new([4], data1).unwrap();
    ///
    /// let data2: Vec<u8> = vec![1, 2, 3, 4];
    /// let t2 = Tensor::<u8, 1>::new([4], data2).unwrap();
    ///
    /// let t3 = t1.mul(&t2);
    /// assert_eq!(t3.data, vec![1, 4, 9, 16]);
    /// ```
    pub fn mul(&self, other: &Tensor<T, N>) -> Tensor<T, N>
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
    /// use kornia_rs::tensor::Tensor;
    ///
    /// let data1: Vec<u8> = vec![1, 2, 3, 4];
    /// let t1 = Tensor::<u8, 1>::new([4], data1).unwrap();
    ///
    /// let data2: Vec<u8> = vec![1, 2, 3, 4];
    /// let t2 = Tensor::<u8, 1>::new([4], data2).unwrap();
    ///
    /// let t3 = t1.div(&t2);
    /// assert_eq!(t3.data, vec![1, 1, 1, 1]);
    /// ```
    pub fn div(&self, other: &Tensor<T, N>) -> Tensor<T, N>
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
    /// use kornia_rs::tensor::Tensor;
    ///
    /// let t = Tensor::<u8, 1>::zeros([4]);
    /// assert_eq!(t.data, vec![0, 0, 0, 0]);
    ///
    /// let t = Tensor::<u8, 2>::zeros([2, 2]);
    /// assert_eq!(t.data, vec![0, 0, 0, 0]);
    /// ```
    pub fn zeros(shape: [usize; N]) -> Self
    where
        T: Default + Copy,
    {
        let data = vec![T::default(); shape.iter().product::<usize>()];
        Tensor {
            shape,
            data,
            strides: get_strides_from_shape(shape),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::tensor::{Tensor, TensorError};

    #[test]
    fn constructor_1d() -> Result<(), TensorError> {
        let data: Vec<u8> = vec![1];
        let t = Tensor::<u8, 1>::new([1], data)?;
        assert_eq!(t.shape, [1]);
        assert_eq!(t.data, vec![1]);
        assert_eq!(t.strides, [1]);
        assert_eq!(t.numel(), 1);
        Ok(())
    }

    #[test]
    fn constructor_2d() -> Result<(), TensorError> {
        let data: Vec<u8> = vec![1, 2];
        let t = Tensor::<u8, 2>::new([1, 2], data)?;
        assert_eq!(t.shape, [1, 2]);
        assert_eq!(t.data, vec![1, 2]);
        assert_eq!(t.strides, [2, 1]);
        assert_eq!(t.numel(), 2);
        Ok(())
    }

    #[test]
    fn get_1d() -> Result<(), TensorError> {
        let data: Vec<u8> = vec![1, 2, 3, 4];
        let t = Tensor::<u8, 1>::new([4], data)?;
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
        let t = Tensor::<u8, 2>::new([2, 2], data)?;
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
        let t = Tensor::<u8, 3>::new([2, 1, 3], data)?;
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
        let t = Tensor::<u8, 1>::new([4], data)?;
        assert_eq!(*t.get_unchecked([0]), 1);
        assert_eq!(*t.get_unchecked([1]), 2);
        assert_eq!(*t.get_unchecked([2]), 3);
        assert_eq!(*t.get_unchecked([3]), 4);
        Ok(())
    }

    #[test]
    fn get_checked_2d() -> Result<(), TensorError> {
        let data: Vec<u8> = vec![1, 2, 3, 4];
        let t = Tensor::<u8, 2>::new([2, 2], data)?;
        assert_eq!(*t.get_unchecked([0, 0]), 1);
        assert_eq!(*t.get_unchecked([0, 1]), 2);
        assert_eq!(*t.get_unchecked([1, 0]), 3);
        assert_eq!(*t.get_unchecked([1, 1]), 4);
        Ok(())
    }

    #[test]
    fn add_1d() -> Result<(), TensorError> {
        let data1: Vec<u8> = vec![1, 2, 3, 4];
        let t1 = Tensor::<u8, 1>::new([4], data1)?;
        let data2: Vec<u8> = vec![1, 2, 3, 4];
        let t2 = Tensor::<u8, 1>::new([4], data2)?;
        let t3 = t1.add(&t2);
        assert_eq!(t3.data, vec![2, 4, 6, 8]);
        Ok(())
    }

    #[test]
    fn add_2d() -> Result<(), TensorError> {
        let data1: Vec<u8> = vec![1, 2, 3, 4];
        let t1 = Tensor::<u8, 2>::new([2, 2], data1)?;
        let data2: Vec<u8> = vec![1, 2, 3, 4];
        let t2 = Tensor::<u8, 2>::new([2, 2], data2)?;
        let t3 = t1.add(&t2);
        assert_eq!(t3.data, vec![2, 4, 6, 8]);
        Ok(())
    }

    #[test]
    fn add_3d() -> Result<(), TensorError> {
        let data1: Vec<u8> = vec![1, 2, 3, 4, 5, 6];
        let t1 = Tensor::<u8, 3>::new([2, 1, 3], data1)?;
        let data2: Vec<u8> = vec![1, 2, 3, 4, 5, 6];
        let t2 = Tensor::<u8, 3>::new([2, 1, 3], data2)?;
        let t3 = t1.add(&t2);
        assert_eq!(t3.data, vec![2, 4, 6, 8, 10, 12]);
        Ok(())
    }

    #[test]
    fn sub_1d() -> Result<(), TensorError> {
        let data1: Vec<u8> = vec![1, 2, 3, 4];
        let t1 = Tensor::<u8, 1>::new([4], data1)?;
        let data2: Vec<u8> = vec![1, 2, 3, 4];
        let t2 = Tensor::<u8, 1>::new([4], data2)?;
        let t3 = t1.sub(&t2);
        assert_eq!(t3.data, vec![0, 0, 0, 0]);
        Ok(())
    }

    #[test]
    fn sub_2d() -> Result<(), TensorError> {
        let data1: Vec<u8> = vec![1, 2, 3, 4];
        let t1 = Tensor::<u8, 2>::new([2, 2], data1)?;
        let data2: Vec<u8> = vec![1, 2, 3, 4];
        let t2 = Tensor::<u8, 2>::new([2, 2], data2)?;
        let t3 = t1.sub(&t2);
        assert_eq!(t3.data, vec![0, 0, 0, 0]);
        Ok(())
    }

    #[test]
    fn div_1d() -> Result<(), TensorError> {
        let data1: Vec<u8> = vec![1, 2, 3, 4];
        let t1 = Tensor::<u8, 1>::new([4], data1)?;
        let data2: Vec<u8> = vec![1, 2, 3, 4];
        let t2 = Tensor::<u8, 1>::new([4], data2)?;
        let t3 = t1.div(&t2);
        assert_eq!(t3.data, vec![1, 1, 1, 1]);
        Ok(())
    }

    #[test]
    fn div_2d() -> Result<(), TensorError> {
        let data1: Vec<u8> = vec![1, 2, 3, 4];
        let t1 = Tensor::<u8, 2>::new([2, 2], data1)?;
        let data2: Vec<u8> = vec![1, 2, 3, 4];
        let t2 = Tensor::<u8, 2>::new([2, 2], data2)?;
        let t3 = t1.div(&t2);
        assert_eq!(t3.data, vec![1, 1, 1, 1]);
        Ok(())
    }

    #[test]
    fn mul_1d() -> Result<(), TensorError> {
        let data1: Vec<u8> = vec![1, 2, 3, 4];
        let t1 = Tensor::<u8, 1>::new([4], data1)?;
        let data2: Vec<u8> = vec![1, 2, 3, 4];
        let t2 = Tensor::<u8, 1>::new([4], data2)?;
        let t3 = t1.mul(&t2);
        assert_eq!(t3.data, vec![1, 4, 9, 16]);
        Ok(())
    }

    #[test]
    fn mul_2d() -> Result<(), TensorError> {
        let data1: Vec<u8> = vec![1, 2, 3, 4];
        let t1 = Tensor::<u8, 2>::new([2, 2], data1)?;
        let data2: Vec<u8> = vec![1, 2, 3, 4];
        let t2 = Tensor::<u8, 2>::new([2, 2], data2)?;
        let t3 = t1.mul(&t2);
        assert_eq!(t3.data, vec![1, 4, 9, 16]);
        Ok(())
    }

    #[test]
    fn reshape_1d() -> Result<(), TensorError> {
        let data: Vec<u8> = vec![1, 2, 3, 4];
        let t = Tensor::<u8, 1>::new([4], data)?;
        let t2 = t.reshape([2, 2])?;
        assert_eq!(t2.shape, [2, 2]);
        assert_eq!(t2.data, vec![1, 2, 3, 4]);
        assert_eq!(t2.strides, [2, 1]);
        assert_eq!(t2.numel(), 4);
        Ok(())
    }

    #[test]
    fn reshape_2d() -> Result<(), TensorError> {
        let data: Vec<u8> = vec![1, 2, 3, 4];
        let t = Tensor::<u8, 2>::new([2, 2], data)?;
        let t2 = t.reshape([4])?;
        assert_eq!(t2.shape, [4]);
        assert_eq!(t2.data, vec![1, 2, 3, 4]);
        assert_eq!(t2.strides, [1]);
        assert_eq!(t2.numel(), 4);
        Ok(())
    }

    #[test]
    fn reshape_get_1d() -> Result<(), TensorError> {
        let data: Vec<u8> = vec![1, 2, 3, 4];
        let t = Tensor::<u8, 1>::new([4], data)?;
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
        let t = Tensor::<u8, 1>::zeros([4]);
        assert_eq!(t.data, vec![0, 0, 0, 0]);
        Ok(())
    }

    #[test]
    fn zeros_2d() -> Result<(), TensorError> {
        let t = Tensor::<u8, 2>::zeros([2, 2]);
        assert_eq!(t.data, vec![0, 0, 0, 0]);
        Ok(())
    }
}
