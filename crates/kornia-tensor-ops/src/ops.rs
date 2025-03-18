use kernels::ops::{cosine_similarity_float_kernel, dot_product1_kernel};
use kornia_tensor::{storage::TensorStorage, CpuAllocator, Tensor, TensorAllocator, TensorError};
use num_traits::{Float, Zero};

use crate::error::TensorOpsError;

/// Compute the sum of the elements in the tensor along dimension `dim`
///
/// # Arguments
///
/// * `tensor` - The tensor to sum the elements of.
/// * `dim` - The index of the dimension/axis to perform the sum operation over.
///
/// # Returns
///
/// A new `Tensor` containing the element sums.
///
/// # Errors
///
/// If the requested dimension is greater than the number of axes of the tensor, an error is returned.
///
/// # Example
///
/// ```
/// use kornia_tensor::{Tensor, CpuAllocator};
/// use kornia_tensor_ops::TensorOps;
///
/// let data: [u8; 6] = [1, 1, 1, 1, 1, 1];
/// let t = Tensor::<u8, 2, CpuAllocator>::from_shape_slice([2, 3], &data, CpuAllocator).unwrap();
/// let agg = Tensor::sum_elements(&t, 1).unwrap();
/// assert_eq!(agg.shape, [2, 1]);
/// assert_eq!(agg.as_slice(), [3, 3]);
/// ```
fn sum_elements<T, const N: usize, A>(
    tensor: &Tensor<T, N, A>,
    dim: usize,
) -> Result<Tensor<T, N, A>, TensorOpsError>
where
    T: Zero + Clone + std::ops::Add<Output = T>,
    A: TensorAllocator + Clone + 'static,
{
    if dim >= N {
        return Err(TensorOpsError::DimOutOfBounds(dim, N - 1));
    }

    let mut out_shape = tensor.shape;
    out_shape[dim] = 1;

    let mut out_strides = tensor.strides;
    if dim > 0 {
        out_strides
            .iter_mut()
            .take(dim)
            .for_each(|s| *s /= tensor.shape[dim]);
    }

    let numel: usize = out_shape.iter().product();
    let mut data = vec![T::zero(); numel];

    for (i, v) in tensor.as_slice().iter().enumerate() {
        let mut out_index = tensor.get_index_unchecked(i);
        out_index[dim] = 0;
        let out_offset = out_index
            .iter()
            .zip(out_strides.iter())
            .fold(0, |acc, (&idx, &stride)| acc + idx * stride);
        let agg = unsafe { data.get_unchecked_mut(out_offset) };
        *agg = agg.clone() + v.clone();
    }

    let storage = TensorStorage::from_vec(data, tensor.storage.alloc().clone());

    Ok(Tensor {
        storage,
        shape: out_shape,
        strides: out_strides,
    })
}

/// Multiply the pixel data by a scalar.
///
/// # Arguments
///
/// * `n` - The scalar to multiply the pixel data by.
///
/// # Returns
///
/// A new image with the pixel data multiplied by the scalar.
fn mul_scalar<T, const N: usize, A>(tensor: &Tensor<T, N, A>, n: T) -> Tensor<T, N, A>
where
    T: Float + Clone,
    A: TensorAllocator + Clone + 'static,
{
    tensor.map(|&x| x * n)
}

/// Raise the pixel data to the power of a float.
///
/// # Arguments
///
/// * `n` - The power to raise the pixel data to.
///
/// # Returns
///
/// A new image with the pixel data raised to the power.
fn powf<T, const N: usize, A>(tensor: &Tensor<T, N, A>, n: T) -> Tensor<T, N, A>
where
    T: Float + Clone,
    A: TensorAllocator + Clone + 'static,
{
    tensor.map(|x| x.powf(n))
}
/// Perform an element-wise minimum operation on two tensors.
///
/// # Arguments
///
/// * `other` - The other tensor to compare.
///
/// # Returns
///
/// A new `Tensor` instance.
fn min<T, const N: usize>(
    tensor: &Tensor<T, N, CpuAllocator>,
    other: &Tensor<T, N, CpuAllocator>,
) -> Result<Tensor<T, N, CpuAllocator>, TensorOpsError>
where
    T: PartialOrd + Clone,
{
    tensor
        .element_wise_op(other, |a, b| if a < b { a.clone() } else { b.clone() })
        .map_err(|_| TensorOpsError::ShapeMismatch(tensor.shape.to_vec(), other.shape.to_vec()))
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
fn powi<T, const N: usize, A>(tensor: &Tensor<T, N, A>, n: i32) -> Tensor<T, N, A>
where
    T: Float + Clone,
    A: TensorAllocator + Clone + 'static,
{
    tensor.map(|x| x.powi(n))
}

/// Compute absolute value of the pixel data.
///
/// # Returns
///
/// A new image with the pixel data absolute value.
fn abs<T, const N: usize, A>(tensor: &Tensor<T, N, A>) -> Tensor<T, N, A>
where
    T: Float + Clone,
    A: TensorAllocator + Clone + 'static,
{
    tensor.map(|x| x.abs())
}

/// Compute the mean of the pixel data.
///
/// # Returns
///
/// The mean of the pixel data.
fn mean<T, const N: usize, A>(tensor: &Tensor<T, N, A>) -> Result<T, TensorError>
where
    T: Float + Clone,
    A: TensorAllocator + Clone + 'static,
{
    let data_acc = tensor.as_slice().iter().fold(T::zero(), |acc, &x| acc + x);
    let mean = data_acc / T::from(tensor.as_slice().len()).ok_or(TensorError::CastError)?;

    Ok(mean)
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
/// use kornia_tensor::{Tensor, CpuAllocator};
/// use kornia_tensor_ops::TensorOps;
///
/// let data1: Vec<u8> = vec![1, 2, 3, 4];
/// let t1 = Tensor::<u8, 1, CpuAllocator>::from_shape_vec([4], data1, CpuAllocator).unwrap();
///
/// let data2: Vec<u8> = vec![1, 2, 3, 4];
/// let t2 = Tensor::<u8, 1, CpuAllocator>::from_shape_vec([4], data2, CpuAllocator).unwrap();
///
/// let t3 = t1.add(&t2).unwrap();
/// assert_eq!(t3.as_slice(), vec![2, 4, 6, 8]);
/// ```
fn add<T, const N: usize>(
    tensor: &Tensor<T, N, CpuAllocator>,
    other: &Tensor<T, N, CpuAllocator>,
) -> Result<Tensor<T, N, CpuAllocator>, TensorOpsError>
where
    T: std::ops::Add<Output = T> + Clone,
{
    tensor
        .element_wise_op(other, |a, b| a.clone() + b.clone())
        .map_err(|_| TensorOpsError::ShapeMismatch(tensor.shape.to_vec(), other.shape.to_vec()))
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
/// use kornia_tensor::{Tensor, CpuAllocator};
/// use kornia_tensor_ops::TensorOps;
///
/// let data1: Vec<u8> = vec![1, 2, 3, 4];
/// let t1 = Tensor::<u8, 1, CpuAllocator>::from_shape_vec([4], data1, CpuAllocator).unwrap();
///
/// let data2: Vec<u8> = vec![1, 2, 3, 4];
/// let t2 = Tensor::<u8, 1, CpuAllocator>::from_shape_vec([4], data2, CpuAllocator).unwrap();
///
/// let t3 = t1.sub(&t2).unwrap();
/// assert_eq!(t3.as_slice(), vec![0, 0, 0, 0]);
/// ```
fn sub<T, const N: usize>(
    tensor: &Tensor<T, N, CpuAllocator>,
    other: &Tensor<T, N, CpuAllocator>,
) -> Result<Tensor<T, N, CpuAllocator>, TensorOpsError>
where
    T: std::ops::Sub<Output = T> + Clone,
{
    tensor
        .element_wise_op(other, |a, b| a.clone() - b.clone())
        .map_err(|_| TensorOpsError::ShapeMismatch(tensor.shape.to_vec(), other.shape.to_vec()))
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
/// use kornia_tensor::{Tensor, CpuAllocator};
/// use kornia_tensor_ops::TensorOps;
///
/// let data1: Vec<u8> = vec![1, 2, 3, 4];
/// let t1 = Tensor::<u8, 1, CpuAllocator>::from_shape_vec([4], data1, CpuAllocator).unwrap();
///
/// let data2: Vec<u8> = vec![1, 2, 3, 4];
/// let t2 = Tensor::<u8, 1, CpuAllocator>::from_shape_vec([4], data2, CpuAllocator).unwrap();
///
/// let t3 = t1.mul(&t2).unwrap();
/// assert_eq!(t3.as_slice(), vec![1, 4, 9, 16]);
/// ```
fn mul<T, const N: usize>(
    tensor: &Tensor<T, N, CpuAllocator>,
    other: &Tensor<T, N, CpuAllocator>,
) -> Result<Tensor<T, N, CpuAllocator>, TensorOpsError>
where
    T: std::ops::Mul<Output = T> + Clone,
{
    tensor
        .element_wise_op(other, |a, b| a.clone() * b.clone())
        .map_err(|_| TensorOpsError::ShapeMismatch(tensor.shape.to_vec(), other.shape.to_vec()))
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
/// use kornia_tensor::{Tensor, CpuAllocator};
/// use kornia_tensor_ops::TensorOps;
///
/// let data1: Vec<u8> = vec![1, 2, 3, 4];
/// let t1 = Tensor::<u8, 1, CpuAllocator>::from_shape_vec([4], data1, CpuAllocator).unwrap();
///
/// let data2: Vec<u8> = vec![1, 2, 3, 4];
/// let t2 = Tensor::<u8, 1, CpuAllocator>::from_shape_vec([4], data2, CpuAllocator).unwrap();
///
/// let t3 = t1.div(&t2).unwrap();
/// assert_eq!(t3.as_slice(), vec![1, 1, 1, 1]);
/// ```
fn div<T, const N: usize>(
    tensor: &Tensor<T, N, CpuAllocator>,
    other: &Tensor<T, N, CpuAllocator>,
) -> Result<Tensor<T, N, CpuAllocator>, TensorOpsError>
where
    T: std::ops::Div<Output = T> + Clone,
{
    tensor
        .element_wise_op(other, |a, b| a.clone() / b.clone())
        .map_err(|_| TensorOpsError::ShapeMismatch(tensor.shape.to_vec(), other.shape.to_vec()))
}
/// Compute the dot product between two 1D tensors
///
/// # Arguments
///
/// * `a` - First tensor
/// * `b` - Second tensor
///
/// # Returns
///
/// A scalar value representing the dot product of the two tensors.
///
/// # Errors
///
/// If the shapes of the tensors don't match, an error is returned.
///
/// # Example
///
/// ```
/// use kornia_tensor::{Tensor, CpuAllocator};
/// use kornia_tensor_ops::TensorOps;
///
/// let a = Tensor::<i32, 1, CpuAllocator>::from_shape_slice([3], &[1, 2, 3], CpuAllocator).unwrap();
/// let b = Tensor::<i32, 1, CpuAllocator>::from_shape_slice([3], &[4, 5, 6], CpuAllocator).unwrap();
/// let result = Tensor::<i32,1,CpuAllocator>::dot_product1(&a, &b).unwrap();
/// assert_eq!(result, 32); // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
/// ```
fn dot_product1<T, A>(a: &Tensor<T, 1, A>, b: &Tensor<T, 1, A>) -> Result<T, TensorOpsError>
where
    T: Zero + Clone + std::ops::Add<Output = T> + std::ops::Mul<Output = T> + Copy,
    A: TensorAllocator + Clone + 'static,
{
    if a.shape != b.shape {
        return Err(TensorOpsError::ShapeMismatch(
            a.shape.to_vec(),
            b.shape.to_vec(),
        ));
    }

    dot_product1_kernel(a.as_slice(), b.as_slice()).map_err(|e| e.into())
}

/// Compute the cosine similarity between two tensors with optimized computation
///
/// This version computes the dot product and magnitudes in a single pass through the data.
///
/// # Arguments
///
/// * `a` - First tensor
/// * `b` - Second tensor
///
/// # Returns
///
/// A scalar value representing the cosine similarity between the two tensors.
///
/// # Errors
///
/// If the shapes of the tensors don't match, an error is returned.
///
/// Example:
/// ```
/// use kornia_tensor::{Tensor, CpuAllocator};
/// use kornia_tensor_ops::TensorOps;
///
/// let a = Tensor::<f32, 1, CpuAllocator>::from_shape_slice([3], &[1.0, 2.0, 3.0], CpuAllocator).unwrap();
/// let b = Tensor::<f32, 1, CpuAllocator>::from_shape_slice([3], &[2.0, 4.0, 6.0], CpuAllocator).unwrap();
/// let result = Tensor::cosine_similarity(&a, &b).unwrap();
/// assert!((result - 1.0).abs() < 1e-6);
/// ```
fn cosine_similarity<T, const N: usize, A>(
    a: &Tensor<T, N, A>,
    b: &Tensor<T, N, A>,
) -> Result<T, TensorOpsError>
where
    T: num_traits::Float,
    A: TensorAllocator + Clone + 'static,
{
    if a.shape != b.shape {
        return Err(TensorOpsError::ShapeMismatch(
            a.shape.to_vec(),
            b.shape.to_vec(),
        ));
    }

    cosine_similarity_float_kernel(a.as_slice(), b.as_slice()).map_err(|e| e.into())
}

/// Compute the cosine distance between two tensors with optimized computation
///
/// This version computes the dot product and magnitudes in a single pass through the data.
///
/// # Arguments
///
/// * `a` - First tensor
/// * `b` - Second tensor
///
/// # Returns
///
/// A scalar value representing the cosine distance between the two tensors.
///
/// # Errors
///
/// If the shapes of the tensors don't match, an error is returned.
///
/// # Example
///
/// ```
/// use kornia_tensor::{Tensor, CpuAllocator};
/// use kornia_tensor_ops::TensorOps;
///
/// let a = Tensor::<f32, 1, CpuAllocator>::from_shape_slice([3], &[1.0, 2.0, 3.0], CpuAllocator).unwrap();
/// let b = Tensor::<f32, 1, CpuAllocator>::from_shape_slice([3], &[2.0, 4.0, 6.0], CpuAllocator).unwrap();
/// let result = Tensor::<f32,1,CpuAllocator>::cosine_distance(&a, &b).unwrap();
/// assert!(result.abs() < 1e-6);
/// ```
fn cosine_distance<T, A>(a: &Tensor<T, 1, A>, b: &Tensor<T, 1, A>) -> Result<T, TensorOpsError>
where
    T: num_traits::Float,
    A: TensorAllocator + Clone + 'static,
{
    let similarity = cosine_similarity(a, b)?;
    Ok(T::one() - similarity)
}

/// Trait providing tensor operations for CPU-based tensors.
///
/// This trait defines a collection of mathematical operations that can be performed on tensors.
/// It serves as an interface for implementing common tensor operations such as element-wise
/// operations (addition, multiplication, etc.), reductions (sum, mean), and transformations.
///
/// The operations exposed by this trait are implemented as static methods that take a reference
/// to the tensor as their first argument, following a functional programming style.
///
/// # Type Parameters
///
/// * `T` - The data type of the tensor elements.
/// * `N` - The number of dimensions (rank) of the tensor as a const generic.
///
/// # Implementation
///
/// This trait is implemented for `Tensor<T, N, CpuAllocator>` where `T` implements `Float`,
/// meaning these operations are available for floating-point tensors on CPU.
///
/// # Examples
///
/// ```
/// use kornia_tensor::{Tensor, CpuAllocator};
/// use kornia_tensor_ops::ops::TensorOps;
///
/// // Create a tensor
/// let data = vec![1.0, 2.0, 3.0, 4.0];
/// let t = Tensor::<f32, 2, CpuAllocator>::from_shape_vec([2, 2], data, CpuAllocator).unwrap();
///
/// // Use operations through the trait
/// let scaled = t.mul_scalar(2.0);
/// let abs_val = t.abs();
/// let mean_val = t.mean().unwrap();
/// ```
pub trait TensorOps<T, const N: usize> {
    /// Compute the sum of the elements in the tensor along dimension `dim`
    fn sum_elements(
        tensor: &Tensor<T, N, CpuAllocator>,
        dim: usize,
    ) -> Result<Tensor<T, N, CpuAllocator>, TensorOpsError>
    where
        T: Zero + Clone + std::ops::Add<Output = T>;

    /// Multiply the pixel data by a scalar.
    fn mul_scalar(&self, n: T) -> Tensor<T, N, CpuAllocator>
    where
        T: Float + Clone;

    /// Raise the pixel data to the power of a float.
    fn powf(&self, n: T) -> Tensor<T, N, CpuAllocator>
    where
        T: Float + Clone;
    /// Perform an element-wise minimum operation on two tensors.
    fn min(
        &self,
        other: &Tensor<T, N, CpuAllocator>,
    ) -> Result<Tensor<T, N, CpuAllocator>, TensorOpsError>
    where
        T: PartialOrd + Clone;

    /// Apply the power function to the pixel data.
    fn powi(&self, n: i32) -> Tensor<T, N, CpuAllocator>
    where
        T: Float + Clone;

    /// Compute absolute value of the pixel data.
    fn abs(&self) -> Tensor<T, N, CpuAllocator>
    where
        T: Float + Clone;

    /// Compute the mean of the pixel data.
    fn mean(&self) -> Result<T, TensorError>
    where
        T: Float + Clone;

    /// Perform an element-wise addition on two tensors.
    fn add(
        &self,
        other: &Tensor<T, N, CpuAllocator>,
    ) -> Result<Tensor<T, N, CpuAllocator>, TensorOpsError>
    where
        T: std::ops::Add<Output = T> + Clone;

    /// Perform an element-wise subtraction on two tensors.
    fn sub(
        &self,
        other: &Tensor<T, N, CpuAllocator>,
    ) -> Result<Tensor<T, N, CpuAllocator>, TensorOpsError>
    where
        T: std::ops::Sub<Output = T> + Clone;

    /// Perform an element-wise division on two tensors.
    fn div(
        &self,
        other: &Tensor<T, N, CpuAllocator>,
    ) -> Result<Tensor<T, N, CpuAllocator>, TensorOpsError>
    where
        T: std::ops::Div<Output = T> + Clone;

    /// Perform an element-wise multiplication on two tensors.
    fn mul(
        &self,
        other: &Tensor<T, N, CpuAllocator>,
    ) -> Result<Tensor<T, N, CpuAllocator>, TensorOpsError>
    where
        T: std::ops::Mul<Output = T> + Clone;

    /// Compute the dot product between two 1D tensors
    fn dot_product1(
        a: &Tensor<T, 1, CpuAllocator>,
        b: &Tensor<T, 1, CpuAllocator>,
    ) -> Result<T, TensorOpsError>
    where
        T: Zero + Clone + std::ops::Add<Output = T> + std::ops::Mul<Output = T> + Copy;

    /// Compute the cosine similarity between two tensors
    fn cosine_similarity(
        a: &Tensor<T, N, CpuAllocator>,
        b: &Tensor<T, N, CpuAllocator>,
    ) -> Result<T, TensorOpsError>
    where
        T: Float;

    /// Compute the cosine distance between two tensors
    fn cosine_distance(
        a: &Tensor<T, 1, CpuAllocator>,
        b: &Tensor<T, 1, CpuAllocator>,
    ) -> Result<T, TensorOpsError>
    where
        T: Float;
}
impl<T, const N: usize> TensorOps<T, N> for Tensor<T, N, CpuAllocator> {
    fn sum_elements(
        tensor: &Tensor<T, N, CpuAllocator>,
        dim: usize,
    ) -> Result<Tensor<T, N, CpuAllocator>, TensorOpsError>
    where
        T: Zero + Clone + std::ops::Add<Output = T>,
    {
        sum_elements(tensor, dim)
    }

    fn mul_scalar(&self, n: T) -> Tensor<T, N, CpuAllocator>
    where
        T: Float + Clone,
    {
        mul_scalar(self, n)
    }

    fn powf(&self, n: T) -> Tensor<T, N, CpuAllocator>
    where
        T: Float + Clone,
    {
        powf(self, n)
    }

    fn min(
        &self,
        other: &Tensor<T, N, CpuAllocator>,
    ) -> Result<Tensor<T, N, CpuAllocator>, TensorOpsError>
    where
        T: PartialOrd + Clone,
    {
        min(self, other)
    }

    fn powi(&self, n: i32) -> Tensor<T, N, CpuAllocator>
    where
        T: Float + Clone,
    {
        powi(self, n)
    }

    fn abs(&self) -> Tensor<T, N, CpuAllocator>
    where
        T: Float + Clone,
    {
        abs(self)
    }

    fn mean(&self) -> Result<T, TensorError>
    where
        T: Float + Clone,
    {
        mean(self)
    }

    fn add(
        &self,
        other: &Tensor<T, N, CpuAllocator>,
    ) -> Result<Tensor<T, N, CpuAllocator>, TensorOpsError>
    where
        T: std::ops::Add<Output = T> + Clone,
    {
        add(self, other)
    }

    fn sub(
        &self,
        other: &Tensor<T, N, CpuAllocator>,
    ) -> Result<Tensor<T, N, CpuAllocator>, TensorOpsError>
    where
        T: std::ops::Sub<Output = T> + Clone,
    {
        sub(self, other)
    }

    fn div(
        &self,
        other: &Tensor<T, N, CpuAllocator>,
    ) -> Result<Tensor<T, N, CpuAllocator>, TensorOpsError>
    where
        T: std::ops::Div<Output = T> + Clone,
    {
        div(self, other)
    }

    fn mul(
        &self,
        other: &Tensor<T, N, CpuAllocator>,
    ) -> Result<Tensor<T, N, CpuAllocator>, TensorOpsError>
    where
        T: std::ops::Mul<Output = T> + Clone,
    {
        mul(self, other)
    }

    fn dot_product1(
        a: &Tensor<T, 1, CpuAllocator>,
        b: &Tensor<T, 1, CpuAllocator>,
    ) -> Result<T, TensorOpsError>
    where
        T: Zero + Clone + std::ops::Add<Output = T> + std::ops::Mul<Output = T> + Copy,
    {
        dot_product1(a, b)
    }

    fn cosine_similarity(
        a: &Tensor<T, N, CpuAllocator>,
        b: &Tensor<T, N, CpuAllocator>,
    ) -> Result<T, TensorOpsError>
    where
        T: Float,
    {
        cosine_similarity(a, b)
    }

    fn cosine_distance(
        a: &Tensor<T, 1, CpuAllocator>,
        b: &Tensor<T, 1, CpuAllocator>,
    ) -> Result<T, TensorOpsError>
    where
        T: Float,
    {
        cosine_distance(a, b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sum_dim_oob() -> Result<(), TensorError> {
        let data: [u8; 4] = [2, 2, 2, 2];
        let t = Tensor::<u8, 2, CpuAllocator>::from_shape_slice([2, 2], &data, CpuAllocator)?;
        let res = sum_elements(&t, 2);
        assert!(res.is_err_and(|e| e == TensorOpsError::DimOutOfBounds(2, 1)));
        Ok(())
    }

    #[test]
    fn test_dot_product_shape_mismatch() {
        let a = Tensor::<i32, 1, CpuAllocator>::from_shape_slice([3], &[1, 2, 3], CpuAllocator)
            .unwrap();
        let b = Tensor::<i32, 1, CpuAllocator>::from_shape_slice([4], &[4, 5, 6, 7], CpuAllocator)
            .unwrap();
        let result = dot_product1(&a, &b);
        assert!(result.is_err());

        if let Err(TensorOpsError::ShapeMismatch(shape_a, shape_b)) = &result {
            assert_eq!(shape_a, &vec![3]);
            assert_eq!(shape_b, &vec![4]);
        } else {
            panic!("Expected ShapeMismatch error");
        }
    }

    #[test]
    fn test_sum_1d() -> Result<(), TensorError> {
        let data: [u8; 4] = [1, 1, 1, 1];
        let t = Tensor::<u8, 1, CpuAllocator>::from_shape_slice([4], &data, CpuAllocator)?;
        let res = sum_elements(&t, 0);

        assert!(res.is_ok_and(|v| v.as_slice() == [4]));
        Ok(())
    }

    #[test]
    fn test_sum_2d() -> Result<(), TensorOpsError> {
        let data: [u8; 6] = [1, 2, 3, 4, 5, 6];
        let t = Tensor::<u8, 2, CpuAllocator>::from_shape_slice([2, 3], &data, CpuAllocator)?;
        let t_f32 = t.cast::<f32>();
        let t_i32 = t.cast::<i32>();

        let agg = sum_elements(&t, 1)?;
        assert_eq!(agg.shape, [2, 1]);
        assert_eq!(agg.as_slice(), [6, 15]);

        assert_eq!(sum_elements(&t_f32, 1)?.as_slice(), [6., 15.]);
        assert_eq!(sum_elements(&t_i32, 1)?.as_slice(), [6, 15]);

        let agg = sum_elements(&t, 0)?;
        assert_eq!(agg.shape, [1, 3]);
        assert_eq!(agg.as_slice(), [5, 7, 9]);

        assert_eq!(sum_elements(&t_f32, 0)?.as_slice(), [5., 7., 9.]);
        assert_eq!(sum_elements(&t_i32, 0)?.as_slice(), [5, 7, 9]);
        Ok(())
    }

    #[test]
    fn test_sum_3d() -> Result<(), TensorOpsError> {
        let data: [u8; 24] = [1; 24];
        let t = Tensor::<u8, 3, CpuAllocator>::from_shape_slice([2, 3, 4], &data, CpuAllocator)?;
        let t_f32 = t.cast::<f32>();
        let t_i32 = t.cast::<i32>();

        let agg = sum_elements(&t, 0)?;
        assert_eq!(agg.shape, [1, 3, 4]);
        assert_eq!(agg.as_slice(), [2; 12]);
        assert_eq!(sum_elements(&t_f32, 0)?.as_slice(), [2.; 12]);
        assert_eq!(sum_elements(&t_i32, 0)?.as_slice(), [2; 12]);

        let agg = sum_elements(&t, 1)?;
        assert_eq!(agg.shape, [2, 1, 4]);
        assert_eq!(agg.as_slice(), [3; 8]);
        assert_eq!(sum_elements(&t_f32, 1)?.as_slice(), [3.; 8]);
        assert_eq!(sum_elements(&t_i32, 1)?.as_slice(), [3; 8]);

        let agg = sum_elements(&t, 2)?;
        assert_eq!(agg.shape, [2, 3, 1]);
        assert_eq!(agg.as_slice(), [4; 6]);
        assert_eq!(sum_elements(&t_f32, 2)?.as_slice(), [4.; 6]);
        assert_eq!(sum_elements(&t_i32, 2)?.as_slice(), [4; 6]);

        Ok(())
    }

    #[test]
    fn test_mul_scalar_f32() -> Result<(), TensorError> {
        let data: [f32; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
        let t = Tensor::<f32, 1, CpuAllocator>::from_shape_slice([5], &data, CpuAllocator)?;
        let result = mul_scalar(&t, 2.0);
        assert_eq!(result.as_slice(), &[2.0, 4.0, 6.0, 8.0, 10.0]);
        Ok(())
    }

    #[test]
    fn test_powf_f32() -> Result<(), TensorError> {
        let data: [f32; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
        let t = Tensor::<f32, 1, CpuAllocator>::from_shape_slice([5], &data, CpuAllocator)?;
        let result = powf(&t, 2.0);
        let expected: Vec<f32> = data.iter().map(|&x| x * x).collect();
        assert_eq!(result.as_slice(), expected.as_slice());
        Ok(())
    }

    #[test]
    fn test_min_f32() -> Result<(), TensorError> {
        let data_a: [f32; 5] = [3.0, 1.0, 4.0, 1.0, 5.0];
        let data_b: [f32; 5] = [2.0, 7.0, 1.0, 8.0, 2.0];
        let tensor_a =
            Tensor::<f32, 1, CpuAllocator>::from_shape_slice([5], &data_a, CpuAllocator)?;
        let tensor_b =
            Tensor::<f32, 1, CpuAllocator>::from_shape_slice([5], &data_b, CpuAllocator)?;
        let result = min(&tensor_a, &tensor_b).unwrap();
        let expected = vec![2.0, 1.0, 1.0, 1.0, 2.0];
        assert_eq!(result.as_slice(), expected.as_slice());
        Ok(())
    }

    #[test]
    fn powi_and_abs() -> Result<(), TensorError> {
        let data: Vec<f32> = vec![-1.0, 2.0, -3.0, 4.0];
        let t = Tensor::<f32, 1, _>::from_shape_vec([4], data, CpuAllocator)?;

        let t_powi = powi(&t, 2);
        assert_eq!(t_powi.as_slice(), &[1.0, 4.0, 9.0, 16.0]);

        let t_abs = abs(&t);
        assert_eq!(t_abs.as_slice(), &[1.0, 2.0, 3.0, 4.0]);

        Ok(())
    }

    #[test]
    fn add_1d() -> Result<(), TensorOpsError> {
        let data1: Vec<u8> = vec![1, 2, 3, 4];
        let t1 = Tensor::<u8, 1, _>::from_shape_vec([4], data1, CpuAllocator)?;
        let data2: Vec<u8> = vec![1, 2, 3, 4];
        let t2 = Tensor::<u8, 1, _>::from_shape_vec([4], data2, CpuAllocator)?;
        let t3 = add(&t1, &t2)?;
        assert_eq!(t3.as_slice(), vec![2, 4, 6, 8]);
        Ok(())
    }

    #[test]
    fn add_2d() -> Result<(), TensorOpsError> {
        let data1: Vec<u8> = vec![1, 2, 3, 4];
        let t1 = Tensor::<u8, 2, _>::from_shape_vec([2, 2], data1, CpuAllocator)?;
        let data2: Vec<u8> = vec![1, 2, 3, 4];
        let t2 = Tensor::<u8, 2, _>::from_shape_vec([2, 2], data2, CpuAllocator)?;
        let t3 = add(&t1, &t2)?;
        assert_eq!(t3.as_slice(), vec![2, 4, 6, 8]);
        Ok(())
    }

    #[test]
    fn add_3d() -> Result<(), TensorOpsError> {
        let data1: Vec<u8> = vec![1, 2, 3, 4, 5, 6];
        let t1 = Tensor::<u8, 3, _>::from_shape_vec([2, 1, 3], data1, CpuAllocator)?;
        let data2: Vec<u8> = vec![1, 2, 3, 4, 5, 6];
        let t2 = Tensor::<u8, 3, _>::from_shape_vec([2, 1, 3], data2, CpuAllocator)?;
        let t3 = add(&t1, &t2)?;
        assert_eq!(t3.as_slice(), vec![2, 4, 6, 8, 10, 12]);
        Ok(())
    }

    #[test]
    fn sub_1d() -> Result<(), TensorOpsError> {
        let data1: Vec<u8> = vec![1, 2, 3, 4];
        let t1 = Tensor::<u8, 1, _>::from_shape_vec([4], data1, CpuAllocator)?;
        let data2: Vec<u8> = vec![1, 2, 3, 4];
        let t2 = Tensor::<u8, 1, _>::from_shape_vec([4], data2, CpuAllocator)?;
        let t3 = sub(&t1, &t2)?;
        assert_eq!(t3.as_slice(), vec![0, 0, 0, 0]);
        Ok(())
    }

    #[test]
    fn sub_2d() -> Result<(), TensorOpsError> {
        let data1: Vec<u8> = vec![1, 2, 3, 4];
        let t1 = Tensor::<u8, 2, _>::from_shape_vec([2, 2], data1, CpuAllocator)?;
        let data2: Vec<u8> = vec![1, 2, 3, 4];
        let t2 = Tensor::<u8, 2, _>::from_shape_vec([2, 2], data2, CpuAllocator)?;
        let t3 = sub(&t1, &t2)?;
        assert_eq!(t3.as_slice(), vec![0, 0, 0, 0]);
        Ok(())
    }

    #[test]
    fn div_1d() -> Result<(), TensorOpsError> {
        let data1: Vec<u8> = vec![1, 2, 3, 4];
        let t1 = Tensor::<u8, 1, _>::from_shape_vec([4], data1, CpuAllocator)?;
        let data2: Vec<u8> = vec![1, 2, 3, 4];
        let t2 = Tensor::<u8, 1, _>::from_shape_vec([4], data2, CpuAllocator)?;
        let t3 = div(&t1, &t2)?;
        assert_eq!(t3.as_slice(), vec![1, 1, 1, 1]);
        Ok(())
    }

    #[test]
    fn div_2d() -> Result<(), TensorOpsError> {
        let data1: Vec<u8> = vec![1, 2, 3, 4];
        let t1 = Tensor::<u8, 2, _>::from_shape_vec([2, 2], data1, CpuAllocator)?;
        let data2: Vec<u8> = vec![1, 2, 3, 4];
        let t2 = Tensor::<u8, 2, _>::from_shape_vec([2, 2], data2, CpuAllocator)?;
        let t3 = div(&t1, &t2)?;
        assert_eq!(t3.as_slice(), vec![1, 1, 1, 1]);
        Ok(())
    }

    #[test]
    fn mul_1d() -> Result<(), TensorOpsError> {
        let data1: Vec<u8> = vec![1, 2, 3, 4];
        let t1 = Tensor::<u8, 1, _>::from_shape_vec([4], data1, CpuAllocator)?;
        let data2: Vec<u8> = vec![1, 2, 3, 4];
        let t2 = Tensor::<u8, 1, _>::from_shape_vec([4], data2, CpuAllocator)?;
        let t3 = mul(&t1, &t2)?;
        assert_eq!(t3.as_slice(), vec![1, 4, 9, 16]);
        Ok(())
    }

    #[test]
    fn mul_2d() -> Result<(), TensorOpsError> {
        let data1: Vec<u8> = vec![1, 2, 3, 4];
        let t1 = Tensor::<u8, 2, _>::from_shape_vec([2, 2], data1, CpuAllocator)?;
        let data2: Vec<u8> = vec![1, 2, 3, 4];
        let t2 = Tensor::<u8, 2, _>::from_shape_vec([2, 2], data2, CpuAllocator)?;
        let t3 = mul(&t1, &t2)?;
        assert_eq!(t3.as_slice(), vec![1, 4, 9, 16]);
        Ok(())
    }

    #[test]
    fn test_dot_product1_f32() -> Result<(), TensorOpsError> {
        let a_f32 =
            Tensor::<f32, 1, CpuAllocator>::from_shape_slice([3], &[1.0, 2.0, 3.0], CpuAllocator)?;
        let b_f32 =
            Tensor::<f32, 1, CpuAllocator>::from_shape_slice([3], &[4.0, 5.0, 6.0], CpuAllocator)?;
        let result = dot_product1(&a_f32, &b_f32)?;
        assert_eq!(result, 32.0);
        Ok(())
    }

    #[test]
    fn test_dot_product1_u8() -> Result<(), TensorOpsError> {
        let a_u8 = Tensor::<u8, 1, CpuAllocator>::from_shape_slice([3], &[1, 2, 3], CpuAllocator)?;
        let b_u8 = Tensor::<u8, 1, CpuAllocator>::from_shape_slice([3], &[4, 5, 6], CpuAllocator)?;
        let result = dot_product1(&a_u8, &b_u8)?;
        assert_eq!(result, 32);
        Ok(())
    }

    #[test]
    fn test_cosine_similarity_orthogonal_vectors() -> Result<(), TensorOpsError> {
        let a =
            Tensor::<f32, 1, CpuAllocator>::from_shape_slice([3], &[1.0, 0.0, 0.0], CpuAllocator)?;
        let b =
            Tensor::<f32, 1, CpuAllocator>::from_shape_slice([3], &[0.0, 1.0, 0.0], CpuAllocator)?;
        let result = cosine_similarity(&a, &b)?;
        assert_eq!(result, 0.0);
        Ok(())
    }

    #[test]
    fn test_cosine_similarity_opposite_vectors() -> Result<(), TensorOpsError> {
        let e =
            Tensor::<f32, 1, CpuAllocator>::from_shape_slice([3], &[1.0, 2.0, 3.0], CpuAllocator)?;
        let f = Tensor::<f32, 1, CpuAllocator>::from_shape_slice(
            [3],
            &[-1.0, -2.0, -3.0],
            CpuAllocator,
        )?;
        let result = cosine_similarity(&e, &f)?;
        assert!((result - -1.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_cosine_similarity_zero_vector() -> Result<(), TensorOpsError> {
        let zero =
            Tensor::<f32, 1, CpuAllocator>::from_shape_slice([3], &[0.0, 0.0, 0.0], CpuAllocator)?;
        let g =
            Tensor::<f32, 1, CpuAllocator>::from_shape_slice([3], &[1.0, 2.0, 3.0], CpuAllocator)?;
        let result = cosine_similarity(&zero, &g)?;
        assert_eq!(result, 0.0);
        Ok(())
    }

    #[test]
    fn test_cosine_similarity_zero_dot_product() -> Result<(), TensorOpsError> {
        let h =
            Tensor::<f32, 1, CpuAllocator>::from_shape_slice([3], &[1.0, 1.0, 0.0], CpuAllocator)?;
        let i =
            Tensor::<f32, 1, CpuAllocator>::from_shape_slice([3], &[-1.0, 1.0, 0.0], CpuAllocator)?;
        let result = cosine_similarity(&h, &i)?;
        assert!(result.abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_cosine_distance() -> Result<(), TensorOpsError> {
        let a =
            Tensor::<f32, 1, CpuAllocator>::from_shape_slice([3], &[1.0, 2.0, 3.0], CpuAllocator)?;
        let b =
            Tensor::<f32, 1, CpuAllocator>::from_shape_slice([3], &[1.0, 2.0, 3.0], CpuAllocator)?;
        let result = cosine_distance(&a, &b)?;
        assert!(result.abs() < 1e-6);

        let c =
            Tensor::<f32, 1, CpuAllocator>::from_shape_slice([3], &[1.0, 0.0, 0.0], CpuAllocator)?;
        let d =
            Tensor::<f32, 1, CpuAllocator>::from_shape_slice([3], &[0.0, 1.0, 0.0], CpuAllocator)?;
        let result = cosine_distance(&c, &d)?;
        assert!((result - 1.0).abs() < 1e-6);

        let e =
            Tensor::<f32, 1, CpuAllocator>::from_shape_slice([3], &[1.0, 2.0, 3.0], CpuAllocator)?;
        let f = Tensor::<f32, 1, CpuAllocator>::from_shape_slice(
            [3],
            &[-1.0, -2.0, -3.0],
            CpuAllocator,
        )?;
        let result = cosine_distance(&e, &f)?;
        assert!((result - 2.0).abs() < 1e-6);

        Ok(())
    }
}
