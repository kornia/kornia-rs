use kornia_tensor::{storage::TensorStorage, CpuAllocator, Tensor, TensorAllocator};
use num_traits::{Float, Zero};
use kernels::ops::{cosine_similarity_float_kernel, dot_product1_kernel};


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
/// use kornia_tensor_ops::ops::sum_elements;
///
/// let data: [u8; 6] = [1, 1, 1, 1, 1, 1];
/// let t = Tensor::<u8, 2, CpuAllocator>::from_shape_slice([2, 3], &data, CpuAllocator).unwrap();
/// let agg = sum_elements(&t, 1).unwrap();
/// assert_eq!(agg.shape, [2, 1]);
/// assert_eq!(agg.as_slice(), [3, 3]);
/// ```
pub fn sum_elements<T, const N: usize, A>(
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
    pub fn mul_scalar<T, const N: usize, A>(
        tensor:&Tensor<T, N, A>, 
        n: T
    ) -> Tensor<T, N, A>
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
    pub fn powf<T, const N: usize, A>(
        tensor:&Tensor<T, N, A>, 
        n: T
    ) -> Tensor<T, N, A>
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
    pub fn min<T, const N: usize, A>(tensor:&Tensor<T, N, CpuAllocator>,other:&Tensor<T, N, CpuAllocator>) -> Tensor<T, N, CpuAllocator>
    where
        T: PartialOrd + Clone,
        A: TensorAllocator + Clone + 'static,
    {
        tensor.element_wise_op(other, |a, b| if a < b { a.clone() } else { b.clone() })
            .expect("Tensor dimension mismatch")
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
/// use kornia_tensor_ops::ops::dot_product1;
///
/// let a = Tensor::<i32, 1, CpuAllocator>::from_shape_slice([3], &[1, 2, 3], CpuAllocator).unwrap();
/// let b = Tensor::<i32, 1, CpuAllocator>::from_shape_slice([3], &[4, 5, 6], CpuAllocator).unwrap();
/// let result = dot_product1(&a, &b).unwrap();
/// assert_eq!(result, 32); // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
/// ```
pub fn dot_product1<T, A>(a: &Tensor<T, 1, A>, b: &Tensor<T, 1, A>) -> Result<T, TensorOpsError>
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
/// use kornia_tensor_ops::ops::cosine_similarity;
///
/// let a = Tensor::<f32, 1, CpuAllocator>::from_shape_slice([3], &[1.0, 2.0, 3.0], CpuAllocator).unwrap();
/// let b = Tensor::<f32, 1, CpuAllocator>::from_shape_slice([3], &[2.0, 4.0, 6.0], CpuAllocator).unwrap();
/// let result = cosine_similarity(&a, &b).unwrap();
/// assert!((result - 1.0).abs() < 1e-6);
/// ```
pub fn cosine_similarity<T, const N: usize, A>(
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
/// use kornia_tensor_ops::ops::cosine_distance;
///
/// let a = Tensor::<f32, 1, CpuAllocator>::from_shape_slice([3], &[1.0, 2.0, 3.0], CpuAllocator).unwrap();
/// let b = Tensor::<f32, 1, CpuAllocator>::from_shape_slice([3], &[2.0, 4.0, 6.0], CpuAllocator).unwrap();
/// let result = cosine_distance(&a, &b).unwrap();
/// assert!(result.abs() < 1e-6);
/// ```
pub fn cosine_distance<T, A>(a: &Tensor<T, 1, A>, b: &Tensor<T, 1, A>) -> Result<T, TensorOpsError>
where
    T: num_traits::Float,
    A: TensorAllocator + Clone + 'static,
{
    let similarity = cosine_similarity(a, b)?;
    Ok(T::one() - similarity)
}


#[cfg(test)]
mod tests {
    use kornia_tensor::{CpuAllocator, TensorError};

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
