use kornia_tensor::{storage::TensorStorage, Tensor, TensorAllocator};
use num_traits::Zero;

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

/// Compute the dot product between two tensors
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
/// use kornia_tensor_ops::ops::dot_product_1d;
///
/// let a = Tensor::<i32, 1, CpuAllocator>::from_shape_slice([3], &[1, 2, 3], CpuAllocator).unwrap();
/// let b = Tensor::<i32, 1, CpuAllocator>::from_shape_slice([3], &[4, 5, 6], CpuAllocator).unwrap();
/// let result = dot_product_1d(&a, &b).unwrap();
/// assert_eq!(result, 32); // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
/// ```
pub fn dot_product_1d<T, const N: usize, A>(
    a: &Tensor<T, N, A>,
    b: &Tensor<T, N, A>,
) -> Result<T, TensorOpsError>
where
    T: Zero + Clone + std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
    A: TensorAllocator + Clone + 'static,
{
    if a.shape != b.shape {
        return Err(TensorOpsError::ShapeMismatch(
            a.shape.to_vec(),
            b.shape.to_vec(),
        ));
    }

    let mut result = T::zero();
    for (a_val, b_val) in a.as_slice().iter().zip(b.as_slice().iter()) {
        result = result + a_val.clone() * b_val.clone();
    }

    Ok(result)
}

/// Compute the cosine similarity between two tensors
///
/// # Arguments
///
/// * `a` - First tensor
/// * `b` - Second tensor
///
/// # Returns
///
/// A scalar value representing the cosine similarity between the two tensors.
/// The value ranges from -1 to 1, where:
/// - 1 means the tensors are identical in direction (maximum similarity)
/// - 0 means the tensors are orthogonal (no similarity)
/// - -1 means the tensors are opposite in direction (maximum dissimilarity)
///
/// # Errors
///
/// If the shapes of the tensors don't match, an error is returned.
///
/// # Example
///
/// ```
/// use kornia_tensor::{Tensor, CpuAllocator};
/// use kornia_tensor_ops::ops::cosine_similarity;
///
/// let a = Tensor::<f32, 1, CpuAllocator>::from_shape_slice([3], &[1.0, 0.0, 0.0], CpuAllocator).unwrap();
/// let b = Tensor::<f32, 1, CpuAllocator>::from_shape_slice([3], &[0.0, 1.0, 0.0], CpuAllocator).unwrap();
/// let result = cosine_similarity(&a, &b).unwrap();
/// assert_eq!(result, 0.0); // Orthogonal vectors have cosine similarity of 0
/// ```
pub fn cosine_similarity<T, const N: usize, A>(
    a: &Tensor<T, N, A>,
    b: &Tensor<T, N, A>,
) -> Result<T, TensorOpsError>
where
    T: Zero
        + Clone
        + std::ops::Add<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Div<Output = T>
        + num_traits::Float,
    A: TensorAllocator + Clone + 'static,
{
    if a.shape != b.shape {
        return Err(TensorOpsError::ShapeMismatch(
            a.shape.to_vec(),
            b.shape.to_vec(),
        ));
    }

    // Calculate dot product
    let ab = dot_product_1d(a, b)?;

    // Calculate squared L2 norms using dot product with self
    let a2 = dot_product_1d(a, a)?;
    let b2 = dot_product_1d(b, b)?;

    // Handle edge cases
    if ab == T::zero() {
        return Ok(T::zero());
    }
    if a2 == T::zero() || b2 == T::zero() {
        return Ok(T::zero());
    }

    // Calculate cosine similarity: ab/(sqrt(a2)*sqrt(b2))
    let denominator = a2.sqrt() * b2.sqrt();
    Ok(ab / denominator)
}

/// Compute the cosine distance between two tensors
///
/// # Arguments
///
/// * `a` - First tensor
/// * `b` - Second tensor
///
/// # Returns
///
/// A scalar value representing the cosine distance between the two tensors.
/// The value ranges from 0 to 2, where:
/// - 0 means the tensors are identical in direction (maximum similarity)
/// - 1 means the tensors are orthogonal (no similarity)
/// - 2 means the tensors are opposite in direction (maximum dissimilarity)
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
/// let b = Tensor::<f32, 1, CpuAllocator>::from_shape_slice([3], &[1.0, 2.0, 3.0], CpuAllocator).unwrap();
/// let result = cosine_distance(&a, &b).unwrap();
/// assert!(result < 1e-6); // Same vectors have cosine distance of 0
/// ```
pub fn cosine_distance<T, const N: usize, A>(
    a: &Tensor<T, N, A>,
    b: &Tensor<T, N, A>,
) -> Result<T, TensorOpsError>
where
    T: Zero
        + Clone
        + std::ops::Add<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Sub<Output = T>
        + num_traits::Float,
    A: TensorAllocator + Clone + 'static,
{
    let similarity = cosine_similarity(a, b)?;
    Ok(T::one() - similarity)
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
pub fn cosine_similarity_optimized<T, const N: usize, A>(
    a: &Tensor<T, N, A>,
    b: &Tensor<T, N, A>,
) -> Result<T, TensorOpsError>
where
    T: Zero
        + Clone
        + std::ops::Add<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Div<Output = T>
        + num_traits::Float,
    A: TensorAllocator + Clone + 'static,
{
    if a.shape != b.shape {
        return Err(TensorOpsError::ShapeMismatch(
            a.shape.to_vec(),
            b.shape.to_vec(),
        ));
    }

    let mut dot_product = T::zero();
    let mut magnitude_a = T::zero();
    let mut magnitude_b = T::zero();

    // Compute dot product and magnitudes in a single pass
    for (a_val, b_val) in a.as_slice().iter().zip(b.as_slice().iter()) {
        let a_clone = a_val.clone();
        let b_clone = b_val.clone();
        
        dot_product = dot_product + a_clone.clone() * b_clone.clone();
        magnitude_a = magnitude_a + a_clone * a_clone;
        magnitude_b = magnitude_b + b_clone * b_clone;
    }

    // Handle edge cases
    if magnitude_a == T::zero() || magnitude_b == T::zero() {
        return Ok(T::zero());
    }

    // Calculate cosine similarity: dot_product/(sqrt(magnitude_a)*sqrt(magnitude_b))
    let denominator = magnitude_a.sqrt() * magnitude_b.sqrt();
    Ok(dot_product / denominator)
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
pub fn cosine_distance_optimized<T, const N: usize, A>(
    a: &Tensor<T, N, A>,
    b: &Tensor<T, N, A>,
) -> Result<T, TensorOpsError>
where
    T: Zero
        + Clone
        + std::ops::Add<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Sub<Output = T>
        + num_traits::Float,
    A: TensorAllocator + Clone + 'static,
{
    let similarity = cosine_similarity_optimized(a, b)?;
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
        let result = dot_product_1d(&a, &b);
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
    fn test_dot_product_1d() -> Result<(), TensorOpsError> {
        // Test with i32 tensors
        let a = Tensor::<i32, 1, CpuAllocator>::from_shape_slice([3], &[1, 2, 3], CpuAllocator)?;
        let b = Tensor::<i32, 1, CpuAllocator>::from_shape_slice([3], &[4, 5, 6], CpuAllocator)?;
        let result = dot_product_1d(&a, &b)?;
        assert_eq!(result, 32); // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32

        // Test with f32 tensors
        let a_f32 =
            Tensor::<f32, 1, CpuAllocator>::from_shape_slice([3], &[1.0, 2.0, 3.0], CpuAllocator)?;
        let b_f32 =
            Tensor::<f32, 1, CpuAllocator>::from_shape_slice([3], &[4.0, 5.0, 6.0], CpuAllocator)?;
        let result_f32 = dot_product_1d(&a_f32, &b_f32)?;
        assert_eq!(result_f32, 32.0);

        // Test with u8 tensors
        let a_u8 = Tensor::<u8, 1, CpuAllocator>::from_shape_slice([3], &[1, 2, 3], CpuAllocator)?;
        let b_u8 = Tensor::<u8, 1, CpuAllocator>::from_shape_slice([3], &[4, 5, 6], CpuAllocator)?;
        let result_u8 = dot_product_1d(&a_u8, &b_u8)?;
        assert_eq!(result_u8, 32);

        Ok(())
    }

    #[test]
    fn test_dot_product_3d() -> Result<(), TensorOpsError> {
        let a = Tensor::<i32, 3, CpuAllocator>::from_shape_slice(
            [2, 2, 2],
            &[1, 2, 3, 4, 5, 6, 7, 8],
            CpuAllocator,
        )?;
        let b = Tensor::<i32, 3, CpuAllocator>::from_shape_slice(
            [2, 2, 2],
            &[8, 7, 6, 5, 4, 3, 2, 1],
            CpuAllocator,
        )?;
        let result = dot_product_1d(&a, &b)?;
        // (1*8 + 2*7 + 3*6 + 4*5 + 5*4 + 6*3 + 7*2 + 8*1) = 8 + 14 + 18 + 20 + 20 + 18 + 14 + 8 = 120
        assert_eq!(result, 120);
        Ok(())
    }

    #[test]
    fn test_cosine_similarity_orthogonal_vectors() -> Result<(), TensorOpsError> {
        // Orthogonal vectors (90 degrees) should have cosine similarity of 0
        let a =
            Tensor::<f32, 1, CpuAllocator>::from_shape_slice([3], &[1.0, 0.0, 0.0], CpuAllocator)?;
        let b =
            Tensor::<f32, 1, CpuAllocator>::from_shape_slice([3], &[0.0, 1.0, 0.0], CpuAllocator)?;
        let result = cosine_similarity(&a, &b)?;
        assert!(result.abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_cosine_similarity_parallel_vectors() -> Result<(), TensorOpsError> {
        // Parallel vectors (0 degrees) should have cosine similarity of 1
        let c =
            Tensor::<f32, 1, CpuAllocator>::from_shape_slice([3], &[1.0, 2.0, 3.0], CpuAllocator)?;
        let d =
            Tensor::<f32, 1, CpuAllocator>::from_shape_slice([3], &[2.0, 4.0, 6.0], CpuAllocator)?;
        let result = cosine_similarity(&c, &d)?;
        assert!((result - 1.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_cosine_similarity_opposite_vectors() -> Result<(), TensorOpsError> {
        // Opposite vectors (180 degrees) should have cosine similarity of -1
        let e =
            Tensor::<f32, 1, CpuAllocator>::from_shape_slice([3], &[1.0, 2.0, 3.0], CpuAllocator)?;
        let f = Tensor::<f32, 1, CpuAllocator>::from_shape_slice(
            [3],
            &[-1.0, -2.0, -3.0],
            CpuAllocator,
        )?;
        let result = cosine_similarity(&e, &f)?;
        assert!((result + 1.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_cosine_similarity_zero_vector() -> Result<(), TensorOpsError> {
        // Test with zero vector
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
        // Test with vectors that have dot product of 0
        let h =
            Tensor::<f32, 1, CpuAllocator>::from_shape_slice([3], &[1.0, 1.0, 0.0], CpuAllocator)?;
        let i =
            Tensor::<f32, 1, CpuAllocator>::from_shape_slice([3], &[-1.0, 1.0, 0.0], CpuAllocator)?;
        let result = cosine_similarity(&h, &i)?;
        assert!(result.abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_cosine_similarity_3d_tensors() -> Result<(), TensorOpsError> {
        // Test with 3d tensors
        let a = Tensor::<f32, 3, CpuAllocator>::from_shape_slice(
            [2, 2, 2],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            CpuAllocator,
        )?;
        let b = Tensor::<f32, 3, CpuAllocator>::from_shape_slice(
            [2, 2, 2],
            &[8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            CpuAllocator,
        )?;
        let result = cosine_similarity(&a, &b)?;
        let expected = 0.5882352;
        assert!((result - expected).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_cosine_distance() -> Result<(), TensorOpsError> {
        // Same vectors should have distance of 0
        let a =
            Tensor::<f32, 1, CpuAllocator>::from_shape_slice([3], &[1.0, 2.0, 3.0], CpuAllocator)?;
        let b =
            Tensor::<f32, 1, CpuAllocator>::from_shape_slice([3], &[1.0, 2.0, 3.0], CpuAllocator)?;
        let result = cosine_distance(&a, &b)?;
        assert!(result.abs() < 1e-6);

        // Orthogonal vectors should have distance of 1
        let c =
            Tensor::<f32, 1, CpuAllocator>::from_shape_slice([3], &[1.0, 0.0, 0.0], CpuAllocator)?;
        let d =
            Tensor::<f32, 1, CpuAllocator>::from_shape_slice([3], &[0.0, 1.0, 0.0], CpuAllocator)?;
        let result = cosine_distance(&c, &d)?;
        assert!((result - 1.0).abs() < 1e-6);

        // Opposite vectors should have distance of 2
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

    #[test]
    fn test_cosine_similarity_optimized() -> Result<(), TensorOpsError> {
        // Test with parallel vectors (should have similarity of 1)
        let a = Tensor::<f32, 1, CpuAllocator>::from_shape_slice([3], &[1.0, 2.0, 3.0], CpuAllocator)?;
        let b = Tensor::<f32, 1, CpuAllocator>::from_shape_slice([3], &[2.0, 4.0, 6.0], CpuAllocator)?;
        
        let result = cosine_similarity_optimized(&a, &b)?;
        assert!((result - 1.0).abs() < 1e-6);
        
        // Compare with original implementation
        let original_result = cosine_similarity(&a, &b)?;
        assert!((result - original_result).abs() < 1e-6);
        
        Ok(())
    }

    #[test]
    fn test_cosine_distance_optimized() -> Result<(), TensorOpsError> {
        // Test with orthogonal vectors (should have distance of 1)
        let a = Tensor::<f32, 1, CpuAllocator>::from_shape_slice([3], &[1.0, 0.0, 0.0], CpuAllocator)?;
        let b = Tensor::<f32, 1, CpuAllocator>::from_shape_slice([3], &[0.0, 1.0, 0.0], CpuAllocator)?;
        
        let result = cosine_distance_optimized(&a, &b)?;
        assert!((result - 1.0).abs() < 1e-6);
        
        // Compare with original implementation
        let original_result = cosine_distance(&a, &b)?;
        assert!((result - original_result).abs() < 1e-6);
        
        Ok(())
    }
}
