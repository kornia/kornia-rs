use crate::error::KernelError;
use num_traits::Zero;

/// Computes the dot product of two slices of floating-point values.
///
/// This function calculates the sum of element-wise products of two slices.
///
/// # Arguments
///
/// * `a` - First slice of values
/// * `b` - Second slice of values
///
/// # Returns
///
/// The dot product of the two slices or an error if the slices have different lengths.
///
/// # Errors
///
/// If the lengths of the slices don't match, a `LengthMismatch` error is returned.
///
/// Example:
/// ```
/// use kernels::ops::dot_product1_kernel;
///
/// let a = [1.0, 2.0, 3.0];
/// let b = [4.0, 5.0, 6.0];
/// let result = dot_product1_kernel(&a, &b).unwrap();
/// assert_eq!(result, 32.0); // (1*4) + (2*5) + (3*6) = 32
/// ```
pub fn dot_product1_kernel<T>(a: &[T], b: &[T]) -> Result<T, KernelError>
where
    T: Zero + Copy + Clone + std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
{
    if a.len() != b.len() {
        return Err(KernelError::LengthMismatch(a.len(), b.len()));
    }

    let result = a
        .iter()
        .zip(b.iter())
        .fold(T::zero(), |acc, (a_val, b_val)| acc + *a_val * *b_val);

    Ok(result)
}

/// Computes the cosine similarity between two vectors.
///
/// Cosine similarity measures the cosine of the angle between two vectors, providing
/// a value between -1 and 1 that indicates their similarity. A value of 1 means the vectors
/// are identical, 0 means they are orthogonal, and -1 means they are diametrically opposed.
///
/// # Arguments
///
/// * `a` - First vector as a slice
/// * `b` - Second vector as a slice
///
/// # Returns
///
/// The cosine similarity between the two vectors or an error.
///
/// # Errors
///
/// * If the vectors have different lengths, a `LengthMismatch` error is returned.
/// * If either vector has zero magnitude, a `ZeroMagnitudeVector` error is returned.
///
/// Example:
/// ```
/// use kernels::ops::cosine_similarity_float_kernel;
///
/// let a = [1.0, 2.0, 3.0];
/// let b = [2.0, 4.0, 6.0];
/// let result = cosine_similarity_float_kernel(&a, &b).unwrap();
/// assert_eq!(result, 1.0);
/// ```
pub fn cosine_similarity_float_kernel<T>(a: &[T], b: &[T]) -> Result<T, KernelError>
where
    T: num_traits::Float,
{
    if a.len() != b.len() {
        return Err(KernelError::LengthMismatch(a.len(), b.len()));
    }

    let (dot_product, magnitude_a, magnitude_b) = a.iter().zip(b.iter()).fold(
        (T::zero(), T::zero(), T::zero()),
        |(dot_product, magnitude_a, magnitude_b), (a_val, b_val)| {
            let a = *a_val;
            let b = *b_val;
            (
                dot_product + a * b,
                magnitude_a + a * a,
                magnitude_b + b * b,
            )
        },
    );

    // Handle edge cases
    if magnitude_a == T::zero() || magnitude_b == T::zero() {
        return Ok(T::zero());
    }

    let denominator = magnitude_a.sqrt() * magnitude_b.sqrt();
    Ok(dot_product / denominator)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_product1_float_kernel_length_mismatch() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0];
        let result = dot_product1_kernel(&a, &b);
        assert!(matches!(result, Err(KernelError::LengthMismatch(3, 2))));
    }

    #[test]
    fn test_dot_product1_kernel_f32() {
        let a: [f32; 3] = [1.0, 2.0, 3.0];
        let b: [f32; 3] = [4.0, 5.0, 6.0];
        let result = dot_product1_kernel(&a, &b).unwrap();
        assert_eq!(result, 32.0);
    }

    #[test]
    fn test_dot_product1_kernel_u8() {
        let a = [1, 2, 3];
        let b = [4, 5, 6];
        let result = dot_product1_kernel(&a, &b).unwrap();
        assert_eq!(result, 32);
    }

    #[test]
    fn test_cosine_similarity_orthogonal_vectors() {
        let a = [1.0, 0.0, 0.0];
        let b = [0.0, 1.0, 0.0];
        let result = cosine_similarity_float_kernel(&a, &b).unwrap();
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_cosine_similarity_identical_vectors() {
        let a = [1.0, 2.0, 3.0];
        let b = [1.0, 2.0, 3.0];
        let result = cosine_similarity_float_kernel(&a, &b).unwrap();
        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_cosine_similarity_opposite_vectors() {
        let a = [1.0, 2.0, 3.0];
        let b = [-1.0, -2.0, -3.0];
        let result = cosine_similarity_float_kernel(&a, &b).unwrap();
        assert_eq!(result, -1.0);
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 2.0, 3.0];
        let result = cosine_similarity_float_kernel(&a, &b);
        assert!(matches!(result, Ok(0.0)));
    }

    #[test]
    fn test_cosine_similarity_arbitrary_vectors() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let expected = 32.0 / (14.0_f64.sqrt() * 77.0_f64.sqrt());
        let result = cosine_similarity_float_kernel(&a, &b).unwrap();
        assert!((result - expected as f32).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_length_mismatch() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0];
        let result = cosine_similarity_float_kernel(&a, &b);
        assert!(matches!(result, Err(KernelError::LengthMismatch(3, 2))));
    }
}
