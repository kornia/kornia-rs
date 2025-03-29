//! Operations for tensors.
//!
//! This module provides operations that can be performed on tensors.

use crate::{
    allocator::TensorAllocator,
    tensor::{Tensor, TensorError},
};

/// Add two tensors.
///
/// This is a convenience function that delegates to the `add` function in the `tensor::ops` module.
///
/// # Arguments
///
/// * `lhs` - Left-hand side tensor.
/// * `rhs` - Right-hand side tensor.
///
/// # Returns
///
/// A new `Tensor` instance with the sum of the two tensors.
pub fn add<T, const N: usize, A: TensorAllocator>(
    lhs: &Tensor<T, N, A>,
    rhs: &Tensor<T, N, A>,
) -> Result<Tensor<T, N, A>, TensorError>
where
    T: std::ops::Add<Output = T> + Clone,
    A: Clone + 'static,
{
    crate::tensor::ops::add(lhs, rhs)
}

/// Subtract two tensors.
///
/// This is a convenience function that delegates to the `sub` function in the `tensor::ops` module.
///
/// # Arguments
///
/// * `lhs` - Left-hand side tensor.
/// * `rhs` - Right-hand side tensor.
///
/// # Returns
///
/// A new `Tensor` instance with the difference of the two tensors.
pub fn sub<T, const N: usize, A: TensorAllocator>(
    lhs: &Tensor<T, N, A>,
    rhs: &Tensor<T, N, A>,
) -> Result<Tensor<T, N, A>, TensorError>
where
    T: std::ops::Sub<Output = T> + Clone,
    A: Clone + 'static,
{
    crate::tensor::ops::sub(lhs, rhs)
}

/// Multiply two tensors element-wise.
///
/// This is a convenience function that delegates to the `mul` function in the `tensor::ops` module.
///
/// # Arguments
///
/// * `lhs` - Left-hand side tensor.
/// * `rhs` - Right-hand side tensor.
///
/// # Returns
///
/// A new `Tensor` instance with the product of the two tensors.
pub fn mul<T, const N: usize, A: TensorAllocator>(
    lhs: &Tensor<T, N, A>,
    rhs: &Tensor<T, N, A>,
) -> Result<Tensor<T, N, A>, TensorError>
where
    T: std::ops::Mul<Output = T> + Clone,
    A: Clone + 'static,
{
    crate::tensor::ops::mul(lhs, rhs)
}

/// Divide two tensors element-wise.
///
/// This is a convenience function that delegates to the `div` function in the `tensor::ops` module.
///
/// # Arguments
///
/// * `lhs` - Left-hand side tensor.
/// * `rhs` - Right-hand side tensor.
///
/// # Returns
///
/// A new `Tensor` instance with the quotient of the two tensors.
pub fn div<T, const N: usize, A: TensorAllocator>(
    lhs: &Tensor<T, N, A>,
    rhs: &Tensor<T, N, A>,
) -> Result<Tensor<T, N, A>, TensorError>
where
    T: std::ops::Div<Output = T> + Clone,
    A: Clone + 'static,
{
    crate::tensor::ops::div(lhs, rhs)
}

/// Add a tensor to another tensor in-place.
///
/// # Arguments
///
/// * `lhs` - Left-hand side tensor that will be modified in-place.
/// * `rhs` - Right-hand side tensor.
///
/// # Returns
///
/// Ok(()) if the operation was successful, or an error if the shapes don't match.
pub fn add_inplace<T, const N: usize, A: TensorAllocator>(
    lhs: &mut Tensor<T, N, A>,
    rhs: &Tensor<T, N, A>,
) -> Result<(), TensorError>
where
    T: std::ops::AddAssign<T> + Clone,
    A: 'static,
{
    lhs.add_inplace(rhs)
}

/// Subtract a tensor from another tensor in-place.
///
/// # Arguments
///
/// * `lhs` - Left-hand side tensor that will be modified in-place.
/// * `rhs` - Right-hand side tensor.
///
/// # Returns
///
/// Ok(()) if the operation was successful, or an error if the shapes don't match.
pub fn sub_inplace<T, const N: usize, A: TensorAllocator>(
    lhs: &mut Tensor<T, N, A>,
    rhs: &Tensor<T, N, A>,
) -> Result<(), TensorError>
where
    T: std::ops::SubAssign<T> + Clone,
    A: 'static,
{
    lhs.sub_inplace(rhs)
}

/// Multiply a tensor by another tensor in-place.
///
/// # Arguments
///
/// * `lhs` - Left-hand side tensor that will be modified in-place.
/// * `rhs` - Right-hand side tensor.
///
/// # Returns
///
/// Ok(()) if the operation was successful, or an error if the shapes don't match.
pub fn mul_inplace<T, const N: usize, A: TensorAllocator>(
    lhs: &mut Tensor<T, N, A>,
    rhs: &Tensor<T, N, A>,
) -> Result<(), TensorError>
where
    T: std::ops::MulAssign<T> + Clone,
    A: 'static,
{
    lhs.mul_inplace(rhs)
}

/// Divide a tensor by another tensor in-place.
///
/// # Arguments
///
/// * `lhs` - Left-hand side tensor that will be modified in-place.
/// * `rhs` - Right-hand side tensor.
///
/// # Returns
///
/// Ok(()) if the operation was successful, or an error if the shapes don't match.
pub fn div_inplace<T, const N: usize, A: TensorAllocator>(
    lhs: &mut Tensor<T, N, A>,
    rhs: &Tensor<T, N, A>,
) -> Result<(), TensorError>
where
    T: std::ops::DivAssign<T> + Clone,
    A: 'static,
{
    lhs.div_inplace(rhs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::allocator::CpuAllocator;

    #[test]
    fn test_add() -> Result<(), TensorError> {
        let a = Tensor::<i32, 2, _>::from_shape_vec([2, 2], vec![1, 2, 3, 4], CpuAllocator)?;
        let b = Tensor::<i32, 2, _>::from_shape_vec([2, 2], vec![5, 6, 7, 8], CpuAllocator)?;

        let c = add(&a, &b)?;

        assert_eq!(c.as_slice(), &[6, 8, 10, 12]);
        Ok(())
    }

    #[test]
    fn test_add_inplace() -> Result<(), TensorError> {
        let mut a = Tensor::<i32, 2, _>::from_shape_vec([2, 2], vec![1, 2, 3, 4], CpuAllocator)?;
        let b = Tensor::<i32, 2, _>::from_shape_vec([2, 2], vec![5, 6, 7, 8], CpuAllocator)?;

        add_inplace(&mut a, &b)?;

        assert_eq!(a.as_slice(), &[6, 8, 10, 12]);
        Ok(())
    }

    #[test]
    fn test_sub() -> Result<(), TensorError> {
        let a = Tensor::<i32, 2, _>::from_shape_vec([2, 2], vec![10, 12, 14, 16], CpuAllocator)?;
        let b = Tensor::<i32, 2, _>::from_shape_vec([2, 2], vec![5, 6, 7, 8], CpuAllocator)?;

        let c = sub(&a, &b)?;

        assert_eq!(c.as_slice(), &[5, 6, 7, 8]);
        Ok(())
    }

    #[test]
    fn test_sub_inplace() -> Result<(), TensorError> {
        let mut a = Tensor::<i32, 2, _>::from_shape_vec([2, 2], vec![10, 12, 14, 16], CpuAllocator)?;
        let b = Tensor::<i32, 2, _>::from_shape_vec([2, 2], vec![5, 6, 7, 8], CpuAllocator)?;

        sub_inplace(&mut a, &b)?;

        assert_eq!(a.as_slice(), &[5, 6, 7, 8]);
        Ok(())
    }

    #[test]
    fn test_mul() -> Result<(), TensorError> {
        let a = Tensor::<i32, 2, _>::from_shape_vec([2, 2], vec![1, 2, 3, 4], CpuAllocator)?;
        let b = Tensor::<i32, 2, _>::from_shape_vec([2, 2], vec![5, 6, 7, 8], CpuAllocator)?;

        let c = mul(&a, &b)?;

        assert_eq!(c.as_slice(), &[5, 12, 21, 32]);
        Ok(())
    }

    #[test]
    fn test_mul_inplace() -> Result<(), TensorError> {
        let mut a = Tensor::<i32, 2, _>::from_shape_vec([2, 2], vec![1, 2, 3, 4], CpuAllocator)?;
        let b = Tensor::<i32, 2, _>::from_shape_vec([2, 2], vec![5, 6, 7, 8], CpuAllocator)?;

        mul_inplace(&mut a, &b)?;

        assert_eq!(a.as_slice(), &[5, 12, 21, 32]);
        Ok(())
    }

    #[test]
    fn test_div() -> Result<(), TensorError> {
        let a = Tensor::<i32, 2, _>::from_shape_vec([2, 2], vec![10, 12, 14, 16], CpuAllocator)?;
        let b = Tensor::<i32, 2, _>::from_shape_vec([2, 2], vec![5, 6, 7, 8], CpuAllocator)?;

        let c = div(&a, &b)?;

        assert_eq!(c.as_slice(), &[2, 2, 2, 2]);
        Ok(())
    }

    #[test]
    fn test_div_inplace() -> Result<(), TensorError> {
        let mut a = Tensor::<i32, 2, _>::from_shape_vec([2, 2], vec![10, 12, 14, 16], CpuAllocator)?;
        let b = Tensor::<i32, 2, _>::from_shape_vec([2, 2], vec![5, 6, 7, 8], CpuAllocator)?;

        div_inplace(&mut a, &b)?;

        assert_eq!(a.as_slice(), &[2, 2, 2, 2]);
        Ok(())
    }
} 