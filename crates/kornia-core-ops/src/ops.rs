use kornia_core::{storage::TensorStorage, SafeTensorType, Tensor, TensorAllocator};
use num_traits::Float;

use crate::error::TensorOpsError;

/// A trait to define operationt that can be performed on a tensor.
pub trait TensorOps<T, const N: usize, A>
where
    T: SafeTensorType,
    A: TensorAllocator,
{
    /// Apply the power function to the tensor data.
    ///
    /// # Arguments
    ///
    /// * `n` - The power to raise the tensor data to.
    ///
    /// # Returns
    ///
    /// A new tensor with the data raised to the power.
    fn powf(&self, n: T) -> Tensor<T, N, A>
    where
        T: Float;

    /// Apply the square root function to the tensor elements.
    ///
    /// # Returns
    ///
    /// A new tensor with the square root values.
    fn sqrt(&self) -> Tensor<T, N, A>
    where
        T: Float;

    /// Compute the sum of the elements in the tensor along dimension `dim`
    ///
    /// # Arguments
    ///
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
    /// use kornia_core::{Tensor, CpuAllocator};
    /// use kornia_core_ops::TensorOps;
    ///
    /// let data: [u8; 6] = [1, 1, 1, 1, 1, 1];
    /// let t = Tensor::<u8, 2>::from_shape_slice([2, 3], &data, CpuAllocator).unwrap();
    /// let agg = t.sum_elements(1).unwrap();
    /// assert_eq!(agg.shape, [2, 1]);
    /// assert_eq!(agg.as_slice(), [3, 3]);
    /// ```
    fn sum_elements(&self, dim: usize) -> Result<Tensor<T, N, A>, TensorOpsError>
    where
        T: std::ops::Add<Output = T>;

    /// Compute the p-norm of the tensor along some dimension.
    ///
    /// # Arguments
    ///
    /// * `p` - The order of the norm.
    /// * `dim` - The index of the dimension/axis to use as vectors for the p-norm calculation.
    ///
    /// # Returns
    ///
    /// A new `Tensor` containing the p-norm values at all the vector locations.
    ///
    /// # Errors
    ///
    /// If the requested dimension is greater than the number of axes of the tensor, an error is returned.
    ///
    /// # Example
    ///
    /// ```
    /// use kornia_core::{Tensor, CpuAllocator};
    /// use kornia_core_ops::TensorOps;
    ///
    /// let data: [f32; 5] = [3., 3., 2., 1., 1.];
    /// let t = Tensor::<f32, 1>::from_shape_slice([5], &data, CpuAllocator).unwrap();
    /// let norm = t.pnorm(3., 0).unwrap();
    /// assert_eq!(norm.shape, [1]);
    /// assert_eq!(norm.as_slice(), [4.]);
    /// ```
    fn pnorm(&self, p: T, dim: usize) -> Result<Tensor<T, N, A>, TensorOpsError>
    where
        T: std::ops::Add<Output = T> + Float;

    /// Compute the euclidean norm of a tensor along some dimension.
    ///
    /// # Arguments
    ///
    /// * `dim` - The index of the dimension/axis to use as vectors for the norm calculation.
    ///
    /// # Returns
    ///
    /// A new `Tensor` containing the norm values at all the vector locations.
    ///
    /// # Errors
    ///
    /// If the requested dimension is greater than the number of axes of the tensor, an error is returned.
    ///
    /// # Example
    ///
    /// ```
    /// use kornia_core::{Tensor, CpuAllocator};
    /// use kornia_core_ops::TensorOps;
    ///
    /// let data: [f32; 5] = [3., 2., 1., 1., 1.];
    /// let t = Tensor::<f32, 1>::from_shape_slice([5], &data, CpuAllocator).unwrap();
    /// let norm = t.euclidean_norm(0).unwrap();
    /// assert_eq!(norm.shape, [1]);
    /// assert_eq!(norm.as_slice(), [4.]);
    /// ```
    fn euclidean_norm(&self, dim: usize) -> Result<Tensor<T, N, A>, TensorOpsError>
    where
        T: std::ops::Add<Output = T> + Float;
}

impl<T, const N: usize, A> TensorOps<T, N, A> for Tensor<T, N, A>
where
    T: SafeTensorType,
    A: TensorAllocator + 'static,
{
    fn powf(&self, n: T) -> Tensor<T, N, A>
    where
        T: Float,
    {
        self.map(|x| x.powf(n))
    }

    fn sqrt(&self) -> Tensor<T, N, A>
    where
        T: Float,
    {
        self.map(|x| x.sqrt())
    }

    fn sum_elements(&self, dim: usize) -> Result<Tensor<T, N, A>, TensorOpsError>
    where
        T: std::ops::Add<Output = T>,
    {
        if dim >= N {
            return Err(TensorOpsError::DimOutOfBounds(dim, N - 1));
        }

        let mut out_shape = self.shape;
        out_shape[dim] = 1;

        let mut out_strides = self.strides;
        if dim > 0 {
            out_strides
                .iter_mut()
                .take(dim)
                .for_each(|s| *s /= self.shape[dim]);
        }

        let numel: usize = out_shape.iter().product();
        let mut data = vec![T::default(); numel];

        for (i, v) in self.as_slice().iter().enumerate() {
            let mut out_index = self.get_index_unchecked(i);
            out_index[dim] = 0;
            let out_offset = out_index
                .iter()
                .zip(out_strides.iter())
                .fold(0, |acc, (&idx, &stride)| acc + idx * stride);
            let agg = unsafe { data.get_unchecked_mut(out_offset) };
            *agg = *agg + *v;
        }

        let storage = TensorStorage::from_vec(data, self.storage.alloc().clone());

        Ok(Tensor {
            storage,
            shape: out_shape,
            strides: out_strides,
        })
    }

    fn pnorm(&self, p: T, dim: usize) -> Result<Tensor<T, N, A>, TensorOpsError>
    where
        T: std::ops::Add<Output = T> + Float,
    {
        let p_inv = T::one() / p;
        Ok(self.powf(p).sum_elements(dim)?.powf(p_inv))
    }

    fn euclidean_norm(&self, dim: usize) -> Result<Tensor<T, N, A>, TensorOpsError>
    where
        T: std::ops::Add<Output = T> + Float,
    {
        Ok(self.powi(2).sum_elements(dim)?.sqrt())
    }
}

#[cfg(test)]
mod tests {
    use kornia_core::{CpuAllocator, TensorError};

    use super::*;

    #[test]
    fn test_sum_dim_oob() -> Result<(), TensorError> {
        let data: [u8; 4] = [2, 2, 2, 2];
        let t = Tensor::<u8, 2>::from_shape_slice([2, 2], &data, CpuAllocator)?;
        let res = t.sum_elements(2);
        assert!(res.is_err_and(|e| e == TensorOpsError::DimOutOfBounds(2, 1)));
        Ok(())
    }

    #[test]
    fn test_sum_1d() -> Result<(), TensorError> {
        let data: [u8; 4] = [1, 1, 1, 1];
        let t = Tensor::<u8, 1>::from_shape_slice([4], &data, CpuAllocator)?;
        let res = t.sum_elements(0);

        assert!(res.is_ok_and(|v| v.as_slice() == [4]));
        Ok(())
    }

    #[test]
    fn test_sum_2d() -> Result<(), TensorOpsError> {
        let data: [u8; 6] = [1, 2, 3, 4, 5, 6];
        let t = Tensor::<u8, 2>::from_shape_slice([2, 3], &data, CpuAllocator)?;
        let t_f32 = t.cast::<f32>();
        let t_i32 = t.cast::<i32>();

        let agg = t.sum_elements(1)?;
        assert_eq!(agg.shape, [2, 1]);
        assert_eq!(agg.as_slice(), [6, 15]);

        assert_eq!(t_f32.sum_elements(1)?.as_slice(), [6., 15.]);
        assert_eq!(t_i32.sum_elements(1)?.as_slice(), [6, 15]);

        let agg = t.sum_elements(0)?;
        assert_eq!(agg.shape, [1, 3]);
        assert_eq!(agg.as_slice(), [5, 7, 9]);

        assert_eq!(t_f32.sum_elements(0)?.as_slice(), [5., 7., 9.]);
        assert_eq!(t_i32.sum_elements(0)?.as_slice(), [5, 7, 9]);
        Ok(())
    }

    #[test]
    fn test_sum_3d() -> Result<(), TensorOpsError> {
        let data: [u8; 24] = [1; 24];
        let t = Tensor::<u8, 3>::from_shape_slice([2, 3, 4], &data, CpuAllocator)?;
        let t_f32 = t.cast::<f32>();
        let t_i32 = t.cast::<i32>();

        let agg = t.sum_elements(0)?;
        assert_eq!(agg.shape, [1, 3, 4]);
        assert_eq!(agg.as_slice(), [2; 12]);
        assert_eq!(t_f32.sum_elements(0)?.as_slice(), [2.; 12]);
        assert_eq!(t_i32.sum_elements(0)?.as_slice(), [2; 12]);

        let agg = t.sum_elements(1)?;
        assert_eq!(agg.shape, [2, 1, 4]);
        assert_eq!(agg.as_slice(), [3; 8]);
        assert_eq!(t_f32.sum_elements(1)?.as_slice(), [3.; 8]);
        assert_eq!(t_i32.sum_elements(1)?.as_slice(), [3; 8]);

        let agg = t.sum_elements(2)?;
        assert_eq!(agg.shape, [2, 3, 1]);
        assert_eq!(agg.as_slice(), [4; 6]);
        assert_eq!(t_f32.sum_elements(2)?.as_slice(), [4.; 6]);
        assert_eq!(t_i32.sum_elements(2)?.as_slice(), [4; 6]);
        Ok(())
    }

    #[test]
    fn test_pnorm_1d() -> Result<(), TensorOpsError> {
        let data: [f32; 5] = [3., 3., 2., 1., 1.];
        let t = Tensor::<f32, 1>::from_shape_slice([5], &data, CpuAllocator).unwrap();
        let norm = t.pnorm(3., 0)?;
        assert_eq!(norm.shape, [1]);
        assert_eq!(norm.as_slice(), [4.]);
        Ok(())
    }

    #[test]
    fn test_euclidean_norm_1d() -> Result<(), TensorOpsError> {
        let data: [f32; 5] = [3., 2., 1., 1., 1.];
        let t = Tensor::<f32, 1>::from_shape_slice([5], &data, CpuAllocator).unwrap();
        let norm = t.euclidean_norm(0)?;
        assert_eq!(norm.shape, [1]);
        assert_eq!(norm.as_slice(), [4.]);
        Ok(())
    }
}
