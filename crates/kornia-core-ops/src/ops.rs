use std::iter::Sum;

use kornia_core::{storage::TensorStorage, SafeTensorType, Tensor, TensorAllocator, TensorError};
use num_traits::Float;

use crate::error::TensorOpsError;

/// Reduce the tensor by some operation `op` along a dimension `dim`
///
pub fn reduce<T, const N: usize, A, F>(
    tensor: &Tensor<T, N, A>,
    op: F,
    dim: usize,
) -> Result<Tensor<T, N, A>, TensorOpsError>
where
    T: SafeTensorType,
    F: Fn(&[T]) -> T,
    A: TensorAllocator + Clone,
{
    if dim >= N {
        return Err(TensorOpsError::DimOutOfBounds(dim, N - 1));
    }
    // if left.shape != right.shape {
    //     return Err(TensorError::DimensionMismatch(format!(
    //         "Shapes {:?} and {:?} are not compatible for element-wise operations",
    //         left.shape, right.shape
    //     )));
    // }

    // tensor.mean()
    let shape = tensor.shape;
    for v in tensor.as_slice() {
        println!("{:?}", v);
    }

    // let data = left
    //     .as_slice()
    //     .iter()
    //     .zip(right.as_slice().iter())
    //     .map(|(a, b)| op(a, b))
    //     .collect();

    // let storage = TensorStorage::from_vec(data, left.storage.alloc().clone());

    // Ok(Tensor {
    //     storage,
    //     shape: left.shape,
    //     strides: left.strides,
    // })
    todo!()
}

/// todo
pub fn pnorm<T, const N: usize, A>(t: &Tensor<T, N, A>, p: T) -> T
where
    T: Float + SafeTensorType + Sum,
    A: TensorAllocator,
{
    t.as_slice()
        .iter()
        .map(|x| x.powf(p))
        .sum::<T>()
        .powf(p.recip())
}

/// todo
pub fn norm<T, const N: usize, A>(t: &Tensor<T, N, A>) -> T
where
    T: Float + SafeTensorType + Sum,
    A: TensorAllocator,
{
    t.as_slice()
        .iter()
        .map(|x| x.powi(2))
        .sum::<T>()
        .powf(T::from(0.5).unwrap())
}

#[cfg(test)]
mod tests {
    use kornia_core::{CpuAllocator, TensorError};

    use super::*;

    #[test]
    fn test_pnorm_f32() -> Result<(), TensorError> {
        let data: [f32; 4] = [2., 2., 2., 2.];
        let t = Tensor::<f32, 2>::from_shape_slice([2, 2], &data, CpuAllocator)?;

        let norm = pnorm(&t, 2.0);

        assert_eq!(norm, 4.);
        Ok(())
    }

    #[test]
    fn test_pnorm_f64() -> Result<(), TensorError> {
        let data: [f64; 4] = [2., 2., 2., 2.];
        let t = Tensor::<f64, 2>::from_shape_slice([2, 2], &data, CpuAllocator)?;

        let norm = pnorm(&t, 2.0);

        assert_eq!(norm, 4.);
        Ok(())
    }

    #[test]
    fn test_norm_f32() -> Result<(), TensorError> {
        let data: [f32; 4] = [2., 2., 2., 2.];
        let t = Tensor::<f32, 2>::from_shape_slice([2, 2], &data, CpuAllocator)?;

        let norm = norm(&t);

        assert_eq!(norm, 4.);
        Ok(())
    }

    #[test]
    fn test_norm_f64() -> Result<(), TensorError> {
        let data: [f64; 4] = [2., 2., 2., 2.];
        let t = Tensor::<f64, 2>::from_shape_slice([2, 2], &data, CpuAllocator)?;

        let norm = norm(&t);

        assert_eq!(norm, 4.);
        Ok(())
    }

    #[test]
    fn test_reduce_dim_oob() -> Result<(), TensorError> {
        let data: [u8; 4] = [2, 2, 2, 2];
        let t = Tensor::<u8, 2>::from_shape_slice([2, 2], &data, CpuAllocator)?;
        let res = reduce(&t, |_| 1, 2);
        assert!(res.is_err_and(|e| e == TensorOpsError::DimOutOfBounds(2, 1)));
        Ok(())
    }

    #[test]
    fn test_reduce_1d() -> Result<(), TensorError> {
        let data: [u8; 4] = [1, 2, 3, 4];
        let t = Tensor::<u8, 1>::from_shape_slice([4], &data, CpuAllocator)?;
        let res = reduce(&t, |_| 1, 0);

        assert!(res.is_ok_and(|v| v.as_slice() == [1]));
        Ok(())
    }

    #[test]
    fn test_reduce_2d() -> Result<(), TensorError> {
        let data: [u8; 6] = [1, 2, 3, 4, 5, 6];
        let t = Tensor::<u8, 2>::from_shape_slice([2, 3], &data, CpuAllocator)?;
        assert_eq!(t.get_unchecked([1, 1]), &5);
        assert_eq!(t.get_unchecked([1, 2]), &6);
        let res = reduce(&t, |_| 1, 0);

        assert!(res.is_ok_and(|v| v.as_slice() == [1, 1, 1]));
        Ok(())
    }
    #[test]
    fn test_reduce_3d() -> Result<(), TensorError> {
        let data: [u8; 12] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let t = Tensor::<u8, 3>::from_shape_slice([2, 3, 2], &data, CpuAllocator)?;
        dbg!(t.get_unchecked([0, 0, 0]));
        dbg!(t.get_unchecked([0, 0, 1]));
        dbg!(t.get_unchecked([0, 1, 0]));
        dbg!(t.get_unchecked([0, 1, 1]));
        dbg!(t.get_unchecked([0, 2, 0]));
        dbg!(t.get_unchecked([0, 2, 1]));
        dbg!(t.get_unchecked([1, 0, 0]));
        dbg!(t.get_unchecked([1, 0, 1]));
        dbg!(t.get_unchecked([1, 1, 0]));
        dbg!(t.get_unchecked([1, 1, 1]));
        dbg!(t.get_unchecked([1, 2, 0]));
        dbg!(t.get_unchecked([1, 2, 1]));
        // assert_eq!(t.get_unchecked([1, 1, 0]), &5);
        // let res = reduce(&t, |_| 1, 0);

        // assert!(res.is_ok_and(|v| v.as_slice() == [1, 1, 1]));

        // dim = 2; [2, 3, 1]
        // 0 -> 0
        // 1 -> 0
        // 2 -> 1
        // 3 -> 1
        // 4 -> 2
        // 5 -> 2
        // 6 -> 3
        // 7 -> 3
        // 8 -> 4
        // 9 -> 4
        // 10 -> 5
        // 11 -> 5
        // * i / dim_size2

        // dim = 1; [2, 1, 2]
        // 0 -> 0
        // 1 -> 1
        // 2 -> 0
        // 3 -> 1
        // 4 -> 0
        // 5 -> 1
        // 6 -> 2
        // 7 -> 3
        // 8 -> 2
        // 9 -> 3
        // 10 -> 2
        // 11 -> 3
        // * i / dim_size2 + i / dim_size1

        // dim = 0; [1, 3, 2]
        // 0 -> 0
        // 1 -> 1
        // 2 -> 2
        // 3 -> 3
        // 4 -> 4
        // 5 -> 5
        // 6 -> 0
        // 7 -> 1
        // 8 -> 2
        // 9 -> 3
        // 10 -> 4
        // 11 -> 5
        // * i % (dim_size1*dim_size2)

        Ok(())
    }
}
