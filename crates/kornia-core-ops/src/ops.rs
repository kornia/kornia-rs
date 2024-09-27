use kornia_core::{
    storage::TensorStorage, tensor::get_strides_from_shape, CpuAllocator, SafeTensorType, Tensor,
    TensorAllocator,
};

use crate::error::TensorOpsError;

trait HasShape {
    type WithShape<New: Shape>: HasShape<Shape = New>;
    type Shape: Shape;
}

trait Shape {}

trait ReduceShapeTo<Dst> {
    fn reduce(self, dim: usize) -> Dst;
}

trait SumTo: Sized + HasShape {
    fn sum<Dst: Shape>(&self, dim: usize) -> Result<Self::WithShape<Dst>, TensorOpsError>
    where
        Self::Shape: ReduceShapeTo<Dst>;
}

impl<T: SafeTensorType + std::ops::Add<Output = T>, A: TensorAllocator>
    ReduceShapeTo<Tensor<T, 2, A>> for Tensor<T, 3, A>
{
    fn reduce(self, dim: usize) -> Tensor<T, 2, A> {
        let shape = match dim {
            0 => [self.shape[1], self.shape[2]],
            1 => [self.shape[0], self.shape[2]],
            2 => [self.shape[0], self.shape[1]],
            _ => panic!("dim out of bounds"),
        };
        let strides = get_strides_from_shape(shape);
        Tensor {
            storage: self.storage,
            shape,
            strides,
        }
    }
}
impl<T: SafeTensorType + std::ops::Add<Output = T>, A: TensorAllocator>
    ReduceShapeTo<Tensor<T, 3, A>> for Tensor<T, 3, A>
{
    fn reduce(self, _dim: usize) -> Tensor<T, 3, A> {
        self
    }
}

impl<T: SafeTensorType + std::ops::Add<Output = T>, const N: usize, A: TensorAllocator> Shape
    for Tensor<T, N, A>
{
}

impl<S: Shape> HasShape for S {
    type WithShape<New: Shape> = New;
    type Shape = Self;
}

impl<T: SafeTensorType + std::ops::Add<Output = T>, const N: usize, A: TensorAllocator> SumTo
    for Tensor<T, N, A>
{
    fn sum<Dst: Shape>(&self, dim: usize) -> Result<Self::WithShape<Dst>, TensorOpsError>
    where
        Self::Shape: ReduceShapeTo<Dst>,
    {
        let out = sum_elements(self, dim)?;
        let out = out.reduce(dim);
        Ok(out)
    }
}

/// Compute the sum of the elements in the tensor along dimension `dim`
///
pub fn sum_elements<T, const N: usize, A>(
    tensor: &Tensor<T, N, A>,
    dim: usize,
) -> Result<Tensor<T, N, A>, TensorOpsError>
where
    T: SafeTensorType + std::ops::Add<Output = T>,
    A: TensorAllocator + Clone,
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
    let mut data = vec![T::default(); numel];

    for (i, v) in tensor.as_slice().iter().enumerate() {
        let mut out_index = tensor.get_index(i);
        out_index[dim] = 0;
        let out_offset = out_index
            .iter()
            .zip(out_strides.iter())
            .fold(0, |acc, (&idx, &stride)| acc + idx * stride);
        let agg = unsafe { data.get_unchecked_mut(out_offset) };
        *agg = *agg + *v;
    }

    let storage = TensorStorage::from_vec(data, tensor.storage.alloc().clone());

    Ok(Tensor {
        storage,
        shape: out_shape,
        strides: out_strides,
    })
}

#[cfg(test)]
mod tests {
    use kornia_core::{CpuAllocator, TensorError};

    use super::*;

    #[test]
    fn test_sum_dim_oob() -> Result<(), TensorError> {
        let data: [u8; 4] = [2, 2, 2, 2];
        let t = Tensor::<u8, 2>::from_shape_slice([2, 2], &data, CpuAllocator)?;
        let res = sum_elements(&t, 2);
        assert!(res.is_err_and(|e| e == TensorOpsError::DimOutOfBounds(2, 1)));
        Ok(())
    }

    #[test]
    fn test_sum_1d() -> Result<(), TensorError> {
        let data: [u8; 4] = [1, 1, 1, 1];
        let t = Tensor::<u8, 1>::from_shape_slice([4], &data, CpuAllocator)?;
        let res = sum_elements(&t, 0);

        assert!(res.is_ok_and(|v| v.as_slice() == [4]));
        Ok(())
    }

    #[test]
    fn test_sum_2d() -> Result<(), TensorError> {
        let data: [u8; 6] = [1, 1, 1, 1, 1, 1];
        let t = Tensor::<u8, 2>::from_shape_slice([2, 3], &data, CpuAllocator)?;
        let t_f32 = t.cast::<f32>();
        let t_i32 = t.cast::<i32>();

        let agg = sum_elements(&t, 1).unwrap();
        assert_eq!(agg.shape, [2, 1]);
        assert_eq!(agg.as_slice(), [3, 3]);

        assert_eq!(sum_elements(&t_f32, 1).unwrap().as_slice(), [3., 3.]);
        assert_eq!(sum_elements(&t_i32, 1).unwrap().as_slice(), [3, 3]);

        let agg = sum_elements(&t, 0).unwrap();
        assert_eq!(agg.shape, [1, 3]);
        assert_eq!(agg.as_slice(), [2, 2, 2]);

        assert_eq!(sum_elements(&t_f32, 0).unwrap().as_slice(), [2., 2., 2.]);
        assert_eq!(sum_elements(&t_i32, 0).unwrap().as_slice(), [2, 2, 2]);
        Ok(())
    }

    #[test]
    fn test_sum_3d() -> Result<(), TensorError> {
        let data: [u8; 24] = [1; 24];
        let t = Tensor::<u8, 3>::from_shape_slice([2, 3, 4], &data, CpuAllocator)?;
        let t_f32 = t.cast::<f32>();
        let t_i32 = t.cast::<i32>();

        let agg = sum_elements(&t, 0).unwrap();
        assert_eq!(agg.shape, [1, 3, 4]);
        assert_eq!(agg.as_slice(), [2; 12]);
        assert_eq!(sum_elements(&t_f32, 0).unwrap().as_slice(), [2.; 12]);
        assert_eq!(sum_elements(&t_i32, 0).unwrap().as_slice(), [2; 12]);

        let agg = sum_elements(&t, 1).unwrap();
        assert_eq!(agg.shape, [2, 1, 4]);
        assert_eq!(agg.as_slice(), [3; 8]);
        assert_eq!(sum_elements(&t_f32, 1).unwrap().as_slice(), [3.; 8]);
        assert_eq!(sum_elements(&t_i32, 1).unwrap().as_slice(), [3; 8]);

        let agg = sum_elements(&t, 2).unwrap();
        assert_eq!(agg.shape, [2, 3, 1]);
        assert_eq!(agg.as_slice(), [4; 6]);
        assert_eq!(sum_elements(&t_f32, 2).unwrap().as_slice(), [4.; 6]);
        assert_eq!(sum_elements(&t_i32, 2).unwrap().as_slice(), [4; 6]);

        Ok(())
    }
}
