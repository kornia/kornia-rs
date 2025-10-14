use cust::{error::CudaError, memory::DeviceCopy, prelude::*};

/// A data structure to represent a multi-dimensional tensor on the GPU.
pub struct CudaTensor<T, const N: usize>
where
    T: DeviceCopy,
{
    /// The buffer containing the data of the tensor.
    pub buffer: DeviceBuffer<T>,
    /// The shape of the tensor.
    pub shape: [usize; N],
}

impl<T, const N: usize> CudaTensor <T, N> 
where
    T: DeviceCopy,
{
    /// Creates a new `CudaTensor` from a vector of data.
    pub fn from_vec(data: Vec<T>, shape: [usize; N]) -> Result<Self, CudaError>
    {
        let buffer = data.as_slice().as_dbuf()?;
        Ok(Self { buffer, shape })
    }

    /// Consumes the tensor and copies the data to a vector.
    ///
    /// # Arguments
    ///
    /// * `dst` - A mutable vector to copy the data to.
    ///
    /// # Returns
    ///
    /// A result containing the error if the copy fails.
    pub fn into_vec(self, dst: &mut Vec<T>) -> Result<(), CudaError> {
        if dst.len() != self.numel() {
            return Err(CudaError::OutOfMemory);
        }
        let data = self.buffer.as_slice();
        data.copy_to(dst)?;
        Ok(())
    }

    /// Returns the number of elements in the tensor.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
    
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_vec() -> Result<(), CudaError> {
        // TODO: keep the context alive ? or shared ?
        let _ctx = cust::quick_init()?;
        let data = vec![1, 2, 3, 4];
        let tensor = CudaTensor::from_vec(data, [2, 2])?;
        assert_eq!(tensor.shape, [2, 2]);

        let mut dst = vec![0; tensor.numel()];
        tensor.into_vec(&mut dst)?;
        assert_eq!(dst, vec![1, 2, 3, 4]);

        Ok(())
    }
}