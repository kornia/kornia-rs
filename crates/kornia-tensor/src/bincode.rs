use crate::{
    allocator::{CpuAllocator, TensorAllocator},
    storage::TensorStorage,
    Tensor,
};

impl<T, const N: usize, A: TensorAllocator + 'static> bincode::enc::Encode for Tensor<T, N, A>
where
    T: bincode::enc::Encode + 'static,
{
    fn encode<E: bincode::enc::Encoder>(
        &self,
        encoder: &mut E,
    ) -> Result<(), bincode::error::EncodeError> {
        bincode::Encode::encode(&self.shape.to_vec(), encoder)?;
        bincode::Encode::encode(&self.strides.to_vec(), encoder)?;
        bincode::Encode::encode(&self.storage.as_slice(), encoder)?;
        Ok(())
    }
}

impl<T, const N: usize> bincode::de::Decode for Tensor<T, N, CpuAllocator>
where
    T: bincode::de::Decode + 'static,
{
    fn decode<D: bincode::de::Decoder>(
        decoder: &mut D,
    ) -> Result<Self, bincode::error::DecodeError> {
        let shape: Vec<usize> = bincode::Decode::decode(decoder)?;
        let strides: Vec<usize> = bincode::Decode::decode(decoder)?;
        let data: Vec<T> = bincode::Decode::decode(decoder)?;
        Ok(Self {
            shape: shape.try_into().expect("shape is not valid"),
            strides: strides.try_into().expect("strides is not valid"),
            storage: TensorStorage::from_vec(data, CpuAllocator),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bincode() -> Result<(), Box<dyn std::error::Error>> {
        let tensor = Tensor::<u8, 2, CpuAllocator>::from_shape_vec(
            [2, 3],
            vec![1, 2, 3, 4, 5, 6],
            CpuAllocator,
        )?;
        let mut serialized = vec![0u8; 100];
        let config = bincode::config::standard();
        let length = bincode::encode_into_slice(&tensor, &mut serialized, config)?;
        let deserialized: (Tensor<u8, 2, CpuAllocator>, usize) =
            bincode::decode_from_slice(&serialized[..length], config)?;
        assert_eq!(tensor.as_slice(), deserialized.0.as_slice());
        Ok(())
    }
}
