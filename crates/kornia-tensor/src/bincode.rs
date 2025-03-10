use crate::{
    allocator::{CpuAllocator, TensorAllocator},
    storage::TensorStorage,
    Tensor,
};

impl<T, const N: usize, A: TensorAllocator + 'static> bincode::enc::Encode for Tensor<T, N, A>
where
    T: bincode::enc::Encode,
{
    fn encode<E: bincode::enc::Encoder>(
        &self,
        encoder: &mut E,
    ) -> Result<(), bincode::error::EncodeError> {
        bincode::Encode::encode(&self.shape, encoder)?;
        bincode::Encode::encode(&self.strides, encoder)?;
        bincode::Encode::encode(&self.storage.as_slice(), encoder)?;
        Ok(())
    }
}

impl<T, const N: usize, C> bincode::de::Decode<C> for Tensor<T, N, CpuAllocator>
where
    T: bincode::de::Decode<C>,
{
    fn decode<D: bincode::de::Decoder<Context = C>>(
        decoder: &mut D,
    ) -> Result<Self, bincode::error::DecodeError> {
        let shape = bincode::Decode::decode(decoder)?;
        let strides = bincode::Decode::decode(decoder)?;
        let data = bincode::Decode::decode(decoder)?;
        Ok(Self {
            shape,
            strides,
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
