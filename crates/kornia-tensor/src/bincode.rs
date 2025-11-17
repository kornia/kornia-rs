use crate::{device_marker::{Cpu, DeviceMarker}, storage::TensorStorage, Tensor};

impl<T, const N: usize, D: DeviceMarker + 'static> bincode::enc::Encode for Tensor<T, N, D>
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

impl<T, const N: usize, C> bincode::de::Decode<C> for Tensor<T, N, Cpu>
where
    T: bincode::de::Decode<C>,
{
    fn decode<D: bincode::de::Decoder<Context = C>>(
        decoder: &mut D,
    ) -> Result<Self, bincode::error::DecodeError> {
        let shape = bincode::Decode::decode(decoder)?;
        let strides = bincode::Decode::decode(decoder)?;
        let data = bincode::Decode::decode(decoder)?;
        let storage = TensorStorage::from_vec(data)
            .map_err(|e| bincode::error::DecodeError::OtherString(format!("Storage error: {}", e)))?;
        Ok(Self {
            shape,
            strides,
            storage,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor2;

    #[test]
    fn test_bincode() -> Result<(), Box<dyn std::error::Error>> {
        let tensor = Tensor2::<u8, Cpu>::from_shape_vec(
            [2, 3],
            vec![1, 2, 3, 4, 5, 6],
        )?;
        let mut serialized = vec![0u8; 100];
        let config = bincode::config::standard();
        let length = bincode::encode_into_slice(&tensor, &mut serialized, config)?;
        let deserialized: (Tensor2<u8, Cpu>, usize) =
            bincode::decode_from_slice(&serialized[..length], config)?;
        assert_eq!(tensor.as_slice(), deserialized.0.as_slice());
        Ok(())
    }
}
