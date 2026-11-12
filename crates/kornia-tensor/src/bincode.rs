use crate::Tensor;

impl<T, const N: usize> bincode::enc::Encode for Tensor<T, N>
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

impl<T, const N: usize, C> bincode::de::Decode<C> for Tensor<T, N>
where
    T: bincode::de::Decode<C>,
{
    fn decode<D: bincode::de::Decoder<Context = C>>(
        decoder: &mut D,
    ) -> Result<Self, bincode::error::DecodeError> {
        let shape: [usize; N] = bincode::Decode::decode(decoder)?;
        let strides: [usize; N] = bincode::Decode::decode(decoder)?;
        let data: Vec<T> = bincode::Decode::decode(decoder)?;
        // The input is untrusted: reject any shape/strides that disagree with `data`.
        Tensor::from_shape_strides_vec(shape, strides, data)
            .map_err(|e| bincode::error::DecodeError::OtherString(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bincode() -> Result<(), Box<dyn std::error::Error>> {
        let tensor = Tensor::<u8, 2>::from_shape_vec([2, 3], vec![1, 2, 3, 4, 5, 6])?;
        let mut serialized = vec![0u8; 100];
        let config = bincode::config::standard();
        let length = bincode::encode_into_slice(&tensor, &mut serialized, config)?;
        let deserialized: (Tensor<u8, 2>, usize) =
            bincode::decode_from_slice(&serialized[..length], config)?;
        assert_eq!(tensor.as_slice(), deserialized.0.as_slice());
        Ok(())
    }

    #[test]
    fn test_bincode_rejects_inconsistent_layout() -> Result<(), Box<dyn std::error::Error>> {
        // Regression: a crafted payload whose shape claims more elements than `data`.
        let config = bincode::config::standard();
        let payload = ([4usize, 4], [4usize, 1], vec![1u8]);
        let bytes = bincode::encode_to_vec(&payload, config)?;
        let res: Result<(Tensor<u8, 2>, usize), _> = bincode::decode_from_slice(&bytes, config);
        assert!(res.is_err());

        // Strides that reach past the data are rejected too.
        let payload = ([2usize, 2], [100usize, 1], vec![1u8, 2, 3, 4]);
        let bytes = bincode::encode_to_vec(&payload, config)?;
        let res: Result<(Tensor<u8, 2>, usize), _> = bincode::decode_from_slice(&bytes, config);
        assert!(res.is_err());
        Ok(())
    }
}
