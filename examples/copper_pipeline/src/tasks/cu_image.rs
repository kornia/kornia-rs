type ImageRGBU8 = kornia::image::Image<u8, 3>;
type ImageGrayU8 = kornia::image::Image<u8, 1>;

#[derive(Clone)]
pub struct ImageRGBU8Msg {
    pub image: ImageRGBU8,
}

impl std::fmt::Debug for ImageRGBU8Msg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ImageRGBU8Msg(size: {:?})", self.image.size())
    }
}

impl Default for ImageRGBU8Msg {
    fn default() -> Self {
        Self {
            image: ImageRGBU8::new([0, 0].into(), vec![]).unwrap(),
        }
    }
}

impl bincode::enc::Encode for ImageRGBU8Msg {
    fn encode<E: bincode::enc::Encoder>(
        &self,
        encoder: &mut E,
    ) -> Result<(), bincode::error::EncodeError> {
        bincode::Encode::encode(&self.image.rows(), encoder)?;
        bincode::Encode::encode(&self.image.cols(), encoder)?;
        bincode::Encode::encode(&self.image.as_slice(), encoder)?;
        Ok(())
    }
}

impl bincode::de::Decode for ImageRGBU8Msg {
    fn decode<D: bincode::de::Decoder>(
        decoder: &mut D,
    ) -> Result<Self, bincode::error::DecodeError> {
        let rows = bincode::Decode::decode(decoder)?;
        let cols = bincode::Decode::decode(decoder)?;
        let data = bincode::Decode::decode(decoder)?;
        let image = ImageRGBU8::new([rows, cols].into(), data)
            .map_err(|e| bincode::error::DecodeError::OtherString(e.to_string()))?;
        Ok(Self { image })
    }
}

#[derive(Clone)]
pub struct ImageGrayU8Msg {
    pub image: ImageGrayU8,
}

impl std::fmt::Debug for ImageGrayU8Msg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ImageGrayU8Msg(size: {:?})", self.image.size())
    }
}

impl Default for ImageGrayU8Msg {
    fn default() -> Self {
        Self {
            image: ImageGrayU8::new([0, 0].into(), vec![]).unwrap(),
        }
    }
}

impl bincode::enc::Encode for ImageGrayU8Msg {
    fn encode<E: bincode::enc::Encoder>(
        &self,
        encoder: &mut E,
    ) -> Result<(), bincode::error::EncodeError> {
        bincode::Encode::encode(&self.image.rows(), encoder)?;
        bincode::Encode::encode(&self.image.cols(), encoder)?;
        bincode::Encode::encode(&self.image.as_slice(), encoder)?;
        Ok(())
    }
}

impl bincode::de::Decode for ImageGrayU8Msg {
    fn decode<D: bincode::de::Decoder>(
        decoder: &mut D,
    ) -> Result<Self, bincode::error::DecodeError> {
        let rows = bincode::Decode::decode(decoder)?;
        let cols = bincode::Decode::decode(decoder)?;
        let data = bincode::Decode::decode(decoder)?;
        let image = ImageGrayU8::new([rows, cols].into(), data)
            .map_err(|e| bincode::error::DecodeError::OtherString(e.to_string()))?;
        Ok(Self { image })
    }
}
