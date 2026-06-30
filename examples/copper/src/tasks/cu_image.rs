type ImageRgb8 = kornia::image::Image<u8, 3>;
type ImageGray8 = kornia::image::Image<u8, 1>;

#[derive(Clone)]
pub struct ImageRgb8Msg(pub ImageRgb8);

impl std::ops::Deref for ImageRgb8Msg {
    type Target = ImageRgb8;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

// TODO: implement in kornia-image
impl std::fmt::Debug for ImageRgb8Msg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ImageRgb8Msg(size: {:?})", self.0.size())
    }
}

// TODO: implement Image::empty()
impl Default for ImageRgb8Msg {
    fn default() -> Self {
        Self(ImageRgb8::new([0, 0].into(), vec![]).unwrap())
    }
}

// TODO: implement in kornia-image
impl bincode::enc::Encode for ImageRgb8Msg {
    fn encode<E: bincode::enc::Encoder>(
        &self,
        encoder: &mut E,
    ) -> Result<(), bincode::error::EncodeError> {
        bincode::Encode::encode(&self.0.rows(), encoder)?;
        bincode::Encode::encode(&self.0.cols(), encoder)?;
        bincode::Encode::encode(&self.0.as_slice(), encoder)?;
        Ok(())
    }
}

// TODO: implement in kornia-image
impl<C> bincode::de::Decode<C> for ImageRgb8Msg {
    fn decode<D: bincode::de::Decoder<Context = C>>(
        decoder: &mut D,
    ) -> Result<Self, bincode::error::DecodeError> {
        let rows = bincode::Decode::decode(decoder)?;
        let cols = bincode::Decode::decode(decoder)?;
        let data = bincode::Decode::decode(decoder)?;
        let image = ImageRgb8::new([rows, cols].into(), data)
            .map_err(|e| bincode::error::DecodeError::OtherString(e.to_string()))?;
        Ok(Self(image))
    }
}

#[derive(Clone)]
pub struct ImageGray8Msg(pub ImageGray8);

impl std::ops::Deref for ImageGray8Msg {
    type Target = ImageGray8;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::fmt::Debug for ImageGray8Msg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ImageGray8Msg(size: {:?})", self.0.size())
    }
}

impl Default for ImageGray8Msg {
    fn default() -> Self {
        Self(ImageGray8::new([0, 0].into(), vec![]).unwrap())
    }
}

impl bincode::enc::Encode for ImageGray8Msg {
    fn encode<E: bincode::enc::Encoder>(
        &self,
        encoder: &mut E,
    ) -> Result<(), bincode::error::EncodeError> {
        bincode::Encode::encode(&self.0.rows(), encoder)?;
        bincode::Encode::encode(&self.0.cols(), encoder)?;
        bincode::Encode::encode(&self.0.as_slice(), encoder)?;
        Ok(())
    }
}

impl<C> bincode::de::Decode<C> for ImageGray8Msg {
    fn decode<D: bincode::de::Decoder<Context = C>>(
        decoder: &mut D,
    ) -> Result<Self, bincode::error::DecodeError> {
        let rows = bincode::Decode::decode(decoder)?;
        let cols = bincode::Decode::decode(decoder)?;
        let data = bincode::Decode::decode(decoder)?;
        let image = ImageGray8::new([rows, cols].into(), data)
            .map_err(|e| bincode::error::DecodeError::OtherString(e.to_string()))?;
        Ok(Self(image))
    }
}
