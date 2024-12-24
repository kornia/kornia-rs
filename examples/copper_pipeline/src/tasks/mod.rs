pub mod rerun_viz;
// pub use rerun_viz::*;

pub mod webcam;
// pub use webcam::*;

type ImageRGBU8 = kornia::image::Image<u8, 3>;

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
