use cu29::bincode::de::{Decode, Decoder};
use cu29::bincode::enc::{Encode, Encoder};
use cu29::bincode::error::{DecodeError, EncodeError};
use serde::de::Error as _;
use serde::ser::SerializeTuple;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

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
impl Encode for ImageRgb8Msg {
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        Encode::encode(&self.0.rows(), encoder)?;
        Encode::encode(&self.0.cols(), encoder)?;
        Encode::encode(&self.0.as_slice(), encoder)?;
        Ok(())
    }
}

// TODO: implement in kornia-image
impl<C> Decode<C> for ImageRgb8Msg {
    fn decode<D: Decoder<Context = C>>(decoder: &mut D) -> Result<Self, DecodeError> {
        let rows = Decode::decode(decoder)?;
        let cols = Decode::decode(decoder)?;
        let data = Decode::decode(decoder)?;
        let image = ImageRgb8::new([rows, cols].into(), data)
            .map_err(|e| DecodeError::OtherString(e.to_string()))?;
        Ok(Self(image))
    }
}

// TODO: implement in kornia-image
impl Serialize for ImageRgb8Msg {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut tup = serializer.serialize_tuple(3)?;
        tup.serialize_element(&self.0.rows())?;
        tup.serialize_element(&self.0.cols())?;
        tup.serialize_element(&self.0.as_slice())?;
        tup.end()
    }
}

// TODO: implement in kornia-image
impl<'de> Deserialize<'de> for ImageRgb8Msg {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let (rows, cols, data): (usize, usize, Vec<u8>) = Deserialize::deserialize(deserializer)?;
        let image = ImageRgb8::new([rows, cols].into(), data).map_err(D::Error::custom)?;
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

impl Encode for ImageGray8Msg {
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        Encode::encode(&self.0.rows(), encoder)?;
        Encode::encode(&self.0.cols(), encoder)?;
        Encode::encode(&self.0.as_slice(), encoder)?;
        Ok(())
    }
}

impl<C> Decode<C> for ImageGray8Msg {
    fn decode<D: Decoder<Context = C>>(decoder: &mut D) -> Result<Self, DecodeError> {
        let rows = Decode::decode(decoder)?;
        let cols = Decode::decode(decoder)?;
        let data = Decode::decode(decoder)?;
        let image = ImageGray8::new([rows, cols].into(), data)
            .map_err(|e| DecodeError::OtherString(e.to_string()))?;
        Ok(Self(image))
    }
}

impl Serialize for ImageGray8Msg {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut tup = serializer.serialize_tuple(3)?;
        tup.serialize_element(&self.0.rows())?;
        tup.serialize_element(&self.0.cols())?;
        tup.serialize_element(&self.0.as_slice())?;
        tup.end()
    }
}

impl<'de> Deserialize<'de> for ImageGray8Msg {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let (rows, cols, data): (usize, usize, Vec<u8>) = Deserialize::deserialize(deserializer)?;
        let image = ImageGray8::new([rows, cols].into(), data).map_err(D::Error::custom)?;
        Ok(Self(image))
    }
}
