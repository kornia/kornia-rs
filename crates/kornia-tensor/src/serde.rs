use crate::Tensor;

use serde::ser::SerializeStruct;
use serde::Deserialize;

impl<T, const N: usize> serde::Serialize for Tensor<T, N>
where
    T: serde::Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut state = serializer.serialize_struct("Tensor", 3)?;
        state.serialize_field("data", self.as_slice())?;
        state.serialize_field("shape", &self.shape.to_vec())?;
        state.serialize_field("strides", &self.strides.to_vec())?;
        state.end()
    }
}

impl<'de, T, const N: usize> serde::Deserialize<'de> for Tensor<T, N>
where
    T: serde::Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct TensorData<T> {
            data: Vec<T>,
            shape: Vec<usize>,
            strides: Vec<usize>,
        }

        let TensorData {
            data,
            shape,
            strides,
        } = TensorData::deserialize(deserializer)?;

        let shape_array: [usize; N] = shape
            .try_into()
            .map_err(|_| serde::de::Error::custom("Invalid shape"))?;

        let strides_array: [usize; N] = strides
            .try_into()
            .map_err(|_| serde::de::Error::custom("Invalid strides"))?;

        // The input is untrusted: reject any shape/strides that disagree with `data`.
        Tensor::from_shape_strides_vec(shape_array, strides_array, data)
            .map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serde() -> Result<(), Box<dyn std::error::Error>> {
        let data = vec![1, 2, 3, 4, 5, 6];
        let tensor = Tensor::<u8, 2>::from_shape_vec([2, 3], data)?;
        let serialized = serde_json::to_string(&tensor)?;
        let deserialized: Tensor<u8, 2> = serde_json::from_str(&serialized)?;
        assert_eq!(tensor.as_slice(), deserialized.as_slice());
        assert_eq!(tensor.shape, deserialized.shape);
        assert_eq!(tensor.strides, deserialized.strides);
        Ok(())
    }

    #[test]
    fn test_serde_rejects_inconsistent_shape() {
        // Regression: 1 element of data claiming a 4x4x1 shape used to deserialize.
        let json = r#"{"data":[1],"shape":[4,4,1],"strides":[4,1,1]}"#;
        assert!(serde_json::from_str::<Tensor<u8, 3>>(json).is_err());
    }

    #[test]
    fn test_serde_rejects_out_of_bounds_strides() {
        let json = r#"{"data":[1,2,3,4],"shape":[2,2],"strides":[100,1]}"#;
        assert!(serde_json::from_str::<Tensor<u8, 2>>(json).is_err());
    }

    #[test]
    fn test_serde_rejects_overflowing_shape() {
        let json = format!(
            r#"{{"data":[],"shape":[{},{}],"strides":[1,1]}}"#,
            1usize << 40,
            1usize << 40
        );
        assert!(serde_json::from_str::<Tensor<u8, 2>>(&json).is_err());
    }
}
