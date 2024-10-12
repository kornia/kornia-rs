use crate::{allocator::TensorAllocator, storage::TensorStorage, Tensor};

use serde::ser::SerializeStruct;
use serde::Deserialize;

impl<T, const N: usize, A> serde::Serialize for Tensor<T, N, A>
where
    T: serde::Serialize,
    A: TensorAllocator + 'static,
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

impl<'de, T, const N: usize, A: TensorAllocator + Default + 'static> serde::Deserialize<'de>
    for Tensor<T, N, A>
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

        let storage_array = TensorStorage::from_vec(data, A::default());

        let shape_array: [usize; N] = shape
            .try_into()
            .map_err(|_| serde::de::Error::custom("Invalid shape"))?;

        let strides_array: [usize; N] = strides
            .try_into()
            .map_err(|_| serde::de::Error::custom("Invalid strides"))?;

        Ok(Tensor {
            storage: storage_array,
            shape: shape_array,
            strides: strides_array,
        })
    }
}
