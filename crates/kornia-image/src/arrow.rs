use crate::{Image, ImageError, ImageSize};
use arrow::{
    array::{ArrayRef, BinaryArray, StructArray, UInt32Array},
    datatypes::{DataType, Field},
};
use kornia_tensor::{allocator::TensorAllocatorError, resource::MemoryResource, TensorAllocator};
use std::{alloc::Layout, sync::Arc};

/// Allocator for Arrow arrays.
///
/// Arrow manages the backing buffer's lifetime via reference-counting.
/// `allocate` returns a [`ForeignResource`] that keeps the `arrow::buffer::Buffer`
/// alive (via its keepalive `Arc`) and performs a no-op free on drop.
#[derive(Clone)]
#[allow(dead_code)]
pub struct ArrowAllocator(arrow::buffer::Buffer);

impl TensorAllocator for ArrowAllocator {
    fn allocate(&self, _layout: Layout) -> Result<Box<dyn MemoryResource>, TensorAllocatorError> {
        // ArrowAllocator is used only as a type tag for foreign Arrow-managed memory.
        // Actual allocation never happens here; the buffer is pre-existing.
        Err(TensorAllocatorError::CannotAllocateForeign)
    }
}

/// Trait for converting to Arrow arrays
pub trait IntoArrow {
    /// Convert the image to an Arrow array including the metadata
    fn into_arrow(self) -> arrow::array::ArrayRef;
}

/// Trait for converting from Arrow arrays
pub trait TryFromArrow: Sized {
    /// Convert an Arrow array to an image
    fn try_from_arrow(array: arrow::array::ArrayRef) -> Result<Self, ImageError>;
}

/// Implementation of IntoArrow for Image
impl<const C: usize> IntoArrow for Image<u8, C> {
    fn into_arrow(self) -> arrow::array::ArrayRef {
        let width = self.width() as u32;
        let height = self.height() as u32;
        let channels = self.num_channels() as u32;
        let data = self.as_slice();

        Arc::new(StructArray::from(vec![
            (
                Arc::new(Field::new("width", DataType::UInt32, false)),
                Arc::new(UInt32Array::from(vec![width])) as ArrayRef,
            ),
            (
                Arc::new(Field::new("height", DataType::UInt32, false)),
                Arc::new(UInt32Array::from(vec![height])) as ArrayRef,
            ),
            (
                Arc::new(Field::new("channels", DataType::UInt32, false)),
                Arc::new(UInt32Array::from(vec![channels])) as ArrayRef,
            ),
            (
                Arc::new(Field::new("data", DataType::Binary, false)),
                Arc::new(BinaryArray::from_vec(vec![data])) as ArrayRef,
            ),
        ]))
    }
}

impl<const C: usize> TryFromArrow for Image<u8, C> {
    fn try_from_arrow(array: arrow::array::ArrayRef) -> Result<Self, ImageError> {
        let struct_array = array
            .as_any()
            .downcast_ref::<StructArray>()
            .ok_or(ImageError::CastError)?;

        let width = struct_array
            .column(0)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .ok_or(ImageError::CastError)?
            .value(0);
        let height = struct_array
            .column(1)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .ok_or(ImageError::CastError)?
            .value(0);
        let channels = struct_array
            .column(2)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .ok_or(ImageError::CastError)?
            .value(0);

        if channels != C as u32 {
            return Err(ImageError::InvalidChannelShape(C, channels as usize));
        }

        // Extract data from BinaryArray
        let buffer = struct_array
            .column(3)
            .as_any()
            .downcast_ref::<BinaryArray>()
            .ok_or(ImageError::CastError)?
            .values();

        // zero copy conversion
        // NOTE: we need to clone the buffer to own it, but it is not deep copying
        // as the internal data is reference counted.
        let buffer_owned = buffer.clone();
        let data_ptr = buffer_owned.as_ptr();
        let data_len = buffer_owned.len();

        use kornia_tensor::AllocHandle;
        use std::sync::Arc;
        let alloc: AllocHandle = Arc::new(ArrowAllocator(buffer_owned));

        let image = unsafe {
            Image::from_raw_parts(
                ImageSize {
                    width: width as usize,
                    height: height as usize,
                },
                data_ptr,
                data_len,
                alloc,
            )
        }?;

        Ok(image)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        arrow::{IntoArrow, TryFromArrow},
        image::Image,
        ImageError, ImageSize,
    };
    use kornia_tensor::host_alloc;

    #[test]
    fn test_image_into_arrow() -> Result<(), ImageError> {
        let image = Image::<u8, 1>::new(
            ImageSize {
                width: 2,
                height: 3,
            },
            vec![0, 1, 2, 3, 4, 5],
            host_alloc(),
        )?;

        let arrow_array = image.into_arrow();

        let image_arr = Image::<u8, 1>::try_from_arrow(arrow_array.clone())?;

        assert_eq!(image_arr.width(), 2);
        assert_eq!(image_arr.height(), 3);
        assert_eq!(image_arr.num_channels(), 1);
        assert_eq!(image_arr.as_slice(), &[0, 1, 2, 3, 4, 5]);

        Ok(())
    }
}
