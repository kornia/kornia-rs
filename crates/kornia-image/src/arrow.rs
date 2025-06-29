use crate::{allocator::ImageAllocator, Image, ImageError, ImageSize};
use arrow::{
    array::{ArrayRef, BinaryArray, StructArray, UInt32Array},
    datatypes::{DataType, Field},
};
use kornia_tensor::{allocator::TensorAllocatorError, TensorAllocator};
use std::sync::Arc;

/// Allocator for Arrow arrays
#[derive(Clone)]
#[allow(dead_code)]
pub struct ArrowAllocator(arrow::buffer::Buffer);

impl TensorAllocator for ArrowAllocator {
    fn alloc(&self, layout: std::alloc::Layout) -> Result<*mut u8, TensorAllocatorError> {
        let ptr = unsafe { std::alloc::alloc(layout) };

        if ptr.is_null() {
            Err(TensorAllocatorError::NullPointer)?
        }

        Ok(ptr)
    }

    fn dealloc(&self, _ptr: *mut u8, _layout: std::alloc::Layout) {
        // Do nothing as the memory is managed by Arrow
    }
}

impl ImageAllocator for ArrowAllocator {}

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
impl<const C: usize, A: ImageAllocator> IntoArrow for Image<u8, C, A> {
    fn into_arrow(self) -> arrow::array::ArrayRef {
        let width = self.width() as u32;
        let height = self.height() as u32;
        let channels = self.num_channels() as u32;
        let data = self.into_vec();

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
                Arc::new(Field::new("pixels", DataType::Binary, false)),
                Arc::new(BinaryArray::from_vec(vec![&data])) as ArrayRef,
            ),
        ]))
    }
}

impl<const C: usize> TryFromArrow for Image<u8, C, ArrowAllocator> {
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

        // Extract pixels from BinaryArray
        let buffer = struct_array
            .column(3)
            .as_any()
            .downcast_ref::<BinaryArray>()
            .ok_or(ImageError::CastError)?
            .values();

        // zero copy conversion
        // NOTE: we need to clone the buffer to avoid the borrow checker, but it is not deep copying
        // as the internal data is reference counted.
        let buffer_owned = buffer.clone();
        let data_ptr = buffer_owned.as_ptr();
        let data_len = buffer_owned.len();

        let alloc = ArrowAllocator(buffer_owned);

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
        allocator::CpuAllocator,
        arrow::{IntoArrow, TryFromArrow},
        image::Image,
        ImageError, ImageSize,
    };

    #[test]
    fn test_image_into_arrow() -> Result<(), ImageError> {
        let image = Image::<u8, 1, CpuAllocator>::new(
            ImageSize {
                width: 2,
                height: 3,
            },
            vec![0, 1, 2, 3, 4, 5],
            CpuAllocator,
        )?;

        let arrow_array = image.into_arrow();

        let image_arr = Image::<u8, 1, _>::try_from_arrow(arrow_array.clone())?;

        assert_eq!(image_arr.width(), 2);
        assert_eq!(image_arr.height(), 3);
        assert_eq!(image_arr.num_channels(), 1);
        assert_eq!(image_arr.into_vec(), vec![0, 1, 2, 3, 4, 5]);

        Ok(())
    }
}
