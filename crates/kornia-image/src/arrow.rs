use crate::{Image, ImageError, ImageSize};
use arrow::array::Array;
use arrow::{
    array::{ArrayRef, BinaryArray, StructArray, UInt32Array},
    datatypes::{DataType, Field},
};
use kornia_tensor::{allocator::TensorAllocatorError, resource::MemoryResource, TensorAllocator};
use std::{alloc::Layout, any::Any, sync::Arc};

/// Former allocator tag for Arrow-backed images.
///
/// [`TryFromArrow`] now borrows the Arrow value buffer read-only through
/// [`Image::from_borrowed_host_readonly`], so this type is no longer used. It cannot
/// allocate: [`TensorAllocator::allocate`] always returns
/// [`TensorAllocatorError::CannotAllocateForeign`].
#[deprecated(
    note = "unused: `TryFromArrow` borrows the Arrow buffer read-only; this type will be removed"
)]
#[derive(Clone)]
pub struct ArrowAllocator(#[allow(dead_code)] arrow::buffer::Buffer);

#[allow(deprecated)]
impl TensorAllocator for ArrowAllocator {
    fn allocate(&self, _layout: Layout) -> Result<Box<dyn MemoryResource>, TensorAllocatorError> {
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

/// Returns column `idx` of `struct_array` downcast to `A`, checking that it exists
/// and that its first element is present.
fn first_value_column<A: Array + 'static>(
    struct_array: &StructArray,
    idx: usize,
) -> Result<&A, ImageError> {
    if idx >= struct_array.num_columns() {
        return Err(ImageError::CastError);
    }
    let col = struct_array
        .column(idx)
        .as_any()
        .downcast_ref::<A>()
        .ok_or(ImageError::CastError)?;
    if col.is_empty() || col.is_null(0) {
        return Err(ImageError::CastError);
    }
    Ok(col)
}

/// Converts an Arrow struct array (as produced by [`IntoArrow`]) back into an image.
///
/// The conversion is zero-copy: the image borrows the Arrow buffer **read-only**
/// (Arrow buffers are immutable and may be shared), and keeps it alive for as long
/// as the image exists. Mutating the returned image (e.g. `as_slice_mut`) panics;
/// clone it first to obtain an owned, writable copy.
///
/// # Errors
///
/// * [`ImageError::CastError`] if the array does not have the expected schema, or
///   any of its columns is empty/null.
/// * [`ImageError::InvalidChannelShape`] if the channel count differs from `C`, or
///   the pixel data length differs from `width * height * C`.
/// * [`ImageError::InvalidImageShape`] if `width * height * C` overflows.
impl<const C: usize> TryFromArrow for Image<u8, C> {
    fn try_from_arrow(array: arrow::array::ArrayRef) -> Result<Self, ImageError> {
        let struct_array = array
            .as_any()
            .downcast_ref::<StructArray>()
            .ok_or(ImageError::CastError)?;

        let width = first_value_column::<UInt32Array>(struct_array, 0)?.value(0);
        let height = first_value_column::<UInt32Array>(struct_array, 1)?.value(0);
        let channels = first_value_column::<UInt32Array>(struct_array, 2)?.value(0);

        if channels as usize != C {
            return Err(ImageError::InvalidChannelShape(C, channels as usize));
        }

        let binary = first_value_column::<BinaryArray>(struct_array, 3)?;

        // Only the first element's bytes belong to the image: `values()` is the whole
        // (possibly shared, possibly offset) value buffer.
        let data = binary.value(0);

        let size = ImageSize {
            width: width as usize,
            height: height as usize,
        };
        let expected = size.checked_len(C)?;
        if data.len() != expected {
            return Err(ImageError::InvalidChannelShape(data.len(), expected));
        }

        // Keep the underlying Arrow value buffer alive (cheap, reference counted) for
        // as long as the image borrows it.
        let keepalive: Arc<dyn Any + Send + Sync> = Arc::new(binary.values().clone());

        // SAFETY: `data` points into the value buffer owned by `keepalive`, which the
        // image keeps alive; it is valid for `expected` (== `data.len()`) bytes of `u8`
        // (alignment 1). Arrow buffers are immutable and may be shared, so the image is
        // created read-only and can never write through this pointer.
        let image = unsafe { Image::from_borrowed_host_readonly(size, data.as_ptr(), keepalive) }?;

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
    use arrow::{
        array::{ArrayRef, BinaryArray, StructArray, UInt32Array},
        datatypes::{DataType, Field},
    };
    use std::sync::Arc;

    #[test]
    fn test_image_into_arrow() -> Result<(), ImageError> {
        let image = Image::<u8, 1>::new(
            ImageSize {
                width: 2,
                height: 3,
            },
            vec![0, 1, 2, 3, 4, 5],
        )?;

        let arrow_array = image.into_arrow();

        let image_arr = Image::<u8, 1>::try_from_arrow(arrow_array.clone())?;

        assert_eq!(image_arr.width(), 2);
        assert_eq!(image_arr.height(), 3);
        assert_eq!(image_arr.num_channels(), 1);
        assert_eq!(image_arr.as_slice(), &[0, 1, 2, 3, 4, 5]);

        Ok(())
    }

    // Builds the image struct array from whole columns (one row per element).
    fn make_array_rows(
        widths: Vec<u32>,
        heights: Vec<u32>,
        channels: Vec<u32>,
        data: BinaryArray,
    ) -> ArrayRef {
        Arc::new(StructArray::from(vec![
            (
                Arc::new(Field::new("width", DataType::UInt32, false)),
                Arc::new(UInt32Array::from(widths)) as ArrayRef,
            ),
            (
                Arc::new(Field::new("height", DataType::UInt32, false)),
                Arc::new(UInt32Array::from(heights)) as ArrayRef,
            ),
            (
                Arc::new(Field::new("channels", DataType::UInt32, false)),
                Arc::new(UInt32Array::from(channels)) as ArrayRef,
            ),
            (
                Arc::new(Field::new("data", DataType::Binary, false)),
                Arc::new(data) as ArrayRef,
            ),
        ]))
    }

    fn make_array(width: u32, height: u32, channels: u32, data: Vec<&[u8]>) -> ArrayRef {
        make_array_rows(
            vec![width],
            vec![height],
            vec![channels],
            BinaryArray::from_vec(data),
        )
    }

    #[test]
    fn test_from_arrow_rejects_size_mismatch() {
        // Regression (F5): width/height were never checked against the data length,
        // so a 1000x1000 header over 4 bytes produced an image reading far OOB.
        let arr = make_array(1000, 1000, 1, vec![&[1, 2, 3, 4]]);
        assert!(Image::<u8, 1>::try_from_arrow(arr).is_err());
    }

    #[test]
    fn test_from_arrow_rejects_overflow() {
        let arr = make_array(u32::MAX, u32::MAX, 3, vec![&[1, 2, 3]]);
        assert!(Image::<u8, 3>::try_from_arrow(arr).is_err());
    }

    #[test]
    fn test_from_arrow_uses_first_value_only() -> Result<(), ImageError> {
        // Regression (F5): `.values()` returned the concatenation of every row, so a
        // 2-row binary array produced an image over both rows.
        let arr = make_array_rows(
            vec![2, 3],
            vec![1, 1],
            vec![1, 1],
            BinaryArray::from_vec(vec![&[7u8, 8][..], &[9, 10, 11]]),
        );
        let img = Image::<u8, 1>::try_from_arrow(arr)?;
        assert_eq!(img.as_slice(), &[7, 8]);

        // A sliced (offset) binary array must use its own first value.
        let bin = BinaryArray::from_vec(vec![&[1u8, 2][..], &[3, 4]]);
        let arr = make_array_rows(vec![2], vec![1], vec![1], bin.slice(1, 1));
        let img = Image::<u8, 1>::try_from_arrow(arr)?;
        assert_eq!(img.as_slice(), &[3, 4]);
        Ok(())
    }

    #[test]
    fn test_from_arrow_empty_arrays_do_not_panic() {
        // Regression (F5): `.value(0)` on an empty array panicked.
        let arr = make_array_rows(
            vec![],
            vec![],
            vec![],
            BinaryArray::from_vec(Vec::<&[u8]>::new()),
        );
        assert!(Image::<u8, 1>::try_from_arrow(arr).is_err());
    }

    #[test]
    #[should_panic(expected = "read-only")]
    fn test_from_arrow_is_read_only() {
        // Regression (F5): the image aliased the immutable, shareable Arrow buffer as
        // writable storage.
        let arr = make_array(2, 1, 1, vec![&[7, 8]]);
        let mut img = match Image::<u8, 1>::try_from_arrow(arr) {
            Ok(img) => img,
            Err(e) => panic!("unexpected error: {e}"),
        };
        let _ = img.as_slice_mut();
    }
}
