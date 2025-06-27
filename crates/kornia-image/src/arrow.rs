use crate::{allocator::ImageAllocator, Image};

/// Trait for converting to Arrow arrays
pub trait IntoArrow {
    /// The type of the Arrow array
    type Output: arrow::array::Array;

    /// Convert the image to an Arrow array
    fn into_arrow(self) -> arrow::array::ArrayRef;
}

/// Implementation of IntoArrow for Image
impl<const C: usize, A: ImageAllocator> IntoArrow for Image<u8, C, A> {
    type Output = arrow::array::UInt8Array;

    // TODO: can we embed also the metadata in the arrow array?
    fn into_arrow(self) -> arrow::array::ArrayRef {
        let data = self.into_vec();
        std::sync::Arc::new(arrow::array::UInt8Array::from(data))
    }
}

/// Implementation of IntoArrow for Vec<u8>
impl IntoArrow for Vec<u8> {
    type Output = arrow::array::UInt8Array;

    fn into_arrow(self) -> arrow::array::ArrayRef {
        std::sync::Arc::new(arrow::array::UInt8Array::from(self))
    }
}
