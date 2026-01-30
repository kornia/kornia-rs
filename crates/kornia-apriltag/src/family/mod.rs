use std::sync::Arc;
use crate::{
    decoder::{QuickDecode, SharpeningBuffer},
    errors::AprilTagError,
};

/// Represents the AprilTag Family
#[derive(Debug, Clone, PartialEq)]
pub struct TagFamily {
    /// The name of the tag
    pub name: String,
    /// The width of the tag including the border, in units.
    pub width_at_border: usize,
    /// Whether the border is reversed.
    pub reversed_border: bool,
    /// The total width of the tag, including border, in units.
    pub total_width: usize,
    /// The number of bits in the tag code.
    pub nbits: usize,
    /// The x-coordinates of each bit in the tag.
    pub bit_x: Vec<i8>,
    /// The y-coordinates of each bit in the tag.
    pub bit_y: Vec<i8>,
    /// The code data for the tag family.
    pub code_data: Vec<usize>,
    /// A table for fast lookup of decoded tag codes and their associated metadata.
    pub quick_decode: QuickDecode,
    /// Buffer used for storing intermediate values during the sharpening process.
    pub sharpening_buffer: SharpeningBuffer,
}

/// Represents a decoded AprilTag.
#[derive(Debug, Clone, PartialEq)]
pub enum TagFamilyKind {
    /// The Tag16H5 Family. [TagFamily::tag16_h5]
    Tag16H5,
    /// The Tag36H11 Family. [TagFamily::tag36_h11]
    Tag36H11,
    /// The Tag36H10 Family. [TagFamily::tag36_h10]
    Tag36H10,
    /// The Tag25H9 Family. [TagFamily::tag25_h9]
    Tag25H9,
    /// The TagCircle21H7 Family. [TagFamily::tagcircle21_h7]
    TagCircle21H7,
    /// The TagCircle49H12 Family. [TagFamily::tagcircle49_h12]
    TagCircle49H12,
    /// The TagCustom48H12 Family. [TagFamily::tagcustom48_h12]
    TagCustom48H12,
    /// The TagStandard41H12 Family. [TagFamily::tagstandard41_h12]
    TagStandard41H12,
    /// The TagStandard52H13 Family. [TagFamily::tagstandard52_h13]
    TagStandard52H13,
    /// A custom tag family, allowing users to supply a fully defined [`TagFamily`] instance.
    ///Use Arc for cloning
    Custom(Arc<TagFamily>),
}

impl TagFamilyKind {
    /// Returns a vector containing all built-in tag family kinds.
    pub fn all() -> Vec<Self> {
        vec![
            Self::Tag16H5,
            Self::Tag36H11,
            Self::Tag36H10,
            Self::Tag25H9,
            Self::TagCircle21H7,
            Self::TagCircle49H12,
            Self::TagCustom48H12,
            Self::TagStandard41H12,
            Self::TagStandard52H13,
        ]
    }
}

impl From<TagFamily> for TagFamilyKind {
    fn from(value: TagFamily) -> Self {
        match value.name.as_str() {
            "tag16_h5" => TagFamilyKind::Tag16H5,
            "tag36_h11" => TagFamilyKind::Tag36H11,
            "tag36_h10" => TagFamilyKind::Tag36H10,
            "tag25_h9" => TagFamilyKind::Tag25H9,
            "tagcircle21_h7" => TagFamilyKind::TagCircle21H7,
            "tagcircle49_h12" => TagFamilyKind::TagCircle49H12,
            "tagcustom48_h12" => TagFamilyKind::TagCustom48H12,
            "tagstandard41_h12" => TagFamilyKind::TagStandard41H12,
            "tagstandard52_h13" => TagFamilyKind::TagStandard52H13,
            // Move value directly into Arc
            _ => TagFamilyKind::Custom(Arc::new(value)),
        }
    }
}

impl From<&TagFamily> for TagFamilyKind {
    fn from(value: &TagFamily) -> Self {
        to_tag_family_kind_impl(value)
    }
}

impl From<&mut TagFamily> for TagFamilyKind {
    fn from(value: &mut TagFamily) -> Self {
        to_tag_family_kind_impl(value)
    }
}

fn to_tag_family_kind_impl(value: &TagFamily) -> TagFamilyKind {
    match value.name.as_str() {
        "tag16_h5" => TagFamilyKind::Tag16H5,
        "tag36_h11" => TagFamilyKind::Tag36H11,
        "tag36_h10" => TagFamilyKind::Tag36H10,
        "tag25_h9" => TagFamilyKind::Tag25H9,
        "tagcircle21_h7" => TagFamilyKind::TagCircle21H7,
        "tagcircle49_h12" => TagFamilyKind::TagCircle49H12,
        "tagcustom48_h12" => TagFamilyKind::TagCustom48H12,
        "tagstandard41_h12" => TagFamilyKind::TagStandard41H12,
        "tagstandard52_h13" => TagFamilyKind::TagStandard52H13,
        _ => TagFamilyKind::Custom(Arc::new(value.clone())),
    }
}

impl TryFrom<TagFamilyKind> for TagFamily {
    type Error = AprilTagError;

    fn try_from(value: TagFamilyKind) -> Result<Self, Self::Error> {
        match value {
            // Keeping the Custom optimization
            TagFamilyKind::Custom(tag_family) => {
                Ok(Arc::try_unwrap(tag_family).unwrap_or_else(|arc| (*arc).clone()))
            }
            // Delegating everything else to the helper function
            _ => to_tag_family_impl(&value),
        }
    }
}

impl TryFrom<&TagFamilyKind> for TagFamily {
    type Error = AprilTagError;

    fn try_from(value: &TagFamilyKind) -> Result<Self, Self::Error> {
        to_tag_family_impl(value)
    }
}

impl TryFrom<&mut TagFamilyKind> for TagFamily {
    type Error = AprilTagError;

    fn try_from(value: &mut TagFamilyKind) -> Result<Self, Self::Error> {
        to_tag_family_impl(value)
    }
}

fn to_tag_family_impl(value: &TagFamilyKind) -> Result<TagFamily, AprilTagError> {
    match value {
        TagFamilyKind::Tag16H5 => TagFamily::tag16_h5(),
        TagFamilyKind::Tag25H9 => TagFamily::tag25_h9(),
        TagFamilyKind::Tag36H10 => TagFamily::tag36_h10(),
        TagFamilyKind::Tag36H11 => TagFamily::tag36_h11(),
        TagFamilyKind::TagCircle21H7 => TagFamily::tagcircle21_h7(),
        TagFamilyKind::TagCircle49H12 => TagFamily::tagcircle49_h12(),
        TagFamilyKind::TagCustom48H12 => TagFamily::tagcustom48_h12(),
        TagFamilyKind::TagStandard41H12 => TagFamily::tagstandard41_h12(),
        TagFamilyKind::TagStandard52H13 => TagFamily::tagstandard52_h13(),
        TagFamilyKind::Custom(tag_family) => Ok((**tag_family).clone()),
    }
}

#[doc(hidden)]
pub mod tag36h11;

#[doc(hidden)]
pub mod tag36h10;

#[doc(hidden)]
pub mod tag16h5;

#[doc(hidden)]
pub mod tag25h9;

#[doc(hidden)]
pub mod tagcircle21h7;

#[doc(hidden)]
pub mod tagcircle49h12;

#[doc(hidden)]
pub mod tagcustom48h12;

#[doc(hidden)]
pub mod tagstandard41h12;

#[doc(hidden)]
pub mod tagstandard52h13;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_custom_family_arc_sharing() {
        let family = TagFamily::tag36_h11().unwrap();
        let custom_kind = TagFamilyKind::Custom(Arc::new(family));
        let custom_kind_clone = custom_kind.clone();

        if let (TagFamilyKind::Custom(arc1), TagFamilyKind::Custom(arc2)) = (custom_kind, custom_kind_clone) {
            assert!(Arc::ptr_eq(&arc1, &arc2), "Cloned Custom TagFamilyKind should share the same Arc pointer");
        } else {
            panic!("Expected Custom variants");
        }
    }
}
