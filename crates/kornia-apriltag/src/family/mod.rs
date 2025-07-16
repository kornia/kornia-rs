use crate::decoder::{QuickDecode, SharpeningBuffer};

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

impl TagFamily {
    /// Returns a vector containing all predefined tag families.
    pub fn all() -> Vec<Self> {
        vec![
            Self::tag16_h5(),
            Self::tag25_h9(),
            Self::tag36_h10(),
            Self::tag36_h11(),
            Self::tagcircle21_h7(),
            Self::tagcircle49_h12(),
            Self::tagstandard41_h12(),
            Self::tagstandard52_h13(),
            Self::tagcustom48_h12(),
        ]
    }
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
    /// The TagCircle49H12 Family. [TagFamily::tagcircle19_h12]
    TagCircle49H12,
    /// The TagCustom48H12 Family. [TagFamily::tagcustom48_h12]
    TagCustom48H12,
    /// The TagStandard41H12 Family. [TagFamily::tagstandard41_h12]
    TagStandard41H12,
    /// The TagStandard52H13 Family. [TagFamily::tagstandard52_h13]
    TagStandard52H13,
    /// A custom tag family, specified by name.
    Custom(String),
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
            _ => TagFamilyKind::Custom(value.name),
        }
    }
}

impl From<&TagFamily> for TagFamilyKind {
    fn from(value: &TagFamily) -> Self {
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
            _ => TagFamilyKind::Custom(value.name.clone()),
        }
    }
}

impl From<&mut TagFamily> for TagFamilyKind {
    fn from(value: &mut TagFamily) -> Self {
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
            _ => TagFamilyKind::Custom(value.name.clone()),
        }
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
