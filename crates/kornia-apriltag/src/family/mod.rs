use crate::{decoder::QuickDecode, quad::FitQuadConfig};

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
    /// TODO
    pub quick_decode: QuickDecode,
}

/// Represents a decoded AprilTag.
#[derive(Debug, Clone, PartialEq)]
pub enum DecodedTag {
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

impl From<TagFamily> for DecodedTag {
    fn from(value: TagFamily) -> Self {
        match value.name.as_str() {
            "tag36_h11" => DecodedTag::Tag36H11,
            "tag36_h10" => DecodedTag::Tag36H10,
            "tag25_h9" => DecodedTag::Tag25H9,
            "tagcircle21_h7" => DecodedTag::TagCircle21H7,
            "tagcircle19_h12" => DecodedTag::TagCircle49H12,
            "tagcustom48_h12" => DecodedTag::TagCustom48H12,
            "tagstandard41_h12" => DecodedTag::TagStandard41H12,
            "tagstandard52_h13" => DecodedTag::TagStandard52H13,
            _ => DecodedTag::Custom(value.name),
        }
    }
}

impl From<&TagFamily> for DecodedTag {
    fn from(value: &TagFamily) -> Self {
        match value.name.as_str() {
            "tag36_h11" => DecodedTag::Tag36H11,
            _ => DecodedTag::Custom(value.name.clone()),
        }
    }
}

/// TODO
pub struct DecodeTagsConfig {
    /// TODO
    pub tag_families: Vec<TagFamily>,
    /// TODO
    pub fit_quad_config: FitQuadConfig,
    /// Whether to enable edge refinement before decoding.
    pub refine_edges_enabled: bool,
    /// Sharpening factor applied during decoding.
    pub decode_sharpening: f32,
    /// TODO
    pub normal_border: bool,
    /// TODO
    pub reversed_border: bool,
    /// TODO
    pub min_tag_width: usize,
    /// TODO
    pub sharpening_buffer_len: usize,
}

impl DecodeTagsConfig {
    /// TODO
    pub fn new(tag_families: Vec<TagFamily>) -> Self {
        let mut normal_border = false;
        let mut reversed_border = false;
        let mut min_tag_width = usize::MAX;
        let mut min_sharpening_buffer_size = 0;

        tag_families.iter().for_each(|family| {
            if family.width_at_border < min_tag_width {
                min_tag_width = family.width_at_border;
            }
            normal_border |= !family.reversed_border;
            reversed_border |= family.reversed_border;

            if min_sharpening_buffer_size < family.total_width {
                min_sharpening_buffer_size = family.total_width;
            }
        });

        min_tag_width = min_tag_width.min(3);

        Self {
            tag_families,
            fit_quad_config: Default::default(),
            normal_border,
            refine_edges_enabled: true,
            decode_sharpening: 0.25,
            reversed_border,
            min_tag_width,
            sharpening_buffer_len: min_sharpening_buffer_size * min_sharpening_buffer_size,
        }
    }

    /// TODO
    pub fn add(&mut self, family: TagFamily) {
        if family.width_at_border < self.min_tag_width {
            self.min_tag_width = family.width_at_border;
        }
        self.normal_border |= !family.reversed_border;
        self.reversed_border |= family.reversed_border;

        let len = family.total_width * family.total_width;
        if self.sharpening_buffer_len < len {
            self.sharpening_buffer_len = len;
        }

        self.tag_families.push(family);
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
