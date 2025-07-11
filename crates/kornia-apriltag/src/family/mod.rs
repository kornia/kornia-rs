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
    pub bit_x: Vec<u8>,
    /// The y-coordinates of each bit in the tag.
    pub bit_y: Vec<u8>,
    /// The code data for the tag family.
    pub code_data: Vec<usize>,
    /// TODO
    pub quick_decode: QuickDecode,
    // TODO: more properties
}

/// Represents a decoded AprilTag.
#[derive(Debug, Clone, PartialEq)]
pub enum DecodedTag {
    /// The Tag36H11 Family. [TagFamily::tag36_h11]
    Tag36H11,
    /// The Tag36H10 Family. [TagFamily::tag36_h10]
    Tag36H10,
    /// A custom tag family, specified by name.
    Custom(String),
}

impl From<TagFamily> for DecodedTag {
    fn from(value: TagFamily) -> Self {
        match value.name.as_str() {
            "tag36_h11" => DecodedTag::Tag36H11,
            "tag36_h10" => DecodedTag::Tag36H10,
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
}

#[doc(hidden)]
pub mod tag36h11;

#[doc(hidden)]
pub mod tag36h10;
