/// Represents the AprilTag Family
#[derive(Debug, Clone, PartialEq, Eq)]
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
    // TODO: more properties
}

/// Represents a decoded AprilTag.
#[derive(Debug, Clone, PartialEq)]
pub enum TagFamilyKind {
    /// The Tag36H11 Family. [TagFamily::tag36_h11]
    Tag36H11,
    /// A custom tag family, specified by name.
    Custom(TagFamily),
}

impl TagFamilyKind {
    /// Convert the tag family kind to a tag family.
    pub fn to_tag_family(&self) -> TagFamily {
        match self {
            TagFamilyKind::Tag36H11 => TagFamily::tag36_h11(),
            TagFamilyKind::Custom(tag_family) => tag_family.clone(),
        }
    }
}

impl From<TagFamily> for TagFamilyKind {
    fn from(value: TagFamily) -> Self {
        match value.name.as_str() {
            "tag36_h11" => TagFamilyKind::Tag36H11,
            _ => TagFamilyKind::Custom(value),
        }
    }
}

impl From<&TagFamily> for TagFamilyKind {
    fn from(value: &TagFamily) -> Self {
        match value.name.as_str() {
            "tag36_h11" => TagFamilyKind::Tag36H11,
            _ => TagFamilyKind::Custom(value.clone()),
        }
    }
}

#[doc(hidden)]
pub mod tag36h11;
