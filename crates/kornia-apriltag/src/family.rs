/// Represents the AprilTag Family
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TagFamily {
    /// The width of the tag including the border, in units.
    pub width_at_border: usize,
    /// Whether the border is reversed.
    pub reversed_border: bool,
    // TODO: more properties
}

impl TagFamily {
    /// The Tag36H11 AprilTag family.
    pub const TAG36_H11: Self = TagFamily {
        width_at_border: 8,
        reversed_border: false,
    };

    // TODO: More Tag Families
}
