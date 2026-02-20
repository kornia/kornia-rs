use crate::{
    decoder::{QuickDecode, SharpeningBuffer},
    errors::AprilTagError,
};
use std::str::FromStr;
use std::sync::Arc;

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
    /// The minimum Hamming distance between any two valid codes in this family.
    ///
    /// This value determines the maximum `max_hamming` that can be safely used
    /// for decoding. To avoid false positives where two different tags could be
    /// confused, `max_hamming` must be less than `min_hamming / 2`.
    pub min_hamming: u8,
    /// A table for fast lookup of decoded tag codes and their associated metadata.
    pub quick_decode: QuickDecode,
    /// Buffer used for storing intermediate values during the sharpening process.
    pub sharpening_buffer: SharpeningBuffer,
}

impl TagFamily {
    /// Returns the maximum safe `max_hamming` value for this tag family.
    ///
    /// This is calculated as `(min_hamming - 1) / 2` to ensure that two different
    /// tags with bit errors cannot be confused with each other.
    pub fn max_safe_hamming(&self) -> u8 {
        (self.min_hamming - 1) / 2
    }

    /// Sets the maximum allowed Hamming distance for decoding and returns self.
    ///
    /// The Hamming distance determines how many bit errors are tolerated when
    /// matching observed codes to valid tag codes. A lower value reduces false
    /// positives but may miss detections in noisy images.
    ///
    /// # Arguments
    ///
    /// * `max_hamming` - The maximum number of bit errors to tolerate.
    ///
    /// # Errors
    ///
    /// Returns `AprilTagError::MaxHammingTooLarge` if `max_hamming` exceeds the safe
    /// limit for this tag family (which is `(min_hamming - 1) / 2`).
    ///
    /// # Example
    ///
    /// ```
    /// use kornia_apriltag::family::TagFamily;
    ///
    /// let family = TagFamily::tag36_h11()?.with_max_hamming(1)?;
    /// # Ok::<(), kornia_apriltag::errors::AprilTagError>(())
    /// ```
    pub fn with_max_hamming(mut self, max_hamming: u8) -> Result<Self, AprilTagError> {
        self.set_max_hamming(max_hamming)?;
        Ok(self)
    }

    /// Sets the maximum allowed Hamming distance for decoding.
    ///
    /// # Arguments
    ///
    /// * `max_hamming` - The maximum number of bit errors to tolerate.
    ///
    /// # Errors
    ///
    /// Returns `AprilTagError::MaxHammingTooLarge` if `max_hamming` exceeds the safe
    /// limit for this tag family (which is `(min_hamming - 1) / 2`).
    pub fn set_max_hamming(&mut self, max_hamming: u8) -> Result<(), AprilTagError> {
        let max_safe = self.max_safe_hamming();
        if max_hamming > max_safe {
            return Err(AprilTagError::MaxHammingTooLarge {
                max_hamming,
                min_hamming: self.min_hamming,
                max_safe,
            });
        }
        self.quick_decode.set_max_hamming(max_hamming)?;
        Ok(())
    }

    /// Returns the current maximum allowed Hamming distance.
    pub fn max_hamming(&self) -> u8 {
        self.quick_decode.max_hamming()
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
    /// The TagCircle49H12 Family. [TagFamily::tagcircle49_h12]
    TagCircle49H12,
    /// The TagCustom48H12 Family. [TagFamily::tagcustom48_h12]
    TagCustom48H12,
    /// The TagStandard41H12 Family. [TagFamily::tagstandard41_h12]
    TagStandard41H12,
    /// The TagStandard52H13 Family. [TagFamily::tagstandard52_h13]
    TagStandard52H13,
    /// A custom tag family, allowing users to supply a fully defined [`TagFamily`] instance.
    /// The [`Arc`] allows cheap cloning and shared ownership of the underlying [`TagFamily`]
    /// between multiple [`TagFamilyKind::Custom`] values without duplicating the tag data.
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

impl FromStr for TagFamilyKind {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if let Some(kind) = builtin_tag(s) {
            Ok(kind)
        } else {
            Err(format!("Unknown builtin tag family: '{}'", s))
        }
    }
}

// Maps names to variants
fn builtin_tag(name: &str) -> Option<TagFamilyKind> {
    match name {
        "tag16_h5" => Some(TagFamilyKind::Tag16H5),
        "tag36_h11" => Some(TagFamilyKind::Tag36H11),
        "tag36_h10" => Some(TagFamilyKind::Tag36H10),
        "tag25_h9" => Some(TagFamilyKind::Tag25H9),
        "tagcircle21_h7" => Some(TagFamilyKind::TagCircle21H7),
        "tagcircle49_h12" => Some(TagFamilyKind::TagCircle49H12),
        "tagcustom48_h12" => Some(TagFamilyKind::TagCustom48H12),
        "tagstandard41_h12" => Some(TagFamilyKind::TagStandard41H12),
        "tagstandard52_h13" => Some(TagFamilyKind::TagStandard52H13),
        _ => None,
    }
}

impl From<TagFamily> for TagFamilyKind {
    fn from(value: TagFamily) -> Self {
        TagFamilyKind::Custom(Arc::new(value))
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
    if let Some(kind) = builtin_tag(&value.name) {
        return kind;
    }
    TagFamilyKind::Custom(Arc::new(value.clone()))
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
            other => to_tag_family_impl(&other),
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

        // Assert that both point to the same memory
        if let (TagFamilyKind::Custom(arc1), TagFamilyKind::Custom(arc2)) =
            (custom_kind, custom_kind_clone)
        {
            assert!(
                Arc::ptr_eq(&arc1, &arc2),
                "Cloned Custom TagFamilyKind should share the same Arc pointer"
            );
        } else {
            panic!("Expected Custom variants");
        }
    }

    #[test]
    fn test_custom_family_try_unwrap_logic() {
        let family = TagFamily::tag36_h11().unwrap();
        let kind = TagFamilyKind::Custom(Arc::new(family));

        // Unique Ownership: try_unwrap should succeed
        let result = TagFamily::try_from(kind);
        assert!(result.is_ok());

        // Shared Ownership
        let family2 = TagFamily::tag36_h11().unwrap();
        let kind2 = TagFamilyKind::Custom(Arc::new(family2));
        let _kind_clone = kind2.clone();

        // This triggers the .unwrap_or_else(|a| (*a).clone()) path
        let result2 = TagFamily::try_from(kind2);
        let extracted = result2.expect("Should succeed via clone fallback");
        assert_eq!(extracted.name, "tag36_h11");
        assert_eq!(extracted.code_data.len(), 587);
        if let TagFamilyKind::Custom(arc) = _kind_clone {
            assert_eq!(arc.name, "tag36_h11");
        }
    }

    #[test]
    fn test_safety_no_silent_data_loss() {
        // Standard tag
        let mut family = TagFamily::tag36_h11().unwrap();
        family.width_at_border = 999;
        assert_eq!(family.name, "tag36_h11");

        // Convert it using From
        let kind = TagFamilyKind::from(family);
        if let TagFamilyKind::Custom(arc) = kind {
            assert_eq!(arc.width_at_border, 999, "User modification was preserved.");
        } else {
            panic!("Safety Fail: The library ignored user data and returned the standard builtin.");
        }
    }
}
