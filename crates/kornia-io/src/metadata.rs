use std::path::Path;

use crate::error::IoError;
use little_exif::filetype::FileExtension;

/// Simple image metadata extracted during decoding.
///
/// For now this only includes the EXIF Orientation tag (if present).
/// See `read_image_metadata` for the helper function that extracts this.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ImageMetadata {
    /// EXIF `Orientation` tag (1..8) if present, otherwise `None`.
    pub exif_orientation: Option<u16>,
}

/// Read only metadata from an image file.
///
/// This does NOT decode the image pixels. It returns `ImageMetadata` with
/// `exif_orientation` set when an EXIF Orientation tag is present and parsable.
/// File I/O errors are returned as `IoError::FileError`.
///
/// # Arguments
///
/// * `path` - The path to the image file (supports JPEG, PNG, TIFF).
///
/// # Returns
///
/// An `ImageMetadata` struct with `exif_orientation` set if found.
///
/// # Examples
///
/// ```no_run
/// use kornia_io::metadata::read_image_metadata;
///
/// let metadata = read_image_metadata("photo.jpg")?;
/// if let Some(orientation) = metadata.exif_orientation {
///     println!("Image has EXIF orientation: {}", orientation);
/// }
/// # Ok::<(), kornia_io::error::IoError>(())
/// ```
pub fn read_image_metadata<P: AsRef<Path>>(path: P) -> Result<ImageMetadata, IoError> {
    // Read the file bytes first. I/O errors are reported as `IoError::FileError`.
    let bytes = std::fs::read(path.as_ref()).map_err(IoError::FileError)?;

    // First try to find a raw EXIF block in the file (useful for synthetic
    // tests which write only the EXIF bytes). If present, parse orientation
    // directly without relying on full-file decoding.
    if let Some(v) = parse_exif_orientation(&bytes) {
        return Ok(ImageMetadata {
            exif_orientation: Some(v),
        });
    }

    // Otherwise, try to detect the file type and parse via `little_exif`.
    // Since little_exif doesn't expose get_file_type, we detect by extension.
    let ext = path
        .as_ref()
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase());

    let file_type = match ext.as_deref() {
        Some("jpg") | Some("jpeg") => Some(FileExtension::JPEG),
        Some("tif") | Some("tiff") => Some(FileExtension::TIFF),
        _ => None,
    };

    let metadata = if let Some(ft) = file_type {
        little_exif::metadata::Metadata::new_from_vec(&bytes, ft).ok()
    } else {
        // Fall back to trying JPEG parsing; if it fails treat as absent EXIF.
        little_exif::metadata::Metadata::new_from_vec(&bytes, FileExtension::JPEG).ok()
    };

    let mut orientation: Option<u16> = None;
    if let Some(metadata) = metadata {
        for tag in metadata.get_tag(&little_exif::exif_tag::ExifTag::Orientation(Vec::new())) {
            if let little_exif::exif_tag::ExifTag::Orientation(values) = tag {
                if let Some(&v) = values.first() {
                    if (1..=8).contains(&v) {
                        orientation = Some(v);
                        break;
                    }
                }
            }
        }
    }

    Ok(ImageMetadata {
        exif_orientation: orientation,
    })
}

// Minimal EXIF extractor for the Orientation tag. This is intentionally small and
// only implements what's required for our tests: locating a leading "Exif\0\0"
// block and reading IFD0 entries for tag 0x0112 with type SHORT (3).
fn parse_exif_orientation(bytes: &[u8]) -> Option<u16> {
    let needle = b"Exif\0\0";
    let pos = bytes.windows(needle.len()).position(|w| w == needle)?;
    let mut offset = pos + needle.len(); // start of TIFF header

    // Need at least 8 bytes for TIFF header (endian + 42 + offset)
    if offset + 8 > bytes.len() {
        return None;
    }

    let le = match &bytes[offset..offset + 2] {
        b"II" => true,
        b"MM" => false,
        _ => return None,
    };
    offset += 2;

    // Check magic number 42
    let magic = if le {
        u16::from_le_bytes([bytes[offset], bytes[offset + 1]])
    } else {
        u16::from_be_bytes([bytes[offset], bytes[offset + 1]])
    };
    if magic != 42 {
        return None;
    }
    offset += 2;

    // Offset to 0th IFD (relative to TIFF header start)
    let ifd_offset = if le {
        u32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ])
    } else {
        u32::from_be_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ])
    } as usize;

    let tiff_start = pos + needle.len();
    let ifd_pos = tiff_start + ifd_offset;
    if ifd_pos + 2 > bytes.len() {
        return None;
    }

    let num_entries = if le {
        u16::from_le_bytes([bytes[ifd_pos], bytes[ifd_pos + 1]])
    } else {
        u16::from_be_bytes([bytes[ifd_pos], bytes[ifd_pos + 1]])
    } as usize;

    let mut entry_pos = ifd_pos + 2;
    for _ in 0..num_entries {
        if entry_pos + 12 > bytes.len() {
            return None;
        }
        let tag = if le {
            u16::from_le_bytes([bytes[entry_pos], bytes[entry_pos + 1]])
        } else {
            u16::from_be_bytes([bytes[entry_pos], bytes[entry_pos + 1]])
        };

        let field_type = if le {
            u16::from_le_bytes([bytes[entry_pos + 2], bytes[entry_pos + 3]])
        } else {
            u16::from_be_bytes([bytes[entry_pos + 2], bytes[entry_pos + 3]])
        };

        let count = if le {
            u32::from_le_bytes([
                bytes[entry_pos + 4],
                bytes[entry_pos + 5],
                bytes[entry_pos + 6],
                bytes[entry_pos + 7],
            ])
        } else {
            u32::from_be_bytes([
                bytes[entry_pos + 4],
                bytes[entry_pos + 5],
                bytes[entry_pos + 6],
                bytes[entry_pos + 7],
            ])
        };

        if tag == 0x0112 && field_type == 3 && count >= 1 {
            // value is stored in the 4-byte value/offset field; for SHORT it's the
            // lower 2 bytes in the same endianness.
            let v0 = bytes[entry_pos + 8];
            let v1 = bytes[entry_pos + 9];
            let val = if le {
                u16::from_le_bytes([v0, v1])
            } else {
                u16::from_be_bytes([v0, v1])
            };
            if (1..=8).contains(&val) {
                return Some(val);
            } else {
                return None;
            }
        }

        entry_pos += 12;
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    /// Helper to write a unique temp file in the system temp dir.
    fn write_temp_file(name_suffix: &str, data: &[u8]) -> std::path::PathBuf {
        let mut path = std::env::temp_dir();
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis();
        path.push(format!("kornia_io_test_{}_{}.jpg", name_suffix, ts));
        fs::write(&path, data).expect("write temp file");
        path
    }

    #[test]
    fn test_missing_exif_returns_none() {
        // Use an existing small JPEG from the repo if available.
        let repo_img = std::path::Path::new("../../tests/data/dog.jpeg");
        if !repo_img.exists() {
            // If not present, skip the test.
            eprintln!("skipping test_missing_exif_returns_none: test image not found");
            return;
        }

        let bytes = fs::read(repo_img).expect("read repo jpeg");
        let tmp = write_temp_file("no_exif", &bytes);

        let meta = read_image_metadata(&tmp).expect("read metadata");
        assert!(meta.exif_orientation.is_none());

        let _ = fs::remove_file(tmp);
    }

    #[test]
    fn test_write_and_read_orientation_values() {
        let repo_img = std::path::Path::new("../../tests/data/dog.jpeg");
        if !repo_img.exists() {
            eprintln!("skipping test_write_and_read_orientation_values: test image not found");
            return;
        }

        let original = fs::read(repo_img).expect("read repo jpeg");

        for &val in &[1u16, 6u16, 8u16] {
            // Create metadata with Orientation tag set
            let mut metadata = little_exif::metadata::Metadata::new();
            metadata.set_tag(little_exif::exif_tag::ExifTag::Orientation(vec![val]));

            let mut buf = original.clone();
            metadata
                .write_to_vec(&mut buf, little_exif::filetype::FileExtension::JPEG)
                .expect("embed exif");

            let tmp = write_temp_file(&format!("orientation_{}", val), &buf);
            let meta = read_image_metadata(&tmp).expect("read metadata");
            assert_eq!(meta.exif_orientation, Some(val));

            let _ = fs::remove_file(tmp);
        }
    }

    #[test]
    fn test_all_orientation_values() {
        let repo_img = std::path::Path::new("../../tests/data/dog.jpeg");
        if !repo_img.exists() {
            eprintln!("skipping test_all_orientation_values: test image not found");
            return;
        }

        let original = fs::read(repo_img).expect("read repo jpeg");

        // Test all 8 valid EXIF orientation values
        for val in 1u16..=8 {
            let mut metadata = little_exif::metadata::Metadata::new();
            metadata.set_tag(little_exif::exif_tag::ExifTag::Orientation(vec![val]));

            let mut buf = original.clone();
            metadata
                .write_to_vec(&mut buf, little_exif::filetype::FileExtension::JPEG)
                .expect("embed exif");

            let tmp = write_temp_file(&format!("all_orientations_{}", val), &buf);
            let meta = read_image_metadata(&tmp).expect("read metadata");
            assert_eq!(
                meta.exif_orientation,
                Some(val),
                "Failed to read orientation value {}",
                val
            );

            let _ = fs::remove_file(tmp);
        }
    }
}
