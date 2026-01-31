use std::fs::File;
use std::io::Read;
use std::num::NonZeroU8;
use std::path::Path;

use crate::error::IoError;
use little_exif::filetype::FileExtension;

/// Simple image metadata extracted during decoding.
/// For now this only includes the EXIF Orientation tag (if present).
/// See `read_image_metadata` for the helper function that extracts this.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ImageMetadata {
    /// EXIF `Orientation` tag (1..8) if present, otherwise `None`.
    /// Uses `NonZeroU8` to compress the value to a single byte.
    pub exif_orientation: Option<NonZeroU8>,
}

/// Read metadata from an image file without decoding pixels.
///
/// # Arguments
///
/// * `path` - Path to image file (JPEG, PNG, or TIFF).
///
/// # Returns
///
/// `ImageMetadata` with `exif_orientation` set if found.
///
/// # Example
///
/// ```no_run
/// use kornia_io::metadata::read_image_metadata;
///
/// let metadata = read_image_metadata("photo.jpg")?;
/// if let Some(orientation) = metadata.exif_orientation {
///     println!("Orientation: {}", orientation.get());
/// }
/// # Ok::<(), kornia_io::error::IoError>(())
/// ```
pub fn read_image_metadata<P: AsRef<Path>>(path: P) -> Result<ImageMetadata, IoError> {
    let path = path.as_ref();
    let mut file = File::open(path).map_err(IoError::FileError)?;

    // Read the first 128KB to check for EXIF in the header.
    // This covers most use cases without reading the full file.
    let mut buffer = vec![0; 128 * 1024];
    let n = file.read(&mut buffer).map_err(IoError::FileError)?;
    buffer.truncate(n);

    // Fast path: try to parse EXIF orientation from the header.
    // This avoids the overhead of the full little_exif parser for the common case.
    if let Some(v) = parse_exif_orientation(&buffer) {
        return Ok(ImageMetadata {
            exif_orientation: Some(v),
        });
    }

    // Slow path: read the rest of the file if needed for little_exif.
    if n == 128 * 1024 {
        file.read_to_end(&mut buffer).map_err(IoError::FileError)?;
    }

    // File type must be determined from extension
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase());

    let file_type = match ext.as_deref() {
        Some("jpg") | Some("jpeg") => FileExtension::JPEG,
        Some("tif") | Some("tiff") => FileExtension::TIFF,
        Some("png") => FileExtension::PNG {
            as_zTXt_chunk: false,
        },
        _ => return Err(IoError::InvalidFileExtension(path.to_path_buf())),
    };

    let metadata = little_exif::metadata::Metadata::new_from_vec(&buffer, file_type).ok();

    let orientation = metadata.and_then(|m| {
        m.get_tag(&little_exif::exif_tag::ExifTag::Orientation(Vec::new()))
            .into_iter()
            .find_map(|tag| {
                if let little_exif::exif_tag::ExifTag::Orientation(values) = tag {
                    values.first().and_then(|&v| {
                        if (1..=8).contains(&v) {
                            NonZeroU8::new(v as u8)
                        } else {
                            None
                        }
                    })
                } else {
                    None
                }
            })
    });
    Ok(ImageMetadata {
        exif_orientation: orientation,
    })
}

fn read_u16(bytes: &[u8], offset: usize, le: bool) -> u16 {
    if le {
        u16::from_le_bytes([bytes[offset], bytes[offset + 1]])
    } else {
        u16::from_be_bytes([bytes[offset], bytes[offset + 1]])
    }
}

fn read_u32(bytes: &[u8], offset: usize, le: bool) -> u32 {
    if le {
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
    }
}

// Fast-path parser extracts orientation from first 128KB without full EXIF parsing
fn parse_exif_orientation(bytes: &[u8]) -> Option<NonZeroU8> {
    // Locate EXIF header marker
    let needle = b"Exif\0\0";
    let pos = bytes.windows(needle.len()).position(|w| w == needle)?;
    let mut offset = pos + needle.len();
    if offset + 8 > bytes.len() {
        return None;
    }

    // Determine byte order (II or MM)
    let le = match &bytes[offset..offset + 2] {
        b"II" => true,
        b"MM" => false,
        _ => return None,
    };
    offset += 2;

    // Check TIFF magic number (42)
    let magic = read_u16(bytes, offset, le);

    if magic != 42 {
        return None;
    }
    offset += 2;

    let ifd_offset = read_u32(bytes, offset, le) as usize;

    let tiff_start = pos + needle.len();
    let ifd_pos = tiff_start + ifd_offset;
    if ifd_pos + 2 > bytes.len() {
        return None;
    }

    let num_entries = read_u16(bytes, ifd_pos, le) as usize;

    let mut entry_pos = ifd_pos + 2;
    for _ in 0..num_entries {
        if entry_pos + 12 > bytes.len() {
            return None;
        }

        let tag = read_u16(bytes, entry_pos, le);
        let field_type = read_u16(bytes, entry_pos + 2, le);
        let count = read_u32(bytes, entry_pos + 4, le);

        // Orientation tag is short type (3)
        if tag == 0x0112 && field_type == 3 && count >= 1 {
            let val = read_u16(bytes, entry_pos + 8, le);
            if let Some(nz) = NonZeroU8::new(val as u8) {
                if (1..=8).contains(&nz.get()) {
                    return Some(nz);
                }
            }
            return None;
        }

        entry_pos += 12;
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_missing_exif_returns_none() {
        let repo_img = std::path::Path::new("../../tests/data/dog.jpeg");
        let bytes = fs::read(repo_img).expect("read repo jpeg");

        let tmp_file = tempfile::Builder::new()
            .suffix(".jpg")
            .tempfile()
            .expect("create temp file");

        fs::write(tmp_file.path(), &bytes).expect("write temp file");

        let meta = read_image_metadata(tmp_file.path()).expect("read metadata");
        assert!(meta.exif_orientation.is_none());
    }

    #[test]
    fn test_write_and_read_orientation_values() {
        let repo_img = std::path::Path::new("../../tests/data/dog.jpeg");
        let original = fs::read(repo_img).expect("read repo jpeg");

        for &val in &[1u16, 6u16, 8u16] {
            // Create metadata with Orientation tag set
            let mut metadata = little_exif::metadata::Metadata::new();
            metadata.set_tag(little_exif::exif_tag::ExifTag::Orientation(vec![val]));

            let mut buf = original.clone();
            metadata
                .write_to_vec(&mut buf, little_exif::filetype::FileExtension::JPEG)
                .expect("embed exif");

            let tmp_file = tempfile::Builder::new()
                .suffix(".jpg")
                .tempfile()
                .expect("create temp file");

            fs::write(tmp_file.path(), &buf).expect("write temp file");

            let meta = read_image_metadata(tmp_file.path()).expect("read metadata");
            assert_eq!(meta.exif_orientation, NonZeroU8::new(val as u8));
        }
    }

    #[test]
    fn test_all_orientation_values() {
        let repo_img = std::path::Path::new("../../tests/data/dog.jpeg");
        let original = fs::read(repo_img).expect("read repo jpeg");

        for val in 1u16..=8 {
            let mut metadata = little_exif::metadata::Metadata::new();
            metadata.set_tag(little_exif::exif_tag::ExifTag::Orientation(vec![val]));

            let mut buf = original.clone();
            metadata
                .write_to_vec(&mut buf, little_exif::filetype::FileExtension::JPEG)
                .expect("embed exif");

            let tmp_file = tempfile::Builder::new()
                .suffix(".jpg")
                .tempfile()
                .expect("create temp file");

            fs::write(tmp_file.path(), &buf).expect("write temp file");

            let meta = read_image_metadata(tmp_file.path()).expect("read metadata");
            assert_eq!(
                meta.exif_orientation,
                NonZeroU8::new(val as u8),
                "Failed to read orientation value {}",
                val
            );
        }
    }
}
