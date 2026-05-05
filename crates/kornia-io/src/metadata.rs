use std::fs::File;
use std::io::Read;
use std::num::NonZeroU8;
use std::path::Path;

use crate::error::IoError;
use kornia_image::allocator::CpuAllocator;
use kornia_image::{Image, ImageSize};
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
    if !path.exists() {
        return Err(IoError::FileDoesNotExist(path.to_path_buf()));
    }
    let mut file = File::open(path).map_err(IoError::FileError)?;

    // Read the first 128KB to check for EXIF in the header.
    let mut buffer = vec![0; 128 * 1024];
    let n = file.read(&mut buffer).map_err(IoError::FileError)?;
    buffer.truncate(n);

    // Fast path: try to parse EXIF orientation from the header.
    if let Some(v) = parse_exif_orientation(&buffer) {
        return Ok(ImageMetadata {
            exif_orientation: Some(v),
        });
    }

    // Always read the rest of the file for little_exif slow path.
    file.read_to_end(&mut buffer).map_err(IoError::FileError)?;

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

#[cfg(test)]
mod orientation_tests {
    use super::*;
    use kornia_image::{allocator::CpuAllocator, Image, ImageSize};
    use std::num::NonZeroU8;

    // 1. make_test_image returns Image<f32, 3, CpuAllocator>
    fn make_test_image() -> Image<f32, 3, CpuAllocator> {
        let mut img = Image::<f32, 3, _>::from_size_val(
            ImageSize {
                width: 3,
                height: 2,
            },
            0.0,
            CpuAllocator,
        )
        .unwrap();
        img.set_pixel(0, 0, 0, 10.0).unwrap();
        img.set_pixel(2, 0, 0, 20.0).unwrap();
        img.set_pixel(0, 1, 0, 30.0).unwrap();
        img.set_pixel(2, 1, 0, 40.0).unwrap();
        img
    }

    // 2. test_orientation_identity — compare pixel values
    #[test]
    fn test_orientation_identity() {
        let img = make_test_image();
        let out = apply_exif_orientation(img, NonZeroU8::new(1).unwrap()).unwrap();
        assert_eq!(out.size().width, 3);
        assert_eq!(out.size().height, 2);
        assert_eq!(*out.get_pixel(0, 0, 0).unwrap(), 10.0_f32);
        assert_eq!(*out.get_pixel(2, 1, 0).unwrap(), 40.0_f32);
    }

    #[test]
    fn test_orientation_6_and_8_swap_dims() {
        let img = make_test_image();
        let out6 = apply_exif_orientation(img.clone(), NonZeroU8::new(6).unwrap()).unwrap();
        let out8 = apply_exif_orientation(img, NonZeroU8::new(8).unwrap()).unwrap();
        assert_eq!(out6.size().width, 2);
        assert_eq!(out6.size().height, 3);
        assert_eq!(out8.size().width, 2);
        assert_eq!(out8.size().height, 3);
    }

    #[test]
    fn test_all_orientations_corners() {
        let expected: [(f32, f32, f32, f32); 8] = [
            (10.0, 20.0, 30.0, 40.0),
            (20.0, 10.0, 40.0, 30.0),
            (40.0, 30.0, 20.0, 10.0),
            (30.0, 40.0, 10.0, 20.0),
            (10.0, 30.0, 20.0, 40.0),
            (30.0, 10.0, 40.0, 20.0),
            (40.0, 20.0, 30.0, 10.0),
            (20.0, 40.0, 10.0, 30.0), // 8: rotate 90 CCW
        ];
        for (i, &(tl, tr, bl, br)) in expected.iter().enumerate() {
            let img = make_test_image();
            let out = apply_exif_orientation(img, NonZeroU8::new((i + 1) as u8).unwrap()).unwrap();
            let w = out.size().width;
            let h = out.size().height;
            assert_eq!(
                *out.get_pixel(0, 0, 0).unwrap(),
                tl,
                "TL failed for orientation {}",
                i + 1
            );
            assert_eq!(
                *out.get_pixel(w - 1, 0, 0).unwrap(),
                tr,
                "TR failed for orientation {}",
                i + 1
            );
            assert_eq!(
                *out.get_pixel(0, h - 1, 0).unwrap(),
                bl,
                "BL failed for orientation {}",
                i + 1
            );
            assert_eq!(
                *out.get_pixel(w - 1, h - 1, 0).unwrap(),
                br,
                "BR failed for orientation {}",
                i + 1
            );
        }
    }

    // End-to-end test for read_image_jpeg_auto_orient would require JPEG IO and EXIF embedding,
    // which is not shown here due to test environment constraints.
}

use crate::jpeg::read_image_jpeg_rgb8;

/// Applies EXIF orientation correction using exact pixel remapping.
pub fn apply_exif_orientation(
    image: Image<f32, 3, CpuAllocator>,
    orientation: NonZeroU8,
) -> Result<Image<f32, 3, CpuAllocator>, IoError> {
    let src_w = image.size().width;
    let src_h = image.size().height;
    let o = orientation.get();

    let out_size = match o {
        1..=4 => ImageSize {
            width: src_w,
            height: src_h,
        },
        5..=8 => ImageSize {
            width: src_h,
            height: src_w,
        },
        _ => return Err(IoError::InvalidOrientation(o as u16)),
    };

    let mut dst = Image::<f32, 3, CpuAllocator>::from_size_val(out_size, 0.0, CpuAllocator)
        .map_err(IoError::ImageCreationError)?;

    for sy in 0..src_h {
        for sx in 0..src_w {
            let (dx, dy) = match o {
                1 => (sx, sy),
                2 => (src_w - 1 - sx, sy),
                3 => (src_w - 1 - sx, src_h - 1 - sy),
                4 => (sx, src_h - 1 - sy),
                5 => (sy, sx),
                6 => (src_h - 1 - sy, sx),
                7 => (src_h - 1 - sy, src_w - 1 - sx),
                8 => (sy, src_w - 1 - sx),
                _ => unreachable!(),
            };
            for c in 0..3 {
                let value = *image
                    .get_pixel(sx, sy, c)
                    .map_err(IoError::ImageCreationError)?;
                dst.set_pixel(dx, dy, c, value)
                    .map_err(IoError::ImageCreationError)?;
            }
        }
    }
    Ok(dst)
}

/// Convenience reader: loads JPEG and applies EXIF orientation if present.
/// Returns an f32 image in [0, 255] range.
pub fn read_image_jpeg_auto_orient<P: AsRef<Path>>(
    path: P,
) -> Result<Image<f32, 3, CpuAllocator>, IoError> {
    let meta = read_image_metadata(&path)?;
    let rgb8 = read_image_jpeg_rgb8(&path)?;
    // Convert Rgb8 → Image<f32, 3>
    let image = rgb8.cast::<f32>().map_err(IoError::ImageCreationError)?;
    if let Some(orientation) = meta.exif_orientation {
        apply_exif_orientation(image, orientation)
    } else {
        Ok(image)
    }
}
