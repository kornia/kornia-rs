use crate::{
    error::IoError,
    limits::{alloc_image, check_image_dimensions, MAX_IMAGE_PIXELS},
};
use kornia_image::{color_spaces::Rgb8, ImageSize};
use std::{fs, path::Path};
use zune_bmp::{
    zune_core::{bytestream::ZCursor, colorspace::ColorSpace, options::DecoderOptions},
    BmpDecoder,
};

/// Reads an uncompressed Windows BMP with a BITMAPINFOHEADER as RGB8.
///
/// Supports 24-bit pixels and 8-bit indexed pixels with a color palette.
///
/// # Arguments
/// * `file_path` - Path to the BMP file.
///
/// # Errors
/// Returns an I/O error if the file cannot be read, a format error for unsupported
/// layouts, or a decoding error for malformed or truncated input.
pub fn read_image_bmp_rgb8(file_path: impl AsRef<Path>) -> Result<Rgb8, IoError> {
    decode_image_bmp_rgb8(&fs::read(file_path)?)
}

fn decode_image_bmp_rgb8(data: &[u8]) -> Result<Rgb8, IoError> {
    let malformed = |reason: &str| IoError::BmpDecodingError(reason.to_owned());
    let Some(header) = data.get(..54).filter(|header| header.starts_with(b"BM")) else {
        return Err(malformed("Missing BMP file or DIB header"));
    };
    let u32_at = |offset| {
        let [a, b, c, d] = [
            header[offset],
            header[offset + 1],
            header[offset + 2],
            header[offset + 3],
        ];
        u32::from_le_bytes([a, b, c, d])
    };
    let bpp = u16::from_le_bytes([header[28], header[29]]);
    if u32_at(14) != 40 || u32_at(30) != 0 || !matches!(bpp, 8 | 24) {
        return Err(IoError::FormatMismatch(
            "Expected an uncompressed 24-bit or indexed 8-bit BMP with a BITMAPINFOHEADER".into(),
        ));
    }
    if header[26..28] != [1, 0] || u32_at(18) as i32 <= 0 || u32_at(22) == 0 {
        return Err(malformed("Invalid BMP dimensions or plane count"));
    }
    let width = u32_at(18) as usize;
    let height = (u32_at(22) as i32).unsigned_abs() as usize;
    check_image_dimensions(width, height)?;

    let pixel_offset = u32_at(10) as usize;
    if pixel_offset < 54 {
        return Err(malformed("Invalid BMP pixel offset"));
    }
    // zune can substitute zeros for missing bytes. Verify the complete padded rows first.
    let row_bytes = width
        .checked_mul(usize::from(bpp / 8))
        .and_then(|bytes| bytes.checked_add(3))
        .map(|bytes| bytes & !3)
        .ok_or_else(|| malformed("BMP row size overflow"))?;
    let pixels = row_bytes
        .checked_mul(height)
        .and_then(|len| data.get(pixel_offset..)?.get(..len))
        .ok_or_else(|| malformed("Truncated BMP pixel data"))?;
    if bpp == 8 {
        if pixel_offset == 54 {
            return Err(IoError::FormatMismatch(
                "Expected an RGB palette for 8-bit BMP".into(),
            ));
        }
        let colors = match u32_at(46) {
            0 => 256,
            count => count as usize,
        };
        if colors > 256 || (pixel_offset - 54) / 4 < colors {
            return Err(malformed("Invalid or truncated BMP palette"));
        }
        if pixels.chunks_exact(row_bytes).any(|row| {
            row[..width]
                .iter()
                .any(|&index| usize::from(index) >= colors)
        }) {
            return Err(malformed("BMP pixel references a missing palette entry"));
        }
    }

    let options = DecoderOptions::default()
        .set_max_width(MAX_IMAGE_PIXELS)
        .set_max_height(MAX_IMAGE_PIXELS)
        .set_strict_mode(true);
    let mut decoder = BmpDecoder::new_with_options(ZCursor::new(data), options);
    decoder
        .decode_headers()
        .map_err(|err| IoError::BmpDecodingError(format!("{err:?}")))?;
    if decoder.colorspace() != Some(ColorSpace::RGB) {
        return Err(IoError::FormatMismatch("Expected RGB BMP output".into()));
    }
    let mut image = alloc_image::<u8, 3>(ImageSize { width, height })?;
    if decoder.output_buf_size() != Some(image.as_slice().len()) {
        return Err(malformed("Inconsistent BMP output size"));
    }
    decoder
        .decode_into(image.as_slice_mut())
        .map_err(|err| IoError::BmpDecodingError(format!("{err:?}")))?;
    Ok(Rgb8(image))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::functional::read_image_any_rgb8;

    const FIXTURE: &[u8] = include_bytes!("../../../tests/data/rgb8-4x2.bmp");

    #[test]
    fn bmp_dispatch_preserves_pixels() -> Result<(), IoError> {
        let img = read_image_any_rgb8("../../tests/data/rgb8-4x2.bmp")?;
        assert_eq!((img.cols(), img.rows()), (4, 2));
        assert_eq!(
            img.as_slice(),
            &[
                255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 255, 255, 0, 255, 0, 255,
                0, 255, 255
            ]
        );
        Ok(())
    }

    #[test]
    fn bmp_rgb_ignores_row_padding() -> Result<(), IoError> {
        let mut bmp = FIXTURE.to_vec();
        bmp[18..22].copy_from_slice(&3i32.to_le_bytes());
        let img = decode_image_bmp_rgb8(&bmp)?;
        assert_eq!((img.cols(), img.rows()), (3, 2));
        assert_eq!(
            img.as_slice(),
            &[255, 0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 0, 255, 0, 255]
        );
        Ok(())
    }

    #[test]
    fn bmp_expands_palette_and_ignores_padding() -> Result<(), IoError> {
        let mut bmp = FIXTURE[..54].to_vec();
        bmp[2..6].copy_from_slice(&70u32.to_le_bytes());
        bmp[10..14].copy_from_slice(&62u32.to_le_bytes());
        bmp[18..22].copy_from_slice(&3i32.to_le_bytes());
        bmp[28..30].copy_from_slice(&8u16.to_le_bytes());
        bmp[34..38].copy_from_slice(&8u32.to_le_bytes());
        bmp[46..50].copy_from_slice(&2u32.to_le_bytes());
        bmp.extend_from_slice(&[0, 0, 255, 0, 255, 0, 0, 0, 1, 0, 1, 99, 0, 1, 0, 99]);
        let img = decode_image_bmp_rgb8(&bmp)?;
        assert_eq!((img.cols(), img.rows()), (3, 2));
        assert_eq!(
            img.as_slice(),
            &[255, 0, 0, 0, 0, 255, 255, 0, 0, 0, 0, 255, 255, 0, 0, 0, 0, 255]
        );
        bmp[62] = 2;
        assert!(matches!(
            decode_image_bmp_rgb8(&bmp),
            Err(IoError::BmpDecodingError(_))
        ));
        Ok(())
    }

    #[test]
    fn bmp_reports_typed_errors() {
        for data in [&b"BM"[..], &FIXTURE[..FIXTURE.len() - 1]] {
            assert!(matches!(
                decode_image_bmp_rgb8(data),
                Err(IoError::BmpDecodingError(_))
            ));
        }

        let mut bmp = FIXTURE.to_vec();
        bmp[28..30].copy_from_slice(&32u16.to_le_bytes());
        assert!(matches!(
            decode_image_bmp_rgb8(&bmp),
            Err(IoError::FormatMismatch(_))
        ));

        let mut bmp = FIXTURE.to_vec();
        bmp[18..22].copy_from_slice(&65_535i32.to_le_bytes());
        bmp[22..26].copy_from_slice(&65_535i32.to_le_bytes());
        assert!(matches!(
            decode_image_bmp_rgb8(&bmp),
            Err(IoError::ImageTooLarge { .. })
        ));
    }
}
