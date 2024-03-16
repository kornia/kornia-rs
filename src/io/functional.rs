use anyhow::Result;
use std::path::Path;

use crate::image::{Image, ImageSize};

use super::jpeg::{ImageDecoder, ImageEncoder};

/// Reads a JPEG image from the given file path.
///
/// The method reads the JPEG image data directly from a file leveraging the libjpeg-turbo library.
///
/// # Arguments
///
/// * `image_path` - The path to the JPEG image.
///
/// # Returns
///
/// An in image containing the JPEG image data.
///
/// # Example
///
/// ```
/// use kornia_rs::image::Image;
/// use kornia_rs::io::functional as F;
///
/// let image_path = std::path::Path::new("tests/data/dog.jpeg");
/// let image: Image<u8, 3> = F::read_image_jpeg(image_path).unwrap();
/// assert_eq!(image.size().width, 258);
/// assert_eq!(image.size().height, 195);
/// assert_eq!(image.num_channels(), 3);
/// ```
pub fn read_image_jpeg(file_path: &Path) -> Result<Image<u8, 3>> {
    // verify the file exists and is a JPEG
    if !file_path.exists() {
        return Err(anyhow::anyhow!(
            "File does not exist: {}",
            file_path.to_str().unwrap()
        ));
    }

    if file_path.extension().map_or(true, |ext| {
        ext.to_ascii_lowercase() != "jpg" && ext.to_ascii_lowercase() != "jpeg"
    }) {
        return Err(anyhow::anyhow!(
            "File is not a JPEG: {}",
            file_path.to_str().unwrap()
        ));
    }

    // open the file and map it to memory
    let file = std::fs::File::open(file_path)?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };

    // decode the data directly from memory
    let image: Image<u8, 3> = {
        let mut decoder = ImageDecoder::new()?;
        decoder.decode(&mmap)?
    };

    Ok(image)
}

/// Writes the given JPEG data to the given file path.
///
/// # Arguments
///
/// * `file_path` - The path to the JPEG image.
/// * `image` - The tensor containing the JPEG image data.
pub fn write_image_jpeg(file_path: &Path, image: &Image<u8, 3>) -> Result<()> {
    // compress the image
    let jpeg_data = ImageEncoder::new()?.encode(image)?;

    // write the data directly to a file
    std::fs::write(file_path, jpeg_data)?;

    Ok(())
}

/// Reads an image from the given file path.
///
/// The method tries to read from any image format supported by the image crate.
///
/// # Arguments
///
/// * `file_path` - The path to the image.
///
/// # Returns
///
/// A tensor containing the image data.
///
/// # Example
///
/// ```
/// use kornia_rs::image::Image;
/// use kornia_rs::io::functional as F;
///
/// let image_path = std::path::Path::new("tests/data/dog.jpeg");
/// let image: Image<u8, 3> = F::read_image_any(image_path).unwrap();
/// assert_eq!(image.size().width, 258);
/// assert_eq!(image.size().height, 195);
/// assert_eq!(image.num_channels(), 3);
/// ```
pub fn read_image_any(file_path: &Path) -> Result<Image<u8, 3>> {
    // verify the file exists
    if !file_path.exists() {
        return Err(anyhow::anyhow!(
            "File does not exist: {}",
            file_path.to_str().unwrap()
        ));
    }

    // open the file and map it to memory
    let file = std::fs::File::open(file_path)?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };

    // decode the data directly from memory
    let img = image::io::Reader::new(std::io::Cursor::new(&mmap))
        .with_guessed_format()?
        .decode()?;

    // return the image data
    let data = img.to_rgb8().to_vec();
    let image = Image::new(
        ImageSize {
            width: img.width() as usize,
            height: img.height() as usize,
        },
        data,
    )?;

    Ok(image)
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;
    use tempfile::tempdir;

    use crate::io::functional::{read_image_any, read_image_jpeg, write_image_jpeg};

    #[test]
    fn read_jpeg() {
        let image_path = Path::new("tests/data/dog.jpeg");
        let image = read_image_jpeg(image_path).unwrap();
        assert_eq!(image.size().width, 258);
        assert_eq!(image.size().height, 195);
    }

    #[test]
    fn read_any() {
        let image_path = Path::new("tests/data/dog.jpeg");
        let image = read_image_any(image_path).unwrap();
        assert_eq!(image.size().width, 258);
        assert_eq!(image.size().height, 195);
    }

    #[test]
    fn read_write_jpeg() {
        let image_path_read = Path::new("tests/data/dog.jpeg");
        let tmp_dir = tempdir().unwrap();
        fs::create_dir_all(tmp_dir.path()).unwrap();
        let file_path = tmp_dir.path().join("dog.jpeg");
        let image_data = read_image_jpeg(image_path_read).unwrap();
        write_image_jpeg(&file_path, &image_data).unwrap();
        let image_data_back = read_image_jpeg(&file_path).unwrap();
        assert!(file_path.exists(), "File does not exist: {:?}", file_path);
        assert_eq!(image_data_back.size().width, 258);
        assert_eq!(image_data_back.size().height, 195);
        assert_eq!(image_data_back.num_channels(), 3);
    }
}
