use crate::io;
use std::path::Path;

/// Image size in pixels
///
/// A struct to represent the size of an image in pixels.
///
/// # Examples
///
/// ```
/// use kornia_rs::image::ImageSize;
///
/// let image_size = ImageSize {
///    width: 10,
///   height: 20,
/// };
/// assert_eq!(image_size.width, 10);
/// assert_eq!(image_size.height, 20);
/// ```
#[derive(Clone, Debug)]
pub struct ImageSize {
    /// Width of the image in pixels
    pub width: usize,
    /// Height of the image in pixels
    pub height: usize,
}

impl std::fmt::Display for ImageSize {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "ImageSize {{ width: {}, height: {} }}",
            self.width, self.height
        )
    }
}

#[derive(Clone)]
/// Represents an image with pixel data.
pub struct Image {
    /// The pixel data of the image.
    pub data: ndarray::Array<u8, ndarray::Dim<[usize; 3]>>,
}

impl Image {
    pub fn new(shape: ImageSize, data: Vec<u8>) -> Image {
        let image =
            match ndarray::Array::<u8, _>::from_shape_vec([shape.height, shape.width, 3], data) {
                Ok(image) => image,
                Err(err) => {
                    panic!("Error converting image: {}", err);
                }
            };
        Image { data: image }
    }

    pub fn image_size(&self) -> ImageSize {
        ImageSize {
            width: self.data.shape()[1],
            height: self.data.shape()[0],
        }
    }

    pub fn num_channels(&self) -> usize {
        self.data.shape()[2]
    }

    pub fn from_shape_vec(shape: [usize; 3], data: Vec<u8>) -> Image {
        let image = match ndarray::Array::<u8, _>::from_shape_vec(shape, data) {
            Ok(image) => image,
            Err(err) => {
                panic!("Error converting image: {}", err);
            }
        };
        Image { data: image }
    }

    pub fn from_file(image_path: &Path) -> Image {
        match image_path.extension().and_then(|ext| ext.to_str()) {
            Some("jpeg") | Some("jpg") => io::functions::read_image_jpeg(image_path),
            _ => io::functions::read_image_any(image_path),
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn image_size() {
        use crate::image::ImageSize;
        let image_size = ImageSize {
            width: 10,
            height: 20,
        };
        assert_eq!(image_size.width, 10);
        assert_eq!(image_size.height, 20);
    }

    #[test]
    fn image_smoke() {
        use crate::image::{Image, ImageSize};
        let image = Image::new(
            ImageSize {
                width: 10,
                height: 20,
            },
            vec![0; 10 * 20 * 3],
        );
        assert_eq!(image.image_size().width, 10);
        assert_eq!(image.image_size().height, 20);
        assert_eq!(image.num_channels(), 3);
    }

    #[test]
    fn image_from_file() {
        use crate::image::Image;
        let image_path = std::path::Path::new("tests/data/dog.jpeg");
        let image = Image::from_file(image_path);
        assert_eq!(image.image_size().width, 258);
        assert_eq!(image.image_size().height, 195);
        assert_eq!(image.num_channels(), 3);
    }

    #[test]
    fn image_from_vec() {
        use crate::image::Image;
        let image = Image::from_shape_vec([2, 2, 3], vec![0; 2 * 2 * 3]);
        assert_eq!(image.image_size().width, 2);
        assert_eq!(image.image_size().height, 2);
        assert_eq!(image.num_channels(), 3);
    }
}
