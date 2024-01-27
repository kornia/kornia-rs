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
#[derive(Clone)]
pub struct ImageSize {
    /// Width of the image in pixels
    pub width: usize,
    /// Height of the image in pixels
    pub height: usize,
}

#[derive(Clone)]
pub struct Image {
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
}
