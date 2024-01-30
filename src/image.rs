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
pub struct Image<T = u8> {
    /// The pixel data of the image. Is mutable so that we can manipulate the image
    /// from the outside.
    pub data: ndarray::Array<T, ndarray::Dim<[usize; 3]>>,
}

// provisionally, we will use the following types:
pub type ImageU8C3 = Image<u8>; // HW3 (u8)
pub type ImageF32C3 = Image<f32>; // HW3 (f32)
pub type ImageF64C3 = Image<f64>; // HW3 (f64)

impl<T> Image<T> {
    pub fn new(shape: ImageSize, data: Vec<T>) -> Image<T> {
        let image =
            match ndarray::Array::<T, _>::from_shape_vec((shape.height, shape.width, 3), data) {
                Ok(image) => image,
                Err(err) => {
                    panic!("Error converting image: {}", err);
                }
            };
        Image { data: image }
    }

    pub fn cast<U>(&self) -> Image<U>
    where
        T: num_traits::cast::AsPrimitive<U>,
        U: num_traits::NumAssign + num_traits::cast::AsPrimitive<T>,
    {
        let image = self.data.map(|x| x.as_());
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

    pub fn from_shape_vec(shape: [usize; 3], data: Vec<T>) -> Image<T> {
        let image = match ndarray::Array::<T, _>::from_shape_vec(shape, data) {
            Ok(image) => image,
            Err(err) => {
                panic!("Error converting image: {}", err);
            }
        };
        Image { data: image }
    }

    pub fn from_file(image_path: &Path) -> Image<u8> {
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
            vec![0u8; 10 * 20 * 3],
        );
        assert_eq!(image.image_size().width, 10);
        assert_eq!(image.image_size().height, 20);
        assert_eq!(image.num_channels(), 3);
    }

    #[test]
    fn image_from_file() {
        use crate::image::Image;
        let image_path = std::path::Path::new("tests/data/dog.jpeg");
        let image = Image::<u8>::from_file(image_path);
        assert_eq!(image.image_size().width, 258);
        assert_eq!(image.image_size().height, 195);
        assert_eq!(image.num_channels(), 3);
    }

    #[test]
    fn image_from_vec() {
        use crate::image::Image;
        let image = Image::from_shape_vec([2, 2, 3], vec![0f32; 2 * 2 * 3]);
        assert_eq!(image.image_size().width, 2);
        assert_eq!(image.image_size().height, 2);
        assert_eq!(image.num_channels(), 3);
    }

    #[test]
    fn image_cast() {
        use crate::image::Image;
        let data = vec![0., 1., 2., 3., 4., 5.];
        let image_f64 = Image::from_shape_vec([2, 1, 3], data);
        assert_eq!(image_f64.data.get((1, 0, 2)).unwrap(), &5.0f64);

        let image_u8 = image_f64.cast::<u8>();
        assert_eq!(image_u8.data.get((1, 0, 2)).unwrap(), &5u8);

        let image_i32: Image<i32> = image_u8.cast();
        assert_eq!(image_i32.data.get((1, 0, 2)).unwrap(), &5i32);
    }
}
