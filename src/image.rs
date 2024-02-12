//use crate::io;
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
pub struct Image<T, const CHANNELS: usize> {
    /// The pixel data of the image. Is mutable so that we can manipulate the image
    /// from the outside.
    pub data: ndarray::Array<T, ndarray::Dim<[ndarray::Ix; 3]>>,
}

// provisionally, we will use the following types:
impl<T, const CHANNELS: usize> Image<T, CHANNELS> {
    pub fn new(shape: ImageSize, data: Vec<T>) -> Result<Self, std::io::Error> {
        // check if the data length matches the image size
        if data.len() != shape.width * shape.height * CHANNELS {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Data length ({}) does not match the image size ({})",
                    data.len(),
                    shape.width * shape.height * CHANNELS
                ),
            ));
        }

        // allocate the image data
        let data =
            ndarray::Array::<T, _>::from_shape_vec((shape.height, shape.width, CHANNELS), data)
                .expect("Failed to create image data");

        Ok(Image { data })
    }

    pub fn from_shape(shape: ImageSize) -> Result<Self, std::io::Error>
    where
        T: Clone + Default,
    {
        let data = vec![T::default(); shape.width * shape.height * CHANNELS];
        let image = Image::new(shape, data)?;

        Ok(image)
    }

    pub fn cast<U>(self) -> Result<Image<U, CHANNELS>, std::io::Error>
    where
        U: Clone + Default + num_traits::NumCast + std::fmt::Debug,
        T: Copy + num_traits::NumCast + std::fmt::Debug,
    {
        let casted_data = self
            .data
            .map(|&x| U::from(x).expect("Failed to cast image data"));

        Ok(Image { data: casted_data })
    }

    pub fn cast_and_scale<U>(self, scale: U) -> Result<Image<U, CHANNELS>, std::io::Error>
    where
        U: Copy
            + Clone
            + Default
            + num_traits::NumCast
            + std::fmt::Debug
            + std::ops::Mul<Output = U>,
        T: Copy + num_traits::NumCast + std::fmt::Debug,
    {
        let casted_data = self.data.map(|&x| {
            let xu = U::from(x).expect("Failed to cast image data");
            xu * scale
        });

        Ok(Image { data: casted_data })
    }

    pub fn image_size(&self) -> ImageSize {
        ImageSize {
            width: self.width(),
            height: self.height(),
        }
    }

    pub fn width(&self) -> usize {
        self.data.shape()[1]
    }

    pub fn height(&self) -> usize {
        self.data.shape()[0]
    }

    pub fn num_channels(&self) -> usize {
        //self.data.shape()[2]
        CHANNELS
    }

    pub fn data(self) -> ndarray::Array<T, ndarray::Dim<[ndarray::Ix; 3]>> {
        self.data
    }

    pub fn data_ref(&self) -> &ndarray::Array<T, ndarray::Dim<[ndarray::Ix; 3]>> {
        &self.data
    }

    //pub fn from_shape_vec(shape: [usize; 2], data: Vec<T>) -> Image<T> {
    //    let image = match ndarray::Array::<T, _>::from_shape_vec(shape, data) {
    //        Ok(image) => image,
    //        Err(err) => {
    //            panic!("Error converting image: {}", err);
    //        }
    //    };
    //    Image { data: image }
    //}

    //pub fn from_file(image_path: &Path) -> Image<u8, 3> {
    //    match image_path.extension().and_then(|ext| ext.to_str()) {
    //        Some("jpeg") | Some("jpg") => io::functions::read_image_jpeg(image_path),
    //        _ => io::functions::read_image_any(image_path),
    //    }
    //}

    // TODO: implement from bytes
    // pub fn from_bytes(bytes: &[u8]) -> Image {
}

#[cfg(test)]
mod tests {
    use crate::image::ImageSize;

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
        let image = Image::<u8, 3>::new(
            ImageSize {
                width: 10,
                height: 20,
            },
            vec![0u8; 10 * 20 * 3],
        )
        .unwrap();
        assert_eq!(image.image_size().width, 10);
        assert_eq!(image.image_size().height, 20);
        assert_eq!(image.num_channels(), 3);
    }

    #[test]
    //fn image_from_file() {
    //    use crate::image::Image;
    //    let image_path = std::path::Path::new("tests/data/dog.jpeg");
    //    let image = Image::<u8, 3>::from_file(image_path);
    //    assert_eq!(image.image_size().width, 258);
    //    assert_eq!(image.image_size().height, 195);
    //    assert_eq!(image.num_channels(), 3);
    //}
    #[test]
    fn image_from_vec() {
        use crate::image::Image;
        let image: Image<f32, 3> = Image::new(
            ImageSize {
                height: 3,
                width: 2,
            },
            vec![0.0; 3 * 2 * 3],
        )
        .unwrap();
        assert_eq!(image.image_size().width, 2);
        assert_eq!(image.image_size().height, 3);
        assert_eq!(image.num_channels(), 3);
    }

    #[test]
    fn image_cast() {
        use crate::image::Image;
        let data = vec![0., 1., 2., 3., 4., 5.];
        let image_f64 = Image::<f64, 3>::new(
            ImageSize {
                height: 2,
                width: 1,
            },
            data,
        )
        .unwrap();
        assert_eq!(image_f64.data.get((1, 0, 2)).unwrap(), &5.0f64);

        let image_u8 = image_f64.cast::<u8>().unwrap();
        assert_eq!(image_u8.data.get((1, 0, 2)).unwrap(), &5u8);

        let image_i32: Image<i32, 3> = image_u8.cast().unwrap();
        assert_eq!(image_i32.data.get((1, 0, 2)).unwrap(), &5i32);
    }

    #[test]
    fn image_rgbd() {
        use crate::image::Image;
        let image = Image::<f32, 4>::new(
            ImageSize {
                height: 2,
                width: 3,
            },
            vec![0f32; 2 * 3 * 4],
        )
        .unwrap();
        assert_eq!(image.image_size().width, 3);
        assert_eq!(image.image_size().height, 2);
        assert_eq!(image.num_channels(), 4);
    }
}
