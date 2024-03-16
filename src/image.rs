//use crate::io;
use anyhow::Result;
use num_traits::Float;

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
#[derive(Clone, Copy, Debug, PartialEq)]
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
///
/// The image is represented as a 3D array with shape (H, W, C), where H is the height of the image,
/// The ownership of the pixel data is mutable so that we can manipulate the image from the outside.
pub struct Image<T, const CHANNELS: usize> {
    /// The pixel data of the image. Is mutable so that we can manipulate the image
    /// from the outside.
    pub data: ndarray::Array<T, ndarray::Dim<[ndarray::Ix; 3]>>,
}

impl<T, const CHANNELS: usize> Image<T, CHANNELS> {
    /// Create a new image from pixel data.
    ///
    /// # Arguments
    ///
    /// * `size` - The size of the image in pixels.
    /// * `data` - The pixel data of the image.
    ///
    /// # Returns
    ///
    /// A new image with the given pixel data.
    ///
    /// # Errors
    ///
    /// If the length of the pixel data does not match the image size, an error is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use kornia_rs::image::{Image, ImageSize};
    ///
    /// let image = Image::<u8, 3>::new(
    ///    ImageSize {
    ///       width: 10,
    ///      height: 20,
    ///  },
    /// vec![0u8; 10 * 20 * 3],
    /// ).unwrap();
    ///
    /// assert_eq!(image.size().width, 10);
    /// assert_eq!(image.size().height, 20);
    /// assert_eq!(image.num_channels(), 3);
    /// ```
    pub fn new(size: ImageSize, data: Vec<T>) -> Result<Self> {
        // check if the data length matches the image size
        if data.len() != size.width * size.height * CHANNELS {
            return Err(anyhow::anyhow!(
                "Data length ({}) does not match the image size ({})",
                data.len(),
                size.width * size.height * CHANNELS
            ));
        }

        // allocate the image data
        let data =
            ndarray::Array::<T, _>::from_shape_vec((size.height, size.width, CHANNELS), data)?;

        Ok(Image { data })
    }

    /// Create a new image with the given size and default pixel data.
    ///
    /// # Arguments
    ///
    /// * `size` - The size of the image in pixels.
    /// * `val` - The default value of the pixel data.
    ///
    /// # Returns
    ///
    /// A new image with the given size and default pixel data.
    ///
    /// # Errors
    ///
    /// If the length of the pixel data does not match the image size, an error is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use kornia_rs::image::{Image, ImageSize};
    ///
    /// let image = Image::<u8, 3>::from_size_val(
    ///   ImageSize {
    ///     width: 10,
    ///    height: 20,
    /// }, 0u8).unwrap();
    ///
    /// assert_eq!(image.size().width, 10);
    /// assert_eq!(image.size().height, 20);
    /// assert_eq!(image.num_channels(), 3);
    /// ```
    pub fn from_size_val(size: ImageSize, val: T) -> Result<Self>
    where
        T: Clone + Default,
    {
        let data = vec![val; size.width * size.height * CHANNELS];
        let image = Image::new(size, data)?;

        Ok(image)
    }

    /// Cast the pixel data to a different type.
    ///
    /// # Arguments
    ///
    /// * `scale` - The scale to multiply the pixel data with.
    ///
    /// # Returns
    ///
    /// A new image with the pixel data cast to the new type.
    ///
    /// # Errors
    ///
    /// If the pixel data cannot be cast to the new type, an error is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use kornia_rs::image::{Image, ImageSize};
    ///
    /// let data = vec![0., 1., 2., 3., 4., 5.];
    ///
    /// let image_f64 = Image::<f64, 3>::new(
    ///  ImageSize {
    ///   height: 2,
    ///  width: 1,
    /// },
    /// data,
    /// ).unwrap();
    ///
    /// assert_eq!(image_f64.data.get((1, 0, 2)).unwrap(), &5.0f64);
    ///
    /// let image_u8 = image_f64.cast::<u8>().unwrap();
    ///
    /// assert_eq!(image_u8.data.get((1, 0, 2)).unwrap(), &5u8);
    ///
    /// let image_i32: Image<i32, 3> = image_u8.cast().unwrap();
    ///
    /// assert_eq!(image_i32.data.get((1, 0, 2)).unwrap(), &5i32);
    /// ```
    pub fn cast<U>(self) -> Result<Image<U, CHANNELS>>
    where
        U: Clone + Default + num_traits::NumCast + std::fmt::Debug,
        T: Copy + num_traits::NumCast + std::fmt::Debug,
    {
        let casted_data = self
            .data
            .map(|&x| U::from(x).expect("Failed to cast image data"));

        Ok(Image { data: casted_data })
    }

    /// Cast the pixel data to a different type and scale it.
    ///
    /// # Arguments
    ///
    /// * `scale` - The scale to multiply the pixel data with.
    ///
    /// # Returns
    ///
    /// A new image with the pixel data cast to the new type and scaled.
    ///
    /// # Errors
    ///
    /// If the pixel data cannot be cast to the new type, an error is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use kornia_rs::image::{Image, ImageSize};
    ///
    /// let data = vec![0u8, 0, 255, 0, 0, 255];
    ///
    /// let image_u8 = Image::<u8, 3>::new(
    /// ImageSize {
    ///   height: 2,
    ///   width: 1,
    /// },
    /// data,
    /// ).unwrap();
    ///
    /// let image_f32 = image_u8.cast_and_scale::<f32>(1. / 255.0).unwrap();
    ///
    /// assert_eq!(image_f32.data.get((1, 0, 2)).unwrap(), &1.0f32);
    /// ```
    pub fn cast_and_scale<U>(self, scale: U) -> Result<Image<U, CHANNELS>>
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

    // TODO: optimize this
    pub fn mul(&self, scale: T) -> Self
    where
        T: Copy + std::ops::Mul<Output = T>,
    {
        let scaled_data = self.data.map(|&x| x * scale);
        Image { data: scaled_data }
    }

    // TODO: optimize this
    pub fn div(&self, scale: T) -> Self
    where
        T: Copy + std::ops::Div<Output = T>,
    {
        let scaled_data = self.data.map(|&x| x / scale);
        Image { data: scaled_data }
    }

    // TODO: optimize this
    pub fn sub(&self, other: &Self) -> Self
    where
        T: Copy + std::ops::Sub<Output = T>,
    {
        let diff = &self.data - &other.data;
        Image { data: diff }
    }

    // TODO: optimize this
    pub fn powi(&self, n: i32) -> Self
    where
        T: Copy + Float,
    {
        let powered_data = self.data.map(|&x| x.powi(n));
        Image { data: powered_data }
    }

    // TODO: optimize this
    pub fn mean(&self) -> T
    where
        T: Copy + Float,
    {
        self.data.fold(T::zero(), |acc, &x| acc + x) / T::from(self.data.len()).unwrap()
    }

    pub fn abs(&self) -> Self
    where
        T: Copy + Float,
    {
        let abs_data = self.data.map(|&x| x.abs());
        Image { data: abs_data }
    }

    /// Get the size of the image in pixels.
    #[deprecated(since = "0.1.2", note = "Use `image.size()` instead")]
    pub fn image_size(&self) -> ImageSize {
        ImageSize {
            width: self.width(),
            height: self.height(),
        }
    }

    /// Get the size of the image in pixels.
    pub fn size(&self) -> ImageSize {
        ImageSize {
            width: self.width(),
            height: self.height(),
        }
    }

    /// Get the width of the image in pixels.
    pub fn width(&self) -> usize {
        self.data.shape()[1]
    }

    /// Get the height of the image in pixels.
    pub fn height(&self) -> usize {
        self.data.shape()[0]
    }

    /// Get the number of channels in the image.
    pub fn num_channels(&self) -> usize {
        CHANNELS
    }

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
        assert_eq!(image.size().width, 10);
        assert_eq!(image.size().height, 20);
        assert_eq!(image.num_channels(), 3);
    }

    #[test]
    //fn image_from_file() {
    //    use crate::image::Image;
    //    let image_path = std::path::Path::new("tests/data/dog.jpeg");
    //    let image = Image::<u8, 3>::from_file(image_path);
    //    assert_eq!(image.size().width, 258);
    //    assert_eq!(image.size().height, 195);
    //    assert_eq!(image.num_channels(), 3);
    //}
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
        assert_eq!(image.size().width, 2);
        assert_eq!(image.size().height, 3);
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
        assert_eq!(image.size().width, 3);
        assert_eq!(image.size().height, 2);
        assert_eq!(image.num_channels(), 4);
    }
}
