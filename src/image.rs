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

/// Trait for image data types.
// Send and Sync is required for ndarray::Zip::par_for_each
pub trait ImageDtype: Copy + Default + Into<f32> + Send + Sync {
    fn from_f32(x: f32) -> Self;
}

impl ImageDtype for f32 {
    fn from_f32(x: f32) -> Self {
        x
    }
}

impl ImageDtype for u8 {
    fn from_f32(x: f32) -> Self {
        x.round().clamp(0.0, 255.0) as u8
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

    /// Get a channel of the image.
    /// # Arguments
    ///
    /// * `channel` - The channel to get.
    ///
    /// # Returns
    ///
    /// A new image with the given channel.
    ///
    /// # Errors
    ///
    /// If the channel index is out of bounds, an error is returned.
    pub fn channel(&self, channel: usize) -> Result<Image<T, 1>>
    where
        T: Clone,
    {
        if channel >= CHANNELS {
            return Err(anyhow::anyhow!(
                "Channel index ({}) out of bounds ({}).",
                channel,
                CHANNELS
            ));
        }

        let channel_data = self.data.slice(ndarray::s![.., .., channel..channel + 1]);

        Ok(Image {
            data: channel_data.to_owned(),
        })
    }

    /// Split the image into its channels.
    ///
    /// # Returns
    ///
    /// A vector of images, each containing one channel of the original image.
    ///
    /// # Examples
    ///
    /// ```
    /// use kornia_rs::image::{Image, ImageSize};
    ///
    /// let image = Image::<f32, 2>::from_size_val(
    ///   ImageSize {
    ///    width: 10,
    ///   height: 20,
    /// },
    /// 0.0f32).unwrap();
    ///
    /// let channels = image.split_channels().unwrap();
    /// assert_eq!(channels.len(), 2);
    /// ```
    pub fn split_channels(&self) -> Result<Vec<Image<T, 1>>>
    where
        T: Clone,
    {
        let mut channels = Vec::with_capacity(CHANNELS);
        for i in 0..CHANNELS {
            channels.push(self.channel(i)?);
        }

        Ok(channels)
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

    pub fn cols(&self) -> usize {
        self.width()
    }

    pub fn rows(&self) -> usize {
        self.height()
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

    /// Get the pixel data of the image as a 4D tensor in NCHW format.
    ///
    /// Internally, the image is stored in HWC format, and the function gives
    /// away ownership of the pixel data.
    pub fn to_tensor_nchw(self) -> ndarray::Array4<T> {
        // add batch axis 1xHxWxC
        let data = self.data.insert_axis(ndarray::Axis(0));

        // permute axes to NHWC -> NCHW
        data.permuted_axes([0, 3, 1, 2])
    }

    /// Get the pixel data of the image as a 4D tensor in NHWC format.
    ///
    /// Internally, the image is stored in HWC format, and the function gives
    /// away ownership of the pixel data.
    pub fn to_tensor_nhwc(self) -> ndarray::Array4<T> {
        self.data.insert_axis(ndarray::Axis(0))
    }

    // NOTE: experimental api
    pub fn set_pixel(&mut self, x: usize, y: usize, ch: usize, val: T) -> Result<()>
    where
        T: Copy,
    {
        if x >= self.width() || y >= self.height() {
            return Err(anyhow::anyhow!(
                "Pixel coordinates ({}, {}) out of bounds ({}, {}).",
                x,
                y,
                self.width(),
                self.height()
            ));
        }

        if ch >= CHANNELS {
            return Err(anyhow::anyhow!(
                "Channel index ({}) out of bounds ({}).",
                ch,
                CHANNELS
            ));
        }

        self.data[[y, x, ch]] = val;

        Ok(())
    }

    // NOTE: experimental api
    pub fn get_pixel(&self, x: usize, y: usize, ch: usize) -> Result<T>
    where
        T: Copy,
    {
        if x >= self.width() || y >= self.height() {
            return Err(anyhow::anyhow!(
                "Pixel coordinates ({}, {}) out of bounds ({}, {}).",
                x,
                y,
                self.width(),
                self.height()
            ));
        }

        if ch >= CHANNELS {
            return Err(anyhow::anyhow!(
                "Channel index ({}) out of bounds ({}).",
                ch,
                CHANNELS
            ));
        }

        Ok(self.data[[y, x, ch]])
    }
}

#[cfg(test)]
mod tests {
    use crate::image::{Image, ImageSize};
    use anyhow::Result;

    #[test]
    fn image_size() {
        let image_size = ImageSize {
            width: 10,
            height: 20,
        };
        assert_eq!(image_size.width, 10);
        assert_eq!(image_size.height, 20);
    }

    #[test]
    fn image_smoke() -> Result<()> {
        let image = Image::<u8, 3>::new(
            ImageSize {
                width: 10,
                height: 20,
            },
            vec![0u8; 10 * 20 * 3],
        )?;
        assert_eq!(image.size().width, 10);
        assert_eq!(image.size().height, 20);
        assert_eq!(image.num_channels(), 3);

        Ok(())
    }

    #[test]
    fn image_from_vec() -> Result<()> {
        let image: Image<f32, 3> = Image::new(
            ImageSize {
                height: 3,
                width: 2,
            },
            vec![0.0; 3 * 2 * 3],
        )?;
        assert_eq!(image.size().width, 2);
        assert_eq!(image.size().height, 3);
        assert_eq!(image.num_channels(), 3);

        Ok(())
    }

    #[test]
    fn image_cast() -> Result<()> {
        let data = vec![0., 1., 2., 3., 4., 5.];
        let image_f64 = Image::<f64, 3>::new(
            ImageSize {
                height: 2,
                width: 1,
            },
            data,
        )?;
        assert_eq!(image_f64.data.get((1, 0, 2)).unwrap(), &5.0f64);

        let image_u8 = image_f64.cast::<u8>()?;
        assert_eq!(image_u8.data.get((1, 0, 2)).unwrap(), &5u8);

        let image_i32: Image<i32, 3> = image_u8.cast()?;
        assert_eq!(image_i32.data.get((1, 0, 2)).unwrap(), &5i32);

        Ok(())
    }

    #[test]
    fn image_rgbd() -> Result<()> {
        let image = Image::<f32, 4>::new(
            ImageSize {
                height: 2,
                width: 3,
            },
            vec![0f32; 2 * 3 * 4],
        )?;
        assert_eq!(image.size().width, 3);
        assert_eq!(image.size().height, 2);
        assert_eq!(image.num_channels(), 4);

        Ok(())
    }

    #[test]
    fn image_channel() -> Result<()> {
        let image = Image::<f32, 3>::new(
            ImageSize {
                height: 2,
                width: 1,
            },
            vec![0., 1., 2., 3., 4., 5.],
        )?;

        let channel = image.channel(2)?;
        assert_eq!(channel.data.get((1, 0, 0)).unwrap(), &5.0f32);

        Ok(())
    }

    #[test]
    fn image_split_channels() -> Result<()> {
        let image = Image::<f32, 3>::new(
            ImageSize {
                height: 2,
                width: 1,
            },
            vec![0., 1., 2., 3., 4., 5.],
        )
        .unwrap();
        let channels = image.split_channels()?;
        assert_eq!(channels.len(), 3);
        assert_eq!(channels[0].data.get((1, 0, 0)).unwrap(), &3.0f32);
        assert_eq!(channels[1].data.get((1, 0, 0)).unwrap(), &4.0f32);
        assert_eq!(channels[2].data.get((1, 0, 0)).unwrap(), &5.0f32);

        Ok(())
    }

    #[test]
    fn convert_to_tensor() -> Result<()> {
        let image = Image::<f32, 3>::new(
            ImageSize {
                height: 2,
                width: 1,
            },
            vec![0., 1., 2., 3., 4., 5.],
        )?;

        let tensor_nchw = image.clone().to_tensor_nchw();
        assert_eq!(tensor_nchw.shape(), &[1, 3, 2, 1]);
        assert_eq!(tensor_nchw[[0, 2, 1, 0]], 5.0f32);

        let tensor_nhwc = image.to_tensor_nhwc();
        assert_eq!(tensor_nhwc.shape(), &[1, 2, 1, 3]);
        assert_eq!(tensor_nhwc[[0, 1, 0, 2]], 5.0f32);

        Ok(())
    }
}
