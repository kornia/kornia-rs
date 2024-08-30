use std::ops;

use kornia_core::{CpuAllocator, SafeTensorType, Tensor3};

use crate::error::ImageError;

/// Image size in pixels
///
/// A struct to represent the size of an image in pixels.
///
/// # Examples
///
/// ```
/// use kornia::image::ImageSize;
///
/// let image_size = ImageSize {
///   width: 10,
///   height: 20,
/// };
///
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

impl From<[usize; 2]> for ImageSize {
    fn from(size: [usize; 2]) -> Self {
        ImageSize {
            width: size[0],
            height: size[1],
        }
    }
}

impl From<ImageSize> for [u32; 2] {
    fn from(size: ImageSize) -> Self {
        [size.width as u32, size.height as u32]
    }
}

/// Trait for image data types.
///
/// Send and Sync is required for ndarray::Zip::par_for_each
// TODO: replace this with ConverTo once we have a better solution
pub trait ImageDtype: Copy + Default + Into<f32> + Send + Sync {
    /// Convert a f32 value to the image data type.
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
/// The image is represented as a 3D Tensor with shape (H, W, C), where H is the height of the image,
pub struct Image<T, const CHANNELS: usize>(pub Tensor3<T>)
where
    T: SafeTensorType;

/// helper to deference the inner tensor
impl<T, const CHANNELS: usize> ops::Deref for Image<T, CHANNELS>
where
    T: SafeTensorType,
{
    type Target = Tensor3<T>;

    // Define the deref method to return a reference to the inner Tensor3<T>.
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// helper to deference the inner tensor
impl<T, const CHANNELS: usize> ops::DerefMut for Image<T, CHANNELS>
where
    T: SafeTensorType,
{
    // Define the deref_mut method to return a mutable reference to the inner Tensor3<T>.
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T, const CHANNELS: usize> Image<T, CHANNELS>
where
    T: SafeTensorType,
{
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
    /// use kornia::image::{Image, ImageSize};
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
    pub fn new(size: ImageSize, data: Vec<T>) -> Result<Self, ImageError> {
        // check if the data length matches the image size
        if data.len() != size.width * size.height * CHANNELS {
            return Err(ImageError::InvalidChannelShape(
                data.len(),
                size.width * size.height * CHANNELS,
            ));
        }

        // allocate the image data
        Ok(Self(Tensor3::from_shape_vec(
            [size.height, size.width, CHANNELS],
            data,
            CpuAllocator,
        )?))
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
    /// use kornia::image::{Image, ImageSize};
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
    pub fn from_size_val(size: ImageSize, val: T) -> Result<Self, ImageError>
    where
        T: Clone + Default,
    {
        let data = vec![val; size.width * size.height * CHANNELS];
        let image = Image::new(size, data)?;

        Ok(image)
    }

    /// Cast the pixel data of the image to a different type.
    ///
    /// # Returns
    ///
    /// A new image with the pixel data cast to the given type.
    pub fn cast<U>(&self) -> Result<Image<U, CHANNELS>, ImageError>
    where
        U: num_traits::NumCast + SafeTensorType,
        T: num_traits::NumCast,
    {
        // TODO: this needs to be optimized and reuse Tensor::cast
        let casted_data = self
            .as_slice()
            .iter()
            .map(|&x| {
                let xu = U::from(x).ok_or(ImageError::CastError)?;
                Ok(xu)
            })
            .collect::<Result<Vec<U>, ImageError>>()?;

        Ok(Image::new(self.size(), casted_data)?)
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
    pub fn channel(&self, channel: usize) -> Result<Image<T, 1>, ImageError>
    where
        T: Clone,
    {
        if channel >= CHANNELS {
            return Err(ImageError::ChannelIndexOutOfBounds(channel, CHANNELS));
        }

        let mut channel_data = vec![];

        for y in 0..self.height() {
            for x in 0..self.width() {
                channel_data.push(*self.get_unchecked([y, x, channel]));
            }
        }

        Image::new(self.size(), channel_data)
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
    /// use kornia::image::{Image, ImageSize};
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
    pub fn split_channels(&self) -> Result<Vec<Image<T, 1>>, ImageError>
    where
        T: Clone,
    {
        let mut channels = Vec::with_capacity(CHANNELS);

        for i in 0..CHANNELS {
            channels.push(self.channel(i)?);
        }

        Ok(channels)
    }

    /// Get the size of the image in pixels.
    pub fn size(&self) -> ImageSize {
        ImageSize {
            width: self.shape[1],
            height: self.shape[0],
        }
    }

    /// Get the number of columns of the image.
    pub fn cols(&self) -> usize {
        self.width()
    }

    /// Get the number of rows of the image.
    pub fn rows(&self) -> usize {
        self.height()
    }

    /// Get the width of the image in pixels.
    pub fn width(&self) -> usize {
        self.shape[1]
    }

    /// Get the height of the image in pixels.
    pub fn height(&self) -> usize {
        self.shape[0]
    }

    /// Get the number of channels in the image.
    pub fn num_channels(&self) -> usize {
        CHANNELS
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
    /// use kornia::image::{Image, ImageSize};
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
    /// assert_eq!(image_f32.get([1, 0, 2]), Some(&1.0f32));
    /// ```
    pub fn cast_and_scale<U>(self, scale: U) -> Result<Image<U, CHANNELS>, ImageError>
    where
        U: num_traits::NumCast + std::ops::Mul<Output = U> + SafeTensorType,
        T: num_traits::NumCast,
    {
        let casted_data = self
            .as_slice()
            .iter()
            .map(|&x| {
                let xu = U::from(x).ok_or(ImageError::CastError)?;
                Ok(xu * scale)
            })
            .collect::<Result<Vec<U>, ImageError>>()?;

        Image::new(self.size(), casted_data)
    }

    /// Get the pixel data of the image.
    ///
    /// NOTE: experimental api
    ///
    /// # Arguments
    ///
    /// * `x` - The x-coordinate of the pixel.
    /// * `y` - The y-coordinate of the pixel.
    /// * `ch` - The channel index of the pixel.
    ///
    /// # Returns
    ///
    /// The pixel value at the given coordinates.
    #[deprecated(since = "0.1.6", note = "Please use the `get` method instead.")]
    pub fn get_pixel(&self, x: usize, y: usize, ch: usize) -> Result<T, ImageError>
    where
        T: Copy,
    {
        if x >= self.width() || y >= self.height() {
            return Err(ImageError::PixelIndexOutOfBounds(
                x,
                y,
                self.width(),
                self.height(),
            ));
        }

        if ch >= CHANNELS {
            return Err(ImageError::ChannelIndexOutOfBounds(ch, CHANNELS));
        }

        let val = match self.get([y, x, ch]) {
            Some(v) => v,
            None => return Err(ImageError::ImageDataNotContiguous),
        };

        Ok(*val)
    }
}

#[cfg(test)]
mod tests {
    use crate::image::{Image, ImageError, ImageSize};

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
    fn image_smoke() -> Result<(), ImageError> {
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
    fn image_from_vec() -> Result<(), ImageError> {
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
    fn image_cast() -> Result<(), ImageError> {
        let data = vec![0, 1, 2, 3, 4, 5];
        let image_u8 = Image::<_, 3>::new(
            ImageSize {
                height: 2,
                width: 1,
            },
            data,
        )?;
        assert_eq!(image_u8.get([1, 0, 2]), Some(&5u8));

        let image_i32: Image<i32, 3> = image_u8.cast()?;
        assert_eq!(image_i32.get([1, 0, 2]), Some(&5i32));

        Ok(())
    }

    #[test]
    fn image_rgbd() -> Result<(), ImageError> {
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
    fn image_channel() -> Result<(), ImageError> {
        let image = Image::<f32, 3>::new(
            ImageSize {
                height: 2,
                width: 1,
            },
            vec![0., 1., 2., 3., 4., 5.],
        )?;

        let channel = image.channel(2)?;
        assert_eq!(channel.get([1, 0, 0]), Some(&5.0f32));

        Ok(())
    }

    #[test]
    fn image_split_channels() -> Result<(), ImageError> {
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
        assert_eq!(channels[0].get([1, 0, 0]), Some(&3.0f32));
        assert_eq!(channels[1].get([1, 0, 0]), Some(&4.0f32));
        assert_eq!(channels[2].get([1, 0, 0]), Some(&5.0f32));

        Ok(())
    }
}
