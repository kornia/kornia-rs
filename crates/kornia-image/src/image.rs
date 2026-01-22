use crate::{allocator::ImageAllocator, error::ImageError};
use kornia_tensor::{Tensor, Tensor2, Tensor3};

/// Image size in pixels
///
/// A struct to represent the size of an image in pixels.
///
/// # Examples
///
/// ```
/// use kornia_image::ImageSize;
///
/// let image_size = ImageSize {
///   width: 10,
///   height: 20,
/// };
///
/// assert_eq!(image_size.width, 10);
/// assert_eq!(image_size.height, 20);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

/// Pixel data type format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PixelFormat {
    /// Unsigned 8-bit samples (0-255).
    U8,
    /// Unsigned 16-bit samples (0-65535).
    U16,
    /// 32-bit floating point samples.
    F32,
}

/// Metadata describing the layout of an image including dimensions, channels, and pixel format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageLayout {
    /// Spatial size of the image (width and height).
    pub image_size: ImageSize,
    /// Number of color channels.
    pub channels: u8,
    /// Scalar data type for pixel values.
    pub pixel_format: PixelFormat,
}

impl ImageLayout {
    /// Creates a new `ImageLayout` with the given size, channel count, and pixel format.
    ///
    /// # Arguments
    ///
    /// * `image_size` - The width and height of the image
    /// * `channels` - Number of color channels
    /// * `pixel_format` - The data type for pixel values
    pub fn new(image_size: ImageSize, channels: u8, pixel_format: PixelFormat) -> Self {
        Self {
            image_size,
            channels,
            pixel_format,
        }
    }
}

#[derive(Clone)]
/// Represents an image with pixel data.
///
/// The image is represented as a 3D Tensor with shape (H, W, C), where H is the height of the image,
pub struct Image<T, const C: usize, A: ImageAllocator>(pub Tensor3<T, A>);

/// helper to deference the inner tensor
impl<T, const C: usize, A: ImageAllocator> std::ops::Deref for Image<T, C, A> {
    type Target = Tensor3<T, A>;

    // Define the deref method to return a reference to the inner Tensor3<T>.
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// helper to deference the inner tensor
impl<T, const C: usize, A: ImageAllocator> std::ops::DerefMut for Image<T, C, A> {
    // Define the deref_mut method to return a mutable reference to the inner Tensor3<T>.
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T, const C: usize, A: ImageAllocator> Image<T, C, A> {
    /// Create a new image from pixel data.
    ///
    /// # Arguments
    ///
    /// * `size` - The size of the image in pixels.
    /// * `data` - The pixel data of the image.
    /// * `alloc` - The allocator of the image
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
    /// use kornia_image::{Image, ImageSize};
    /// use kornia_image::allocator::CpuAllocator;
    ///
    /// let image = Image::<u8, 3, _>::new(
    ///    ImageSize {
    ///       width: 10,
    ///      height: 20,
    ///  },
    /// vec![0u8; 10 * 20 * 3],
    /// CpuAllocator
    /// ).unwrap();
    ///
    /// assert_eq!(image.size().width, 10);
    /// assert_eq!(image.size().height, 20);
    /// assert_eq!(image.num_channels(), 3);
    /// ```
    pub fn new(size: ImageSize, data: Vec<T>, alloc: A) -> Result<Self, ImageError>
    where
        T: Clone, // TODO: remove this bound
    {
        // check if the data length matches the image size
        if data.len() != size.width * size.height * C {
            return Err(ImageError::InvalidChannelShape(
                data.len(),
                size.width * size.height * C,
            ));
        }

        // allocate the image data
        Ok(Self(Tensor3::from_shape_vec(
            [size.height, size.width, C],
            data,
            alloc,
        )?))
    }

    /// Create a new image with the given size and default pixel data.
    ///
    /// # Arguments
    ///
    /// * `size` - The size of the image in pixels.
    /// * `val` - The default value of the pixel data.
    /// * `alloc` - The allocator of the image
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
    /// use kornia_image::{Image, ImageSize};
    /// use kornia_image::allocator::CpuAllocator;
    ///
    /// let image = Image::<u8, 3, _>::from_size_val(
    ///   ImageSize {
    ///     width: 10,
    ///    height: 20,
    /// }, 0u8, CpuAllocator).unwrap();
    ///
    /// assert_eq!(image.size().width, 10);
    /// assert_eq!(image.size().height, 20);
    /// assert_eq!(image.num_channels(), 3);
    /// ```
    pub fn from_size_val(size: ImageSize, val: T, alloc: A) -> Result<Self, ImageError>
    where
        T: Clone + Default,
    {
        let data = vec![val; size.width * size.height * C];
        let image = Image::new(size, data, alloc)?;

        Ok(image)
    }

    /// Create a new image from raw parts.
    ///
    /// # Arguments
    ///
    /// * `size` - The size of the image in pixels.
    /// * `data` - A pointer to the pixel data.
    /// * `len` - The length of the pixel data.
    ///
    /// # Returns
    ///
    /// A new image created from the given size and pixel data.
    ///
    /// # Safety
    ///
    /// The pointer must be non-null and the length must be valid.
    pub unsafe fn from_raw_parts(
        size: ImageSize,
        data: *const T,
        len: usize,
        alloc: A,
    ) -> Result<Self, ImageError>
    where
        T: Clone,
    {
        Tensor::from_raw_parts([size.height, size.width, C], data, len, alloc)?.try_into()
    }

    /// Create a new image from a slice of pixel data.
    ///
    /// # Arguments
    ///
    /// * `size` - The size of the image in pixels.
    /// * `data` - A slice containing the pixel data.
    ///
    /// # Returns
    ///
    /// A new image created from the given size and pixel data.
    ///
    /// # Errors
    ///
    /// Returns an error if the length of the data slice doesn't match the image dimensions,
    /// or if there's an issue creating the tensor or image.
    pub fn from_size_slice(size: ImageSize, data: &[T], alloc: A) -> Result<Self, ImageError>
    where
        T: Clone,
    {
        let tensor: Tensor3<T, A> =
            Tensor::from_shape_slice([size.height, size.width, C], data, alloc)?;
        Image::try_from(tensor)
    }

    /// Map the pixel data of the image to a different type.
    ///
    /// # Arguments
    ///
    /// * `f` - A function that takes a pixel value and returns a new pixel value.
    ///
    /// # Returns
    ///
    /// A new image with the pixel data mapped to the new type.
    pub fn map<U>(&self, f: impl Fn(&T) -> U) -> Result<Image<U, C, A>, ImageError>
    where
        U: Clone,
    {
        let data = self.as_slice().iter().map(f).collect::<Vec<U>>();
        let alloc = self.storage.alloc();
        Image::<U, C, A>::new(self.size(), data, alloc.clone())
    }

    /// Cast the pixel data of the image to a different type.
    ///
    /// # Returns
    ///
    /// A new image with the pixel data cast to the given type.
    pub fn cast<U>(&self) -> Result<Image<U, C, A>, ImageError>
    where
        U: num_traits::NumCast + Clone + Copy, // TODO: remove this bound
        T: num_traits::NumCast + Clone + Copy, // TODO: remove this bound
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

        let alloc = self.storage.alloc();

        Image::new(self.size(), casted_data, alloc.clone())
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
    pub fn channel(&self, channel: usize) -> Result<Image<T, 1, A>, ImageError>
    where
        T: Clone,
    {
        if channel >= C {
            return Err(ImageError::ChannelIndexOutOfBounds(channel, C));
        }

        let channel_data = self
            .as_slice()
            .iter()
            .skip(channel)
            .step_by(C)
            .cloned()
            .collect();

        let alloc = self.storage.alloc();

        Image::new(self.size(), channel_data, alloc.clone())
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
    /// use kornia_image::{Image, ImageSize};
    /// use kornia_image::allocator::CpuAllocator;
    ///
    /// let image = Image::<f32, 2, _>::from_size_val(
    ///   ImageSize {
    ///    width: 10,
    ///   height: 20,
    /// },
    /// 0.0f32,
    /// CpuAllocator).unwrap();
    ///
    /// let channels = image.split_channels().unwrap();
    /// assert_eq!(channels.len(), 2);
    /// ```
    pub fn split_channels(&self) -> Result<Vec<Image<T, 1, A>>, ImageError>
    where
        T: Clone + Copy, // TODO: remove this bound
    {
        let mut channels = Vec::with_capacity(C);

        for i in 0..C {
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
        self.shape[1]
    }

    /// Get the number of rows of the image.
    pub fn rows(&self) -> usize {
        self.shape[0]
    }

    /// Get the width of the image in pixels.
    pub fn width(&self) -> usize {
        self.cols()
    }

    /// Get the height of the image in pixels.
    pub fn height(&self) -> usize {
        self.rows()
    }

    /// Get the number of channels in the image.
    pub fn num_channels(&self) -> usize {
        C
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
    /// use kornia_image::{Image, ImageSize};
    /// use kornia_image::allocator::CpuAllocator;
    ///
    /// let data = vec![0u8, 0, 255, 0, 0, 255];
    ///
    /// let image_u8 = Image::<u8, 3, _>::new(
    /// ImageSize {
    ///   height: 2,
    ///   width: 1,
    /// },
    /// data,
    /// CpuAllocator
    /// ).unwrap();
    ///
    /// let image_f32 = image_u8.cast_and_scale::<f32>(1. / 255.0).unwrap();
    ///
    /// assert_eq!(image_f32.get([1, 0, 2]), Some(&1.0f32));
    /// ```
    pub fn cast_and_scale<U>(self, scale: U) -> Result<Image<U, C, A>, ImageError>
    where
        U: num_traits::NumCast + std::ops::Mul<Output = U> + Clone + Copy,
        T: num_traits::NumCast + Clone + Copy,
    {
        let casted_data = self
            .as_slice()
            .iter()
            .map(|&x| {
                let xu = U::from(x).ok_or(ImageError::CastError)?;
                Ok(xu * scale)
            })
            .collect::<Result<Vec<U>, ImageError>>()?;

        let alloc = self.storage.alloc();

        Image::new(self.size(), casted_data, alloc.clone())
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
    pub fn scale_and_cast<U>(&self, scale: T) -> Result<Image<U, C, A>, ImageError>
    where
        U: num_traits::NumCast + Clone + Copy,
        T: num_traits::NumCast + std::ops::Mul<Output = T> + Clone + Copy,
    {
        let casted_data = self
            .as_slice()
            .iter()
            .map(|&x| {
                let xu = U::from(x * scale).ok_or(ImageError::CastError)?;
                Ok(xu)
            })
            .collect::<Result<Vec<U>, ImageError>>()?;

        let alloc = self.storage.alloc();

        Image::new(self.size(), casted_data, alloc.clone())
    }

    /// Get the pixel data of the image.
    ///
    /// NOTE: this is method is for convenience and not optimized for performance.
    /// We recommend using iterators over the data slice.
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
    pub fn get_pixel(&self, x: usize, y: usize, ch: usize) -> Result<&T, ImageError> {
        if x >= self.width() || y >= self.height() {
            return Err(ImageError::PixelIndexOutOfBounds(
                x,
                y,
                self.width(),
                self.height(),
            ));
        }

        if ch >= C {
            return Err(ImageError::ChannelIndexOutOfBounds(ch, C));
        }

        let val = match self.get([y, x, ch]) {
            Some(v) => v,
            None => return Err(ImageError::ImageDataNotContiguous),
        };

        Ok(val)
    }

    /// Set the pixel value at the given coordinates.
    ///
    /// NOTE: this is method is for convenience and not optimized for performance.
    /// We recommend creating a mutable slice and operating on it directly.
    ///
    /// # Arguments
    ///
    /// * `x` - The x-coordinate of the pixel.
    /// * `y` - The y-coordinate of the pixel.
    /// * `ch` - The channel index of the pixel.
    /// * `val` - The value to set the pixel to.
    ///
    /// # Returns
    ///
    /// The pixel value at the given coordinates.
    pub fn set_pixel(&mut self, x: usize, y: usize, ch: usize, val: T) -> Result<(), ImageError> {
        if x >= self.width() || y >= self.height() {
            return Err(ImageError::PixelIndexOutOfBounds(
                x,
                y,
                self.width(),
                self.height(),
            ));
        }

        if ch >= C {
            return Err(ImageError::ChannelIndexOutOfBounds(ch, C));
        }

        let idx = y * self.width() * C + x * C + ch;
        self.as_slice_mut()[idx] = val;

        Ok(())
    }

    /// Convert the image to a vector.
    pub fn into_vec(self) -> Vec<T> {
        self.0.into_vec()
    }

    /// Get a copy of the image data as a vector.
    pub fn to_vec(&self) -> Vec<T>
    where
        T: Clone,
    {
        self.as_slice().to_vec()
    }
}

/// helper to convert an single channel tensor to a kornia image with try into
impl<T, A: ImageAllocator> TryFrom<Tensor2<T, A>> for Image<T, 1, A>
where
    T: Clone,
{
    type Error = ImageError;

    fn try_from(value: Tensor2<T, A>) -> Result<Self, Self::Error> {
        let alloc = value.storage.alloc();

        Self::from_size_slice(
            ImageSize {
                width: value.shape[1],
                height: value.shape[0],
            },
            value.as_slice(),
            alloc.clone(),
        )
    }
}

/// helper to convert an multi channel tensor to a kornia image with try into
impl<T, const C: usize, A: ImageAllocator> TryFrom<Tensor3<T, A>> for Image<T, C, A> {
    type Error = ImageError;

    fn try_from(value: Tensor3<T, A>) -> Result<Self, Self::Error> {
        if value.shape[2] != C {
            return Err(ImageError::InvalidChannelShape(value.shape[2], C));
        }
        Ok(Self(value))
    }
}

impl<T, const C: usize, A: ImageAllocator> TryInto<Tensor3<T, A>> for Image<T, C, A> {
    type Error = ImageError;

    fn try_into(self) -> Result<Tensor3<T, A>, Self::Error> {
        Ok(self.0)
    }
}

#[cfg(test)]
mod tests {
    use crate::image::{Image, ImageError, ImageLayout, ImageSize, PixelFormat};
    use kornia_tensor::{CpuAllocator, Tensor};

    #[test]
    fn test_image_size() {
        let image_size = ImageSize {
            width: 10,
            height: 20,
        };
        assert_eq!(image_size.width, 10);
        assert_eq!(image_size.height, 20);
    }

    #[test]
    fn test_image_layout_creation() {
        let size = ImageSize {
            width: 258,
            height: 195,
        };

        let layout = ImageLayout::new(size, 3, PixelFormat::U8);

        assert_eq!(layout.image_size.width, 258);
        assert_eq!(layout.image_size.height, 195);
        assert_eq!(layout.channels, 3);
        assert_eq!(layout.pixel_format, PixelFormat::U8);
    }

    #[test]
    fn test_image_layout_pixel_formats() {
        let size = ImageSize {
            width: 100,
            height: 100,
        };

        let layout_u8 = ImageLayout::new(size, 1, PixelFormat::U8);
        assert_eq!(layout_u8.pixel_format, PixelFormat::U8);

        let layout_u16 = ImageLayout::new(size, 1, PixelFormat::U16);
        assert_eq!(layout_u16.pixel_format, PixelFormat::U16);

        let layout_f32 = ImageLayout::new(size, 4, PixelFormat::F32);
        assert_eq!(layout_f32.pixel_format, PixelFormat::F32);
        assert_eq!(layout_f32.channels, 4);
    }

    #[test]
    fn test_image_smoke() -> Result<(), ImageError> {
        let image = Image::<u8, 3, CpuAllocator>::new(
            ImageSize {
                width: 10,
                height: 20,
            },
            vec![0u8; 10 * 20 * 3],
            CpuAllocator,
        )?;
        assert_eq!(image.size().width, 10);
        assert_eq!(image.size().height, 20);
        assert_eq!(image.num_channels(), 3);

        Ok(())
    }

    #[test]
    fn test_image_from_vec() -> Result<(), ImageError> {
        let image: Image<f32, 3, CpuAllocator> = Image::new(
            ImageSize {
                height: 3,
                width: 2,
            },
            vec![0.0; 3 * 2 * 3],
            CpuAllocator,
        )?;
        assert_eq!(image.size().width, 2);
        assert_eq!(image.size().height, 3);
        assert_eq!(image.num_channels(), 3);

        Ok(())
    }

    #[test]
    fn test_image_cast() -> Result<(), ImageError> {
        let data = vec![0, 1, 2, 3, 4, 5];
        let image_u8 = Image::<_, 3, CpuAllocator>::new(
            ImageSize {
                height: 2,
                width: 1,
            },
            data,
            CpuAllocator,
        )?;
        assert_eq!(image_u8.get([1, 0, 2]), Some(&5u8));

        let image_i32: Image<i32, 3, CpuAllocator> = image_u8.cast()?;
        assert_eq!(image_i32.get([1, 0, 2]), Some(&5i32));

        Ok(())
    }

    #[test]
    fn test_image_rgbd() -> Result<(), ImageError> {
        let image = Image::<f32, 4, CpuAllocator>::new(
            ImageSize {
                height: 2,
                width: 3,
            },
            vec![0f32; 2 * 3 * 4],
            CpuAllocator,
        )?;
        assert_eq!(image.size().width, 3);
        assert_eq!(image.size().height, 2);
        assert_eq!(image.num_channels(), 4);

        Ok(())
    }

    #[test]
    fn test_image_channel() -> Result<(), ImageError> {
        let image = Image::<f32, 3, CpuAllocator>::new(
            ImageSize {
                height: 2,
                width: 1,
            },
            vec![0., 1., 2., 3., 4., 5.],
            CpuAllocator,
        )?;

        let channel = image.channel(2)?;
        assert_eq!(channel.get([1, 0, 0]), Some(&5.0f32));

        Ok(())
    }

    #[test]
    fn test_image_split_channels() -> Result<(), ImageError> {
        let image = Image::<f32, 3, CpuAllocator>::new(
            ImageSize {
                height: 2,
                width: 1,
            },
            vec![0., 1., 2., 3., 4., 5.],
            CpuAllocator,
        )
        .unwrap();
        let channels = image.split_channels()?;
        assert_eq!(channels.len(), 3);
        assert_eq!(channels[0].get([1, 0, 0]), Some(&3.0f32));
        assert_eq!(channels[1].get([1, 0, 0]), Some(&4.0f32));
        assert_eq!(channels[2].get([1, 0, 0]), Some(&5.0f32));

        Ok(())
    }

    #[test]
    fn test_scale_and_cast() -> Result<(), ImageError> {
        let data = vec![0u8, 0, 255, 0, 0, 255];
        let image_u8 = Image::<u8, 3, CpuAllocator>::new(
            ImageSize {
                height: 2,
                width: 1,
            },
            data,
            CpuAllocator,
        )?;
        let image_f32 = image_u8.scale_and_cast::<f32>(1)?;
        assert_eq!(image_f32.get([1, 0, 2]), Some(&255.0f32));

        Ok(())
    }

    #[test]
    fn test_cast_and_scale() -> Result<(), ImageError> {
        let data = vec![0u8, 0, 255, 0, 0, 255];
        let image_u8 = Image::<u8, 3, CpuAllocator>::new(
            ImageSize {
                height: 2,
                width: 1,
            },
            data,
            CpuAllocator,
        )?;
        let image_f32 = image_u8.cast_and_scale::<f32>(1. / 255.0)?;
        assert_eq!(image_f32.get([1, 0, 2]), Some(&1.0f32));

        Ok(())
    }

    #[test]
    fn test_image_from_tensor() -> Result<(), ImageError> {
        let data = vec![0u8, 1, 2, 3, 4, 5];
        let tensor = Tensor::<u8, 2, _>::from_shape_vec([2, 3], data, CpuAllocator)?;

        let image = Image::<u8, 1, CpuAllocator>::try_from(tensor.clone())?;
        assert_eq!(image.size().width, 3);
        assert_eq!(image.size().height, 2);
        assert_eq!(image.num_channels(), 1);

        let image_2: Image<u8, 1, CpuAllocator> = tensor.try_into()?;
        assert_eq!(image_2.size().width, 3);
        assert_eq!(image_2.size().height, 2);
        assert_eq!(image_2.num_channels(), 1);

        Ok(())
    }

    #[test]
    fn test_image_from_tensor_3d() -> Result<(), ImageError> {
        let tensor = Tensor::<u8, 3, CpuAllocator>::from_shape_vec(
            [2, 3, 4],
            vec![0u8; 2 * 3 * 4],
            CpuAllocator,
        )?;

        let image = Image::<u8, 4, CpuAllocator>::try_from(tensor.clone())?;
        assert_eq!(image.size().width, 3);
        assert_eq!(image.size().height, 2);
        assert_eq!(image.num_channels(), 4);

        let image_2: Image<u8, 4, CpuAllocator> = tensor.try_into()?;
        assert_eq!(image_2.size().width, 3);
        assert_eq!(image_2.size().height, 2);
        assert_eq!(image_2.num_channels(), 4);

        Ok(())
    }

    #[test]
    fn test_image_from_raw_parts() -> Result<(), ImageError> {
        let data = vec![0u8, 1, 2, 3, 4, 5];
        let image = unsafe {
            Image::<_, 1, CpuAllocator>::from_raw_parts(
                [2, 3].into(),
                data.as_ptr(),
                data.len(),
                CpuAllocator,
            )?
        };
        std::mem::forget(data);
        assert_eq!(image.size().width, 2);
        assert_eq!(image.size().height, 3);
        assert_eq!(image.num_channels(), 1);
        Ok(())
    }

    #[test]
    fn test_get_pixel() -> Result<(), ImageError> {
        let image = Image::<u8, 3, CpuAllocator>::new(
            ImageSize {
                height: 2,
                width: 1,
            },
            vec![1, 2, 5, 19, 255, 128],
            CpuAllocator,
        )?;
        assert_eq!(image.get_pixel(0, 0, 0)?, &1);
        assert_eq!(image.get_pixel(0, 0, 1)?, &2);
        assert_eq!(image.get_pixel(0, 0, 2)?, &5);
        assert_eq!(image.get_pixel(0, 1, 0)?, &19);
        assert_eq!(image.get_pixel(0, 1, 1)?, &255);
        assert_eq!(image.get_pixel(0, 1, 2)?, &128);
        Ok(())
    }

    #[test]
    fn test_set_pixel() -> Result<(), ImageError> {
        let mut image = Image::<u8, 3, CpuAllocator>::new(
            ImageSize {
                height: 2,
                width: 1,
            },
            vec![1, 2, 5, 19, 255, 128],
            CpuAllocator,
        )?;

        image.set_pixel(0, 0, 0, 128)?;
        image.set_pixel(0, 1, 1, 25)?;

        assert_eq!(image.get_pixel(0, 0, 0)?, &128);
        assert_eq!(image.get_pixel(0, 1, 1)?, &25);

        Ok(())
    }

    #[test]
    fn test_image_map() -> Result<(), ImageError> {
        let image_u8 = Image::<u8, 1, CpuAllocator>::new(
            ImageSize {
                height: 2,
                width: 1,
            },
            vec![0, 128],
            CpuAllocator,
        )?;

        let image_f32 = image_u8.map(|x| (x + 2) as f32)?;

        assert_eq!(image_f32.size().width, 1);
        assert_eq!(image_f32.size().height, 2);
        assert_eq!(image_f32.num_channels(), 1);
        assert_eq!(image_f32.get([0, 0, 0]), Some(&2.0f32));
        assert_eq!(image_f32.get([1, 0, 0]), Some(&130.0f32));

        Ok(())
    }
}
