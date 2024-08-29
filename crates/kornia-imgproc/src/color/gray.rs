use kornia_core::SafeTensorType;
use kornia_image::{Image, ImageError};

/// Define the RGB weights for the grayscale conversion.
const RW: f64 = 0.299;
const GW: f64 = 0.587;
const BW: f64 = 0.114;

/// Convert an RGB image to grayscale using the formula:
///
/// Y = 0.299 * R + 0.587 * G + 0.114 * B
///
/// # Arguments
///
/// * `src` - The input RGB image.
/// * `dst` - The output grayscale image.
///
/// Precondition: the input image must have 3 channels.
/// Precondition: the output image must have 1 channel.
/// Precondition: the input and output images must have the same size.
///
/// # Example
///
/// ```
/// use kornia::image::{Image, ImageSize};
/// use kornia::imgproc::color::gray_from_rgb;
///
/// let image = Image::<f32, 3>::new(
///     ImageSize {
///         width: 4,
///         height: 5,
///     },
///     vec![0f32; 4 * 5 * 3],
/// )
/// .unwrap();
///
/// let mut gray = Image::<f32, 1>::from_size_val(image.size(), 0.0).unwrap();
///
/// gray_from_rgb(&image, &mut gray).unwrap();
/// assert_eq!(gray.num_channels(), 1);
/// assert_eq!(gray.size().width, 4);
/// assert_eq!(gray.size().height, 5);
/// ```
pub fn gray_from_rgb<T>(src: &Image<T, 3>, dst: &mut Image<T, 1>) -> Result<(), ImageError>
where
    T: Default + Copy + Clone + Send + Sync + num_traits::Float + SafeTensorType,
{
    if src.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            src.size().width,
            src.size().height,
            dst.size().width,
            dst.size().height,
        ));
    }

    if src.num_channels() != 3 {
        return Err(ImageError::ChannelIndexOutOfBounds(3, src.num_channels()));
    }

    if dst.num_channels() != 1 {
        return Err(ImageError::ChannelIndexOutOfBounds(1, dst.num_channels()));
    }

    let rw = T::from(RW).ok_or(ImageError::CastError)?;
    let gw = T::from(GW).ok_or(ImageError::CastError)?;
    let bw = T::from(BW).ok_or(ImageError::CastError)?;

    let src_data = unsafe {
        ndarray::ArrayView3::from_shape_ptr((src.height(), src.width(), 3), src.as_ptr())
    };

    let dst_data = unsafe {
        ndarray::ArrayView3::from_shape_ptr((dst.height(), dst.width(), 1), dst.as_ptr())
    };
    let mut dst_data = dst_data.to_owned();

    ndarray::Zip::from(dst_data.rows_mut())
        .and(src_data.rows())
        .par_for_each(|mut out, inp| {
            assert_eq!(inp.len(), 3);
            let r = inp[0];
            let g = inp[1];
            let b = inp[2];
            out[0] = rw * r + gw * g + bw * b;
        });

    // copy the data from the temporary array to the output image
    dst.as_slice_mut()
        .copy_from_slice(dst_data.as_slice().unwrap());

    Ok(())
}

#[cfg(test)]
mod tests {
    use kornia_image::{ops, Image};
    use kornia_io::functional as F;

    #[test]
    fn gray_from_rgb() -> Result<(), Box<dyn std::error::Error>> {
        let image = F::read_image_any("../../tests/data/dog.jpeg")?;

        let mut image_norm = Image::from_size_val(image.size(), 0.0)?;
        ops::cast_and_scale(&image, &mut image_norm, 1. / 255.0)?;

        let mut gray = Image::<f32, 1>::from_size_val(image_norm.size(), 0.0)?;
        super::gray_from_rgb(&image_norm, &mut gray)?;

        assert_eq!(gray.num_channels(), 1);
        assert_eq!(gray.size().width, 258);
        assert_eq!(gray.size().height, 195);
        Ok(())
    }
}
