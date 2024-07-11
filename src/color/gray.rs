use crate::{
    image::Image,
    tensor::{Tensor3, TensorError},
};
use anyhow::{Ok, Result};
use arrow_buffer::{Buffer, MutableBuffer};
use rayon::prelude::*;

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
/// * `image` - The input RGB image assumed to have 3 channels.
///
/// # Returns
///
/// The grayscale image.
///
/// Precondition: the input image must have 3 channels.
///
/// # Example
///
/// ```
/// use kornia_rs::image::{Image, ImageSize};
///
/// let image = Image::<f32, 3>::new(
///     ImageSize {
///         width: 4,
///         height: 5,
///     },
///     vec![0f32; 4 * 5 * 3],
/// )
/// .unwrap();
/// let gray: Image<f32, 1> = kornia_rs::color::gray_from_rgb(&image).unwrap();
/// assert_eq!(gray.num_channels(), 1);
/// assert_eq!(gray.size().width, 4);
/// assert_eq!(gray.size().height, 5);
/// ```
pub fn gray_from_rgb<T>(image: &Image<T, 3>) -> Result<Image<T, 1>>
where
    T: Default + Copy + Clone + Send + Sync + num_traits::Float,
{
    assert_eq!(image.num_channels(), 3);

    let rw = T::from(RW).ok_or(anyhow::anyhow!("Failed to convert RW"))?;
    let gw = T::from(GW).ok_or(anyhow::anyhow!("Failed to convert GW"))?;
    let bw = T::from(BW).ok_or(anyhow::anyhow!("Failed to convert BW"))?;

    let mut output = Image::<T, 1>::from_size_val(image.size(), T::default())?;

    ndarray::Zip::from(output.data.rows_mut())
        .and(image.data.rows())
        .par_for_each(|mut out, inp| {
            assert_eq!(inp.len(), 3);
            let r = inp[0];
            let g = inp[1];
            let b = inp[2];
            out[0] = rw * r + gw * g + bw * b;
        });

    Ok(output)
}

/// Convert an RGB image to grayscale using the formula:
///
/// Y = 77 * R + 150 * G + 29 * B / 256
///
/// # Arguments
///
/// * `src` - The input RGB image assumed to have 3 channels.
/// * `dst` - The output grayscale image.
///
/// # Returns
///
/// A result indicating success or failure.
///
/// Precondition: the input image must have 3 channels.
/// Precondition: the output image must have 1 channel.
pub fn gray_from_rgb_new(
    src: &Tensor3<u8>,
    dst: &mut Tensor3<u8>,
) -> std::result::Result<(), TensorError> {
    if src.shape[0] != dst.shape[0] || src.shape[1] != dst.shape[1] {
        Err(TensorError::ShapeMismatch)?
    }

    if src.shape[2] != 3 {
        Err(TensorError::InvalidNumDimensions(src.shape[2]))?
    }

    if dst.shape[2] != 1 {
        Err(TensorError::InvalidNumDimensions(dst.shape[2]))?
    }

    src.as_slice()
        .par_chunks_exact(3)
        .zip(dst.as_slice_mut().par_iter_mut())
        .for_each(|(pixel, out)| {
            let r = pixel[0] as u32;
            let g = pixel[1] as u32;
            let b = pixel[2] as u32;
            *out = ((77 * r + 150 * g + 29 * b) / 256) as u8;
        });

    std::result::Result::Ok(())
}

#[cfg(test)]
mod tests {
    use crate::io::functional as F;
    use crate::tensor::{CpuAllocator, Tensor3};
    use anyhow::Result;

    #[test]
    fn gray_from_rgb() -> Result<()> {
        let image_path = std::path::Path::new("tests/data/dog.jpeg");
        let image = F::read_image_any(image_path)?;
        let image_norm = image.cast_and_scale::<f32>(1. / 255.0)?;
        let gray = super::gray_from_rgb(&image_norm.cast::<f64>()?)?;
        assert_eq!(gray.num_channels(), 1);
        assert_eq!(gray.size().width, 258);
        assert_eq!(gray.size().height, 195);
        Ok(())
    }

    #[test]
    fn gray_from_rgb_new() -> Result<()> {
        let image_path = std::path::Path::new("tests/data/dog.jpeg");
        let image = F::read_image_any(image_path)?;

        let rbg = Tensor3::<u8>::from_shape_vec(
            [image.rows(), image.cols(), 3],
            image.data.as_slice().unwrap().to_vec(),
            CpuAllocator,
        )?;

        let mut gray = Tensor3::<u8>::new_uninitialized(
            [image.rows(), image.cols(), 1],
            rbg.storage.alloc().clone(),
        )?;

        super::gray_from_rgb_new(&rbg, &mut gray)?;

        Ok(())
    }
}
