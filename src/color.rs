use crate::image::Image;
use ndarray::{s, stack, Axis};

pub fn grayscale_from_rgb(image: Image) -> Image {
    // convert to f32
    let mut image_f32 = image.data.mapv(|x| x as f32);

    // get channels
    let mut binding = image_f32.view_mut();
    let (r, g, b) = binding.multi_slice_mut((s![.., .., 0], s![.., .., 1], s![.., .., 2]));

    // weighted sum
    // TODO: check data type, for u8 or f32/f64
    let gray_f32 = (&r * 76.0 + &g * 150.0 + &b * 29.0) / 255.0;
    let gray_u8 = gray_f32.mapv(|x| x as u8);

    // TODO: ideally we stack the channels. Not working yet.
    let gray_stacked = match stack(Axis(2), &[gray_u8.view(), gray_u8.view(), gray_u8.view()]) {
        Ok(gray_stacked) => gray_stacked,
        Err(err) => {
            panic!("Error stacking channels: {}", err);
        }
    };
    Image { data: gray_stacked }
}
