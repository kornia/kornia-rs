use kornia_rs::image::Image;
use kornia_rs::io::functional as F;
use kornia_rs::tensor::{CpuAllocator, Tensor3};
use ndarray::ShapeBuilder;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // read the image
    let image_path = std::path::Path::new("/home/edgar/software/kornia-rs/tests/data/dog.jpeg");
    let image: Image<u8, 3> = F::read_image_any(image_path)?;

    let rbg = Tensor3::<u8>::from_shape_vec(
        [image.rows(), image.cols(), 3],
        image
            .data
            .as_slice()
            .expect("failed to get image data")
            .to_vec(),
        CpuAllocator,
    )?;

    let mut gray = Tensor3::<u8>::new_uninitialized(
        [image.rows(), image.cols(), 1],
        rbg.storage.alloc().clone(),
    )?;

    //let image_f32: Image<f32, 3> = image.cast_and_scale::<f32>(1.0 / 255.0)?;

    // convert the image to grayscale
    //let gray: Image<f32, 1> = kornia_rs::color::gray_from_rgb(&image_f32)?;

    //let gray_resize: Image<f32, 1> = kornia_rs::resize::resize_native(
    //    &gray,
    //    kornia_rs::image::ImageSize {
    //        width: 128,
    //        height: 128,
    //    },
    //    kornia_rs::interpolation::InterpolationMode::Bilinear,
    //)?;

    //println!("gray_resize: {:?}", gray_resize.size());

    // create a Rerun recording stream
    let rec = rerun::RecordingStreamBuilder::new("Kornia App").spawn()?;

    // log the images
    //rec.log("image", &rerun::Image::try_from(image_f32.data)?)?;
    //rec.log("gray", &rerun::Image::try_from(gray.data)?)?;
    //rec.log("gray_resize", &rerun::Image::try_from(gray_resize.data)?)?;

    //let mut gray_buf = arrow_buffer::MutableBuffer::from_len_zeroed(image.rows() * image.cols());

    kornia_rs::color::gray_from_rgb_new(&rbg, &mut gray)?;

    //let mut tensor_data: rerun::TensorData = gray_buf.as_slice().into();
    // THIS DOES NOT WORK
    // tensor_data.shape = vec![image.rows(), image.cols(), 1].into();

    // THIS WORKS WANT TO AVOID THIS
    let img_vis = ndarray::Array3::<u8>::from_shape_vec(
        (image.rows(), image.cols(), 1),
        gray.as_slice().to_vec(),
    )?;

    rec.log("gray", &rerun::Image::try_from(img_vis)?)?;
    Ok(())
}
