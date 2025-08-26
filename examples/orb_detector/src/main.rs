use argh::FromArgs;
use std::path::PathBuf;

use kornia::{
    image::{Image, ImageSize},
    imgproc::{
        self,
        color::gray_from_rgb_u8,
        draw::draw_line,
        features::{match_descriptors, OrbDectector},
        flip::vertical_flip,
    },
    io::{functional::read_image_any_rgb8, png::read_image_png_mono8},
    tensor::CpuAllocator,
};

/// TODO
#[derive(FromArgs)]
struct Args {
    /// TODO
    #[argh(option, short = 'f')]
    image_path: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Args = argh::from_env();

    // start the recording stream
    let rec = rerun::RecordingStreamBuilder::new("Kornia Fast Detector App").spawn()?;

    // read the image
    let first_image = read_image_any_rgb8(args.image_path)?;
    let mut second_image = Image::from_size_val(first_image.size(), 0, CpuAllocator)?;
    vertical_flip(&first_image, &mut second_image)?;

    let mut first_image_gray = Image::from_size_val(first_image.size(), 0, CpuAllocator)?;
    let mut second_image_gray = Image::from_size_val(first_image.size(), 0, CpuAllocator)?;
    gray_from_rgb_u8(&first_image, &mut first_image_gray)?;
    gray_from_rgb_u8(&second_image, &mut second_image_gray)?;

    let mut first_image_f32: Image<_, 1, _> =
        Image::from_size_val(first_image.size(), 0f32, CpuAllocator)?;
    let mut second_image_f32: Image<_, 1, _> =
        Image::from_size_val(first_image.size(), 0f32, CpuAllocator)?;

    let f_f32 = first_image_f32.as_slice_mut();
    let s_f32 = second_image_f32.as_slice_mut();

    for (i, (&f_px, &s_px)) in first_image_gray
        .as_slice()
        .iter()
        .zip(second_image_gray.as_slice())
        .enumerate()
    {
        f_f32[i] = f_px as f32;
        s_f32[i] = s_px as f32;
    }

    let joined_image = join_images(&first_image_f32, &second_image_f32);
    rec.log_static(
        "image",
        &rerun::Image::from_elements(
            joined_image.as_slice(),
            joined_image.size().into(),
            rerun::ColorModel::L,
        ),
    )?;

    let orb_detector = OrbDectector::default();

    let first_detection = orb_detector.detect(&first_image_f32)?;
    let first_extract = orb_detector.extract(
        &first_image_f32,
        &first_detection.0,
        &first_detection.1,
        &first_detection.2,
    )?;

    let second_detection = orb_detector.detect(&second_image_f32)?;
    let second_extract = orb_detector.extract(
        &second_image_f32,
        &second_detection.0,
        &second_detection.1,
        &second_detection.2,
    )?;

    let matches = match_descriptors(&first_extract.0, &second_extract.0, None, true, None);

    // Converting keypoint coordinates to (W, H) for drawing match lines
    let mut coords = Vec::new();
    for &(i1, i2) in matches.iter() {
        let kp1 = &first_detection.0[i1];
        let kp2 = &second_detection.0[i2];

        let coords1 = (kp1.1, kp1.0);
        let coords2 = (kp2.1 + first_image_f32.width() as f32, kp2.0);

        coords.push([coords1, coords2]);
    }

    rec.log("image/matches", &rerun::LineStrips2D::new(coords))?;

    let keypoints1: Vec<[f32; 2]> = first_detection.0.iter().map(|kp| [kp.1, kp.0]).collect();
    let keypoints2: Vec<[f32; 2]> = second_detection
        .0
        .iter()
        .map(|kp| [kp.1 + first_image_f32.width() as f32, kp.0])
        .collect();

    rec.log("image/keypoints1", &rerun::Points2D::new(&keypoints1))?;
    rec.log("image/keypoints2", &rerun::Points2D::new(&keypoints2))?;

    Ok(())
}

fn join_images(
    image1: &Image<f32, 1, CpuAllocator>,
    image2: &Image<f32, 1, CpuAllocator>,
) -> Image<f32, 1, CpuAllocator> {
    let size = ImageSize {
        width: image1.width() + image2.width(),
        height: image1.height(),
    };

    let mut data = Vec::with_capacity(size.height * size.width);

    for (px1, px2) in image1
        .as_slice()
        .chunks(image1.width())
        .zip(image2.as_slice().chunks(image2.width()))
    {
        data.extend(px1);
        data.extend(px2);
    }

    Image::new(size, data, CpuAllocator).unwrap()
}
