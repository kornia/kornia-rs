use argh::FromArgs;
use std::path::PathBuf;

use kornia::{
    image::{Image, ImageSize},
    imgproc::{
        self,
        color::gray_from_rgb_u8,
        features::{match_descriptors, OrbDectector},
    },
    io::{
        functional::read_image_any_rgb8,
        jpeg,
        v4l::{PixelFormat, V4LCameraConfig, V4lVideoCapture},
    },
    tensor::CpuAllocator,
};

/// TODO
#[derive(FromArgs)]
struct Args {
    /// TODO
    #[argh(positional)]
    image_path: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Args = argh::from_env();

    println!("{}", args.image_path.to_string_lossy());

    // start the recording stream
    let rec = rerun::RecordingStreamBuilder::new("Kornia Fast Detector App").spawn()?;

    let webcam_size = ImageSize {
        width: 640,
        height: 480,
    };
    let mut webcam = V4lVideoCapture::new(V4LCameraConfig {
        size: webcam_size,
        ..Default::default()
    })?;

    let mut webcam_frame = Image::from_size_val(webcam_size, 0, CpuAllocator)?;
    let mut webcam_frame_gray = Image::from_size_val(webcam_size, 0, CpuAllocator)?;
    let mut webcam_frame_grayf32 = Image::from_size_val(webcam_size, 0.0, CpuAllocator)?;

    let webcam_orb_detector = OrbDectector::new();

    println!("Webcam done");

    // read the image
    let img_rgb = read_image_any_rgb8(args.image_path)?;

    let mut img_gray = Image::from_size_val(img_rgb.size(), 0, CpuAllocator)?;
    let mut img_grayf32 = Image::from_size_val(img_rgb.size(), 0.0, CpuAllocator)?;
    gray_from_rgb_u8(&img_rgb, &mut img_gray)?;
    u8_to_f32_image(&img_gray, &mut img_grayf32);

    let mut joined_image = Image::from_size_val(
        ImageSize {
            width: webcam_size.width + img_rgb.width(),
            height: webcam_size.height.max(img_rgb.height()),
        },
        0,
        CpuAllocator,
    )?;

    let img_orb_detector = OrbDectector::new();
    let img_detection = img_orb_detector.detect(&img_grayf32)?;
    let img_extract = img_orb_detector.extract(
        &img_grayf32,
        &img_detection.0,
        &img_detection.1,
        &img_detection.2,
    )?;

    loop {
        let Some(frame) = webcam.grab_frame()? else {
            continue;
        };

        let buf = frame.buffer.as_slice();
        match frame.pixel_format {
            PixelFormat::YUYV => {
                imgproc::color::convert_yuyv_to_rgb_u8(
                    buf,
                    &mut webcam_frame,
                    imgproc::color::YuvToRgbMode::Bt601Full,
                )?;
            }
            PixelFormat::MJPG => {
                jpeg::decode_image_jpeg_rgb8(buf, &mut webcam_frame)?;
            }
            _ => return Err(format!("Unsupported format: {}", frame.pixel_format).into()),
        }

        gray_from_rgb_u8(&webcam_frame, &mut webcam_frame_gray)?;
        u8_to_f32_image(&webcam_frame_gray, &mut webcam_frame_grayf32);

        join_images_inplace(&webcam_frame_gray, &img_gray, &mut joined_image);

        rec.log(
            "image",
            &rerun::Image::from_elements(
                joined_image.as_slice(),
                joined_image.size().into(),
                rerun::ColorModel::L,
            ),
        )?;

        let webcam_detection = webcam_orb_detector.detect(&webcam_frame_grayf32)?;
        let webcam_extraction = webcam_orb_detector.extract(
            &webcam_frame_grayf32,
            &webcam_detection.0,
            &webcam_detection.1,
            &webcam_detection.2,
        )?;

        let matches = match_descriptors(&webcam_extraction.0, &img_extract.0, None, true, None);

        // Converting keypoint coordinates to (W, H) for drawing match lines
        let mut coords = Vec::new();
        for &(i1, i2) in matches.iter() {
            let kp1 = &webcam_detection.0[i1];
            let kp2 = &img_detection.0[i2];

            let coords1 = (kp1.1, kp1.0);
            let coords2 = (kp2.1 + webcam_frame_grayf32.width() as f32, kp2.0);

            coords.push([coords1, coords2]);
        }

        rec.log("image/matches", &rerun::LineStrips2D::new(coords))?;

        let keypoints1: Vec<[f32; 2]> = webcam_detection.0.iter().map(|kp| [kp.1, kp.0]).collect();
        let keypoints2: Vec<[f32; 2]> = img_detection
            .0
            .iter()
            .map(|kp| [kp.1 + webcam_frame.width() as f32, kp.0])
            .collect();

        rec.log("image/keypoints1", &rerun::Points2D::new(&keypoints1))?;
        rec.log("image/keypoints2", &rerun::Points2D::new(&keypoints2))?;
    }
}

fn join_images_inplace(
    image1: &Image<u8, 1, CpuAllocator>,
    image2: &Image<u8, 1, CpuAllocator>,
    out: &mut Image<u8, 1, CpuAllocator>,
) {
    let width1 = image1.width();
    let width2 = image2.width();
    let height = image1.height();

    assert_eq!(image1.height(), image2.height());
    assert_eq!(out.width(), width1 + width2);
    assert_eq!(out.height(), height);

    let row_len1 = width1;
    let row_len2 = width2;
    let out_row_len = width1 + width2;

    let src1 = image1.as_slice();
    let src2 = image2.as_slice();
    let dst = out.as_slice_mut();

    for row in 0..height {
        let src1_start = row * row_len1;
        let src2_start = row * row_len2;
        let dst_start = row * out_row_len;

        // Copy first image row
        dst[dst_start..dst_start + row_len1]
            .copy_from_slice(&src1[src1_start..src1_start + row_len1]);
        // Copy second image row
        dst[dst_start + row_len1..dst_start + row_len1 + row_len2]
            .copy_from_slice(&src2[src2_start..src2_start + row_len2]);
    }
}

fn u8_to_f32_image(src: &Image<u8, 1, CpuAllocator>, dst: &mut Image<f32, 1, CpuAllocator>) {
    src.as_slice()
        .iter()
        .zip(dst.as_slice_mut())
        .for_each(|(&s, d)| {
            *d = s as f32 / 255.0;
        });
}
