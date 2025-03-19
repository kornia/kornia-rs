use argh::FromArgs;
use std::path::PathBuf;

use kornia::{
    image::{Image, ImageSize},
    imgproc,
    imgproc::calibration::{
        distortion::{generate_correction_map_polynomial, PolynomialDistortion},
        {CameraExtrinsic, CameraIntrinsic},
    },
    io::functional as F,
};

#[derive(FromArgs)]
/// Undistort an image and log it to Rerun
struct Args {
    /// path to an input image
    #[argh(option, short = 'i')]
    image_path: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Args = argh::from_env();

    // read the image
    let img = F::read_image_any_rgb8(args.image_path)?;

    // the intrinsic parameters of an Oak-D camera
    let intrinsic = CameraIntrinsic {
        fx: 577.48583984375,
        fy: 652.8748779296875,
        cx: 577.48583984375,
        cy: 386.1428833007813,
    };

    // the distortion parameters of an Oak-D camera
    let distortion = PolynomialDistortion {
        k1: 1.7547749280929563,
        k2: 0.0097926277667284,
        k3: -0.027250492945313457,
        k4: 2.1092164516448975,
        k5: 0.462927520275116,
        k6: -0.08215277642011642,
        p1: -0.00005457743463921361,
        p2: 0.00003006766564794816,
    };

    let extrinsic = CameraExtrinsic {
        rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        translation: [0.0, 0.0, 0.0],
    };

    // create a base grid

    let (map_x, map_y) = generate_correction_map_polynomial(
        &intrinsic,
        &extrinsic,
        &intrinsic,
        &distortion,
        &ImageSize {
            width: img.cols(),
            height: img.rows(),
        },
    )?;

    // apply the remap
    let mut img_undistorted = Image::from_size_val(img.size(), 0.0)?;
    imgproc::interpolation::remap(
        &img.clone().cast_and_scale(1.0 / 255.0)?,
        &mut img_undistorted,
        &map_x,
        &map_y,
        imgproc::interpolation::InterpolationMode::Bilinear,
    )?;

    // create a Rerun recording stream
    let rec = rerun::RecordingStreamBuilder::new("Kornia App").spawn()?;

    rec.log(
        "img",
        &rerun::Image::from_elements(img.as_slice(), img.size().into(), rerun::ColorModel::RGB),
    )?;

    rec.log(
        "img_undistorted",
        &rerun::Image::from_elements(
            img_undistorted.as_slice(),
            img_undistorted.size().into(),
            rerun::ColorModel::RGB,
        ),
    )?;

    Ok(())
}
