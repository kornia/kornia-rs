use clap::Parser;
use std::path::PathBuf;

use kornia::{
    image::{Image, ImageSize},
    imgproc::{
        self,
        calibration::{
            distortion::{
                distort_point_polynomial, generate_correction_map_polynomial, PolynomialDistortion,
            },
            CameraExtrinsic, CameraIntrinsic,
        },
        interpolation::grid::meshgrid_from_fn,
    },
    io::functional as F,
};

#[derive(Parser)]
struct Args {
    #[arg(short, long)]
    image_path: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // read the image
    let img = F::read_image_any(args.image_path)?;

    // the intrinsic parameters of an Oak-D camera
    let intrinsic = CameraIntrinsic {
        fx: 1548.24072265625,
        fy: 1546.2818603515625,
        cx: 965.2394409179688,
        cy: 538.1027221679688,
    };

    // the distortion parameters of an Oak-D camera
    let distortion = PolynomialDistortion {
        k1: -0.9245772957801819,
        k2: -59.177528381347656,
        k3: 339.5682067871094,
        k4: -1.0775686502456665,
        k5: -57.51222229003906,
        k6: 333.1765441894531,
        p1: 0.0010584688279777765,
        p2: 0.001023623626679182,
    };

    // create the correction map
    let count = std::sync::atomic::AtomicUsize::new(0);

    let (map_x, map_y) = meshgrid_from_fn(img.cols(), img.rows(), |x, y| {
        // apply the distortion model to the pixel
        let (x_dst, y_dst) = distort_point_polynomial(x as f64, y as f64, &intrinsic, &distortion);

        //println!("x_dst, y_dst: {} {}", x_dst, y_dst);

        // clamp the pixel coordinates to the image boundaries

        if (x_dst < 0.0) || (x_dst > img.cols() as f64) {
            println!("x_dst out of bounds: {}", x_dst);
            count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        };

        if (y_dst < 0.0) || (y_dst > img.rows() as f64) {
            println!("y_dst out of bounds: {}", y_dst);
            count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        };

        let x_dst = x_dst.clamp(0.0, img.cols() as f64 - 1.0);
        let y_dst = y_dst.clamp(0.0, img.rows() as f64 - 1.0);

        Ok((x_dst as f32, y_dst as f32))
    })?;

    println!("count: {}", count.load(std::sync::atomic::Ordering::SeqCst));

    // apply the remap
    let mut img_undistorted = Image::from_size_val(img.size(), 0.0)?;
    imgproc::interpolation::remap(
        &img.clone().cast_and_scale(1.0 / 255.0)?,
        &mut img_undistorted,
        &map_x,
        &map_y,
        imgproc::interpolation::InterpolationMode::Bilinear,
    )?;

    let img_undistorted = img_undistorted.scale_and_cast::<u8>(255.0)?;

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
