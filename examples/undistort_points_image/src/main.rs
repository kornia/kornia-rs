use argh::FromArgs;
use std::path::PathBuf;

use kornia::{
    image::Image,
    imgproc::calibration::{
        distortion::{undistort_points, PolynomialDistortion, TermCriteria},
        CameraIntrinsic,
    },
    io::functional as F,
    tensor::{CpuAllocator, Tensor},
};

#[derive(FromArgs)]
/// Undistort an image using Brown–Conrady model
struct Args {
    /// path to an input image
    #[argh(option, short = 'i')]
    image_path: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Args = argh::from_env();

    // read the image
    let img_u8 = F::read_image_any_rgb8(args.image_path)?;
    let size = img_u8.size();
    let (h, w) = (size.height, size.width);

    // the intrinsic parameters
    let intrinsic = CameraIntrinsic {
        fx: w as f64,
        fy: w as f64,
        cx: w as f64 / 2.0,
        cy: h as f64 / 2.0,
    };

    // the distortion parameters
    let distortion = PolynomialDistortion {
        k1: -0.2,
        k2: 0.05,
        k3: 0.0,
        k4: 0.0,
        k5: 0.0,
        k6: 0.0,
        p1: 0.0,
        p2: 0.0,
    };

    // create grid of all distorted pixel coordinates
    let num_pixels = w * h;
    let mut distorted_points = vec![0.0; num_pixels * 2];
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 2;
            distorted_points[idx] = x as f64;
            distorted_points[idx + 1] = y as f64;
        }
    }

    // undistort all points
    let src = Tensor::<f64, 2, _>::from_shape_vec([num_pixels, 2], distorted_points, CpuAllocator)?;
    let mut dst = Tensor::<f64, 2, _>::from_shape_val([num_pixels, 2], 0.0, CpuAllocator);

    let criteria = TermCriteria {
        max_iter: 20,
        eps: 1e-6,
    };

    undistort_points(
        &src,
        &mut dst,
        &intrinsic,
        &distortion,
        None,
        None,
        None,
        criteria,
    )?;

    // bilinear splatting (distorted is mapped on undistorted)
    let img_f32 = img_u8.cast::<f32>()?;
    let mut img_undistorted = Image::<f32, 3, _>::from_size_val(size, 0.0, CpuAllocator)?;
    let mut weight_map = vec![0.0f32; num_pixels];

    let dst_slice = dst.as_slice();
    let src_img = img_f32.as_slice();
    let dst_img = img_undistorted.as_slice_mut();

    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            let x_norm = dst_slice[idx * 2];
            let y_norm = dst_slice[idx * 2 + 1];

            // convert normalized to pixel coordinates
            let undist_x = x_norm * intrinsic.fx + intrinsic.cx;
            let undist_y = y_norm * intrinsic.fy + intrinsic.cy;

            // bilinear splatting (distribute to 4 nearest neighbors)
            let x0 = undist_x.floor() as isize;
            let y0 = undist_y.floor() as isize;
            let dx = undist_x - x0 as f64;
            let dy = undist_y - y0 as f64;

            let src_r = src_img[idx * 3];
            let src_g = src_img[idx * 3 + 1];
            let src_b = src_img[idx * 3 + 2];

            for dy_offset in 0..2 {
                for dx_offset in 0..2 {
                    let nx = x0 + dx_offset;
                    let ny = y0 + dy_offset;

                    if nx >= 0 && nx < w as isize && ny >= 0 && ny < h as isize {
                        let weight = (if dx_offset == 0 { 1.0 - dx } else { dx })
                            * (if dy_offset == 0 { 1.0 - dy } else { dy });

                        let target_idx = (ny as usize) * w + (nx as usize);

                        dst_img[target_idx * 3] += src_r * weight as f32;
                        dst_img[target_idx * 3 + 1] += src_g * weight as f32;
                        dst_img[target_idx * 3 + 2] += src_b * weight as f32;
                        weight_map[target_idx] += weight as f32;
                    }
                }
            }
        }
    }

    for idx in 0..num_pixels {
        if weight_map[idx] > 0.0 {
            dst_img[idx * 3] /= weight_map[idx];
            dst_img[idx * 3 + 1] /= weight_map[idx];
            dst_img[idx * 3 + 2] /= weight_map[idx];
        }
    }

    // create and log a Rerun recording stream
    let rec = rerun::RecordingStreamBuilder::new("Kornia Undistortion").spawn()?;
    let res = [w as u32, h as u32];

    rec.log(
        "distorted",
        &rerun::Image::from_elements(img_u8.as_slice(), res, rerun::ColorModel::RGB),
    )?;

    let final_u8 = img_undistorted.cast_and_scale::<u8>(255.0 as u8)?;
    rec.log(
        "undistorted",
        &rerun::Image::from_elements(final_u8.as_slice(), res, rerun::ColorModel::RGB),
    )?;

    Ok(())
}
