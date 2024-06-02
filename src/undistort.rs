use crate::image::{Image, ImageSize};
use crate::resize::{interpolate_pixel, meshgrid, InterpolationMode};
use anyhow::Result;

type IntrinsicMatrix = [f32; 9];
type DistortionCoefficients = [f32; 12];

fn distort_point(
    u: f32,
    v: f32,
    intrinsics: IntrinsicMatrix,
    distortion: DistortionCoefficients,
) -> (f32, f32) {
    let (fx, fy, cx, cy) = (intrinsics[0], intrinsics[4], intrinsics[2], intrinsics[5]);
    let (k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4) = (
        distortion[0],
        distortion[1],
        distortion[2],
        distortion[3],
        distortion[4],
        distortion[5],
        distortion[6],
        distortion[7],
        distortion[8],
        distortion[9],
        distortion[10],
        distortion[11],
    );
    return (0.0, 0.0);
}

fn undistort_image(
    src: &Image<f32, 1>,
    intrinsics_matrix: IntrinsicMatrix,
    distortion_coefficients: DistortionCoefficients,
) -> Result<Image<f32, 1>> {
    let mut dst = Image::from_size_val(src.size(), 0.0)?;

    for v in 0..src.size().height {
        for u in 0..src.size().width {
            let (u_distorted, v_distorted) = distort_point(
                u as f32,
                v as f32,
                intrinsics_matrix,
                distortion_coefficients,
            );

            let pixel = interpolate_pixel(
                &src.data,
                u_distorted,
                v_distorted,
                0,
                InterpolationMode::Bilinear,
            );
            dst.set_pixel(u, v, 0, pixel)?;
        }
    }

    Ok(dst)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn undistort_image_smoke() -> Result<()> {
        let intrinsic_matrix = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let distortion_coefficients = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        let image = Image::<_, 1>::new(
            ImageSize {
                width: 4,
                height: 5,
            },
            vec![0f32; 4 * 5],
        )?;

        let image_undistorted = undistort_image(&image, intrinsic_matrix, distortion_coefficients)?;

        assert_eq!(image_undistorted.num_channels(), 1);
        assert_eq!(image_undistorted.size().width, 4);
        assert_eq!(image_undistorted.size().height, 5);

        println!("{:?}", image_undistorted.data);

        Ok(())
    }
}
