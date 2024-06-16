use super::{CameraExtrinsic, CameraIntrinsic};
use crate::{
    image::{Image, ImageSize},
    interpolation::meshgrid,
};

pub struct PolynomialDistortion {
    pub k1: f64,
    pub k2: f64,
    pub k3: f64,
    pub k4: f64,
    pub k5: f64,
    pub k6: f64,
    pub p1: f64,
    pub p2: f64,
}

pub fn distort_point_polynomial(
    x: f64,
    y: f64,
    intrinsic: &CameraIntrinsic,
    distortion: &PolynomialDistortion,
) -> (f64, f64) {
    // unpack the intrinsic and distortion parameters
    let (fx, fy, cx, cy) = (intrinsic.fx, intrinsic.fy, intrinsic.cx, intrinsic.cy);
    let (k1, k2, k3, k4, k5, k6, p1, p2) = (
        distortion.k1,
        distortion.k2,
        distortion.k3,
        distortion.k4,
        distortion.k5,
        distortion.k6,
        distortion.p1,
        distortion.p2,
    );

    // normalize the coordinates
    let x = (x - cx) / fx;
    let y = (y - cy) / fy;

    // calculate the radial distance
    let r2 = x * x + y * y;

    // radial distortion
    let kr = (1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2)
        / (1.0 + k4 * r2 + k5 * r2 * r2 + k6 * r2 * r2 * r2);

    // tangential distortion
    let xd = x * kr + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
    let yd = y * kr + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;

    // denormalize the coordinates
    let xdst = fx * xd + cx;
    let ydst = fy * yd + cy;

    (xdst, ydst)
}

pub fn undistort_rectify_map_polynomial(
    intrinsic: &CameraIntrinsic,
    _extrinsic: &CameraExtrinsic,
    _new_intrinsic: &CameraIntrinsic,
    distortion: &PolynomialDistortion,
    size: &ImageSize,
) -> (Image<f32, 1>, Image<f32, 1>) {
    // generate the meshgrid with coordinates
    let x = ndarray::Array::linspace(0., size.width as f32 - 1., size.width)
        .insert_axis(ndarray::Axis(0));

    let y = ndarray::Array::linspace(0., size.height as f32 - 1., size.height)
        .insert_axis(ndarray::Axis(0));

    // combine the x and y coordinates to create a meshgrid
    let (map_x, map_y) = meshgrid(&x, &y);
    let (mut map_x, mut map_y) = (
        map_x.insert_axis(ndarray::Axis(2)),
        map_y.insert_axis(ndarray::Axis(2)),
    ); // add a channel axis  (HxWx1, HxWx1)

    ndarray::Zip::from(map_x.rows_mut())
        .and(map_y.rows_mut())
        .par_for_each(|mut xarr, mut yarr| {
            let (x, y) = (xarr[0], yarr[0]);
            let (xdst, ydst) = distort_point_polynomial(x as f64, y as f64, intrinsic, distortion);
            xarr[0] = xdst as f32;
            yarr[0] = ydst as f32;
        });

    let map_x = Image { data: map_x };
    let map_y = Image { data: map_y };

    (map_x, map_y)
}
