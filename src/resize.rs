use crate::image::{Image, ImageSize};
use ndarray::{stack, Array, Array2, Array3, Ix2, Zip};

fn meshgrid(x: &Array<f32, Ix2>, y: &Array<f32, Ix2>) -> (Array2<f32>, Array2<f32>) {
    let nx = x.len_of(ndarray::Axis(1));
    let ny = y.len_of(ndarray::Axis(1));
    //println!("nx: {:?}", nx);
    //println!("ny: {:?}", ny);

    //println!("x: {:?}", x.shape());
    let xx = x.broadcast((ny, nx)).unwrap().to_owned();
    //println!("xx: {:?}", xx);

    //println!("y: {:?}", y.shape());
    let yy = y.broadcast((nx, ny)).unwrap().t().to_owned();
    //println!("yy: {:?}", yy);

    (xx, yy)
}

fn bilinear_interpolation(image: Image, u: f32, v: f32, c: usize) -> f32 {
    let image_size = image.image_size();
    let height = image_size.height;
    let width = image_size.width;

    let iu = u.trunc() as usize;
    let iv = v.trunc() as usize;
    let frac_u = u.fract();
    let frac_v = v.fract();
    let val00 = image.data[[iv, iu, c]] as f32;
    let val01 = if iu + 1 < width {
        image.data[[iv, iu + 1, c]] as f32
    } else {
        val00
    };
    let val10 = if iv + 1 < height {
        image.data[[iv + 1, iu, c]] as f32
    } else {
        val00
    };
    let val11 = if iu + 1 < width && iv + 1 < height {
        image.data[[iv + 1, iu + 1, c]] as f32
    } else {
        val00
    };

    val00 * (1. - frac_u) * (1. - frac_v)
        + val01 * frac_u * (1. - frac_v)
        + val10 * (1. - frac_u) * frac_v
        + val11 * frac_u * frac_v
}

fn nearest_neighbor_interpolation(image: Image, u: f32, v: f32, c: usize) -> f32 {
    let image_size = image.image_size();
    let height = image_size.height;
    let width = image_size.width;

    let iu = u.round() as usize;
    let iv = v.round() as usize;

    let iu = iu.clamp(0, width - 1);
    let iv = iv.clamp(0, height - 1);

    let val = image.data[[iv, iu, c]] as f32;

    val
}

#[derive(Debug, Clone, Copy)]
pub enum InterpolationMode {
    Bilinear,
    NearestNeighbor,
}

pub struct ResizeOptions {
    pub interpolation: InterpolationMode,
}

impl Default for ResizeOptions {
    fn default() -> Self {
        ResizeOptions {
            interpolation: InterpolationMode::Bilinear,
        }
    }
}

pub fn resize(image: Image, new_size: ImageSize, optional_args: ResizeOptions) -> Image {
    let image_size = image.image_size();

    // create the output image
    let mut output = Array3::<u8>::zeros((new_size.height, new_size.width, 3));

    // create a grid of x and y coordinates for the output image
    // and interpolate the values from the input image.
    let x = ndarray::Array::linspace(0., (image_size.width - 1) as f32, new_size.width)
        .insert_axis(ndarray::Axis(0));
    let y = ndarray::Array::linspace(0., (image_size.height - 1) as f32, new_size.height)
        .insert_axis(ndarray::Axis(0));

    let (xx, yy) = meshgrid(&x, &y);
    //println!("xx: {:?}", xx);
    //println!("yy: {:?}", yy);

    // TODO: parallelize this
    //for i in 0..xx.shape()[0] {
    //    for j in 0..xx.shape()[1] {
    //        let x = xx[[i, j]];
    //        let y = yy[[i, j]];
    //        //println!("x: {:?}", x);
    //        //println!("y: {:?}", y);
    //        //println!("###########3");

    //        for k in 0..3 {
    //            //output[[i, j, k]] = image_data[[y as usize, x as usize, k]];
    //            output[[i, j, k]] = bilinear_interpolation(image.clone(), x, y, k) as u8;
    //        }
    //    }
    //}

    // TODO: benchmark this
    let xy = stack![ndarray::Axis(2), xx, yy];

    Zip::from(xy.rows())
        .and(output.rows_mut())
        .par_for_each(|xy, mut out| {
            assert_eq!(xy.len(), 2);
            let x = xy[0];
            let y = xy[1];
            //println!("x: {:?}", x);
            //println!("y: {:?}", y);
            //println!("###########3");
            //println!("out: {:?}", out.shape());
            for k in [0, 1, 2].iter() {
                //output[[i, j, k]] = image_data[[y as usize, x as usize, k]];
                //out[*k] = bilinear_interpolation(image.clone(), x, y, *k) as u8;
                //out[*k] = nearest_neighbor_interpolation(image.clone(), x, y, *k) as u8;
                //out[*k] = match interpolation {
                //    Interpolation::Bilinear => bilinear_interpolation(image.clone(), x, y, *k) as u8,
                //    Interpolation::NearestNeighbor => nearest_neighbor_interpolation(image.clone(), x, y, *k) as u8,
                //};
                match optional_args.interpolation {
                    InterpolationMode::Bilinear => {
                        out[*k] = bilinear_interpolation(image.clone(), x, y, *k) as u8
                    }
                    InterpolationMode::NearestNeighbor => {
                        out[*k] = nearest_neighbor_interpolation(image.clone(), x, y, *k) as u8
                    }
                }
            }
            //for k in 0..3 {
            //    out[k] = bilinear_interpolation(image.clone(), x, y, k) as u8;
            //}
        });

    Image { data: output }
}

#[cfg(test)]
mod tests {

    #[test]
    fn resize_smoke() {
        use crate::image::{Image, ImageSize};
        let image = Image::from_shape_vec([4, 5, 3], vec![0; 4 * 5 * 3]);
        let image_resized = super::resize(
            image,
            ImageSize {
                width: 2,
                height: 3,
            },
            super::ResizeOptions::default(),
        );
        assert_eq!(image_resized.num_channels(), 3);
        assert_eq!(image_resized.image_size().width, 2);
        assert_eq!(image_resized.image_size().height, 3);
    }
}
