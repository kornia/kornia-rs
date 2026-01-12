use kornia::image::Image;
use kornia_tensor::{CpuAllocator, Tensor};

/// preprocess image for siglip2
/// input: u8 rgb image [224,224,3]
/// output: f32 tensor [1,3,224,224]
pub fn preprocess(
    img: Image<u8, 3, CpuAllocator>,
) -> Tensor<f32, 4, CpuAllocator> {
    let size = img.size();
    assert_eq!(size.width, 224);
    assert_eq!(size.height, 224);

    // convert to f32 and scale to [0,1]
    let mut img_f32 =
        Image::<f32, 3, _>::from_size_val(size, 0.0, CpuAllocator).unwrap();

    kornia::image::ops::cast_and_scale(&img, &mut img_f32, 1.0 / 255.0).unwrap();

    let chw = img_f32.permute_axes([2, 0, 1]).as_contiguous();

    // normalize (SigLIP2)
    let mean = [0.5f32, 0.5, 0.5];
    let std = [0.5f32, 0.5, 0.5];

    let mut data = chw.into_vec();
    let hw = 224 * 224;

    for c in 0..3 {
        let offset = c * hw;
        for i in 0..hw {
            data[offset + i] = (data[offset + i] - mean[c]) / std[c];
        }
    }
    Tensor::from_shape_vec([1, 3, 224, 224], data, CpuAllocator).unwrap()
}