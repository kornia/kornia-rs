use kornia::image::Image;
use kornia::image::ops;
use kornia_tensor::{CpuAllocator, Tensor};

/// preprocess image for SigLIP2
/// input: u8 RGB image [224, 224, 3]
/// output: f32 tensor [1, 3, 224, 224]
pub fn preprocess(
    img: Image<u8, 3, CpuAllocator>,
) -> Tensor<f32, 4, CpuAllocator> {
    let size = img.size();
    assert_eq!(size.width, 224);
    assert_eq!(size.height, 224);

    let mut img_f32 =
        Image::<f32, 3, _>::from_size_val(size, 0.0, CpuAllocator).unwrap();
    ops::cast_and_scale(&img, &mut img_f32, 1.0 / 255.0).unwrap();

    let chw = img_f32.permute_axes([2, 0, 1]).as_contiguous();

    // SigLIP2 normalization using Kornia
    let mean = Tensor::from_slice([3], &[0.5f32, 0.5, 0.5], CpuAllocator).unwrap();
    let std = Tensor::from_slice([3], &[0.5f32, 0.5, 0.5], CpuAllocator).unwrap();

    let chw_norm = kornia_tensor::ops::normalize(&chw, &mean, &std).unwrap();

    chw_norm.unsqueeze(0)
}
