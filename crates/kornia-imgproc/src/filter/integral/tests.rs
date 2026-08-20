use crate::filter::integral_image_u8;
use kornia_image::{Image, ImageSize};

#[test]
fn test_integral_image_u8() {
    let src = Image::<u8, 1>::new(
        ImageSize {
            width: 3,
            height: 3,
        },
        vec![1, 2, 3, 4, 5, 6, 7, 8, 9],
    )
    .unwrap();
    let mut dst = Image::<f32, 1>::from_size_val(src.size(), 0.0).unwrap();

    integral_image_u8(&src, &mut dst).unwrap();

    let out = dst.as_slice();
    assert_eq!(out[0], 1.0);
    assert_eq!(out[1], 3.0);
    assert_eq!(out[2], 6.0);
    assert_eq!(out[3], 5.0);
    assert_eq!(out[4], 12.0);
    assert_eq!(out[5], 21.0);
    assert_eq!(out[6], 12.0);
    assert_eq!(out[7], 27.0);
    assert_eq!(out[8], 45.0);
}

#[cfg(feature = "cuda")]
#[test]
fn test_integral_image_cuda() {
    use cudarc::driver::CudaContext;

    let src = Image::<u8, 1>::new(
        ImageSize {
            width: 3,
            height: 3,
        },
        vec![1, 2, 3, 4, 5, 6, 7, 8, 9],
    )
    .unwrap();
    let dst = Image::<f32, 1>::from_size_val(src.size(), 0.0).unwrap();

    let ctx = CudaContext::new(0).unwrap();
    let stream = std::sync::Arc::new(ctx.default_stream());

    let src_dev = src.to_cuda(&stream).unwrap();
    let mut dst_dev = dst.to_cuda(&stream).unwrap();

    integral_image_u8(&src_dev, &mut dst_dev).unwrap();
    stream.synchronize().unwrap();

    let dst_host = dst_dev.to_host(&stream).unwrap();
    let out = dst_host.as_slice();
    assert_eq!(out[0], 1.0);
    assert_eq!(out[1], 3.0);
    assert_eq!(out[2], 6.0);
    assert_eq!(out[3], 5.0);
    assert_eq!(out[4], 12.0);
    assert_eq!(out[5], 21.0);
    assert_eq!(out[6], 12.0);
    assert_eq!(out[7], 27.0);
    assert_eq!(out[8], 45.0);
}
