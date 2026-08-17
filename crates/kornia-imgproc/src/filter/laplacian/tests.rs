use crate::filter::laplacian_u8;
use kornia_image::{Image, ImageSize};

#[test]
fn test_laplacian_u8() {
    let src = Image::<u8, 1>::new(
        ImageSize {
            width: 3,
            height: 3,
        },
        vec![1, 2, 3, 4, 5, 6, 7, 8, 9],
    )
    .unwrap();
    let mut dst = Image::<i16, 1>::from_size_val(src.size(), 0).unwrap();

    laplacian_u8(&src, &mut dst).unwrap();

    let out = dst.as_slice();
    // [1, 2, 3] -> pad [1, 1, 2, 3, 3]
    // [4, 5, 6] -> pad [4, 4, 5, 6, 6]
    // [7, 8, 9] -> pad [7, 7, 8, 9, 9]
    // Laplacian 3x3 for center (5):
    // v_up=2, v_down=8, v_left=4, v_right=6, v_center=5 -> 2+8+4+6 - 4*5 = 20 - 20 = 0.
    assert_eq!(out[4], 0);

    // Top-left (1): v_up=1, v_down=4, v_left=1, v_right=2, v_center=1 -> 1+4+1+2 - 4 = 4.
    assert_eq!(out[0], 4);

    // Bottom-right (9): v_up=6, v_down=9, v_left=8, v_right=9, v_center=9 -> 6+9+8+9 - 36 = -4.
    assert_eq!(out[8], -4);
}

#[cfg(feature = "cuda")]
#[test]
fn test_laplacian_cuda() {
    use cudarc::driver::CudaContext;

    let src = Image::<u8, 1>::new(
        ImageSize {
            width: 3,
            height: 3,
        },
        vec![1, 2, 3, 4, 5, 6, 7, 8, 9],
    )
    .unwrap();
    let dst = Image::<i16, 1>::from_size_val(src.size(), 0).unwrap();

    let ctx = CudaContext::new(0).unwrap();
    let stream = std::sync::Arc::new(ctx.default_stream());

    let src_dev = src.to_cuda(&stream).unwrap();
    let mut dst_dev = dst.to_cuda(&stream).unwrap();

    laplacian_u8(&src_dev, &mut dst_dev).unwrap();
    stream.synchronize().unwrap();

    let dst_host = dst_dev.to_host(&stream).unwrap();
    let out = dst_host.as_slice();

    assert_eq!(out[4], 0);
    assert_eq!(out[0], 4);
    assert_eq!(out[8], -4);
}
