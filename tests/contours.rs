use kornia_image::{Image, ImageSize};
use kornia_tensor::CpuAllocator;
use kornia_imgproc::contours::{find_contours, RetrievalMode, ContourApproximationMode};

#[test]
fn test_find_contours_simple_square() {
    // 5x5 image with a 3x3 square in the middle
    let data = vec![
        0, 0, 0, 0, 0,
        0, 1, 1, 1, 0,
        0, 1, 1, 1, 0,
        0, 1, 1, 1, 0,
        0, 0, 0, 0, 0,
    ];
    let image = Image::<u8, 1, _>::new(
        ImageSize { width: 5, height: 5 },
        data,
        CpuAllocator,
    ).unwrap();

    let (contours, _) = find_contours(&image, RetrievalMode::External, ContourApproximationMode::None).unwrap();

    println!("Contours found: {:?}", contours);

    assert_eq!(contours.len(), 1);
    // Expected contour: (1,1), (1,2), (1,3), (2,3), (3,3), (3,2), (3,1), (2,1) (order depends on starting point and direction)
    // Actually typically starts at top-left (1,1).
    // Trace CCW? 
}
