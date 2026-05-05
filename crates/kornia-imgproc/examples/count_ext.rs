
use kornia_image::{allocator::CpuAllocator, Image, ImageSize};
use kornia_imgproc::contours::{find_contours, RetrievalMode, ContourApproximationMode};

fn main() {
    let path = std::env::args().nth(1).unwrap();
    let rgb = kornia_io::png::read_image_png_rgb8(&path).unwrap();
    let (w, h) = (rgb.width(), rgb.height());
    let mut gray = Image::<u8, 1, _>::from_size_val(ImageSize { width: w, height: h }, 0, CpuAllocator).unwrap();
    kornia_imgproc::color::gray_from_rgb_u8(&rgb, &mut gray).unwrap();
    let mut bw = Image::<u8, 1, _>::from_size_val(ImageSize { width: w, height: h }, 0, CpuAllocator).unwrap();
    kornia_imgproc::threshold::threshold_binary(&gray, &mut bw, 127, 1).unwrap();
    let mode = match std::env::args().nth(2).as_deref() {
        Some("list") => RetrievalMode::List,
        Some("ccomp") => RetrievalMode::CComp,
        Some("tree") => RetrievalMode::Tree,
        _ => RetrievalMode::External,
    };
    let r = find_contours(&bw, mode, ContourApproximationMode::Simple).unwrap();
    println!("{}", r.contours.len());
}
