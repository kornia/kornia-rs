
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
    println!("count: {}", r.contours.len());
    if std::env::var("DUMP_BOXES").is_ok() {
        for c in &r.contours {
            if c.is_empty() { continue; }
            let mut xmin = c[0][0]; let mut ymin = c[0][1];
            let mut xmax = c[0][0]; let mut ymax = c[0][1];
            for &[x, y] in c {
                if x < xmin { xmin = x; }
                if x > xmax { xmax = x; }
                if y < ymin { ymin = y; }
                if y > ymax { ymax = y; }
            }
            println!("[{},{},{},{}]", xmin, ymin, xmax, ymax);
        }
    }
    if std::env::var("DUMP_HIER").is_ok() {
        let mut parent_count = std::collections::BTreeMap::new();
        let mut frame_outer = 0u64;
        let mut frame_other = 0u64;
        for h in &r.hierarchy {
            *parent_count.entry(h[3]).or_insert(0u64) += 1;
            if h[3] <= 0 { frame_outer += 1; }
        }
        for (parent, n) in parent_count.iter().take(10) {
            println!("  parent={parent}: {n}");
        }
        println!("  total parent<=0: {}", frame_outer);
        let _ = frame_other;
    }
}
