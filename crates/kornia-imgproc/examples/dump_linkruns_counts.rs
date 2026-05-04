//! Single-fixture helper for check_linkruns_quality.py — given a fixture
//! name + size (+ optional path or seed), emit the kornia linkruns
//! contour count to stdout as `fixture,WxH,count`.

use kornia_image::{allocator::CpuAllocator, Image, ImageSize};
use kornia_imgproc::contours_linkruns::find_external_contours_linkruns;

fn make_filled_square(w: usize, h: usize) -> Vec<u8> {
    let mw = w / 8;
    let mh = h / 8;
    let mut d = vec![0u8; w * h];
    for r in mh..(h - mh) {
        for c in mw..(w - mw) {
            d[r * w + c] = 1;
        }
    }
    d
}

fn make_hollow_square(w: usize, h: usize) -> Vec<u8> {
    let ow = w / 8;
    let oh = h / 8;
    let iw = w / 4;
    let ih = h / 4;
    let mut d = vec![0u8; w * h];
    for r in oh..(h - oh) {
        for c in ow..(w - ow) {
            d[r * w + c] = 1;
        }
    }
    for r in ih..(h - ih) {
        for c in iw..(w - iw) {
            d[r * w + c] = 0;
        }
    }
    d
}

fn make_noise(w: usize, h: usize, seed: u64) -> Vec<u8> {
    let mut state = seed;
    (0..w * h)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) & 1) as u8
        })
        .collect()
}

fn load_real(path: &str) -> Option<(usize, usize, Vec<u8>)> {
    let rgb = kornia_io::png::read_image_png_rgb8(path).ok()?;
    let (w, h) = (rgb.width(), rgb.height());
    let mut gray = Image::<u8, 1, _>::from_size_val(
        ImageSize { width: w, height: h }, 0, CpuAllocator,
    ).unwrap();
    kornia_imgproc::color::gray_from_rgb_u8(&rgb, &mut gray).ok()?;
    let mut bw = Image::<u8, 1, _>::from_size_val(
        ImageSize { width: w, height: h }, 0, CpuAllocator,
    ).unwrap();
    kornia_imgproc::threshold::threshold_binary(&gray, &mut bw, 127, 1).ok()?;
    Some((w, h, bw.as_slice().to_vec()))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let fixture = args.get(1).expect("fixture name required");
    let w: usize = args.get(2).expect("width required").parse().unwrap();
    let h: usize = args.get(3).expect("height required").parse().unwrap();
    let extra = args.get(4);

    let data = match fixture.as_str() {
        "filled_square" => make_filled_square(w, h),
        "hollow_square" => make_hollow_square(w, h),
        "sparse_noise" => make_noise(w, h, 0xC0FFEE),
        "real" => {
            let (_w, _h, d) = load_real(extra.expect("path required for real"))
                .expect("failed to load");
            d
        }
        _ => panic!("unknown fixture: {fixture}"),
    };

    let contours = find_external_contours_linkruns(&data, w, h);
    println!("{fixture},{w}x{h},{}", contours.len());
}
