use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use kornia_image::{allocator::CpuAllocator, Image, ImageSize};
use kornia_imgproc::contours::{
    find_contours, ContourApproximationMode, FindContoursExecutor, RetrievalMode,
};
use opencv::{core, imgproc, prelude::*};

/// Filled square with a margin of size/8 on each side.
fn make_filled_square(width: usize, height: usize) -> Vec<u8> {
    let margin_w = width / 8;
    let margin_h = height / 8;
    let mut data = vec![0u8; width * height];
    for r in margin_h..(height - margin_h) {
        for c in margin_w..(width - margin_w) {
            data[r * width + c] = 1;
        }
    }
    data
}

/// Hollow square (ring)
fn make_hollow_square(width: usize, height: usize) -> Vec<u8> {
    let outer_w = width / 8;
    let outer_h = height / 8;
    let inner_w = width / 4;
    let inner_h = height / 4;
    let mut data = vec![0u8; width * height];
    for r in outer_h..(height - outer_h) {
        for c in outer_w..(width - outer_w) {
            if r < inner_h || r >= (height - inner_h) || c < inner_w || c >= (width - inner_w) {
                data[r * width + c] = 1;
            }
        }
    }
    data
}

/// Sparse random noise via a simple LCG
fn make_noise(width: usize, height: usize, seed: u64) -> Vec<u8> {
    let mut state = seed;
    (0..width * height)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) & 1) as u8
        })
        .collect()
}

fn kornia_image(width: usize, height: usize, data: Vec<u8>) -> Image<u8, 1, CpuAllocator> {
    Image::<u8, 1, _>::new(ImageSize { width, height }, data, CpuAllocator).expect("kornia image")
}

/// OpenCV expects 0/255 for binary images, not 0/1.
fn opencv_mat(width: usize, height: usize, data: &[u8]) -> core::Mat {
    let mut mat = core::Mat::zeros(height as i32, width as i32, core::CV_8UC1)
        .unwrap()
        .to_mat()
        .unwrap();
    for r in 0..height {
        for c in 0..width {
            *mat.at_2d_mut::<u8>(r as i32, c as i32).unwrap() =
                if data[r * width + c] != 0 { 255 } else { 0 };
        }
    }
    mat
}

/// Register all six variants
fn register_variants(
    group: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>,
    parameter_string: &str,
    kornia_img: &Image<u8, 1, CpuAllocator>,
    cv_mat: &core::Mat,
) {
    group.bench_with_input(
        BenchmarkId::new("kornia_none", parameter_string),
        kornia_img,
        |b, img| {
            b.iter(|| {
                std::hint::black_box(
                    find_contours(img, RetrievalMode::List, ContourApproximationMode::None).ok(),
                )
            })
        },
    );

    group.bench_with_input(
        BenchmarkId::new("kornia_simple", parameter_string),
        kornia_img,
        |b, img| {
            b.iter(|| {
                std::hint::black_box(
                    find_contours(img, RetrievalMode::List, ContourApproximationMode::Simple).ok(),
                )
            })
        },
    );

    group.bench_with_input(
        BenchmarkId::new("kornia_exec_none", parameter_string),
        kornia_img,
        |b, img| {
            let mut exec = FindContoursExecutor::new();
            b.iter(|| {
                std::hint::black_box(
                    exec.find_contours(img, RetrievalMode::List, ContourApproximationMode::None)
                        .ok(),
                )
            })
        },
    );

    group.bench_with_input(
        BenchmarkId::new("kornia_exec_simple", parameter_string),
        kornia_img,
        |b, img| {
            let mut exec = FindContoursExecutor::new();
            b.iter(|| {
                std::hint::black_box(
                    exec.find_contours(img, RetrievalMode::List, ContourApproximationMode::Simple)
                        .ok(),
                )
            })
        },
    );

    let mut cv_contours: core::Vector<core::Vector<core::Point>> = core::Vector::new();
    let mut cv_hierarchy: core::Vector<core::Vec4i> = core::Vector::new();

    group.bench_function(BenchmarkId::new("opencv_none", parameter_string), |b| {
        b.iter(|| {
            imgproc::find_contours_with_hierarchy(
                cv_mat,
                &mut cv_contours,
                &mut cv_hierarchy,
                imgproc::RETR_LIST,
                imgproc::CHAIN_APPROX_NONE,
                core::Point::new(0, 0),
            )
            .unwrap()
        })
    });

    group.bench_function(BenchmarkId::new("opencv_simple", parameter_string), |b| {
        b.iter(|| {
            imgproc::find_contours_with_hierarchy(
                cv_mat,
                &mut cv_contours,
                &mut cv_hierarchy,
                imgproc::RETR_LIST,
                imgproc::CHAIN_APPROX_SIMPLE,
                core::Point::new(0, 0),
            )
            .unwrap()
        })
    });
}

// Benchmark groups

fn bench_filled_square(c: &mut Criterion) {
    let mut group = c.benchmark_group("contours_filled_square");

    for (width, height) in [(128, 128), (256, 256), (512, 512), (1024, 1024)].iter() {
        group.throughput(Throughput::Elements((*width * *height) as u64));
        let parameter_string = format!("{width}x{height}");

        let data = make_filled_square(*width, *height);
        let kornia_img = kornia_image(*width, *height, data.clone());
        let cv_mat = opencv_mat(*width, *height, &data);

        register_variants(&mut group, &parameter_string, &kornia_img, &cv_mat);
    }

    group.finish();
}

fn bench_hollow_square(c: &mut Criterion) {
    let mut group = c.benchmark_group("contours_hollow_square");

    for (width, height) in [(128, 128), (256, 256), (512, 512), (1024, 1024)].iter() {
        group.throughput(Throughput::Elements((*width * *height) as u64));
        let parameter_string = format!("{width}x{height}");

        let data = make_hollow_square(*width, *height);
        let kornia_img = kornia_image(*width, *height, data.clone());
        let cv_mat = opencv_mat(*width, *height, &data);

        register_variants(&mut group, &parameter_string, &kornia_img, &cv_mat);
    }

    group.finish();
}

fn bench_sparse_noise(c: &mut Criterion) {
    let mut group = c.benchmark_group("contours_sparse_noise");

    for (width, height) in [(128, 128), (256, 256), (512, 512), (1024, 1024)].iter() {
        group.throughput(Throughput::Elements((*width * *height) as u64));
        let parameter_string = format!("{width}x{height}");

        let data = make_noise(*width, *height, 0xDEAD_BEEF_u64);
        let kornia_img = kornia_image(*width, *height, data.clone());
        let cv_mat = opencv_mat(*width, *height, &data);

        register_variants(&mut group, &parameter_string, &kornia_img, &cv_mat);
    }

    group.finish();
}

fn bench_real_data(c: &mut Criterion) {
    let mut group = c.benchmark_group("contours_real_data");

    for img_name in ["dog.jpeg", "apriltags_tag36h11.jpg"].iter() {
        let img_path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../../tests/data/{img_name}"));
        let img_rgb8 = kornia_io::functional::read_image_any_rgb8(&img_path)
            .unwrap_or_else(|_| panic!("Failed to load {img_name}"));

        let new_size = img_rgb8.size();
        let width = new_size.width;
        let height = new_size.height;

        let mut img_gray8 = Image::from_size_val(new_size, 0u8, CpuAllocator).unwrap();
        kornia_imgproc::color::gray_from_rgb_u8(&img_rgb8, &mut img_gray8).unwrap();
        
        let mut img_bin8 = Image::from_size_val(new_size, 0u8, CpuAllocator).unwrap();
        kornia_imgproc::threshold::threshold_binary(&img_gray8, &mut img_bin8, 127, 255).unwrap();

        group.throughput(Throughput::Elements((width * height) as u64));
        let parameter_string = format!("{img_name}_{width}x{height}");

        let cv_mat = opencv_mat(width, height, img_bin8.as_slice());

        register_variants(&mut group, &parameter_string, &img_bin8, &cv_mat);
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_filled_square,
    bench_hollow_square,
    bench_sparse_noise,
    bench_real_data
);
criterion_main!(benches);
