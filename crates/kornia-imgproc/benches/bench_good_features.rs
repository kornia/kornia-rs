//! `features::good_features_to_track` and, with `opencv_bench`,
//! OpenCV's `goodFeaturesToTrack` with a border mask. One refill of one
//! 640×480 camera: 48 corners, 23 px apart, a 9 px border,
//! at half and full resolution, into an empty image and around 40 live tracks.
//!
//! `RAYON_NUM_THREADS=1 taskset -c 8 cargo bench -j2 -p kornia-imgproc
//! --features opencv_bench --bench bench_good_features`

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};
use kornia_image::{Image, ImageSize};
use kornia_imgproc::features::{good_features_to_track, GoodFeaturesParams, GoodFeaturesScratch};

const WIDTH: usize = 640;
const HEIGHT: usize = 480;
const CORNERS: usize = 48;
const QUALITY: f32 = 0.01;
const SPACING: f32 = 23.0;
const BORDER: usize = 9;

/// Blocks of hashed brightness with a smooth ripple: corners everywhere.
fn scene() -> Image<u8, 1> {
    let data = (0..WIDTH * HEIGHT)
        .map(|i| {
            let (x, y) = (i % WIDTH, i / WIDTH);
            let block = ((x / 12) * 7_919 + (y / 12) * 104_729) % 251;
            let ripple = 30.0 * (0.21 * x as f32).sin() * (0.17 * y as f32).cos();
            (block as f32 * 0.8 + 40.0 + ripple).clamp(0.0, 255.0) as u8
        })
        .collect();
    Image::new(
        ImageSize {
            width: WIDTH,
            height: HEIGHT,
        },
        data,
    )
    .unwrap()
}

/// 40 live tracks on an 8×5 grid.
fn occupied() -> Vec<[f32; 2]> {
    (0..5)
        .flat_map(|j| (0..8).map(move |i| [40.0 + i as f32 * 80.0, 50.0 + j as f32 * 95.0]))
        .collect()
}

fn bench_good_features(c: &mut Criterion) {
    let img = scene();
    let live = occupied();
    let mut group = c.benchmark_group("good_features");
    for (name, half_resolution, occupied) in [
        ("half_empty_640x480", true, &[][..]),
        ("half_around_40_tracks_640x480", true, &live[..]),
        ("full_empty_640x480", false, &[][..]),
    ] {
        let params = GoodFeaturesParams {
            max_corners: CORNERS,
            quality: QUALITY,
            min_spacing_px: SPACING,
            border_px: BORDER,
            half_resolution,
        };
        let mut scratch = GoodFeaturesScratch::default();
        let mut out = Vec::with_capacity(CORNERS);
        group.bench_function(name, |b| {
            b.iter(|| {
                out.clear();
                good_features_to_track(&img, black_box(occupied), &params, &mut scratch, &mut out);
                black_box(&out);
            });
        });
    }

    #[cfg(feature = "opencv_bench")]
    opencv_arm::bench(&mut group, &img);

    group.finish();
}

#[cfg(feature = "opencv_bench")]
mod opencv_arm {
    use super::*;
    use criterion::measurement::WallTime;
    use criterion::BenchmarkGroup;
    use opencv::{core, imgproc, prelude::*};

    fn mat(img: &Image<u8, 1>) -> opencv::Result<core::Mat> {
        let mut mat =
            core::Mat::zeros(img.height() as i32, img.width() as i32, core::CV_8UC1)?.to_mat()?;
        for (i, &v) in img.as_slice().iter().enumerate() {
            *mat.at_2d_mut::<u8>((i / img.width()) as i32, (i % img.width()) as i32)? = v;
        }
        Ok(mat)
    }

    /// 255 inside a `border` band, as the fixture script's mask.
    fn mask(width: usize, height: usize, border: usize) -> opencv::Result<core::Mat> {
        let mut mask = core::Mat::zeros(height as i32, width as i32, core::CV_8UC1)?.to_mat()?;
        for y in border..height - border {
            for x in border..width - border {
                *mask.at_2d_mut::<u8>(y as i32, x as i32)? = 255;
            }
        }
        Ok(mask)
    }

    fn bench_one(
        group: &mut BenchmarkGroup<'_, WallTime>,
        name: &str,
        img: &core::Mat,
        mask: &core::Mat,
        spacing: f32,
    ) {
        let mut corners = core::Vector::<core::Point2f>::new();
        group.bench_function(name, |b| {
            b.iter(|| {
                imgproc::good_features_to_track(
                    img,
                    &mut corners,
                    CORNERS as i32,
                    f64::from(QUALITY),
                    f64::from(spacing),
                    mask,
                    3,
                    false,
                    0.04,
                )
                .unwrap();
                black_box(&corners);
            });
        });
    }

    pub(super) fn bench(group: &mut BenchmarkGroup<'_, WallTime>, img: &Image<u8, 1>) {
        let full = mat(img).unwrap();
        bench_one(
            group,
            "opencv_full_masked_640x480",
            &full,
            &mask(WIDTH, HEIGHT, BORDER).unwrap(),
            SPACING,
        );
        // OpenCV on an image halved outside the timing; ours halves inside it.
        let mut half = core::Mat::default();
        imgproc::resize(
            &full,
            &mut half,
            core::Size::new(0, 0),
            0.5,
            0.5,
            imgproc::INTER_AREA,
        )
        .unwrap();
        let half_mask = mask(WIDTH / 2, HEIGHT / 2, BORDER / 2).unwrap();
        bench_one(
            group,
            "opencv_half_masked_320x240",
            &half,
            &half_mask,
            SPACING / 2.0,
        );
    }
}

criterion_group!(benches, bench_good_features);
criterion_main!(benches);
