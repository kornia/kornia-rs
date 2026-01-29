use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use kornia_image::Image;
use kornia_imgproc::draw::{draw_line, draw_polygon};
use kornia_tensor::CpuAllocator;

fn bench_draw(c: &mut Criterion) {
    let sizes = [(512, 512), (1024, 1024)];
    let thicknesses = [1, 3, 5];
    let color = [255u8];

    //line benchmarks
    {
        let mut group = c.benchmark_group("Draw Line");

        for (width, height) in sizes.iter() {
            let image_size = [*width, *height].into();

            let p0_h = (0, (*height as i64) / 2);
            let p1_h = ((*width as i64) - 1, (*height as i64) / 2);
            let p0_d = (0, 0);
            let p1_d = ((*width as i64) - 1, (*height as i64) - 1);

            for thickness in thicknesses.iter() {
                let parameter_string = format!("{}x{}_t{}", width, height, thickness);

                // Horizontal Line bench
                group.bench_with_input(
                    BenchmarkId::new("Horizontal", &parameter_string),
                    &thickness,
                    |b, &t| {
                        let mut img =
                            Image::<u8, 1, _>::from_size_val(image_size, 0, CpuAllocator).unwrap();
                        b.iter(|| {
                            draw_line(&mut img, p0_h, p1_h, color, *t);
                        })
                    },
                );

                // Diagonal Line bench
                group.bench_with_input(
                    BenchmarkId::new("Diagonal", &parameter_string),
                    &thickness,
                    |b, &t| {
                        let mut img =
                            Image::<u8, 1, _>::from_size_val(image_size, 0, CpuAllocator).unwrap();
                        b.iter(|| {
                            draw_line(&mut img, p0_d, p1_d, color, *t);
                        })
                    },
                );
            }
        }
        group.finish();
    }

    //polygon benchmarks
    {
        let mut group = c.benchmark_group("Draw Polygon");

        for (width, height) in sizes.iter() {
            let image_size = [*width, *height].into();

            let margin = 50;
            let bbox_points = [
                (margin, margin),
                ((*width as i64) - margin, margin),
                ((*width as i64) - margin, (*height as i64) - margin),
                (margin, (*height as i64) - margin),
            ];

            let center_x = (*width as f64) / 2.0;
            let center_y = (*height as f64) / 2.0;
            let radius = ((*height as f64) / 2.0) * 0.8;
            let circle_points: Vec<(i64, i64)> = (0..64)
                .map(|i| {
                    let angle = (i as f64) * 2.0 * std::f64::consts::PI / 64.0;
                    (
                        (center_x + radius * angle.cos()) as i64,
                        (center_y + radius * angle.sin()) as i64,
                    )
                })
                .collect();

            for thickness in thicknesses.iter() {
                let parameter_string = format!("{}x{}_t{}", width, height, thickness);

                // Bounding Box bench
                group.bench_with_input(
                    BenchmarkId::new("BBox", &parameter_string),
                    &thickness,
                    |b, &t| {
                        let mut img =
                            Image::<u8, 1, _>::from_size_val(image_size, 0, CpuAllocator).unwrap();
                        b.iter(|| {
                            draw_polygon(&mut img, &bbox_points, color, *t);
                        })
                    },
                );

                // Circle Approximation bench
                group.bench_with_input(
                    BenchmarkId::new("Circle", &parameter_string),
                    &thickness,
                    |b, &t| {
                        let mut img =
                            Image::<u8, 1, _>::from_size_val(image_size, 0, CpuAllocator).unwrap();
                        b.iter(|| {
                            draw_polygon(&mut img, &circle_points, color, *t);
                        })
                    },
                );
            }
        }
        group.finish();
    }
}

criterion_group!(benches, bench_draw);
criterion_main!(benches);
