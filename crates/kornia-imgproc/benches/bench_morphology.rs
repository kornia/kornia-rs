use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

use kornia_image::Image;
use kornia_imgproc::morphology::{close, dilate, erode, open, Kernel, KernelShape};
use kornia_imgproc::padding::PaddingMode;
use kornia_tensor::CpuAllocator;

fn bench_morphology(c: &mut Criterion) {
    let mut group = c.benchmark_group("Morphology");

    for (width, height) in [(256, 224), (512, 448), (1024, 896)].iter() {
        for kernel_size in [3, 5, 7].iter() {
            let parameter_string = format!("{width}x{height}_k{kernel_size}");
            let image_size = [*width, *height].into();

            let image = Image::<u8, 1, _>::from_size_val(image_size, 128, CpuAllocator).unwrap();
            let dst = Image::<u8, 1, _>::from_size_val(image_size, 0, CpuAllocator).unwrap();

            let kernel = Kernel::new(KernelShape::Box { size: *kernel_size });

            group.bench_with_input(
                BenchmarkId::new("dilate", &parameter_string),
                &(&image, &dst),
                |b, i| {
                    let (src, mut out) = (i.0, i.1.clone());
                    b.iter(|| {
                        std::hint::black_box(dilate(
                            src,
                            &mut out,
                            &kernel,
                            PaddingMode::Reflect101,
                            [0],
                        ))
                    })
                },
            );

            group.bench_with_input(
                BenchmarkId::new("erode", &parameter_string),
                &(&image, &dst),
                |b, i| {
                    let (src, mut out) = (i.0, i.1.clone());
                    b.iter(|| {
                        std::hint::black_box(erode(
                            src,
                            &mut out,
                            &kernel,
                            PaddingMode::Reflect101,
                            [0],
                        ))
                    })
                },
            );

            group.bench_with_input(
                BenchmarkId::new("open", &parameter_string),
                &(&image, &dst),
                |b, i| {
                    let (src, mut out) = (i.0, i.1.clone());
                    b.iter(|| {
                        std::hint::black_box(open(
                            src,
                            &mut out,
                            &kernel,
                            PaddingMode::Reflect101,
                            [0],
                        ))
                    })
                },
            );

            group.bench_with_input(
                BenchmarkId::new("close", &parameter_string),
                &(&image, &dst),
                |b, i| {
                    let (src, mut out) = (i.0, i.1.clone());
                    b.iter(|| {
                        std::hint::black_box(close(
                            src,
                            &mut out,
                            &kernel,
                            PaddingMode::Reflect101,
                            [0],
                        ))
                    })
                },
            );
        }
    }

    group.finish();
}

criterion_group!(benches, bench_morphology);
criterion_main!(benches);
