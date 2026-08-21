use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

use kornia_image::{Image, ImageError};
use kornia_imgproc::morphology::{close, dilate, erode, open, Kernel, KernelShape};
use kornia_imgproc::padding::PaddingMode;

type MorphOp =
    fn(&Image<u8, 1>, &mut Image<u8, 1>, &Kernel, PaddingMode, [u8; 1]) -> Result<(), ImageError>;

const OPS: [(&str, MorphOp); 4] = [
    ("dilate", dilate::<u8, 1>),
    ("erode", erode::<u8, 1>),
    ("open", open::<u8, 1>),
    ("close", close::<u8, 1>),
];

/// Deterministic pseudo-random bytes.
///
/// A constant-valued image is close to the best case for any running-min/max
/// (van Herk / Gil-Werman) implementation, so benchmarking morphology on one
/// would report speedups that do not hold on real images. The current
/// `Ord`-based engine is content-independent, but the benchmark should not be
/// the thing that hides that when the engine changes.
fn pseudo_random(len: usize, seed: u32) -> Vec<u8> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 24) as u8
        })
        .collect()
}

fn bench_morphology(c: &mut Criterion) {
    let mut group = c.benchmark_group("Morphology");

    for (width, height) in [(256, 224), (512, 448), (1024, 896), (1920, 1080)].iter() {
        group.throughput(criterion::Throughput::Elements((*width * *height) as u64));

        let image_size = [*width, *height].into();
        let image =
            Image::<u8, 1>::new(image_size, pseudo_random(width * height, 0x1234_5678)).unwrap();

        for kernel_size in [3, 5, 7].iter() {
            let parameter_string = format!("{width}x{height}_k{kernel_size}");
            let kernel = Kernel::new(KernelShape::Box { size: *kernel_size });

            for (name, op) in OPS.iter() {
                group.bench_with_input(
                    BenchmarkId::new(*name, &parameter_string),
                    &image,
                    |b, src| {
                        let mut out = Image::<u8, 1>::from_size_val(image_size, 0).unwrap();
                        b.iter(|| {
                            std::hint::black_box(op(
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
    }

    group.finish();
}

criterion_group!(benches, bench_morphology);
criterion_main!(benches);
