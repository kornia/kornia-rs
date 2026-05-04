//! NEON baseline vs cubecl-{cuda,cpu} × {kernel-only, end-to-end}.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use kornia_cubecl::resize::resize_bilinear_u8_rgb;
use kornia_cubecl::runtime;
use kornia_image::{Image, ImageSize};
use kornia_imgproc::{interpolation::InterpolationMode, resize};
use kornia_tensor::CpuAllocator;
use rand::{rngs::StdRng, RngCore, SeedableRng};

const SIZES: &[(usize, usize)] = &[(512, 256), (1024, 512), (2048, 1024), (4096, 2048)];

fn make_image(w: usize, h: usize) -> Image<u8, 3, CpuAllocator> {
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    let mut buf = vec![0u8; w * h * 3];
    rng.fill_bytes(&mut buf);
    Image::new(ImageSize { width: w, height: h }, buf, CpuAllocator).unwrap()
}

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("resize_u8_rgb_2x_downscale");

    #[cfg(feature = "cpu")]
    let cpu_client = runtime::init_cpu();
    #[cfg(feature = "cuda")]
    let cuda_client = runtime::init_cuda().ok();

    for &(src_w, src_h) in SIZES {
        let (dst_w, dst_h) = (src_w / 2, src_h / 2);
        group.throughput(Throughput::Elements((dst_w * dst_h) as u64));
        let id = format!("{src_w}x{src_h}");
        let src = make_image(src_w, src_h);
        let mut dst_neon = Image::<u8, 3, _>::from_size_val(
            ImageSize { width: dst_w, height: dst_h }, 0, CpuAllocator,
        ).unwrap();

        // --- NEON baseline (fast_image_resize via kornia-imgproc) ---
        group.bench_function(BenchmarkId::new("neon", &id), |b| {
            b.iter(|| {
                resize::resize_fast_rgb(
                    std::hint::black_box(&src),
                    std::hint::black_box(&mut dst_neon),
                    InterpolationMode::Bilinear,
                ).unwrap();
            });
        });

        // --- cubecl-cpu arms (skipped if feature off) ---
        #[cfg(feature = "cpu")]
        {
            let src_h_cpu = cpu_client.create_from_slice(src.as_slice());
            let dst_h_cpu = cpu_client.empty(dst_w * dst_h * 3);
            group.bench_function(BenchmarkId::new("cubecl_cpu_kernel", &id), |b| {
                b.iter(|| {
                    resize_bilinear_u8_rgb::<runtime::CpuRuntime>(
                        &cpu_client, &src_h_cpu,
                        ImageSize { width: src_w, height: src_h },
                        &dst_h_cpu,
                        ImageSize { width: dst_w, height: dst_h },
                    ).unwrap();
                    let _ = cubecl::future::block_on(cpu_client.sync());
                });
            });
            group.bench_function(BenchmarkId::new("cubecl_cpu_e2e", &id), |b| {
                b.iter(|| {
                    let s = cpu_client.create_from_slice(src.as_slice());
                    let d = cpu_client.empty(dst_w * dst_h * 3);
                    resize_bilinear_u8_rgb::<runtime::CpuRuntime>(
                        &cpu_client, &s,
                        ImageSize { width: src_w, height: src_h },
                        &d,
                        ImageSize { width: dst_w, height: dst_h },
                    ).unwrap();
                    let _ = cpu_client.read_one(d).unwrap();
                });
            });
        }

        // --- cubecl-cuda arms ---
        #[cfg(feature = "cuda")]
        if let Some(ref cuda) = cuda_client {
            let src_h_cu = cuda.create_from_slice(src.as_slice());
            let dst_h_cu = cuda.empty(dst_w * dst_h * 3);
            group.bench_function(BenchmarkId::new("cubecl_cuda_kernel", &id), |b| {
                b.iter(|| {
                    resize_bilinear_u8_rgb::<runtime::CudaRuntime>(
                        cuda, &src_h_cu,
                        ImageSize { width: src_w, height: src_h },
                        &dst_h_cu,
                        ImageSize { width: dst_w, height: dst_h },
                    ).unwrap();
                    let _ = cubecl::future::block_on(cuda.sync());
                });
            });
            group.bench_function(BenchmarkId::new("cubecl_cuda_e2e", &id), |b| {
                b.iter(|| {
                    let s = cuda.create_from_slice(src.as_slice());
                    let d = cuda.empty(dst_w * dst_h * 3);
                    resize_bilinear_u8_rgb::<runtime::CudaRuntime>(
                        cuda, &s,
                        ImageSize { width: src_w, height: src_h },
                        &d,
                        ImageSize { width: dst_w, height: dst_h },
                    ).unwrap();
                    let _ = cuda.read_one(d).unwrap();
                });
            });
        } else {
            eprintln!("skipping cuda arms for {id}: no device");
        }
    }

    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
