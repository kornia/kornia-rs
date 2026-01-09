use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use candle_core::{Device, DType};
use kornia_image::{Image, ImageSize};
use kornia_tensor::CpuAllocator;
use kornia_vlm::video::VideoSample;

fn bench_video_to_tensor(c: &mut Criterion) {
    let mut group = c.benchmark_group("VideoToTensor");

    // Test different video configurations
    let configs = vec![
        (8, 224, 224, "8_frames_224x224"),
        (16, 224, 224, "16_frames_224x224"),
        (32, 224, 224, "32_frames_224x224"),
        (8, 640, 480, "8_frames_640x480"),
        (32, 640, 480, "32_frames_640x480"),
    ];

    for (num_frames, width, height, name) in configs {
        let size = ImageSize { width, height };
        
        // Calculate throughput (pixels processed)
        let total_pixels = (num_frames * width * height * 3) as u64;
        group.throughput(Throughput::Elements(total_pixels));

        // Create test video
        let mut video = VideoSample::<32, CpuAllocator>::new();
        for i in 0..num_frames {
            let frame = Image::from_size_val(size, (i % 256) as u8, CpuAllocator).unwrap();
            video.add_frame(frame, i as u32 * 33); // 33ms per frame (~30 FPS)
        }

        let device = Device::Cpu;

        // Benchmark F32 conversion
        group.bench_with_input(
            BenchmarkId::new("f32", name),
            &(&video, &device),
            |b, (vid, dev)| {
                b.iter(|| {
                    black_box(vid.into_tensor(black_box(DType::F32), black_box(dev)).unwrap())
                })
            },
        );

        // Benchmark U8 conversion
        group.bench_with_input(
            BenchmarkId::new("u8", name),
            &(&video, &device),
            |b, (vid, dev)| {
                b.iter(|| {
                    black_box(vid.into_tensor(black_box(DType::U8), black_box(dev)).unwrap())
                })
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_video_to_tensor);
criterion_main!(benches);
