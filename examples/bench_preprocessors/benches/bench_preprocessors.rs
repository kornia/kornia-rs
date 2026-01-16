use anyhow::Result;
use candle_core::{Device, Tensor};
use criterion::{criterion_group, criterion_main, Criterion};
use kornia_image::{allocator::CpuAllocator, Image, ImageSize};
use kornia_imgproc::{interpolation::InterpolationMode, resize::resize_fast_rgb};
use kornia_io::jpeg::read_image_jpeg_rgb8;
use kornia_vlm::preprocessor::{Model, PreprocessConfig, Preprocessor};
use kornia_vlm::smolvlm::preprocessor::SmolVlmImagePreprocessor;
use std::path::Path;

// Minimal normalization to match PaliGemma pipeline cost
fn image_to_normalized_tensor(img: &Image<u8, 3, CpuAllocator>, device: &Device) -> Result<Tensor> {
    let tensor = Tensor::from_slice(img.as_slice(), (img.height(), img.width(), 3), device)?
        .permute((2, 0, 1))? // (H, W, C) -> (C, H, W)
        .to_dtype(candle_core::DType::F32)? // Cast to f32
        .affine(2. / 255., -1.)?; // Normalize 0..255 -> -1..1
    Ok(tensor)
}

fn load_test_image() -> Image<u8, 3, CpuAllocator> {
    let paths = [
        "tests/data/apriltags_tag36h11.jpg",
        "../../tests/data/apriltags_tag36h11.jpg",
    ];

    for p in paths {
        let path = Path::new(p);
        if path.exists() {
            println!("Loading image from {:?}", path);
            return read_image_jpeg_rgb8(path)
                .expect("Failed to read image")
                .into_inner();
        }
    }
    panic!("Image not found in search paths: {:?}", paths);
}

fn bench_preprocessors(c: &mut Criterion) {
    let image = load_test_image();
    let device = Device::Cpu;

    // --- 1. Generic SmolVLM ---
    let config_smol = PreprocessConfig {
        model_type: Model::SmolVLM,
        target_size: 384,
        mean: [0.5, 0.5, 0.5],
        std: [0.5, 0.5, 0.5],
        interpolation: InterpolationMode::Lanczos,
    };
    let preprocessor_smol = Preprocessor::new(config_smol.clone());

    c.bench_function("Generic SmolVLM (Fused Bicubic + Tiling)", |b| {
        b.iter(|| preprocessor_smol.process(&image, &device).unwrap())
    });

    // --- 2. Specific SmolVLM ---
    let mut smol_pre = SmolVlmImagePreprocessor::new(2048, 384, &device).unwrap();

    c.bench_function("Specific SmolVLM (Resize + Pad + Norm)", |b| {
        b.iter(|| smol_pre.preprocess(&image, &device, CpuAllocator).unwrap())
    });

    // --- 3. Generic PaliGemma ---
    let config_pali = PreprocessConfig {
        model_type: Model::PaliGemma,
        target_size: 224,
        mean: [0.5, 0.5, 0.5],
        std: [0.5, 0.5, 0.5],
        interpolation: InterpolationMode::Bicubic,
    };
    let preprocessor_pali = Preprocessor::new(config_pali.clone());

    c.bench_function("Generic PaliGemma (Fused Bicubic)", |b| {
        b.iter(|| preprocessor_pali.process(&image, &device).unwrap())
    });

    // --- 4. Specific PaliGemma (Simulated) ---
    let mut pali_specific_out = Image::from_size_val(
        ImageSize {
            width: 224,
            height: 224,
        },
        0u8,
        CpuAllocator,
    )
    .unwrap();

    c.bench_function("Specific PaliGemma (Resize Bilinear + Norm)", |b| {
        b.iter(|| {
            resize_fast_rgb(&image, &mut pali_specific_out, InterpolationMode::Bilinear).unwrap();
            let _t = image_to_normalized_tensor(&pali_specific_out, &device).unwrap();
        })
    });

    // --- 5. Generic Clip ---
    // Default config is Clip (224x224, Center Crop)
    let config_clip = PreprocessConfig::default();
    let preprocessor_clip = Preprocessor::new(config_clip);

    c.bench_function("Generic Clip (Center Crop + Resize + Norm)", |b| {
        b.iter(|| preprocessor_clip.process(&image, &device).unwrap())
    });

    // --- 6. Native Clip (Simulated) ---
    // Clip native pipeline: Center Crop -> Resize -> To f32 -> Normalize
    let (w, h) = (image.width(), image.height());
    let crop_size = w.min(h);
    let x = (w - crop_size) / 2;
    let y = (h - crop_size) / 2;

    use kornia_vlm::preprocessor::{CLIP_MEAN, CLIP_STD};

    c.bench_function("Native Clip (Crop + Resize Bicubic + Norm)", |b| {
        b.iter(|| {
            // Allocate on every call
            let mut clip_cropped = Image::<u8, 3, _>::from_size_val(
                ImageSize {
                    width: crop_size,
                    height: crop_size,
                },
                0u8,
                CpuAllocator,
            )
            .unwrap();

            let mut clip_resized = Image::<u8, 3, _>::from_size_val(
                ImageSize {
                    width: 224,
                    height: 224,
                },
                0u8,
                CpuAllocator,
            )
            .unwrap();

            let mut clip_f32 = Image::<f32, 3, _>::from_size_val(
                ImageSize {
                    width: 224,
                    height: 224,
                },
                0.0,
                CpuAllocator,
            )
            .unwrap();

            let mut clip_normalized = Image::<f32, 3, _>::from_size_val(
                ImageSize {
                    width: 224,
                    height: 224,
                },
                0.0,
                CpuAllocator,
            )
            .unwrap();

            // 1. Center Crop
            kornia_imgproc::crop::crop_image(&image, &mut clip_cropped, x, y).unwrap();

            // 2. Resize
            resize_fast_rgb(&clip_cropped, &mut clip_resized, InterpolationMode::Bicubic).unwrap();

            // 3. To f32 (scale to 0-1)
            let inv_255 = 1.0 / 255.0;
            clip_f32
                .as_slice_mut()
                .iter_mut()
                .zip(clip_resized.as_slice().iter())
                .for_each(|(dst, &src)| {
                    *dst = src as f32 * inv_255;
                });

            // 4. Normalize
            kornia_imgproc::normalize::normalize_mean_std(
                &clip_f32,
                &mut clip_normalized,
                &CLIP_MEAN,
                &CLIP_STD,
            )
            .unwrap();
        })
    });
}

criterion_group!(benches, bench_preprocessors);
criterion_main!(benches);
