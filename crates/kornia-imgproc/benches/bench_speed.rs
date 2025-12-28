use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;
use rayon::prelude::*;

// =======================
// Kornia imports
// =======================
use kornia_image::{Image, ImageSize};
use kornia_image::allocator::{ImageAllocator, CpuAllocator};

// =======================================================================
// PART 1: Fused CLIP Preprocess (resize + crop + normalize)
// =======================================================================

pub struct ClipPreprocessConfig {
    pub target_size: usize,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl Default for ClipPreprocessConfig {
    fn default() -> Self {
        Self {
            target_size: 224,
            mean: [0.48145466, 0.4578275, 0.40821073],
            std: [0.26862954, 0.26130258, 0.27577711],
        }
    }
}

const ROWS_PER_TASK: usize = 8; // Process 8 rows per thread-task

#[derive(Copy, Clone)]
#[repr(C)]
struct XMapOpt {
    off0: u32,
    off1: u32,
    wx0: f32,
    wx1: f32,
}

#[derive(Copy, Clone)]
struct YMapOpt {
    y0_offset: usize,
    y1_offset: usize,
    wy0: f32,
    wy1: f32,
}

pub fn clip_preprocess_fused<A: ImageAllocator>(
    image: &Image<u8, 3, A>,
    config: &ClipPreprocessConfig,
) -> Image<f32, 3, CpuAllocator> {
    let out_size = config.target_size;
    let w = image.width();
    let h = image.height();
    let src = image.as_slice();

    // 1. Output Setup
    let plane_size = out_size * out_size;
    let mut data: Vec<f32> = Vec::with_capacity(plane_size * 3);
    unsafe { data.set_len(plane_size * 3) };

    let (r_plane, rest) = data.split_at_mut(plane_size);
    let (g_plane, b_plane) = rest.split_at_mut(plane_size);

    // 2. Geometry Setup
    let scale = if w < h { w as f32 / out_size as f32 } else { h as f32 / out_size as f32 };
    let start_x = (w as f32 - out_size as f32 * scale) * 0.5;
    let start_y = (h as f32 - out_size as f32 * scale) * 0.5;

    // 3. X-Map Precomputation
    let mut x_map: Vec<XMapOpt> = Vec::with_capacity(out_size);
    let safe_limit_x = if scale > 0.0 { (((w as f32 - 1.0 - start_x) / scale).floor() as usize).min(out_size) } else { 0 };

    let mut sx = start_x;
    for _ in 0..safe_limit_x {
        let x0 = sx as usize;
        let dx = sx - x0 as f32;
        x_map.push(XMapOpt { off0: (x0 * 3) as u32, off1: (x0 * 3 + 3) as u32, wx0: 1.0 - dx, wx1: dx });
        sx += scale;
    }
    for _ in safe_limit_x..out_size {
        let x0 = sx as usize;
        let x1 = (x0 + 1).min(w - 1);
        let dx = sx - x0 as f32;
        x_map.push(XMapOpt { off0: (x0 * 3) as u32, off1: (x1 * 3) as u32, wx0: 1.0 - dx, wx1: dx });
        sx += scale;
    }

    // 4. Y-Map Precomputation
    let stride = w * 3;
    let y_map: Vec<YMapOpt> = (0..out_size).map(|y| {
        let sy = start_y + y as f32 * scale;
        let y0 = sy as usize;
        let y1 = (y0 + 1).min(h - 1);
        let dy = sy - y0 as f32;
        YMapOpt { y0_offset: y0 * stride, y1_offset: y1 * stride, wy0: 1.0 - dy, wy1: dy }
    }).collect();

    // 5. LUT Generation
    let inv255 = 1.0 / 255.0;
    let scale_r = inv255 / config.std[0];
    let bias_r = -(config.mean[0] / config.std[0]);
    let scale_g = inv255 / config.std[1];
    let bias_g = -(config.mean[1] / config.std[1]);
    let scale_b = inv255 / config.std[2];
    let bias_b = -(config.mean[2] / config.std[2]);

    let mut lut_r = [0f32; 256];
    let mut lut_g = [0f32; 256];
    let mut lut_b = [0f32; 256];
    for i in 0..256 {
        let v = i as f32;
        lut_r[i] = v * scale_r;
        lut_g[i] = v * scale_g;
        lut_b[i] = v * scale_b;
    }

    // 6. Blocked Parallel Execution
    // We chunk the output planes into blocks of (out_size * ROWS_PER_TASK) floats
    r_plane.par_chunks_mut(out_size * ROWS_PER_TASK)
        .zip(g_plane.par_chunks_mut(out_size * ROWS_PER_TASK))
        .zip(b_plane.par_chunks_mut(out_size * ROWS_PER_TASK))
        .zip(y_map.par_chunks(ROWS_PER_TASK)) // We also chunk the Y-map
        .for_each(|(((r_block, g_block), b_block), y_block)| unsafe {

            let src_ptr = src.as_ptr();
            let x_map_ptr = x_map.as_ptr();

            // Inner Loop: Iterate through the rows in this specific block
            // (usually 0..8, but fewer for the very last block)
            for row_idx in 0..y_block.len() {
                let y_data = y_block.get_unchecked(row_idx);

                // Row base pointers from source
                let row0_base = src_ptr.add(y_data.y0_offset);
                let row1_base = src_ptr.add(y_data.y1_offset);

                // Destination pointers for this specific row
                // We add (row_idx * out_size) to the start of the *block* pointer
                let r_row_ptr = r_block.as_mut_ptr().add(row_idx * out_size);
                let g_row_ptr = g_block.as_mut_ptr().add(row_idx * out_size);
                let b_row_ptr = b_block.as_mut_ptr().add(row_idx * out_size);

                let wy0 = y_data.wy0;
                let wy1 = y_data.wy1;

                // Horizontal Loop (Same as before)
                for x in 0..out_size {
                    let xm = *x_map_ptr.add(x);

                    let w0 = wy0 * xm.wx0;
                    let w1 = wy0 * xm.wx1;
                    let w2 = wy1 * xm.wx0;
                    let w3 = wy1 * xm.wx1;

                    let p00 = row0_base.add(xm.off0 as usize);
                    let p01 = row0_base.add(xm.off1 as usize);
                    let p10 = row1_base.add(xm.off0 as usize);
                    let p11 = row1_base.add(xm.off1 as usize);

                    let r = lut_r[*p00 as usize] * w0 + lut_r[*p01 as usize] * w1 + lut_r[*p10 as usize] * w2 + lut_r[*p11 as usize] * w3 + bias_r;
                    let g = lut_g[*p00.add(1) as usize] * w0 + lut_g[*p01.add(1) as usize] * w1 + lut_g[*p10.add(1) as usize] * w2 + lut_g[*p11.add(1) as usize] * w3 + bias_g;
                    let b = lut_b[*p00.add(2) as usize] * w0 + lut_b[*p01.add(2) as usize] * w1 + lut_b[*p10.add(2) as usize] * w2 + lut_b[*p11.add(2) as usize] * w3 + bias_b;

                    r_row_ptr.add(x).write(r);
                    g_row_ptr.add(x).write(g);
                    b_row_ptr.add(x).write(b);
                }
            }
        });

    Image::new(
        ImageSize { width: out_size, height: out_size },
        data,
        CpuAllocator,
    ).expect("Failed")
}

// =======================================================================
// PART 2: Standard Kornia Pipeline
// =======================================================================

use kornia_imgproc::{
    crop,
    interpolation::InterpolationMode,
    normalize,
    resize,
};

fn kornia_pipeline_op(
    image: &Image<u8, 3, CpuAllocator>,
    config: &ClipPreprocessConfig,
) -> Image<f32, 3, CpuAllocator> {
    
    let src_w = image.width();
    let src_h = image.height();
    let target = config.target_size;

    let scale = if src_w < src_h {
        target as f32 / src_w as f32
    } else {
        target as f32 / src_h as f32
    };

    let new_w = (src_w as f32 * scale).ceil() as usize;
    let new_h = (src_h as f32 * scale).ceil() as usize;

    // 2. Cast to f32
    let image_f32 = image
        .clone() 
        .cast_and_scale::<f32>(1.0 / 255.0)
        .unwrap();

    // 3. Resize
    let mut resized = Image::<f32, 3, _>::from_size_val(
        ImageSize { width: new_w, height: new_h },
        0.0f32,
        CpuAllocator
    ).unwrap();

    resize::resize_native(
        &image_f32,
        &mut resized,
        InterpolationMode::Bilinear,
    ).unwrap();

    // 4. Center Crop
    let crop_x = (new_w.saturating_sub(target)) / 2;
    let crop_y = (new_h.saturating_sub(target)) / 2;
    
    let mut cropped = Image::<f32, 3, _>::from_size_val(
        ImageSize { width: target, height: target },
        0.0f32,
        CpuAllocator
    ).unwrap();

    crop::crop_image(
        &resized,
        &mut cropped,
        crop_x, 
        crop_y,
    ).unwrap();

    // 5. Normalize
    let mut normalized = Image::<f32, 3, _>::from_size_val(
        cropped.size(),
        0.0f32,
        CpuAllocator
    ).unwrap();

    normalize::normalize_mean_std(
        &cropped,
        &mut normalized,
        &config.mean,
        &config.std,
    )
    .unwrap();

    // 6. Permute HWC -> CHW
    let num_pixels = target * target;
    let mut chw_data = Vec::with_capacity(num_pixels * 3);
    
    // Safe because we fill every index below
    unsafe { chw_data.set_len(num_pixels * 3) };

    let src_data = normalized.as_slice();

    // Iterate over pixels and split channels to different planes
    for i in 0..num_pixels {
        let r = src_data[i * 3];
        let g = src_data[i * 3 + 1];
        let b = src_data[i * 3 + 2];

        // R plane is at offset 0
        chw_data[i] = r;
        // G plane is at offset num_pixels
        chw_data[i + num_pixels] = g;
        // B plane is at offset 2 * num_pixels
        chw_data[i + 2 * num_pixels] = b;
    }

    Image::new(
        ImageSize {
            width: target,
            height: target,
        },
        chw_data,
        CpuAllocator,
    )
    .expect("Failed to create image from buffer")

}

// =======================================================================
// PART 3: Benchmark Harness
// =======================================================================

fn benchmark_preprocessing(c: &mut Criterion) {
    // 1. Load Image
    let source = kornia_io::functional::read_image_any_rgb8("../../sample.jpg")
        .unwrap()
        .into_inner();

    let config = ClipPreprocessConfig::default();
    let mut group = c.benchmark_group(format!(
        "Preprocessing_{}x{}",
        source.width(),
        source.height()
    ));

    group.bench_function("Fused Kernel", |b| {
        b.iter(|| {
            black_box(clip_preprocess_fused(
                black_box(&source),
                black_box(&config),
            ))
        })
    });

    group.bench_function("Kornia Pipeline", |b| {
        b.iter(|| {
            black_box(kornia_pipeline_op(
                black_box(&source),
                black_box(&config),
            ))
        })
    });


    group.finish();
}
criterion_group!(benches, benchmark_preprocessing);
criterion_main!(benches);