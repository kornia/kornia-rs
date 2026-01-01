use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;
use rayon::prelude::*;
use kornia_image::{Image, ImageSize};
use kornia_image::allocator::{ImageAllocator, CpuAllocator};
use wide::f32x8;

/// Standard configuration for CLIP input (224x224, standard mean/std).
pub struct ClipPreprocessConfig {
    pub target_size: usize,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl Default for ClipPreprocessConfig {
    fn default() -> Self {
        Self {
            target_size: 224,
            // CLIP standard Mean and STD
            mean: [0.48145466, 0.4578275, 0.40821073],
            std: [0.26862954, 0.26130258, 0.27577711],
        }
    }
}

/// Optimized Precomputed State (Structure of Arrays for SIMD)
pub struct ClipPreprocessState {
    // X-Map: Split into separate vectors for SIMD loading
    pub x_off0: Vec<usize>,
    pub x_off1: Vec<usize>,
    pub x_w0: Vec<f32>, 
    
    // Y-Map: Kept as struct since we access it once per row
    pub y_map: Vec<YMapOpt>,
    
    // Normalization constants
    pub scale: [f32; 3],
    pub bias: [f32; 3],
    
    pub target_size: usize,
}

#[derive(Copy, Clone)]
pub struct YMapOpt {
    pub y0_offset: usize,
    pub y1_offset: usize,
    pub wy0: f32,
    pub wy1: f32,
}

impl ClipPreprocessState {
    pub fn new(src_size: ImageSize, config: &ClipPreprocessConfig) -> Self {
        let out_size = config.target_size;
        let w = src_size.width;
        let h = src_size.height;
        
        // 1. Geometry Setup (Center Crop Logic)
        let scale_geom = if w < h { 
            w as f32 / out_size as f32 
        } else { 
            h as f32 / out_size as f32 
        };
        
        let start_x = (w as f32 - out_size as f32 * scale_geom) * 0.5;
        let start_y = (h as f32 - out_size as f32 * scale_geom) * 0.5;

        // 2. X-Map Precomputation
        let mut x_off0 = Vec::with_capacity(out_size);
        let mut x_off1 = Vec::with_capacity(out_size);
        let mut x_w0 = Vec::with_capacity(out_size);

        let mut sx = start_x;
        for _ in 0..out_size {
            let x0 = sx as usize;
            let x1 = (x0 + 1).min(w - 1);
            let dx = sx - x0 as f32;
            
            x_off0.push(x0 * 3);
            x_off1.push(x1 * 3);
            x_w0.push(1.0 - dx);
            
            sx += scale_geom;
        }

        // 3. Y-Map Precomputation
        let stride = w * 3;
        let y_map: Vec<YMapOpt> = (0..out_size).map(|y| {
            let sy = start_y + y as f32 * scale_geom;
            let y0 = sy as usize;
            let y1 = (y0 + 1).min(h - 1);
            let dy = sy - y0 as f32;
            
            YMapOpt { 
                y0_offset: y0 * stride, 
                y1_offset: y1 * stride, 
                wy0: 1.0 - dy, 
                wy1: dy 
            }
        }).collect();

        // 4. Precompute Normalization Constants
        let inv255 = 1.0 / 255.0;
        let scale = [
            inv255 / config.std[0],
            inv255 / config.std[1],
            inv255 / config.std[2],
        ];
        let bias = [
            -(config.mean[0] / config.std[0]),
            -(config.mean[1] / config.std[1]),
            -(config.mean[2] / config.std[2]),
        ];

        Self {
            x_off0, x_off1, x_w0,
            y_map,
            scale, bias,
            target_size: out_size,
        }
    }
}

// =======================================================================
// PART 2: Optimized Function (SIMD + Rayon)
// =======================================================================

const ROWS_PER_TASK: usize = 8; 

pub fn clip_preprocess_with_state<A: ImageAllocator>(
    image: &Image<u8, 3, A>, 
    state: &ClipPreprocessState
) -> Image<f32, 3, CpuAllocator> {
    
let out_size = state.target_size;
    let plane_size = out_size * out_size;
    let src = image.as_slice(); 
    
    // FIX 1: Explicit type annotation added here
    let mut data: Vec<f32> = Vec::with_capacity(plane_size * 3);
    unsafe { data.set_len(plane_size * 3) };

    let (r_plane, rest) = data.split_at_mut(plane_size);
    let (g_plane, b_plane) = rest.split_at_mut(plane_size);

    let scale_r = f32x8::splat(state.scale[0]);
    let bias_r  = f32x8::splat(state.bias[0]);
    let scale_g = f32x8::splat(state.scale[1]);
    let bias_g  = f32x8::splat(state.bias[1]);
    let scale_b = f32x8::splat(state.scale[2]);
    let bias_b  = f32x8::splat(state.bias[2]);
    let one     = f32x8::splat(1.0);

    r_plane.par_chunks_mut(out_size * ROWS_PER_TASK)
        .zip(g_plane.par_chunks_mut(out_size * ROWS_PER_TASK))
        .zip(b_plane.par_chunks_mut(out_size * ROWS_PER_TASK))
        .zip(state.y_map.par_chunks(ROWS_PER_TASK)) 
        .for_each(|(((r_blk, g_blk), b_blk), y_blk)| unsafe {

            let src_ptr = src.as_ptr();

            for (row_idx, y_data) in y_blk.iter().enumerate() {
                let row0 = src_ptr.add(y_data.y0_offset);
                let row1 = src_ptr.add(y_data.y1_offset);
                
                let r_dst = r_blk.as_mut_ptr().add(row_idx * out_size);
                let g_dst = g_blk.as_mut_ptr().add(row_idx * out_size);
                let b_dst = b_blk.as_mut_ptr().add(row_idx * out_size);

                let wy0 = f32x8::splat(y_data.wy0);
                let wy1 = f32x8::splat(y_data.wy1);

                let mut x = 0;
                
                while x + 8 <= out_size {
                    let wx0 = f32x8::from(&state.x_w0[x..x+8]);
                    let wx1 = one - wx0;

                    let w00 = wy0 * wx0;
                    let w01 = wy0 * wx1;
                    let w10 = wy1 * wx0;
                    let w11 = wy1 * wx1;

                    let mut p00_r = [0.0; 8]; let mut p01_r = [0.0; 8];
                    let mut p10_r = [0.0; 8]; let mut p11_r = [0.0; 8];
                    
                    let mut p00_g = [0.0; 8]; let mut p01_g = [0.0; 8];
                    let mut p10_g = [0.0; 8]; let mut p11_g = [0.0; 8];

                    let mut p00_b = [0.0; 8]; let mut p01_b = [0.0; 8];
                    let mut p10_b = [0.0; 8]; let mut p11_b = [0.0; 8];

                    for i in 0..8 {
                        let off0 = *state.x_off0.get_unchecked(x + i);
                        let off1 = *state.x_off1.get_unchecked(x + i);

                        p00_r[i] = *row0.add(off0) as f32;
                        p01_r[i] = *row0.add(off1) as f32;
                        p10_r[i] = *row1.add(off0) as f32;
                        p11_r[i] = *row1.add(off1) as f32;

                        p00_g[i] = *row0.add(off0 + 1) as f32;
                        p01_g[i] = *row0.add(off1 + 1) as f32;
                        p10_g[i] = *row1.add(off0 + 1) as f32;
                        p11_g[i] = *row1.add(off1 + 1) as f32;

                        p00_b[i] = *row0.add(off0 + 2) as f32;
                        p01_b[i] = *row0.add(off1 + 2) as f32;
                        p10_b[i] = *row1.add(off0 + 2) as f32;
                        p11_b[i] = *row1.add(off1 + 2) as f32;
                    }

                    // FIX 2: Use .to_array() and .copy_from_slice()
                    // Red
                    let v_r = f32x8::from(p00_r) * w00 + f32x8::from(p01_r) * w01 + 
                              f32x8::from(p10_r) * w10 + f32x8::from(p11_r) * w11;
                    let res_r = v_r.mul_add(scale_r, bias_r);
                    std::slice::from_raw_parts_mut(r_dst.add(x), 8)
                        .copy_from_slice(&res_r.to_array());

                    // Green
                    let v_g = f32x8::from(p00_g) * w00 + f32x8::from(p01_g) * w01 + 
                              f32x8::from(p10_g) * w10 + f32x8::from(p11_g) * w11;
                    let res_g = v_g.mul_add(scale_g, bias_g);
                    std::slice::from_raw_parts_mut(g_dst.add(x), 8)
                        .copy_from_slice(&res_g.to_array());

                    // Blue
                    let v_b = f32x8::from(p00_b) * w00 + f32x8::from(p01_b) * w01 + 
                              f32x8::from(p10_b) * w10 + f32x8::from(p11_b) * w11;
                    let res_b = v_b.mul_add(scale_b, bias_b);
                    std::slice::from_raw_parts_mut(b_dst.add(x), 8)
                        .copy_from_slice(&res_b.to_array());

                    x += 8;
                }

                while x < out_size {
                    let wx0 = *state.x_w0.get_unchecked(x);
                    let wx1 = 1.0 - wx0;
                    
                    let w00 = y_data.wy0 * wx0;
                    let w01 = y_data.wy0 * wx1;
                    let w10 = y_data.wy1 * wx0;
                    let w11 = y_data.wy1 * wx1;

                    let off0 = *state.x_off0.get_unchecked(x);
                    let off1 = *state.x_off1.get_unchecked(x);

                    let p00 = row0.add(off0);
                    let p01 = row0.add(off1);
                    let p10 = row1.add(off0);
                    let p11 = row1.add(off1);

                    let r = (*p00 as f32) * w00 + (*p01 as f32) * w01 + 
                            (*p10 as f32) * w10 + (*p11 as f32) * w11;
                    r_dst.add(x).write(r * state.scale[0] + state.bias[0]);

                    let g = (*p00.add(1) as f32) * w00 + (*p01.add(1) as f32) * w01 + 
                            (*p10.add(1) as f32) * w10 + (*p11.add(1) as f32) * w11;
                    g_dst.add(x).write(g * state.scale[1] + state.bias[1]);

                    let b = (*p00.add(2) as f32) * w00 + (*p01.add(2) as f32) * w01 + 
                            (*p10.add(2) as f32) * w10 + (*p11.add(2) as f32) * w11;
                    b_dst.add(x).write(b * state.scale[2] + state.bias[2]);

                    x += 1;
                }
            }
        });

    Image::new(
        ImageSize { width: out_size, height: out_size },
        data,
        CpuAllocator,
    ).expect("Failed to create Kornia Image")
}

// =======================================================================
// PART 4: Standard Kornia Pipeline (Unchanged)
// =======================================================================
use kornia_imgproc::{crop, interpolation::InterpolationMode, normalize, resize};

fn kornia_pipeline_op(
    image: &Image<u8, 3, CpuAllocator>,
    config: &ClipPreprocessConfig,
) -> Image<f32, 3, CpuAllocator> {
    let src_w = image.width();
    let src_h = image.height();
    let target = config.target_size;
    let scale = if src_w < src_h { target as f32 / src_w as f32 } else { target as f32 / src_h as f32 };
    let new_w = (src_w as f32 * scale).ceil() as usize;
    let new_h = (src_h as f32 * scale).ceil() as usize;

    let image_f32 = image.clone().cast_and_scale::<f32>(1.0 / 255.0).unwrap();
    let mut resized = Image::<f32, 3, _>::from_size_val(ImageSize { width: new_w, height: new_h }, 0.0f32, CpuAllocator).unwrap();
    resize::resize_native(&image_f32, &mut resized, InterpolationMode::Bilinear).unwrap();
    
    let crop_x = (new_w.saturating_sub(target)) / 2;
    let crop_y = (new_h.saturating_sub(target)) / 2;
    let mut cropped = Image::<f32, 3, _>::from_size_val(ImageSize { width: target, height: target }, 0.0f32, CpuAllocator).unwrap();
    crop::crop_image(&resized, &mut cropped, crop_x, crop_y).unwrap();

    let mut normalized = Image::<f32, 3, _>::from_size_val(cropped.size(), 0.0f32, CpuAllocator).unwrap();
    normalize::normalize_mean_std(&cropped, &mut normalized, &config.mean, &config.std).unwrap();

    let num_pixels = target * target;
    let mut chw_data = Vec::with_capacity(num_pixels * 3);
    unsafe { chw_data.set_len(num_pixels * 3) };
    let src_data = normalized.as_slice();
    for i in 0..num_pixels {
        chw_data[i] = src_data[i * 3];
        chw_data[i + num_pixels] = src_data[i * 3 + 1];
        chw_data[i + 2 * num_pixels] = src_data[i * 3 + 2];
    }
    Image::new(ImageSize { width: target, height: target }, chw_data, CpuAllocator).expect("Failed")
}

// =======================================================================
// PART 5: Benchmark Harness
// =======================================================================
fn benchmark_preprocessing(c: &mut Criterion) {
    let source = kornia_io::functional::read_image_any_rgb8("../../sample.jpg")
        .unwrap()
        .into_inner();

    let config = ClipPreprocessConfig::default();
    
    let state = ClipPreprocessState::new(source.size(), &config);

    let mut group = c.benchmark_group(format!(
        "Preprocessing_{}x{}",
        source.width(),
        source.height()
    ));

    group.bench_function("Fused Kernel (Baked Bias)", |b| {
        b.iter(|| {
            black_box(clip_preprocess_with_state(
                black_box(&source),
                black_box(&state),
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