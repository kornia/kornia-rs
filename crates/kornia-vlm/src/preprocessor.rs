use candle_core::{Device, Tensor};
use kornia_image::{allocator::ImageAllocator, Image};
use kornia_imgproc::interpolation::InterpolationMode;
use rayon::prelude::*;

// Constants
pub const CLIP_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
pub const CLIP_STD: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];
pub const SIGLIP_MEAN: [f32; 3] = [0.5, 0.5, 0.5];
pub const SIGLIP_STD: [f32; 3] = [0.5, 0.5, 0.5];

#[derive(Clone, Copy, Debug)]
pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl Rect {
    pub fn full() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            w: 1.0,
            h: 1.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Model {
    Clip,
    SmolVLM,
    LLaVA,
    PaliGemma,
}

#[derive(Clone, Debug)]
pub struct PreprocessConfig {
    pub model_type: Model,
    pub target_size: usize,
    pub mean: [f32; 3],
    pub std: [f32; 3],
    pub interpolation: InterpolationMode,
}

impl Default for PreprocessConfig {
    fn default() -> Self {
        Self {
            model_type: Model::Clip,
            target_size: 224,
            mean: CLIP_MEAN,
            std: CLIP_STD,
            interpolation: InterpolationMode::Bicubic,
        }
    }
}

#[derive(Clone, Debug)]
pub struct CropTransform {
    pub src_roi: Rect,
    pub dst_batch_idx: usize,
    pub dst_valid_roi: Rect,
    pub normalization_mean: [f32; 3],
    pub normalization_std: [f32; 3],
}

pub struct Preprocessor {
    config: PreprocessConfig,
}

fn get_center_crop_uv(w: u32, h: u32) -> Rect {
    let w_f = w as f32;
    let h_f = h as f32;
    if w < h {
        let crop_h = w_f / h_f;
        Rect {
            x: 0.0,
            y: (1.0 - crop_h) / 2.0,
            w: 1.0,
            h: crop_h,
        }
    } else {
        let crop_w = h_f / w_f;
        Rect {
            x: (1.0 - crop_w) / 2.0,
            y: 0.0,
            w: crop_w,
            h: 1.0,
        }
    }
}

fn get_grid_tiles(w: u32, h: u32, target_size: usize) -> Vec<(Rect, Rect)> {
    let w_f = w as f32;
    let h_f = h as f32;
    let target_f = target_size as f32;

    let cols = (w_f / target_f).ceil() as u32;
    let rows = (h_f / target_f).ceil() as u32;

    let mut tiles = Vec::with_capacity((rows * cols) as usize);

    for r in 0..rows {
        for c in 0..cols {
            let src_x = c as f32 * target_f;
            let src_y = r as f32 * target_f;

            let valid_w = (w_f - src_x).min(target_f);
            let valid_h = (h_f - src_y).min(target_f);

            let src_roi = Rect {
                x: src_x / w_f,
                y: src_y / h_f,
                w: valid_w / w_f,
                h: valid_h / h_f,
            };

            let dst_valid_roi = Rect {
                x: 0.0,
                y: 0.0,
                w: valid_w / target_f,
                h: valid_h / target_f,
            };

            tiles.push((src_roi, dst_valid_roi));
        }
    }
    tiles
}

#[inline(always)]
fn sinc(x: f32) -> f32 {
    if x == 0.0 {
        1.0
    } else {
        let x_pi = x * std::f32::consts::PI;
        x_pi.sin() / x_pi
    }
}

#[inline(always)]
fn get_lanzcos_weight(x: f32) -> f32 {
    const A: f32 = 3.0;
    if x.abs() < A {
        sinc(x) * sinc(x / A)
    } else {
        0.0
    }
}

#[inline(always)]
fn get_bilinear_weight(x: f32) -> f32 {
    1.0 - x.abs().min(1.0)
}

#[inline(always)]
fn get_nearest_weight(x: f32) -> f32 {
    if x.abs() <= 0.5 {
        1.0
    } else {
        0.0
    }
}

#[inline(always)]
fn get_cubic_weight(x: f32) -> f32 {
    const A: f32 = -0.75;
    if x.abs() <= 1.0 {
        (A + 2.0) * x.abs().powi(3) - (A + 3.0) * x.abs().powi(2) + 1.0
    } else if x.abs() < 2.0 {
        A * x.abs().powi(3) - 5.0 * A * x.abs().powi(2) + 8.0 * A * x.abs() - 4.0 * A
    } else {
        0.0
    }
}

// A reusable workspace to avoid re-allocating scratch buffers
pub struct ResizerWorkspace {
    pub scratch: Vec<f32>, // Intermediate planar buffer

    // X-dimension map buffers
    pub x_map_offsets: Vec<usize>,
    pub x_map_weights: Vec<f32>,
    pub x_map_counts: Vec<usize>,
    pub x_map_starts: Vec<usize>,

    // Y-dimension map buffers
    pub y_map_offsets: Vec<usize>,
    pub y_map_weights: Vec<f32>,
    pub y_map_counts: Vec<usize>,
    pub y_map_starts: Vec<usize>,
}

impl ResizerWorkspace {
    pub fn new() -> Self {
        Self {
            scratch: Vec::new(),
            x_map_offsets: Vec::new(),
            x_map_weights: Vec::new(),
            x_map_counts: Vec::new(),
            x_map_starts: Vec::new(),
            y_map_offsets: Vec::new(),
            y_map_weights: Vec::new(),
            y_map_counts: Vec::new(),
            y_map_starts: Vec::new(),
        }
    }
}

struct MapMetadata<'a> {
    offsets: &'a [usize],
    weights: &'a [f32],
    counts: &'a [usize],
    starts: &'a [usize],
    min_offset: usize,
    max_offset: usize,
}

#[allow(clippy::too_many_arguments)]
fn compute_weight_map_into_workspace<'a>(
    out_size: usize,
    src_dim: usize,
    src_start: f32,
    src_roi_dim: f32,
    stride_mul: usize,
    interpolation: InterpolationMode,
    offsets_buf: &'a mut Vec<usize>,
    weights_buf: &'a mut Vec<f32>,
    counts_buf: &'a mut Vec<usize>,
    starts_buf: &'a mut Vec<usize>,
) -> MapMetadata<'a> {
    offsets_buf.clear();
    weights_buf.clear();
    counts_buf.clear();
    starts_buf.clear();

    let scale = (src_roi_dim * src_dim as f32) / out_size as f32;
    let estimated_kernel = if scale > 1.0 {
        scale.ceil() as usize + 2
    } else {
        match interpolation {
            InterpolationMode::Bicubic => 4,
            InterpolationMode::Lanczos => 6,
            InterpolationMode::Bilinear => 2,
            InterpolationMode::Nearest => 1,
        }
    };

    // Reserve if needed to avoid reallocs during loop
    offsets_buf.reserve(out_size * estimated_kernel);
    weights_buf.reserve(out_size * estimated_kernel);
    counts_buf.reserve(out_size);
    starts_buf.reserve(out_size);

    let mut s_pos = src_start * src_dim as f32;
    let mut min_offset = usize::MAX;
    let mut max_offset = 0;

    for _ in 0..out_size {
        let center = s_pos + 0.5 * scale - 0.5;
        let start_idx = offsets_buf.len();
        let mut count = 0;

        if scale > 1.0 {
            // --- DOWNSCALING (Area-based) ---
            let radius = scale / 2.0;
            let left = center - radius;
            let right = center + radius;

            let first_pixel = left.ceil() as isize;
            let last_pixel = right.floor() as isize;

            // 1. Left Partial
            let left_weight = (first_pixel as f32) - left;
            if left_weight > 0.001 {
                let idx = (left.floor() as isize).clamp(0, src_dim as isize - 1) as usize;
                let off = idx * stride_mul;
                offsets_buf.push(off);
                weights_buf.push(left_weight);
                min_offset = min_offset.min(off);
                max_offset = max_offset.max(off);
                count += 1;
            }

            // 2. Center Full
            for i in first_pixel..=last_pixel {
                if (i as f32) < right {
                    let idx = i.clamp(0, src_dim as isize - 1) as usize;
                    let off = idx * stride_mul;
                    offsets_buf.push(off);
                    weights_buf.push(1.0);
                    min_offset = min_offset.min(off);
                    max_offset = max_offset.max(off);
                    count += 1;
                }
            }

            // 3. Right Partial
            if right > (last_pixel as f32) + 0.001 {
                let right_weight = right - (right.floor());
                let idx = (right.floor() as isize).clamp(0, src_dim as isize - 1) as usize;
                let off = idx * stride_mul;
                offsets_buf.push(off);
                weights_buf.push(right_weight);
                min_offset = min_offset.min(off);
                max_offset = max_offset.max(off);
                count += 1;
            }
        } else {
            // --- UPSCALING (Kernel-based) ---
            let x_int = center.floor() as isize;
            let dx = center - x_int as f32;

            match interpolation {
                InterpolationMode::Bicubic => {
                    for k in 0..4 {
                        let idx = (x_int - 1 + k as isize).clamp(0, src_dim as isize - 1) as usize;
                        let off = idx * stride_mul;
                        offsets_buf.push(off);
                        weights_buf.push(get_cubic_weight((k as isize - 1) as f32 - dx));
                        min_offset = min_offset.min(off);
                        max_offset = max_offset.max(off);
                        count += 1;
                    }
                }
                InterpolationMode::Lanczos => {
                    for k in 0..6 {
                        let idx = (x_int - 2 + k as isize).clamp(0, src_dim as isize - 1) as usize;
                        let off = idx * stride_mul;
                        offsets_buf.push(off);
                        weights_buf.push(get_lanzcos_weight(dx + 2.0 - k as f32));
                        min_offset = min_offset.min(off);
                        max_offset = max_offset.max(off);
                        count += 1;
                    }
                }
                InterpolationMode::Bilinear => {
                    for k in 0..2 {
                        let idx = (x_int + k as isize).clamp(0, src_dim as isize - 1) as usize;
                        let off = idx * stride_mul;
                        offsets_buf.push(off);
                        weights_buf.push(get_bilinear_weight(k as f32 - dx));
                        min_offset = min_offset.min(off);
                        max_offset = max_offset.max(off);
                        count += 1;
                    }
                }
                InterpolationMode::Nearest => {
                    let x_int = center.floor() as isize;
                    let dx = center - x_int as f32;
                    
                    for k in 0..1 {
                        let idx = (x_int + k as isize).clamp(0, src_dim as isize - 1) as usize;
                        let off = idx * stride_mul;
                        offsets_buf.push(off);
                        weights_buf.push(get_nearest_weight(k as f32 - dx));
                        min_offset = min_offset.min(off);
                        max_offset = max_offset.max(off);
                        count += 1;
                    }
                }
            }
        }

        // Normalize weights slice
        let w_slice = &mut weights_buf[start_idx..start_idx + count];
        let w_sum: f32 = w_slice.iter().sum();
        if w_sum > 0.0 {
            for w in w_slice {
                *w /= w_sum;
            }
        }

        starts_buf.push(start_idx);
        counts_buf.push(count);
        s_pos += scale;
    }

    // Handle edge case where no pixels were mapped (should not happen for valid images)
    if min_offset == usize::MAX {
        min_offset = 0;
        max_offset = 0;
    }

    MapMetadata {
        offsets: offsets_buf,
        weights: weights_buf,
        counts: counts_buf,
        starts: starts_buf,
        min_offset,
        max_offset,
    }
}

pub fn process_transform_separable<A: ImageAllocator>(
    image: &Image<u8, 3, A>,
    transform: &CropTransform,
    out_size: usize,
    interpolation: InterpolationMode,
    workspace: &mut ResizerWorkspace,
    device: &Device,
) -> Result<Tensor, candle_core::Error> {
    // --- 1. Geometry Setup ---
    let dst_x_min = (transform.dst_valid_roi.x * out_size as f32).round() as usize;
    let dst_y_min = (transform.dst_valid_roi.y * out_size as f32).round() as usize;
    let dst_w = ((transform.dst_valid_roi.w * out_size as f32).round() as usize)
        .min(out_size.saturating_sub(dst_x_min));
    let dst_h = ((transform.dst_valid_roi.h * out_size as f32).round() as usize)
        .min(out_size.saturating_sub(dst_y_min));

    let [mr, mg, mb] = transform.normalization_mean;
    let [sr, sg, sb] = transform.normalization_std;
    let fill_vals = [(0.0 - mr) / sr, (0.0 - mg) / sg, (0.0 - mb) / sb];

    let plane_size = out_size * out_size;
    let mut final_data = Vec::with_capacity(plane_size * 3);

    // Fast Exit: Return full padding if crop is invalid
    if dst_w == 0 || dst_h == 0 {
        final_data.resize(plane_size, fill_vals[0]);
        final_data.resize(plane_size * 2, fill_vals[1]);
        final_data.resize(plane_size * 3, fill_vals[2]);
        return Tensor::from_vec(final_data, (3, out_size, out_size), device);
    }

    // --- 2. Compute Maps (Using Workspace) ---
    let x_map_data = compute_weight_map_into_workspace(
        dst_w,
        image.width(),
        transform.src_roi.x,
        transform.src_roi.w,
        3,
        interpolation,
        &mut workspace.x_map_offsets,
        &mut workspace.x_map_weights,
        &mut workspace.x_map_counts,
        &mut workspace.x_map_starts,
    );

    let y_map_data = compute_weight_map_into_workspace(
        dst_h,
        image.height(),
        transform.src_roi.y,
        transform.src_roi.h,
        image.width() * 3,
        interpolation,
        &mut workspace.y_map_offsets,
        &mut workspace.y_map_weights,
        &mut workspace.y_map_counts,
        &mut workspace.y_map_starts,
    );

    // --- 3. Intermediate Scratch Buffer Setup ---
    let min_src_row = y_map_data.min_offset / (image.width() * 3);
    let max_src_row = y_map_data.max_offset / (image.width() * 3);
    let scratch_h = max_src_row - min_src_row + 1;
    let scratch_size = scratch_h * dst_w * 3;

    workspace.scratch.clear();
    workspace.scratch.resize(scratch_size, 0.0);

    let (s_r, rest) = workspace.scratch.split_at_mut(scratch_h * dst_w);
    let (s_g, s_b) = rest.split_at_mut(scratch_h * dst_w);

    let src_slice = image.as_slice();
    let src_stride = image.width() * 3;

    // LUT Init
    let inv255 = 1.0 / 255.0;
    let mut lut_r = [0f32; 256];
    let mut lut_g = [0f32; 256];
    let mut lut_b = [0f32; 256];
    for i in 0..256 {
        let v = i as f32 * inv255;
        lut_r[i] = (v - mr) / sr;
        lut_g[i] = (v - mg) / sg;
        lut_b[i] = (v - mb) / sb;
    }

    // --- 4. PASS 1: Horizontal Resize ---
    // Parallelize over rows using chunking to avoid capture issues
    s_r.par_chunks_mut(dst_w)
        .zip(s_g.par_chunks_mut(dst_w))
        .zip(s_b.par_chunks_mut(dst_w))
        .enumerate()
        .for_each(|(i, ((out_r, out_g), out_b))| {
            let src_row_ptr = unsafe { src_slice.as_ptr().add((min_src_row + i) * src_stride) };

            for x in 0..dst_w {
                let start = x_map_data.starts[x];
                let count = x_map_data.counts[x];

                let mut acc_r = 0.0;
                let mut acc_g = 0.0;
                let mut acc_b = 0.0;

                for k in 0..count {
                    unsafe {
                        let off = *x_map_data.offsets.get_unchecked(start + k);
                        let w = *x_map_data.weights.get_unchecked(start + k);
                        let p = src_row_ptr.add(off);

                        acc_r += lut_r[*p as usize] * w;
                        acc_g += lut_g[*p.add(1) as usize] * w;
                        acc_b += lut_b[*p.add(2) as usize] * w;
                    }
                }
                out_r[x] = acc_r;
                out_g[x] = acc_g;
                out_b[x] = acc_b;
            }
        });

    // --- 5. PASS 2: Vertical Resize ---
    final_data.resize(plane_size, fill_vals[0]);
    final_data.resize(plane_size * 2, fill_vals[1]);
    final_data.resize(plane_size * 3, fill_vals[2]);

    let (f_r, rest) = final_data.split_at_mut(plane_size);
    let (f_g, f_b) = rest.split_at_mut(plane_size);

    f_r.par_chunks_mut(out_size)
        .zip(f_g.par_chunks_mut(out_size))
        .zip(f_b.par_chunks_mut(out_size))
        .skip(dst_y_min)
        .take(dst_h)
        .enumerate()
        .for_each(|(y_idx, ((row_r, row_g), row_b))| {
            let start = y_map_data.starts[y_idx];
            let count = y_map_data.counts[y_idx];

            let valid_r = &mut row_r[dst_x_min..dst_x_min + dst_w];
            let valid_g = &mut row_g[dst_x_min..dst_x_min + dst_w];
            let valid_b = &mut row_b[dst_x_min..dst_x_min + dst_w];

            valid_r.fill(0.0);
            valid_g.fill(0.0);
            valid_b.fill(0.0);

            for k in 0..count {
                unsafe {
                    let global_off = *y_map_data.offsets.get_unchecked(start + k);
                    let w = *y_map_data.weights.get_unchecked(start + k);

                    let scratch_row_idx = (global_off / src_stride) - min_src_row;

                    let src_r =
                        s_r.get_unchecked(scratch_row_idx * dst_w..(scratch_row_idx + 1) * dst_w);
                    let src_g =
                        s_g.get_unchecked(scratch_row_idx * dst_w..(scratch_row_idx + 1) * dst_w);
                    let src_b =
                        s_b.get_unchecked(scratch_row_idx * dst_w..(scratch_row_idx + 1) * dst_w);

                    for x in 0..dst_w {
                        valid_r[x] += src_r[x] * w;
                        valid_g[x] += src_g[x] * w;
                        valid_b[x] += src_b[x] * w;
                    }
                }
            }
        });

    Tensor::from_vec(final_data, (3, out_size, out_size), device)
}

impl Preprocessor {
    pub fn new(mut config: PreprocessConfig) -> Self {
        if config.model_type == Model::SmolVLM {
            config.interpolation = InterpolationMode::Lanczos;
        }
        Self { config }
    }

    pub fn compute_transforms(&self, image_dims: (u32, u32)) -> Vec<CropTransform> {
        let (w, h) = image_dims;
        let mut instructions = Vec::new();
        match self.config.model_type {
            Model::Clip => {
                let roi = get_center_crop_uv(w, h);
                instructions.push(CropTransform {
                    src_roi: roi,
                    dst_batch_idx: 0,
                    dst_valid_roi: Rect::full(),
                    normalization_mean: self.config.mean,
                    normalization_std: self.config.std,
                });
            }
            Model::SmolVLM | Model::LLaVA => {
                let w_f = w as f32;
                let h_f = h as f32;
                let target_f = self.config.target_size as f32;

                let global_scale = if w_f <= target_f && h_f <= target_f {
                    1.0
                } else {
                    target_f / w_f.max(h_f)
                };

                instructions.push(CropTransform {
                    src_roi: Rect::full(),
                    dst_batch_idx: 0,
                    dst_valid_roi: Rect {
                        x: 0.0,
                        y: 0.0,
                        w: (w_f * global_scale) / target_f,
                        h: (h_f * global_scale) / target_f,
                    },
                    normalization_mean: self.config.mean,
                    normalization_std: self.config.std,
                });

                let tiles = get_grid_tiles(w, h, self.config.target_size);

                for (i, (src_roi, dst_valid_roi)) in tiles.into_iter().enumerate() {
                    instructions.push(CropTransform {
                        src_roi,
                        dst_batch_idx: i + 1,
                        dst_valid_roi,
                        normalization_mean: self.config.mean,
                        normalization_std: self.config.std,
                    });
                }
            }
            Model::PaliGemma => {
                instructions.push(CropTransform {
                    src_roi: Rect::full(),
                    dst_batch_idx: 0,
                    dst_valid_roi: Rect::full(),
                    normalization_mean: self.config.mean,
                    normalization_std: self.config.std,
                });
            }
        }
        instructions
    }

    /// Process an image generating all necessary crops/tiles as specified by the model config.
    pub fn process<A: ImageAllocator>(
        &self,
        image: &Image<u8, 3, A>,
        device: &Device,
    ) -> Result<Vec<Tensor>, candle_core::Error> {
        let transforms = self.compute_transforms((image.width() as u32, image.height() as u32));
        let mut workspace = ResizerWorkspace::new();
        transforms
            .iter()
            .map(|t| {
                process_transform_separable(
                    image,
                    t,
                    self.config.target_size,
                    self.config.interpolation,
                    &mut workspace,
                    device,
                )
            })
            .collect()
    }
}
