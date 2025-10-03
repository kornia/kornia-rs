use std::time::Duration;

use candle_core::{DType, Device, Shape, Tensor};
use kornia_image::{allocator::ImageAllocator, Image, ImageSize};
use kornia_imgproc::{interpolation::InterpolationMode, resize::resize_fast_rgb};
use log::warn;
use num2words::Num2Words;

const DEFAULT_VIDEO_INTRO: &str = "You are provided the following series of {frame_count} frames from a {video_duration} [H:MM:SS] video.\n";
const DEFAULT_MEDIA_OUTTRO: &str = "\n\n";
const FRAME_TIMESTAMP_MESSAGE: &str = "\nFrame from {timestamp}:";
const MAX_IMAGE_SIZE: usize = 4096; // 4k resolution as absolute maximum

use crate::{
    smolvlm2::{text_processor::TextProcessor, SmolVlm2Error},
    video::{Video, VideoError, VideoMetadata},
};

pub struct VideoProcessorConfig {
    // assumes the given video is sampled at 1 FPS
    pub max_frames: usize,
    pub video_size_longest_edge: usize,
    pub image_token: &'static str,
    pub video_token: &'static str,
    pub frame_mean: [f32; 3],
    pub frame_std: [f32; 3],
    pub rescale_factor: f32,
}

pub struct VideoProcessor {
    config: VideoProcessorConfig,
    processed_videos: Vec<(Tensor, Tensor)>,
    image_token_tensor: Option<Tensor>,
}

impl VideoProcessor {
    pub fn new(
        config: VideoProcessorConfig,
        device: &Device,
        txt_processor: &TextProcessor,
    ) -> Result<Self, SmolVlm2Error> {
        let image_token_tensor = if let Err(SmolVlm2Error::MissingTokenizer) =
            txt_processor.encode(config.image_token)
        {
            None
        } else {
            let image_token = txt_processor.encode(config.image_token)?;
            Some(Tensor::from_slice(&[image_token], &[1], device)?)
        };

        Ok(Self {
            config,
            processed_videos: Vec::new(),
            image_token_tensor,
        })
    }

    pub fn binding_videos_to_prompt<A: ImageAllocator>(
        &mut self,
        prompt: &mut String,
        videos: Vec<&mut Video<A>>,
        device: &Device,
        alloc: A,
    ) -> Result<(), SmolVlm2Error> {
        let cloned_prompt = prompt.clone();
        let video_tags_pos = cloned_prompt
            .match_indices(&self.config.video_token)
            .collect::<Vec<_>>();

        if video_tags_pos.len() != videos.len() {
            return Err(SmolVlm2Error::MismatchedVideoCount {
                tags: video_tags_pos.len(),
                videos: videos.len(),
            });
        }

        // Get the actual number of frames from the first video before processing
        let actual_frames = if !videos.is_empty() {
            videos[0].frames().len()
        } else {
            self.config.max_frames
        };

        let mut video_metadatas = vec![];
        for mut video in videos.into_iter() {
            video_metadatas.push(video.metadata().clone());
            let img_patches = self.preprocess(&mut video, device, alloc.clone())?;
            self.processed_videos
                .push((img_patches, Tensor::zeros(&[0], DType::F32, device)?));
        }

        self.expand_text_with_video_tokens(
            prompt,
            actual_frames,
            &video_metadatas,
            &video_tags_pos,
        )?;

        Ok(())
    }

    pub fn get_video_token_mask(&self, input: &Tensor) -> Result<Tensor, SmolVlm2Error> {
        Ok(input.broadcast_eq(
            self.image_token_tensor
                .as_ref()
                .ok_or(SmolVlm2Error::MissingTokenizer)?,
        )?)
    }

    pub fn get_processed_videos(&self) -> Vec<(&Tensor, &Tensor)> {
        self.processed_videos.iter().map(|(a, b)| (a, b)).collect()
    }

    pub fn clear_processed_videos(&mut self) {
        self.processed_videos.clear();
    }

    /// Preprocess a video for SmolVLM2 model inference
    /// We assume images are RGB.
    pub fn preprocess<A: ImageAllocator>(
        &mut self,
        video: &mut Video<A>,
        device: &Device,
        alloc: A,
    ) -> Result<Tensor, SmolVlm2Error> {
        let new_size = get_resize_output_image_size(
            video.frames()[0].size(),
            self.config.video_size_longest_edge,
        );

        video.process_frames(|mut frame| {
            Self::resize(
                &mut frame,
                new_size,
                InterpolationMode::Bicubic,
                alloc.clone(),
            )?;
            Self::resize(
                &mut frame,
                ImageSize {
                    width: self.config.video_size_longest_edge,
                    height: self.config.video_size_longest_edge,
                },
                InterpolationMode::Bicubic,
                alloc.clone(),
            )?;

            // pad
            let padded_size = ImageSize {
                width: self.config.video_size_longest_edge,
                height: self.config.video_size_longest_edge,
            };
            // TODO: masking
            Self::pad(&mut frame, padded_size, 0, alloc.clone())?;

            Ok(())
        })?;

        // 2. resize (no splitting)

        // Self::resize(
        //     &mut video.frames,
        //     new_size,
        //     InterpolationMode::Bicubic,
        //     alloc.clone(),
        // )?;
        // Self::resize(
        //     &mut video.frames,
        //     ImageSize {
        //         width: self.config.video_size_longest_edge,
        //         height: self.config.video_size_longest_edge,
        //     },
        //     InterpolationMode::Bicubic,
        //     alloc.clone(),
        // )?;

        // // pad
        // let padded_size = ImageSize {
        //     width: self.config.video_size_longest_edge,
        //     height: self.config.video_size_longest_edge,
        // };
        // // TODO: masking
        // Self::pad(
        //     &mut video.frames,
        //     padded_size,
        //     self.config.max_frames,
        //     0,
        //     alloc,
        // )?;

        // temporal padding - only pad if we really need consistent batch size
        // For memory efficiency, avoid padding when not necessary
        let video_tensor = video.into_tensor(device)?;
        let video_tensor_shape = video_tensor.shape();

        if video_tensor_shape.dim(0)? > self.config.max_frames {
            return Err(SmolVlm2Error::VideoProcessingError);
        }
        // Skip padding for now to avoid OOM issues with single frame videos
        // TODO: Add padding back when we need batch processing
        // else if video_tensor_shape.dim(0)? < self.config.max_frames {
        //     let pad_tensor = Tensor::zeros(
        //         // TODO: cache
        //         &[
        //             self.config.max_frames - video_tensor_shape.dim(0)?,
        //             3,
        //             video_tensor_shape.dim(2)?,
        //             video_tensor_shape.dim(3)?,
        //         ],
        //         DType::F32,
        //         device,
        //     )?;
        //     video_tensor = Tensor::cat(&[video_tensor, pad_tensor], 0)?;
        // }

        // normalize && rescale (must be the last step)
        let frames_tensor = Self::normalize_and_rescale(
            video_tensor,
            self.config.frame_mean,
            self.config.frame_std,
            self.config.rescale_factor,
            device,
        )?;

        Ok(frames_tensor)
    }

    /// Expands a single text prompt by replacing video tokens with video prompt strings using metadata.
    pub fn expand_text_with_video_tokens(
        &self,
        prompt: &mut String,
        num_frames: usize,
        video_metadata: &[VideoMetadata],
        video_tags_pos: &[(usize, &str)],
    ) -> Result<(), SmolVlm2Error> {
        let mut offset: isize = 0;
        for (meta_idx, (start, _)) in video_tags_pos.iter().enumerate() {
            let metadata = video_metadata
                .get(meta_idx)
                .cloned()
                .unwrap_or(VideoMetadata::default());
            // let mut metadata = metadata;
            // if metadata.fps.is_none() {
            //     warn!("SmolVLM requires frame timestamps to construct prompts, but the `fps` of the input video could not be inferred. Defaulting to `fps=24`. Please provide `video_metadata` for more accurate results.");
            //     metadata.fps = Some(24);
            // }
            let timestamps: Vec<(u32, u32)> = metadata
                .timestamps
                .iter()
                .map(|&second| (second / 60, second % 60))
                .collect();
            let duration = metadata
                .duration
                .unwrap_or_else(|| *metadata.timestamps.last().unwrap_or(&0));
            let duration_td = Duration::from_secs(duration as u64);
            let duration_str = format!(
                "{:01}:{:02}:{:02}",
                duration_td.as_secs() / 3600,
                (duration_td.as_secs() % 3600) / 60,
                duration_td.as_secs() % 60
            );
            let mut image_prompt_strings = DEFAULT_VIDEO_INTRO
                .replace(
                    "{frame_count}",
                    &Num2Words::new(num_frames as i64)
                        .cardinal()
                        .to_words()
                        .map_err(|_| SmolVlm2Error::VideoProcessingError)?,
                )
                .replace("{video_duration}", &duration_str);
            for timestamp in timestamps {
                let image_prompt_string = get_prompt_single_image(81);
                let timestamp_str = format!("{:02}:{:02}", timestamp.0, timestamp.1);
                let image_prompt_string = FRAME_TIMESTAMP_MESSAGE
                    .replace("{timestamp}", &timestamp_str)
                    + &image_prompt_string;
                image_prompt_strings += &image_prompt_string;
            }
            image_prompt_strings += DEFAULT_MEDIA_OUTTRO;
            let start = (*start as isize + offset) as usize;
            let end = start + self.config.video_token.len();
            prompt.replace_range(start..end, &image_prompt_strings);
            offset += image_prompt_strings.len() as isize - self.config.video_token.len() as isize;
        }

        Ok(())
    }

    /// Resize all frames in the video to a new size.
    ///
    /// This method resizes each frame in the video to the specified dimensions
    /// using the provided interpolation method.
    ///
    /// # Arguments
    ///
    /// * `new_size` - Target size for all frames
    /// * `interpolation` - Interpolation method to use for resizing
    /// * `alloc` - Allocator for creating new resized frames
    ///
    /// # Returns
    ///
    /// * `Result<(), VideoError>` - Ok if successful, VideoError if resizing fails
    pub fn resize<A: ImageAllocator>(
        frame: &mut Image<u8, 3, A>,
        new_size: ImageSize,
        interpolation: InterpolationMode,
        alloc: A,
    ) -> Result<(), VideoError> {
        // for i in 0..frames.len() {
        let mut buf = Image::<u8, 3, A>::from_size_val(new_size, 0, alloc.clone())?;
        resize_fast_rgb(&frame, &mut buf, interpolation)?;
        // frames[i] = buf;
        *frame = buf;
        // }
        Ok(())
    }

    /// Pad video frames spatially and temporally to target dimensions.
    ///
    /// This method performs two types of padding:
    /// 1. Spatial padding: Pads each frame to the target width and height
    /// 2. Temporal padding: Adds blank frames if fewer than max_num_frames exist
    ///
    /// Spatial padding preserves the original image content in the top-left corner
    /// and fills the remaining area with the specified fill value.
    ///
    /// # Arguments
    ///
    /// * `padded_size` - Target spatial dimensions (height, width) for frames
    /// * `max_num_frames` - Target number of frames in the video
    /// * `fill` - Value to use for padding pixels (e.g., 0 for black)
    /// * `alloc` - Allocator for creating new padded frames
    ///
    /// # Returns
    ///
    /// * `Result<(), VideoError>` - Ok if successful, VideoError if padding fails
    pub fn pad<A: ImageAllocator>(
        frame: &mut Image<u8, 3, A>,
        padded_size: ImageSize,
        fill: u8,
        alloc: A,
    ) -> Result<(), VideoError> {
        // Pad each frame spatially if needed
        // for i in 0..frames.len() {
        // let frame = frames;
        let size = frame.size();
        if size.width < padded_size.width || size.height < padded_size.height {
            let mut padded = Image::<u8, 3, A>::from_size_val(padded_size, fill, alloc.clone())?;
            let img_slice = frame.as_slice();
            let padded_img_slice = padded.as_slice_mut();
            let width = size.width;
            let height = size.height;
            let new_width = padded_size.width;
            for y in 0..height.min(padded_size.height) {
                let src_offset = y * width * 3;
                let dst_offset = y * new_width * 3;
                let row_bytes = width.min(new_width) * 3;
                padded_img_slice[dst_offset..dst_offset + row_bytes]
                    .copy_from_slice(&img_slice[src_offset..src_offset + row_bytes]);
            }
            // frames[i] = padded;
            *frame = padded;
        }
        // }

        // Pad temporally (add blank frames if needed)
        // let cur_frames = frames.len();
        // if cur_frames < max_num_frames {
        //     let blank = Image::<u8, 3, A>::from_size_val(padded_size, fill, alloc.clone())?;
        //     for _ in 0..(max_num_frames - cur_frames) {
        //         frames.push(blank.clone());
        //     }
        // }

        Ok(())
    }

    /// Normalize and rescale video frames for neural network processing.
    ///
    /// This method converts frames to tensors, applies rescaling (typically from [0,255] to [0,1]),
    /// and then normalizes using mean and standard deviation values. The resulting tensor
    /// is stored in `frames_tensor` with shape [T, C, H, W] where T is the number of frames.
    ///
    /// # Arguments
    ///
    /// * `mean` - Mean values for each color channel [R, G, B]
    /// * `std` - Standard deviation values for each color channel [R, G, B]
    /// * `rescale_factor` - Factor to rescale pixel values (typically 1.0/255.0)
    /// * `device` - Device to place the resulting tensor on
    ///
    /// # Returns
    ///
    /// * `Result<(), VideoError>` - Ok if successful, VideoError if tensor operations fail
    pub fn normalize_and_rescale(
        frames: Tensor,
        mean: [f32; 3],
        std: [f32; 3],
        rescale_factor: f32,
        device: &Device,
    ) -> Result<Tensor, SmolVlm2Error> {
        let mean_tensor = Tensor::from_slice(&mean, &[1, 3, 1, 1], device)?;
        let std_tensor = Tensor::from_slice(&std, &[1, 3, 1, 1], device)?;
        let rescale_tensor = Tensor::from_slice(&[rescale_factor], &[1, 1, 1, 1], device)?;

        Ok(frames
            .to_dtype(DType::F32)?
            .broadcast_mul(&rescale_tensor)?
            .broadcast_sub(&mean_tensor)?
            .broadcast_div(&std_tensor)?)
        // let mut tensors = vec![];
        // for i in 0..frames.len() {
        //     let mut tensor = Tensor::from_vec(
        //         frames[i].to_vec(),
        //         Shape::from_dims(&[frames[i].size().height, frames[i].size().width, 3]),
        //         device,
        //     )?
        //     .permute(vec![2, 0, 1])?
        //     .to_dtype(candle_core::DType::F32)?;

        //     // Normalize: scale to [0,1]
        //     tensor = tensor.broadcast_mul(&rescale_tensor)?;
        //     tensor = tensor
        //         .broadcast_sub(&mean_tensor)?
        //         .broadcast_div(&std_tensor)?;

        //     tensors.push(tensor);
        // }

        // Ok(Tensor::stack(&tensors, 0)?)
    }
}

pub fn get_prompt_single_image(img_seq_len: usize) -> String {
    format!(
        "<fake_token_around_image><global-img>{}<fake_token_around_image>",
        "<image>".repeat(img_seq_len)
    )
}

/// Get the output size of the video after resizing, given the max side length.
/// The longest edge of the video will be resized to this value, preserving aspect ratio.
/// Returns (height, width).
pub fn get_resize_output_image_size(size: ImageSize, resolution_max_side: usize) -> ImageSize {
    // Find the output size, when rescaling the longest edge to max_len and preserving the aspect ratio
    let resolution_max_side = resolution_max_side.min(MAX_IMAGE_SIZE);
    let mut height = size.height;
    let mut width = size.width;
    let aspect_ratio = width as f32 / height as f32;

    if width >= height {
        width = resolution_max_side;
        height = (width as f32 / aspect_ratio).round() as usize;
        if height % 2 != 0 {
            height += 1;
        }
    } else {
        height = resolution_max_side;
        width = (height as f32 * aspect_ratio).round() as usize;
        if width % 2 != 0 {
            width += 1;
        }
    }

    height = height.max(1);
    width = width.max(1);

    ImageSize { width, height }
}
