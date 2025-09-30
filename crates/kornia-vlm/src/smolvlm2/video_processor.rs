use std::time::Duration;

use candle_core::{DType, Device, Tensor};
use kornia_image::{allocator::ImageAllocator, ImageSize};
use kornia_imgproc::interpolation::InterpolationMode;
use log::warn;
use num2words::Num2Words;

const DEFAULT_VIDEO_INTRO: &str = "You are provided the following series of {frame_count} frames from a {video_duration} [H:MM:SS] video.\n";
const DEFAULT_MEDIA_OUTTRO: &str = "\n\n";
const FRAME_TIMESTAMP_MESSAGE: &str = "\nFrame from {timestamp}:";
const MAX_IMAGE_SIZE: usize = 4096; // 4k resolution as absolute maximum

use crate::{
    smolvlm2::{text_processor::TextProcessor, utils::SmolVlm2Error},
    video::{Video, VideoMetadata},
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
        videos: Vec<Video<A>>,
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

        let mut video_metadatas = vec![];
        for mut video in videos.into_iter() {
            video_metadatas.push(video.metadata.clone());
            let img_patches = self.preprocess(&mut video, device, alloc.clone())?;
            self.processed_videos
                .push((img_patches, Tensor::zeros(&[0], DType::F32, device)?));
        }

        self.expand_text_with_video_tokens(
            prompt,
            self.config.max_frames,
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
        // 2. resize (no splitting)
        let new_size = get_resize_output_image_size(
            video.frames[0].size(),
            self.config.video_size_longest_edge,
        );

        video.resize(new_size, InterpolationMode::Bicubic, alloc.clone())?;
        video.resize(
            // resize to a potentially distorted square
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
        video.pad(padded_size, self.config.max_frames, 0, alloc)?;

        // normalize && rescale (must be the last step)
        video.normalize_and_rescale(
            self.config.frame_mean,
            self.config.frame_std,
            self.config.rescale_factor,
            device,
        )?;

        video
            .frames_tensor
            .clone()
            .ok_or(SmolVlm2Error::VideoProcessingError)
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
                .unwrap_or(VideoMetadata {
                    fps: None,
                    timestamps: vec![],
                    duration: None,
                });
            let mut metadata = metadata;
            if metadata.fps.is_none() {
                warn!("SmolVLM requires frame timestamps to construct prompts, but the `fps` of the input video could not be inferred. Defaulting to `fps=24`. Please provide `video_metadata` for more accurate results.");
                metadata.fps = Some(24);
            }
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
