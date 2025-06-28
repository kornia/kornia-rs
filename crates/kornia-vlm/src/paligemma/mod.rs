mod model;
mod utils;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::paligemma::{Config, Model};
use hf_hub::{api::sync::Api, Repo, RepoType};
use kornia_image::{
    allocator::{CpuAllocator, ImageAllocator},
    Image,
};
use kornia_imgproc::{interpolation::InterpolationMode, resize::resize_fast};
use model::{TextGeneration, TextGenerationConfig};
use tokenizers::Tokenizer;
use utils::hub_load_safetensors;

#[derive(thiserror::Error, Debug)]
pub enum PaligemmaError {
    #[error(transparent)]
    FailedToLoadModel(#[from] hf_hub::api::sync::ApiError),

    #[error(transparent)]
    CandleError(#[from] candle_core::Error),

    #[error(transparent)]
    ImageError(#[from] kornia_image::ImageError),

    #[error(transparent)]
    TokenizerError(#[from] tokenizers::Error),

    #[error(transparent)]
    IoError(#[from] std::io::Error),

    #[error("Cannot find the <eos> token")]
    EosTokenNotFound,
}

/// Configuration for the Paligemma model
pub struct PaligemmaConfig {
    pub seed: u64,
    pub temp: Option<f64>,
    pub top_p: Option<f64>,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
}

impl From<PaligemmaConfig> for TextGenerationConfig {
    fn from(config: PaligemmaConfig) -> Self {
        TextGenerationConfig {
            seed: config.seed,
            temp: config.temp,
            top_p: config.top_p,
            repeat_penalty: config.repeat_penalty,
            repeat_last_n: config.repeat_last_n,
        }
    }
}

impl Default for PaligemmaConfig {
    fn default() -> Self {
        Self {
            seed: 299792458,
            temp: Some(0.7),
            top_p: Some(0.9),
            repeat_penalty: 1.1,
            repeat_last_n: 64,
        }
    }
}

/// The Paligemma model
///
/// This struct contains the Google Paligemma model for text generation from an image
/// and a given text prompt.
///
/// NOTE: to run the model with Cuda, you need to pass the `--features cuda` flag to the `cargo run` command.
pub struct Paligemma {
    pipeline: TextGeneration,
    img_buf: Image<u8, 3, CpuAllocator>,
    dtype: DType,
}

impl Paligemma {
    /// Create a new Paligemma model
    ///
    /// # Arguments
    ///
    /// * `config` - The configuration for the Paligemma model
    ///
    /// # Returns
    pub fn new(config: PaligemmaConfig) -> Result<Self, PaligemmaError> {
        #[cfg(feature = "cuda")]
        let (device, dtype) = match Device::cuda_if_available(0) {
            Ok(device) => (device, DType::BF16),
            Err(e) => {
                log::warn!("CUDA not available, defaulting to CPU: {e}");
                (Device::Cpu, DType::F32)
            }
        };

        #[cfg(not(feature = "cuda"))]
        let (device, dtype) = (Device::Cpu, DType::F32);

        let (model, tokenizer) = Self::load_model(dtype, &device)?;
        let img_buf = Image::from_size_val([224, 224].into(), 0, CpuAllocator)?;
        let pipeline = TextGeneration::new(model, tokenizer, device, config.into());

        Ok(Self {
            pipeline,
            img_buf,
            dtype,
        })
    }

    /// Run the inference of the Paligemma model
    ///
    /// # Arguments
    ///
    /// * `image` - The rgb8 image to generate a caption for with shape [H, W, 3]
    /// * `prompt` - The prompt to generate a caption for
    /// * `sample_len` - The length of the generated caption
    /// * `stdout_debug` - Whether to print the debug information to the stdout
    ///
    /// # Returns
    ///
    /// * `caption` - The generated caption
    pub fn inference<A: ImageAllocator>(
        &mut self,
        image: &Image<u8, 3, A>,
        prompt: &str,
        sample_len: usize,
        stdout_debug: bool,
    ) -> Result<String, PaligemmaError> {
        // resize image to 224x224
        resize_fast(image, &mut self.img_buf, InterpolationMode::Bilinear)?;

        // convert to tensor with shape [1, 3, 224, 224]
        let image_t = Tensor::from_raw_buffer(
            self.img_buf.as_slice(),
            DType::U8,
            &[self.img_buf.rows(), self.img_buf.cols(), 3],
            self.pipeline.device(),
        )?
        .to_dtype(self.dtype)?
        .permute((2, 0, 1))?
        .affine(2. / 255., -1.)?
        .unsqueeze(0)?;

        let response = self
            .pipeline
            .run(&image_t, prompt, sample_len, stdout_debug)
            .expect("Failed to generate text");

        Ok(response)
    }

    // utility function to load the model
    fn load_model(dtype: DType, device: &Device) -> Result<(Model, Tokenizer), PaligemmaError> {
        let api = Api::new()?;
        // TODO: add a way to load the model from a local path
        let model_id = "google/paligemma-3b-mix-224".to_string();
        let repo = {
            let revision = "main".to_string();
            api.repo(Repo::with_revision(model_id, RepoType::Model, revision))
        };

        let tokenizer_filename = repo.get("tokenizer.json")?;
        let filenames = hub_load_safetensors(&repo, "model.safetensors.index.json")?;

        let tokenizer = Tokenizer::from_file(tokenizer_filename)?;

        let config = Config::paligemma_3b_224();
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, device)? };

        let model = Model::new(&config, vb)?;

        Ok((model, tokenizer))
    }
}
