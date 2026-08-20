mod model;
mod utils;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::moondream::{Config, Model};
use hf_hub::{api::sync::Api, Repo, RepoType};
use kornia_image::Image;
use kornia_imgproc::{interpolation::InterpolationMode, resize::resize_fast_rgb_aa};
use tokenizers::Tokenizer;

use model::TextGeneration;

pub use model::GenerationStats;
pub use utils::{MoondreamConfig, MoondreamError};

/// Side length of the square input the vision encoder expects.
///
/// The encoder splits the image into 14x14 patches and
/// [`Config::v2`]'s `embed_len` is 729, so the input has to be
/// 27 * 14 = 378 pixels per side (27 * 27 = 729 patches).
const IMAGE_SIZE: usize = 378;

/// Hugging Face repository holding the weights that match [`Config::v2`].
///
/// NOTE: the `moondream2` repository has since changed its checkpoint layout, so
/// the weights compatible with candle's architecture live under `moondream1`.
const MODEL_ID: &str = "vikhyatk/moondream1";

/// Pinned revision. `main` must not be used here: the upstream repository has
/// rewritten its weight layout, and a mismatched checkpoint surfaces as an opaque
/// "cannot find tensor" from `VarBuilder` rather than a useful error.
const MODEL_REVISION: &str = "f6e9da68e8f1b78b8f3ee10905d56826db7a5802";

/// The Moondream model
///
/// Moondream is a small (~1.6B parameter) vision-language model for answering
/// questions about images. It pairs a ViT-style vision encoder with a Phi text
/// decoder, which makes it cheap enough to run on a CPU and a good fit for edge
/// deployments.
///
/// NOTE: to run the model on a GPU, build with `--features cuda` (NVIDIA) or
/// `--features metal` (Apple Silicon). The default build is CPU-only.
///
/// # Example
///
/// ```no_run
/// use kornia_vlm::moondream::{Moondream, MoondreamConfig};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let image = kornia_io::jpeg::read_image_jpeg_rgb8("dog.jpeg")?;
///
/// let mut moondream = Moondream::new(MoondreamConfig::default())?;
/// let answer = moondream.inference(&image, "What animal is this?", 100, false)?;
/// println!("{answer}");
/// # Ok(())
/// # }
/// ```
pub struct Moondream {
    pipeline: TextGeneration,
    img_buf: Image<u8, 3>,
    dtype: DType,
}

impl Moondream {
    /// Create a new Moondream model, downloading the weights from Hugging Face
    /// on first use.
    ///
    /// # Arguments
    ///
    /// * `config` - The configuration for the Moondream model
    pub fn new(config: MoondreamConfig) -> Result<Self, MoondreamError> {
        use crate::device::get_device_and_dtype;
        let (device, dtype) = get_device_and_dtype();

        let (model, tokenizer) = Self::load_model(dtype, &device)?;
        let img_buf = Image::from_size_val([IMAGE_SIZE, IMAGE_SIZE].into(), 0)?;
        let pipeline = TextGeneration::new(model, tokenizer, device, config.into())?;

        Ok(Self {
            pipeline,
            img_buf,
            dtype,
        })
    }

    /// Answer a question about an image.
    ///
    /// # Arguments
    ///
    /// * `image` - The rgb8 image to ask about, with shape [H, W, 3]
    /// * `prompt` - The question to ask, e.g. "What is the person holding?"
    /// * `sample_len` - The maximum number of tokens to generate
    /// * `stdout_debug` - Whether to stream the generated tokens to stdout
    ///
    /// # Returns
    ///
    /// * `answer` - The generated answer
    pub fn inference(
        &mut self,
        image: &Image<u8, 3>,
        prompt: &str,
        sample_len: usize,
        stdout_debug: bool,
    ) -> Result<String, MoondreamError> {
        let image_t = self.preprocess(image)?;

        // Moondream has no chat template; it was trained on a plain Q/A scaffold.
        let full_prompt = format!("\n\nQuestion: {prompt}\n\nAnswer:");

        self.pipeline
            .run(&image_t, &full_prompt, sample_len, stdout_debug)
    }

    /// Statistics for the most recent [`Self::inference`] call.
    #[inline]
    pub fn stats(&self) -> GenerationStats {
        self.pipeline.stats()
    }

    /// Resize and normalize an rgb8 image into the [1, 3, 378, 378] tensor the
    /// vision encoder expects, with values in [-1, 1].
    fn preprocess(&mut self, image: &Image<u8, 3>) -> Result<Tensor, MoondreamError> {
        // Antialiased resize: camera frames are usually much larger than 378px,
        // and aliasing on the way down costs real accuracy.
        resize_fast_rgb_aa(image, &mut self.img_buf, InterpolationMode::Bilinear, true)?;

        // Moondream normalizes with mean = std = 0.5, which is exactly
        // x * 2/255 - 1 — the same affine map paligemma uses.
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

        Ok(image_t)
    }

    // utility function to load the model
    fn load_model(dtype: DType, device: &Device) -> Result<(Model, Tokenizer), MoondreamError> {
        let api = Api::new()?;
        // TODO: add a way to load the model from a local path
        let repo = api.repo(Repo::with_revision(
            MODEL_ID.to_string(),
            RepoType::Model,
            MODEL_REVISION.to_string(),
        ));

        let tokenizer_filename = repo.get("tokenizer.json")?;
        // Single-file checkpoint, so there is no safetensors index to walk.
        let weights_filename = repo.get("model.safetensors")?;

        let tokenizer = Tokenizer::from_file(tokenizer_filename)?;

        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], dtype, device)? };
        let model = Model::new(&Config::v2(), vb)?;

        Ok((model, tokenizer))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The vision encoder's patch grid has to tile the input exactly, otherwise
    /// the reshape inside `VisionEncoder::forward` fails at runtime.
    #[test]
    fn image_size_matches_the_patch_grid() {
        const PATCH: usize = 14;
        const EMBED_LEN: usize = 729; // Config::v2().vision_config.embed_len
        assert_eq!(IMAGE_SIZE % PATCH, 0);
        let patches_per_side = IMAGE_SIZE / PATCH;
        assert_eq!(patches_per_side * patches_per_side, EMBED_LEN);
    }

    /// The affine map has to send the u8 range onto [-1, 1], matching the
    /// mean = std = 0.5 normalization Moondream was trained with.
    #[test]
    fn normalization_maps_u8_range_to_signed_unit() {
        let device = Device::Cpu;
        let pixels = Tensor::from_slice(&[0u8, 128, 255], &[3], &device).unwrap();
        let normalized = pixels
            .to_dtype(DType::F32)
            .unwrap()
            .affine(2. / 255., -1.)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        assert!((normalized[0] - (-1.0)).abs() < 1e-6);
        assert!((normalized[2] - 1.0).abs() < 1e-6);
        // Same result as subtracting a 0.5 mean and dividing by a 0.5 std.
        let expected_mid = (128.0 / 255.0 - 0.5) / 0.5;
        assert!((normalized[1] - expected_mid).abs() < 1e-6);
    }

    /// Preprocessing must produce the exact shape the encoder expects,
    /// regardless of the input aspect ratio.
    #[test]
    fn preprocess_produces_the_expected_tensor_shape() {
        let mut img_buf =
            Image::<u8, 3>::from_size_val([IMAGE_SIZE, IMAGE_SIZE].into(), 0).unwrap();
        let src = Image::<u8, 3>::from_size_val([640, 480].into(), 128).unwrap();

        resize_fast_rgb_aa(&src, &mut img_buf, InterpolationMode::Bilinear, true).unwrap();

        let device = Device::Cpu;
        let image_t = Tensor::from_raw_buffer(
            img_buf.as_slice(),
            DType::U8,
            &[img_buf.rows(), img_buf.cols(), 3],
            &device,
        )
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .permute((2, 0, 1))
        .unwrap()
        .affine(2. / 255., -1.)
        .unwrap()
        .unsqueeze(0)
        .unwrap();

        assert_eq!(image_t.dims(), &[1, 3, IMAGE_SIZE, IMAGE_SIZE]);
    }

    /// The prompt scaffold is what the model was trained on; a drifting format
    /// degrades answers silently.
    #[test]
    fn prompt_uses_the_question_answer_scaffold() {
        let prompt = format!("\n\nQuestion: {}\n\nAnswer:", "What is this?");
        assert_eq!(prompt, "\n\nQuestion: What is this?\n\nAnswer:");
    }
}
