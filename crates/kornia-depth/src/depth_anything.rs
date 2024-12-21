use std::{path::PathBuf, sync::Arc};

use candle_core::{DType, Device, Module, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::depth_anything_v2::{DepthAnythingV2, DepthAnythingV2Config};
use candle_transformers::models::dinov2;

use kornia_image::{ops, Image, ImageError};
use kornia_imgproc::{
    interpolation::InterpolationMode,
    normalize::{normalize_mean_std, normalize_min_max},
    resize::resize_fast,
};

#[derive(thiserror::Error, Debug)]
pub enum DepthAnythingError {
    #[error("Failed to load preprocess the image")]
    PreprocessError(#[from] ImageError),

    #[error("Failed to convert the image to a tensor")]
    TensorError(#[from] candle_core::Error),
}

struct DepthAnythingV2Preprocessor {
    resized_image: Image<u8, 3>,
    resized_image_f32: Image<f32, 3>,
    normalized_image: Image<f32, 3>,
}

impl DepthAnythingV2Preprocessor {
    // taken these from: https://huggingface.co/spaces/depth-anything/Depth-Anything-V2/blob/main/depth_anything_v2/dpt.py#L207
    const MAGIC_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
    const MAGIC_STD: [f32; 3] = [0.229, 0.224, 0.225];

    const DINO_IMG_SIZE: usize = 518;

    pub fn new() -> Result<Self, DepthAnythingError> {
        let new_size = [Self::DINO_IMG_SIZE, Self::DINO_IMG_SIZE].into();
        Ok(Self {
            resized_image: Image::from_size_val(new_size, 0u8)?,
            resized_image_f32: Image::from_size_val(new_size, 0f32)?,
            normalized_image: Image::from_size_val(new_size, 0f32)?,
        })
    }

    pub fn preprocess(
        &mut self,
        image: &Image<u8, 3>,
        device: &Device,
    ) -> Result<Tensor, DepthAnythingError> {
        // resize the image to the desired size
        resize_fast(image, &mut self.resized_image, InterpolationMode::Bilinear)?;

        // cast the image to f32 and scale it to the range [0, 1]
        ops::cast_and_scale(
            &self.resized_image,
            &mut self.resized_image_f32,
            1.0 / 255.0,
        )?;

        // normalize the image to the mean and std
        normalize_mean_std(
            &self.resized_image_f32,
            &mut self.normalized_image,
            &Self::MAGIC_MEAN,
            &Self::MAGIC_STD,
        )?;

        // convert the image to a candle tensor

        let img_t = Tensor::from_slice(
            self.normalized_image.as_slice(),
            &[Self::DINO_IMG_SIZE, Self::DINO_IMG_SIZE, 3],
            device,
        )?;

        // permute the image to the shape (1, c, h, w)
        let img_t = img_t.permute((2, 0, 1))?.unsqueeze(0)?;

        Ok(img_t)
    }
}

struct DepthAnythingV2Postprocessor {
    normalized_depth: Image<f32, 1>,
}

impl DepthAnythingV2Postprocessor {
    const OUTPUT_IMG_SIZE: usize = 520;

    pub fn new() -> Result<Self, DepthAnythingError> {
        Ok(Self {
            normalized_depth: Image::from_size_val(
                [Self::OUTPUT_IMG_SIZE, Self::OUTPUT_IMG_SIZE].into(),
                0f32,
            )?,
        })
    }

    pub fn postprocess(&mut self, depth: &Tensor) -> Result<&Image<f32, 1>, DepthAnythingError> {
        // convert the depth tensor to an image of shape (h, w, 1)
        let (_, _, rows, cols) = depth.dims4()?;

        let depth_data = depth.flatten_all()?.to_vec1::<f32>()?;
        let depth_image = Image::from_size_slice([rows, cols].into(), depth_data.as_slice())?;

        // normalize the depth image to the range [0, 1]
        normalize_min_max(&depth_image, &mut self.normalized_depth, 0.0, 1.0)?;

        Ok(&self.normalized_depth)
    }
}

pub struct DepthAnything {
    #[allow(unused)]
    dinov2: Arc<dinov2::DinoVisionTransformer>,
    depth_anything: DepthAnythingV2,
    preprocessor: DepthAnythingV2Preprocessor,
    postprocessor: DepthAnythingV2Postprocessor,
    device: Device,
}

impl DepthAnything {
    pub fn new(
        dinov2_model: Option<PathBuf>,
        depth_anything_v2_model: Option<PathBuf>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // set the device to cuda if available, otherwise use cpu
        let device = match Device::cuda_if_available(0) {
            Ok(device) => device,
            Err(e) => {
                log::warn!("Failed to use CUDA, using CPU instead: {}", e);
                Device::Cpu
            }
        };

        let dinov2_model_file = match dinov2_model {
            None => {
                let api = hf_hub::api::sync::Api::new()?;
                let api = api.model("lmz/candle-dino-v2".into());
                api.get("dinov2_vits14.safetensors")?
            }
            Some(dinov2_model) => dinov2_model,
        };

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[dinov2_model_file], DType::F32, &device)?
        };
        let dinov2 = Arc::new(dinov2::vit_small(vb)?);

        let depth_anything_model_file = match depth_anything_v2_model {
            None => {
                let api = hf_hub::api::sync::Api::new()?;
                let api = api.model("jeroenvlek/depth-anything-v2-safetensors".into());
                api.get("depth_anything_v2_vits.safetensors")?
            }
            Some(depth_anything_model) => depth_anything_model,
        };

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[depth_anything_model_file], DType::F32, &device)?
        };

        let config = DepthAnythingV2Config::vit_small();
        let depth_anything = DepthAnythingV2::new(dinov2.clone(), config, vb)?;

        let preprocessor = DepthAnythingV2Preprocessor::new()?;
        let postprocessor = DepthAnythingV2Postprocessor::new()?;

        Ok(Self {
            dinov2,
            depth_anything,
            preprocessor,
            postprocessor,
            device,
        })
    }

    pub fn forward(&mut self, image: &Image<u8, 3>) -> Result<&Image<f32, 1>, DepthAnythingError> {
        let img_t = self.preprocessor.preprocess(image, &self.device)?;
        let depth_t = self.depth_anything.forward(&img_t)?;
        let depth = self.postprocessor.postprocess(&depth_t)?;
        Ok(depth)
    }
}
