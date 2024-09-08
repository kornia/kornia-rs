//! # RT-DETR
//!
//! This module contains the RT-DETR model.
//!
//! The RT-DETR model is a state-of-the-art object detection model.

use std::{env::current_exe, path::PathBuf};

use crate::error::DnnError;
use crate::Detection;
use kornia_core::{CpuAllocator, Tensor};
use kornia_image::Image;
use ort::{CUDAExecutionProvider, GraphOptimizationLevel, Session};

/// Builder for the RT-DETR detector.
///
/// This struct provides a convenient way to configure and create an `RTDETRDetector` instance.
pub struct RTDETRDetectorBuilder {
    /// Path to the RT-DETR model file.
    pub model_path: PathBuf,
    /// Number of threads to use for inference.
    pub num_threads: usize,
}

impl RTDETRDetectorBuilder {
    /// Creates a new `RTDETRDetectorBuilder` with default settings.
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to the RT-DETR model file.
    ///
    /// # Returns
    ///
    /// A `Result` containing the `RTDETRDetectorBuilder` if successful, or a `DnnError` if an error occurred.
    pub fn new(model_path: PathBuf) -> Result<Self, DnnError> {
        Ok(Self {
            model_path,
            num_threads: 4,
        })
    }

    /// Sets the number of threads to use for inference.
    ///
    /// # Arguments
    ///
    /// * `num_threads` - The number of threads to use.
    ///
    /// # Returns
    ///
    /// The updated `RTDETRDetectorBuilder` instance.
    pub fn with_num_threads(mut self, num_threads: usize) -> Self {
        self.num_threads = num_threads;
        self
    }

    /// Builds and returns an `RTDETRDetector` instance.
    ///
    /// # Returns
    ///
    /// A `Result` containing the `RTDETRDetector` if successful, or a `DnnError` if an error occurred.
    pub fn build(self) -> Result<RTDETRDetector, DnnError> {
        RTDETRDetector::new(self.model_path, self.num_threads)
    }
}

/// RT-DETR object detector.
///
/// This struct represents an instance of the RT-DETR object detection model.
pub struct RTDETRDetector {
    session: Session,
}

impl RTDETRDetector {
    // TODO: default to hf hub
    /// Creates a new `RTDETRDetector` instance.
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to the RT-DETR model file.
    /// * `num_threads` - Number of threads to use for inference.
    ///
    /// # Returns
    ///
    /// A `Result` containing the `RTDETRDetector` if successful, or a `DnnError` if an error occurred.
    ///
    /// Pre-requisites:
    /// - ORT_DYLIB_PATH environment variable must be set to the path of the ORT dylib.
    pub fn new(model_path: PathBuf, num_threads: usize) -> Result<Self, DnnError> {
        // get the ort dylib path from the environment variable
        let dylib_path =
            std::env::var("ORT_DYLIB_PATH").map_err(|e| DnnError::OrtDylibError(e.to_string()))?;

        // set the ort dylib path
        ort::init_from(dylib_path).commit()?;

        // create the ort session
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(num_threads)?
            .commit_from_file(model_path)?;

        Ok(Self { session })
    }

    /// Runs object detection on the given image.
    ///
    /// # Arguments
    ///
    /// * `image` - The input image as an `Image<u8, 3>`.
    ///
    /// # Returns
    ///
    /// A `Result` containing a vector of `Detection` objects if successful, or a `DnnError` if an error occurred.
    pub fn run(&self, image: &Image<u8, 3>) -> Result<Vec<Detection>, DnnError> {
        // TODO: explore pre-allocating memory for the image
        // cast and scale the image to f32
        let mut image_hwc_f32 = Image::from_size_val(image.size(), 0.0f32)?;
        kornia_image::ops::cast_and_scale(image, &mut image_hwc_f32, 1.0 / 255.)?;

        // convert to HWC -> CHW
        let image_chw = image_hwc_f32.permute_axes([2, 0, 1]).as_contiguous();

        // TODO: create a Tensor::insert_axis in kornia-rs
        let image_nchw = Tensor::from_shape_vec(
            [
                1,
                image_chw.shape[0],
                image_chw.shape[1],
                image_chw.shape[2],
            ],
            image_chw.into_vec(),
            CpuAllocator,
        )?;

        // make the ort tensor
        let ort_tensor = ort::Tensor::from_array((image_nchw.shape, image_nchw.into_vec()))?;

        // run the model
        let outputs = self.session.run(ort::inputs!["input" => ort_tensor]?)?;

        // extract the output tensor
        let (out_shape, out_ort) = outputs[0].try_extract_raw_tensor::<f32>()?;

        let out_tensor = Tensor::<f32, 3>::from_shape_vec(
            [
                out_shape[0] as usize,
                out_shape[1] as usize,
                out_shape[2] as usize,
            ],
            out_ort.to_vec(),
            CpuAllocator,
        )?;

        // parse the output tensor
        // we expect the output tensor to be a tensor of shape [1, N, 6]
        // where each element is a detection [label, score, x, y, w, h]
        let detections = out_tensor
            .as_slice()
            .chunks_exact(6)
            .map(|chunk| Detection {
                label: chunk[0] as u32,
                score: chunk[1],
                x: chunk[2],
                y: chunk[3],
                w: chunk[4],
                h: chunk[5],
            })
            .collect::<Vec<_>>();

        Ok(detections)
    }
}
