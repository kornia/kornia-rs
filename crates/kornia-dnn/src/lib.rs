//! # Kornia DNN
//!
//! This module contains DNN (Deep Neural Network) related functionality.

/// Error type for the dnn module.
pub mod error;

/// This module contains the RT-DETR model.
pub mod rtdetr;

// TODO: put this in to some sort of structs pool module
/// Represents a detected object in an image.
#[derive(Debug)]
pub struct Detection {
    /// The class label of the detected object.
    pub label: u32,
    /// The confidence score of the detection (typically between 0 and 1).
    pub score: f32,
    /// The x-coordinate of the top-left corner of the bounding box.
    pub x: f32,
    /// The y-coordinate of the top-left corner of the bounding box.
    pub y: f32,
    /// The width of the bounding box.
    pub w: f32,
    /// The height of the bounding box.
    pub h: f32,
}
