//! Stereo geometry: rectification of non-row-aligned camera pairs, and sparse
//! keypoint correspondence on the rectified pair.

mod matcher;
#[cfg(feature = "cuda")]
mod matcher_cuda;
mod rectify;

pub use matcher::{
    SadRefine, StereoDescriptors, StereoKeypoints, StereoMatchConfig, StereoMatchError,
    StereoMatcher, StereoMatches, SubPixelFit, MAX_LEVELS, MAX_SAD_RADIUS,
};
pub use rectify::{CameraCalib, StereoError, StereoRectifier};

#[cfg(feature = "cuda")]
pub use matcher_cuda::{
    CudaStereoDescriptors, CudaStereoKeypoints, CudaStereoMatcher, CudaStereoMatches, KeypointCount,
};
#[cfg(feature = "cuda")]
pub use rectify::CudaStereoRectifier;
