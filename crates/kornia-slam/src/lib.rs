#![deny(missing_docs)]
#![doc = env!("CARGO_PKG_DESCRIPTION")]

pub mod core;
pub mod frontend;
pub mod map;
pub mod relocalization;
pub mod rig;
pub mod utils;

pub use core::{VisualInertialSLAM, SlamConfig};
pub use rig::{CameraSensor, ImuSensor, RigError, SensorRig};
pub use kornia_3d::camera::AnyCamera;

#[cfg(test)]
mod tests;
