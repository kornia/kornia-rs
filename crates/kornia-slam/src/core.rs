use kornia_algebra::{SE3F32, Vec3AF32};
use thiserror::Error;

use crate::frontend::{extract_features, match_features, FrameFeatures};
use crate::map::Map;
use crate::rig::SensorRig;

/// SLAM configuration.
pub struct SlamConfig {
    pub max_keyframes: usize,
    pub min_matches: usize,
}

/// SLAM pipeline error classes.
#[derive(Error, Debug)]
pub enum SlamError {
    #[error("not enough features to initialize")]
    NotEnoughFeatures,
    #[error("estimation failed: {0}")]
    EstimationFailure(String),
}

/// A minimal skeleton for a visual-inertial SLAM pipeline.
pub struct VisualInertialSLAM {
    pub config: SlamConfig,
    pub rig: SensorRig,
    pub map: Map,
    pub initialized: bool,
    last_features: Vec<Option<FrameFeatures>>,
    pub pose: SE3F32,
    pub velocity: Vec3AF32,
    pub imu_bias: (Vec3AF32, Vec3AF32),
}

impl VisualInertialSLAM {
    /// Create a new SLAM instance with an associated sensor rig.
    pub fn new(config: SlamConfig, rig: SensorRig) -> Self {
        let camera_count = rig.cameras.len();
        Self {
            config,
            rig,
            map: Map::new(),
            initialized: false,
            last_features: vec![None; camera_count],
            pose: SE3F32::IDENTITY,
            velocity: Vec3AF32::ZERO,
            imu_bias: (Vec3AF32::ZERO, Vec3AF32::ZERO),
        }
    }

    /// Initialize with first frame + IMU for one camera.
    pub fn initialize(
        &mut self,
        camera_idx: usize,
        image: &[u8],
        width: usize,
        height: usize,
        imu: &[f32],
    ) -> Result<(), SlamError> {
        if image.is_empty() || imu.is_empty() {
            return Err(SlamError::NotEnoughFeatures);
        }

        if camera_idx >= self.rig.cameras.len() {
            return Err(SlamError::EstimationFailure(format!(
                "invalid camera index {camera_idx}"
            )));
        }

        let features = extract_features(image, width, height)
            .map_err(|e| SlamError::EstimationFailure(format!("feature init error: {e:?}")))?;

        if features.keypoints.len() < self.config.min_matches {
            return Err(SlamError::NotEnoughFeatures);
        }

        self.initialized = true;
        self.last_features[camera_idx] = Some(features);

        Ok(())
    }

    /// Run a single VO+IMU update step for one camera.
    pub fn track(
        &mut self,
        camera_idx: usize,
        image: &[u8],
        width: usize,
        height: usize,
        imu: &[f32],
    ) -> Result<usize, SlamError> {
        if !self.initialized {
            return Err(SlamError::EstimationFailure("not initialized".into()));
        }

        if image.is_empty() || imu.is_empty() {
            return Err(SlamError::NotEnoughFeatures);
        }

        if camera_idx >= self.rig.cameras.len() {
            return Err(SlamError::EstimationFailure(format!(
                "invalid camera index {camera_idx}"
            )));
        }

        let current = extract_features(image, width, height)
            .map_err(|e| SlamError::EstimationFailure(format!("feature track error: {e:?}")))?;

        let matches = if let Some(prev) = &self.last_features[camera_idx] {
            match_features(prev, &current)
        } else {
            Vec::new()
        };

        if matches.len() < self.config.min_matches {
            return Err(SlamError::EstimationFailure("not enough matches".into()));
        }

        // Placeholder pose update: realistic VO would use matched keypoints + IMU to estimate.
        self.pose.t.x += imu.get(0).copied().unwrap_or(0.0) * 0.01;
        self.pose.t.y += imu.get(1).copied().unwrap_or(0.0) * 0.01;
        self.pose.t.z += imu.get(2).copied().unwrap_or(0.0) * 0.01;

        // Add matched keypoints to map as landmarks.
        for &(_, idx_cur) in &matches {
            if let Some(kp) = current.keypoints.get(idx_cur) {
                let xyz = [kp[0], kp[1], 0.0];
                self.map.add_landmark(xyz);
            }
        }

        self.last_features[camera_idx] = Some(current);

        Ok(matches.len())
    }

    /// Query landmarks within a certain radius of the current pose.
    pub fn query_nearby_landmarks(&self, radius: f32) -> Vec<crate::map::Landmark> {
        self.map
            .nearby_landmarks([self.pose.t.x, self.pose.t.y, self.pose.t.z], radius)
            .into_iter()
            .cloned()
            .collect()
    }
}
