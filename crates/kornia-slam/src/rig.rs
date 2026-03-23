use kornia_3d::camera::AnyCamera;
use kornia_algebra::{SE3F32, SO3F32, Vec3AF32};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use thiserror::Error;

/// A camera sensor in the rig.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CameraSensor {
    pub id: String,
    pub model: AnyCamera,
    pub extrinsics: SE3F32,
}

/// IMU parameters and extrinsics.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ImuSensor {
    pub id: String,
    pub extrinsics: SE3F32,
    pub accel_noise_density: Option<f32>,
    pub gyro_noise_density: Option<f32>,
    pub time_offset_s: f32,
}

/// A single camera image sample with timestamp.
#[derive(Debug, Clone, PartialEq)]
pub struct CameraFrame {
    pub sensor_id: String,
    pub timestamp_s: f64,
    pub image: Vec<u8>,
}

/// A single IMU sample with timestamp.
#[derive(Debug, Clone, PartialEq)]
pub struct ImuSample {
    pub timestamp_s: f64,
    pub accel: [f32; 3],
    pub gyro: [f32; 3],
}

/// Synchronized snapshot for a fixed timestamp.
#[derive(Debug, Clone, PartialEq)]
pub struct SynchronizedSnapshot {
    pub timestamp_s: f64,
    pub camera_frames: Vec<CameraFrame>,
    pub imu_samples: Vec<ImuSample>,
}

/// Time synchronization helper for rig sensors.
#[derive(Debug, Default)]
pub struct TimeSynchronizer {
    camera_queues: HashMap<String, VecDeque<CameraFrame>>,
    imu_queue: VecDeque<ImuSample>,
}

/// Rigid sensor rig containing cameras and optional IMU.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct SensorRig {
    pub cameras: Vec<CameraSensor>,
    pub imu: Option<ImuSensor>,
}

/// Errors while building a sensor rig.
#[derive(Debug, Error)]
pub enum RigError {
    #[error("invalid YAML: {0}")]
    InvalidYaml(#[from] serde_yaml::Error),
    #[error("rig is empty: needs at least one camera")]
    EmptyRig,
}

impl SensorRig {
    pub fn from_kalibr_yaml(yaml: &str) -> Result<Self, RigError> {
        #[derive(Deserialize)]
        struct KalibrExtrinsics {
            rodrigues: [f32; 3],
            translation: [f32; 3],
        }

        #[derive(Deserialize)]
        struct KalibrCamera {
            model: String,
            intrinsics: Vec<f32>,
            image_width: usize,
            image_height: usize,
            #[serde(default)]
            rostopic: Option<String>,
        }

        #[derive(Deserialize)]
        struct KalibrYaml {
            cameras: Vec<KalibrCamera>,
            imu: Option<serde_yaml::Value>,
            extrinsics: Vec<KalibrExtrinsics>,
        }

        let parsed: KalibrYaml = serde_yaml::from_str(yaml)?;

        if parsed.cameras.is_empty() {
            return Err(RigError::EmptyRig);
        }

        let mut cameras = Vec::new();
        for (idx, cam) in parsed.cameras.into_iter().enumerate() {
            let model = match cam.model.as_str() {
                "pinhole" => AnyCamera::Pinhole(kornia_3d::camera::PinholeCamera {
                    fx: cam.intrinsics[0] as f64,
                    fy: cam.intrinsics[1] as f64,
                    cx: cam.intrinsics[2] as f64,
                    cy: cam.intrinsics[3] as f64,
                }),
                "kannala-brandt" | "kb" => {
                    AnyCamera::Fisheye(kornia_3d::camera::FisheyeCamera {
                        fx: cam.intrinsics[0] as f64,
                        fy: cam.intrinsics[1] as f64,
                        cx: cam.intrinsics[2] as f64,
                        cy: cam.intrinsics[3] as f64,
                        k1: cam.intrinsics[4] as f64,
                        k2: cam.intrinsics[5] as f64,
                        k3: cam.intrinsics[6] as f64,
                        k4: cam.intrinsics[7] as f64,
                    })
                }
                other => {
                    return Err(RigError::InvalidYaml(serde_yaml::Error::custom(format!(
                        "unsupported camera model {other}"
                    ))));
                }
            };

            let extrinsics = parsed
                .extrinsics
                .get(idx)
                .map(|e| SE3F32 {
                    r: SO3F32::exp(Vec3AF32::from_array(e.rodrigues)),
                    t: Vec3AF32::from_array(e.translation),
                })
                .unwrap_or(SE3F32::IDENTITY);

            cameras.push(CameraSensor {
                id: cam.rostopic.unwrap_or_else(|| format!("camera_{idx}")),
                model,
                extrinsics,
            });
        }

        let imu_sensor = parsed.imu.map(|v| {
            let accel_noise_density = v
                .get("accelerometer_noise_density")
                .and_then(|x| x.as_f64())
                .map(|x| x as f32);
            let gyro_noise_density = v
                .get("gyroscope_noise_density")
                .and_then(|x| x.as_f64())
                .map(|x| x as f32);

            ImuSensor {
                id: "imu0".to_string(),
                extrinsics: SE3F32::IDENTITY,
                accel_noise_density,
                gyro_noise_density,
                time_offset_s: v
                    .get("time_offset")
                    .and_then(|x| x.as_f64())
                    .map(|x| x as f32)
                    .unwrap_or(0.0),
            }
        });

        Ok(SensorRig {
            cameras,
            imu: imu_sensor,
        })
    }

    pub fn single_camera_rig(model: AnyCamera) -> Self {
        SensorRig {
            cameras: vec![CameraSensor {
                id: "camera0".to_string(),
                model,
                extrinsics: SE3F32::IDENTITY,
            }],
            imu: None,
        }
    }

    pub fn fps(&self) -> f32 {
        30.0
    }

    pub fn correct_timestamp(&self, sensor_id: &str, timestamp_s: f64) -> f64 {
        if let Some(imu) = &self.imu {
            if imu.id == sensor_id {
                return timestamp_s + imu.time_offset_s as f64;
            }
        }
        timestamp_s
    }
}

impl TimeSynchronizer {
    pub fn push_camera_frame(&mut self, frame: CameraFrame) {
        self.camera_queues
            .entry(frame.sensor_id.clone())
            .or_default()
            .push_back(frame);
    }

    pub fn push_imu_sample(&mut self, sample: ImuSample) {
        self.imu_queue.push_back(sample);
    }

    pub fn next_synchronized_snapshot(
        &mut self,
        camera_ids: &[String],
    ) -> Option<SynchronizedSnapshot> {
        if camera_ids.is_empty() {
            return None;
        }

        let mut target_time = 0.0f64;
        for camera_id in camera_ids {
            let queue = self.camera_queues.get(camera_id)?;
            let first = queue.front()?;
            target_time = target_time.max(first.timestamp_s);
        }

        let mut camera_frames = Vec::new();
        for camera_id in camera_ids {
            if let Some(queue) = self.camera_queues.get_mut(camera_id) {
                while let Some(front) = queue.front() {
                    if front.timestamp_s < target_time {
                        queue.pop_front();
                    } else {
                        break;
                    }
                }
                if let Some(frame) = queue.front() {
                    camera_frames.push(frame.clone());
                } else {
                    return None;
                }
            } else {
                return None;
            }
        }

        let mut imu_samples = Vec::new();
        while let Some(sample) = self.imu_queue.front() {
            if sample.timestamp_s <= target_time {
                imu_samples.push(self.imu_queue.pop_front().unwrap());
            } else {
                break;
            }
        }

        Some(SynchronizedSnapshot {
            timestamp_s: target_time,
            camera_frames,
            imu_samples,
        })
    }
}

pub enum CameraModel {
    Pinhole {
        fx: f32,
        fy: f32,
        cx: f32,
        cy: f32,
        width: usize,
        height: usize,
    },
    KannalaBrandt {
        fx: f32,
        fy: f32,
        cx: f32,
        cy: f32,
        k1: f32,
        k2: f32,
        k3: f32,
        k4: f32,
        width: usize,
        height: usize,
    },
}

impl From<CameraModel> for AnyCamera {
    fn from(model: CameraModel) -> Self {
        match model {
            CameraModel::Pinhole {
                fx,
                fy,
                cx,
                cy,
                ..
            } => AnyCamera::Pinhole(kornia_3d::camera::PinholeCamera {
                fx: fx as f64,
                fy: fy as f64,
                cx: cx as f64,
                cy: cy as f64,
            }),
            CameraModel::KannalaBrandt {
                fx,
                fy,
                cx,
                cy,
                k1,
                k2,
                k3,
                k4,
                ..
            } => AnyCamera::Fisheye(kornia_3d::camera::FisheyeCamera {
                fx: fx as f64,
                fy: fy as f64,
                cx: cx as f64,
                cy: cy as f64,
                k1: k1 as f64,
                k2: k2 as f64,
                k3: k3 as f64,
                k4: k4 as f64,
            }),
        }
    }
}
