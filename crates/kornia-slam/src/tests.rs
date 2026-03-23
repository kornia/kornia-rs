#[cfg(test)]
mod tests {
    use kornia_3d::camera::{AnyCamera, PinholeCamera};
    use kornia_algebra::{SE3F32, Vec3AF32};
    use crate::core::{SlamConfig, VisualInertialSLAM};
    use crate::rig::SensorRig;

    #[test]
    fn slam_initialize_and_track() {
        let width = 32;
        let height = 32;
        let mut image = vec![0u8; width * height];

        // Synthetic high-contrast grid.
        for y in 0..height {
            for x in 0..width {
                if x % 8 == 0 || y % 8 == 0 || (x == y) {
                    image[y * width + x] = 255;
                }
            }
        }

        let rig = SensorRig::single_camera_rig(AnyCamera::Pinhole(PinholeCamera {
            fx: 50.0,
            fy: 50.0,
            cx: 16.0,
            cy: 16.0,
        }));

        let mut slam = VisualInertialSLAM::new(
            SlamConfig {
                max_keyframes: 5,
                min_matches: 5,
            },
            rig,
        );

        assert!(
            slam
                .initialize(0, &image, width, height, &[0.1, 0.1, 0.1])
                .is_ok()
        );

        let matches = slam
            .track(0, &image, width, height, &[0.1, 0.1, 0.1])
            .expect("track should succeed");

        assert!(matches >= 5);

        let nearby = slam.query_nearby_landmarks(100.0);
        assert!(!nearby.is_empty());

        // Check if pose is SE3F32 and updated
        assert!(slam.pose.t.x > 0.0);
    }

    #[test]
    fn rig_from_kalibr_yaml() {
        let yaml = r#"
        cameras:
          - model: pinhole
            intrinsics: [50.0, 50.0, 16.0, 16.0]
            image_width: 32
            image_height: 32
            rostopic: /cam0
        extrinsics:
          - rodrigues: [0.0, 0.0, 0.0]
            translation: [0.0, 0.0, 0.0]
        imu:
          accelerometer_noise_density: 0.01
          gyroscope_noise_density: 0.001
          time_offset: 0.001
        "#;

        let rig = SensorRig::from_kalibr_yaml(yaml).expect("rig parse");
        assert_eq!(rig.cameras.len(), 1);
        assert!(rig.imu.is_some());
        
        let cam = &rig.cameras[0];
        match &cam.model {
            AnyCamera::Pinhole(p) => {
                assert_eq!(p.fx, 50.0);
            }
            _ => panic!("Expected pinhole"),
        }
    }
}
