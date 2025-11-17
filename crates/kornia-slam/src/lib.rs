mod types;
mod ukf;
mod mpc;
mod trajectory;

pub use types::{State9, SlamConfig};
pub use glam::{Vec2, Vec3};

use types::Mat9x9;
use trajectory::Trajectory;

/// Main SLAM filter combining UKF sensor fusion with MPC trajectory following
pub struct SlamFilter {
    state: State9,
    covariance: Mat9x9,
    trajectory: Option<Trajectory>,
    config: SlamConfig,
}

impl SlamFilter {
    /// Create a new SLAM filter with the given configuration
    pub fn new(config: SlamConfig) -> Self {
        Self {
            state: State9::default(),
            covariance: Mat9x9::identity(),
            trajectory: None,
            config,
        }
    }

    /// Handle odometry measurement (twist in body frame)
    pub fn handle_odom(&mut self, linear_x: f32, linear_y: f32, angular_z: f32, dt: f32) {
        ukf::update_odom(
            &mut self.state,
            &mut self.covariance,
            linear_x,
            linear_y,
            angular_z,
            &self.config,
        );
        ukf::predict(&mut self.state, &mut self.covariance, dt, &self.config);
    }

    /// Handle IMU measurement (acceleration and gyroscope)
    pub fn handle_imu(&mut self, accel: Vec3, gyro: Vec3, dt: f32) {
        ukf::update_imu(
            &mut self.state,
            &mut self.covariance,
            accel,
            gyro,
            &self.config,
        );
        ukf::predict(&mut self.state, &mut self.covariance, dt, &self.config);
    }

    /// Handle GPS measurement (position in world frame)
    pub fn handle_gps(&mut self, position: Vec2, accuracy: f32) {
        ukf::update_gps(
            &mut self.state,
            &mut self.covariance,
            position,
            accuracy,
            &self.config,
        );
    }

    /// Set the trajectory to follow
    pub fn set_trajectory(&mut self, waypoints: &[Vec2]) {
        self.trajectory = Some(Trajectory::new(waypoints.to_vec()));
    }

    /// Compute control command using MPC
    pub fn compute_control(&self) -> Option<(f32, f32)> {
        let trajectory = self.trajectory.as_ref()?;
        mpc::compute_control(&self.state, trajectory, &self.config)
    }

    /// Get the current estimated state
    pub fn get_state(&self) -> &State9 {
        &self.state
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slam_filter_creation() {
        let config = SlamConfig::default();
        let filter = SlamFilter::new(config);
        assert_eq!(filter.get_state().position, Vec2::ZERO);
    }

    #[test]
    fn test_handle_odom() {
        let config = SlamConfig::default();
        let mut filter = SlamFilter::new(config);
        
        // Send some odometry measurements
        filter.handle_odom(1.0, 0.0, 0.0, 0.1);
        filter.handle_odom(1.0, 0.0, 0.0, 0.1);
        
        // State should have changed
        let state = filter.get_state();
        assert!(state.position.x > 0.0);
    }

    #[test]
    fn test_handle_imu() {
        let config = SlamConfig::default();
        let mut filter = SlamFilter::new(config);
        
        let accel = Vec3::new(0.0, 0.0, 9.81);
        let gyro = Vec3::new(0.0, 0.0, 0.0);
        
        filter.handle_imu(accel, gyro, 0.1);
        
        // Should not crash
        assert!(true);
    }

    #[test]
    fn test_set_trajectory() {
        let config = SlamConfig::default();
        let mut filter = SlamFilter::new(config);
        
        let waypoints = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(10.0, 0.0),
            Vec2::new(10.0, 10.0),
        ];
        
        filter.set_trajectory(&waypoints);
        
        // Should be able to compute control now
        let control = filter.compute_control();
        assert!(control.is_some());
    }
}

