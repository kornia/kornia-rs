use glam::{Vec2, Vec3};

/// 9-dimensional state vector for the robot
#[derive(Debug, Clone, Copy)]
pub struct State9 {
    /// Position in world frame (x, y)
    pub position: Vec2,
    /// Heading angle in radians
    pub heading: f32,
    /// Linear velocity in body frame (vx, vy)
    pub linear_vel: Vec2,
    /// Angular velocity (yaw rate)
    pub angular_vel: f32,
    /// Accelerometer bias
    pub accel_bias: Vec2,
    /// Gyroscope bias
    pub gyro_bias: f32,
}

impl Default for State9 {
    fn default() -> Self {
        Self {
            position: Vec2::ZERO,
            heading: 0.0,
            linear_vel: Vec2::ZERO,
            angular_vel: 0.0,
            accel_bias: Vec2::ZERO,
            gyro_bias: 0.0,
        }
    }
}

impl State9 {
    /// Convert state to a 9-element array
    pub fn to_array(&self) -> [f32; 9] {
        [
            self.position.x,
            self.position.y,
            self.heading,
            self.linear_vel.x,
            self.linear_vel.y,
            self.angular_vel,
            self.accel_bias.x,
            self.accel_bias.y,
            self.gyro_bias,
        ]
    }

    /// Create state from a 9-element array
    pub fn from_array(arr: [f32; 9]) -> Self {
        Self {
            position: Vec2::new(arr[0], arr[1]),
            heading: arr[2],
            linear_vel: Vec2::new(arr[3], arr[4]),
            angular_vel: arr[5],
            accel_bias: Vec2::new(arr[6], arr[7]),
            gyro_bias: arr[8],
        }
    }

    /// Add two states element-wise
    pub fn add(&self, other: &State9) -> State9 {
        let arr1 = self.to_array();
        let arr2 = other.to_array();
        let mut result = [0.0; 9];
        for i in 0..9 {
            result[i] = arr1[i] + arr2[i];
        }
        State9::from_array(result)
    }

    /// Subtract two states element-wise
    pub fn sub(&self, other: &State9) -> State9 {
        let arr1 = self.to_array();
        let arr2 = other.to_array();
        let mut result = [0.0; 9];
        for i in 0..9 {
            result[i] = arr1[i] - arr2[i];
        }
        State9::from_array(result)
    }

    /// Scale state by a scalar
    pub fn scale(&self, s: f32) -> State9 {
        let arr = self.to_array();
        let mut result = [0.0; 9];
        for i in 0..9 {
            result[i] = arr[i] * s;
        }
        State9::from_array(result)
    }
}

/// 9x9 covariance matrix
#[derive(Debug, Clone, Copy)]
pub struct Mat9x9 {
    pub data: [[f32; 9]; 9],
}

impl Mat9x9 {
    /// Create identity matrix
    pub fn identity() -> Self {
        let mut data = [[0.0; 9]; 9];
        for i in 0..9 {
            data[i][i] = 1.0;
        }
        Self { data }
    }

    /// Create zero matrix
    pub fn zeros() -> Self {
        Self { data: [[0.0; 9]; 9] }
    }

    /// Create diagonal matrix from array
    pub fn from_diagonal(diag: [f32; 9]) -> Self {
        let mut data = [[0.0; 9]; 9];
        for i in 0..9 {
            data[i][i] = diag[i];
        }
        Self { data }
    }

    /// Matrix-vector multiplication
    pub fn mul_vec(&self, v: &[f32; 9]) -> [f32; 9] {
        let mut result = [0.0; 9];
        for i in 0..9 {
            for j in 0..9 {
                result[i] += self.data[i][j] * v[j];
            }
        }
        result
    }

    /// Matrix-matrix addition
    pub fn add(&self, other: &Mat9x9) -> Mat9x9 {
        let mut result = [[0.0; 9]; 9];
        for i in 0..9 {
            for j in 0..9 {
                result[i][j] = self.data[i][j] + other.data[i][j];
            }
        }
        Mat9x9 { data: result }
    }

    /// Matrix-matrix multiplication
    pub fn mul(&self, other: &Mat9x9) -> Mat9x9 {
        let mut result = [[0.0; 9]; 9];
        for i in 0..9 {
            for j in 0..9 {
                for k in 0..9 {
                    result[i][j] += self.data[i][k] * other.data[k][j];
                }
            }
        }
        Mat9x9 { data: result }
    }

    /// Transpose
    pub fn transpose(&self) -> Mat9x9 {
        let mut result = [[0.0; 9]; 9];
        for i in 0..9 {
            for j in 0..9 {
                result[i][j] = self.data[j][i];
            }
        }
        Mat9x9 { data: result }
    }

    /// Scale matrix by scalar
    pub fn scale(&self, s: f32) -> Mat9x9 {
        let mut result = [[0.0; 9]; 9];
        for i in 0..9 {
            for j in 0..9 {
                result[i][j] = self.data[i][j] * s;
            }
        }
        Mat9x9 { data: result }
    }

    /// Compute Cholesky decomposition (simplified - assumes positive definite)
    pub fn cholesky(&self) -> Option<Mat9x9> {
        let mut l = [[0.0; 9]; 9];
        
        for i in 0..9 {
            for j in 0..=i {
                let mut sum = 0.0;
                if i == j {
                    for k in 0..j {
                        sum += l[j][k] * l[j][k];
                    }
                    let val = self.data[j][j] - sum;
                    if val <= 0.0 {
                        return None;
                    }
                    l[j][j] = val.sqrt();
                } else {
                    for k in 0..j {
                        sum += l[i][k] * l[j][k];
                    }
                    if l[j][j] == 0.0 {
                        return None;
                    }
                    l[i][j] = (self.data[i][j] - sum) / l[j][j];
                }
            }
        }
        
        Some(Mat9x9 { data: l })
    }
}

/// Configuration for the SLAM filter
#[derive(Debug, Clone)]
pub struct SlamConfig {
    /// UKF alpha parameter (spread of sigma points)
    pub ukf_alpha: f32,
    /// UKF beta parameter (prior knowledge, 2 for Gaussian)
    pub ukf_beta: f32,
    /// UKF kappa parameter (secondary scaling)
    pub ukf_kappa: f32,
    /// Process noise for each state dimension
    pub process_noise: [f32; 9],
    /// Measurement noise for odometry (vx, vy, omega)
    pub measurement_noise_odom: [f32; 3],
    /// Measurement noise for IMU (ax, ay, gz)
    pub measurement_noise_imu: [f32; 3],
    /// Measurement noise for GPS (x, y)
    pub measurement_noise_gps: [f32; 2],
    /// MPC prediction horizon steps
    pub mpc_horizon: usize,
    /// MPC time step
    pub mpc_dt: f32,
    /// MPC state tracking weights (x, y, theta)
    pub mpc_q_weights: [f32; 3],
    /// MPC control effort weights (v, omega)
    pub mpc_r_weights: [f32; 2],
    /// Maximum linear velocity
    pub max_linear_vel: f32,
    /// Maximum angular velocity
    pub max_angular_vel: f32,
}

impl Default for SlamConfig {
    fn default() -> Self {
        Self {
            ukf_alpha: 0.001,
            ukf_beta: 2.0,
            ukf_kappa: 0.0,
            process_noise: [0.1, 0.1, 0.05, 0.2, 0.2, 0.1, 0.01, 0.01, 0.01],
            measurement_noise_odom: [0.1, 0.1, 0.05],
            measurement_noise_imu: [0.5, 0.5, 0.1],
            measurement_noise_gps: [1.0, 1.0],
            mpc_horizon: 10,
            mpc_dt: 0.1,
            mpc_q_weights: [1.0, 1.0, 0.5],
            mpc_r_weights: [0.1, 0.1],
            max_linear_vel: 2.5,
            max_angular_vel: std::f32::consts::PI,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_state9_to_from_array() {
        let state = State9 {
            position: Vec2::new(1.0, 2.0),
            heading: 0.5,
            linear_vel: Vec2::new(0.5, 0.0),
            angular_vel: 0.1,
            accel_bias: Vec2::new(0.01, 0.02),
            gyro_bias: 0.001,
        };

        let arr = state.to_array();
        let state2 = State9::from_array(arr);

        assert_relative_eq!(state.position.x, state2.position.x);
        assert_relative_eq!(state.position.y, state2.position.y);
        assert_relative_eq!(state.heading, state2.heading);
    }

    #[test]
    fn test_state9_operations() {
        let s1 = State9 {
            position: Vec2::new(1.0, 2.0),
            heading: 0.5,
            ..Default::default()
        };
        let s2 = State9 {
            position: Vec2::new(2.0, 1.0),
            heading: 0.3,
            ..Default::default()
        };

        let sum = s1.add(&s2);
        assert_relative_eq!(sum.position.x, 3.0);
        assert_relative_eq!(sum.position.y, 3.0);
        assert_relative_eq!(sum.heading, 0.8);

        let diff = s1.sub(&s2);
        assert_relative_eq!(diff.position.x, -1.0);
        assert_relative_eq!(diff.position.y, 1.0);

        let scaled = s1.scale(2.0);
        assert_relative_eq!(scaled.position.x, 2.0);
        assert_relative_eq!(scaled.position.y, 4.0);
    }

    #[test]
    fn test_mat9x9_identity() {
        let id = Mat9x9::identity();
        for i in 0..9 {
            for j in 0..9 {
                if i == j {
                    assert_relative_eq!(id.data[i][j], 1.0);
                } else {
                    assert_relative_eq!(id.data[i][j], 0.0);
                }
            }
        }
    }

    #[test]
    fn test_mat9x9_mul_vec() {
        let mat = Mat9x9::identity();
        let v = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let result = mat.mul_vec(&v);
        
        for i in 0..9 {
            assert_relative_eq!(result[i], v[i]);
        }
    }

    #[test]
    fn test_mat9x9_operations() {
        let m1 = Mat9x9::from_diagonal([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let m2 = Mat9x9::from_diagonal([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        
        let sum = m1.add(&m2);
        assert_relative_eq!(sum.data[0][0], 2.0);
        assert_relative_eq!(sum.data[1][1], 3.0);

        let scaled = m1.scale(2.0);
        assert_relative_eq!(scaled.data[0][0], 2.0);
        assert_relative_eq!(scaled.data[4][4], 10.0);
    }

    #[test]
    fn test_mat9x9_cholesky() {
        // Test with a simple positive definite matrix
        let mut mat = Mat9x9::identity();
        mat.data[0][0] = 4.0;
        mat.data[1][1] = 4.0;
        mat.data[2][2] = 4.0;
        
        let chol = mat.cholesky();
        assert!(chol.is_some());
        
        if let Some(l) = chol {
            assert_relative_eq!(l.data[0][0], 2.0);
        }
    }
}

