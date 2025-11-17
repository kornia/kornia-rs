mod sigma;

use crate::types::{Mat9x9, SlamConfig, State9};
use glam::{Vec2, Vec3};
use sigma::{compute_covariance, compute_mean, compute_weights, generate_sigma_points};

/// Process model: propagate state forward in time using constant velocity
fn process_model(state: &State9, dt: f32) -> State9 {
    let cos_theta = state.heading.cos();
    let sin_theta = state.heading.sin();

    // Transform body velocity to world frame
    let vx_world = state.linear_vel.x * cos_theta - state.linear_vel.y * sin_theta;
    let vy_world = state.linear_vel.x * sin_theta + state.linear_vel.y * cos_theta;

    State9 {
        position: state.position + Vec2::new(vx_world, vy_world) * dt,
        heading: state.heading + state.angular_vel * dt,
        linear_vel: state.linear_vel, // Constant velocity assumption
        angular_vel: state.angular_vel, // Constant angular velocity
        accel_bias: state.accel_bias, // Biases slowly varying
        gyro_bias: state.gyro_bias,
    }
}

/// Prediction step: propagate state and covariance forward in time
pub fn predict(state: &mut State9, covariance: &mut Mat9x9, dt: f32, config: &SlamConfig) {
    // Generate sigma points
    let sigma_points = generate_sigma_points(state, covariance, config);

    // Propagate sigma points through process model
    let mut propagated_sigma_points = [State9::default(); 19];
    for (i, sp) in sigma_points.iter().enumerate() {
        propagated_sigma_points[i] = process_model(sp, dt);
    }

    // Compute predicted mean and covariance
    let (wm, wc) = compute_weights(config);
    let predicted_mean = compute_mean(&propagated_sigma_points, &wm);
    let predicted_cov = compute_covariance(&propagated_sigma_points, &predicted_mean, &wc);

    // Add process noise
    let process_noise = Mat9x9::from_diagonal(config.process_noise);
    let predicted_cov_with_noise = predicted_cov.add(&process_noise);

    *state = predicted_mean;
    *covariance = predicted_cov_with_noise;
}

/// Update step for odometry measurements
pub fn update_odom(
    state: &mut State9,
    covariance: &mut Mat9x9,
    linear_x: f32,
    linear_y: f32,
    angular_z: f32,
    config: &SlamConfig,
) {
    // Measurement model: odometry directly measures velocity states
    let measurement = [linear_x, linear_y, angular_z];

    // Generate sigma points
    let sigma_points = generate_sigma_points(state, covariance, config);

    // Transform sigma points through measurement model
    let mut predicted_measurements = [[0.0; 3]; 19];
    for (i, sp) in sigma_points.iter().enumerate() {
        predicted_measurements[i] = [sp.linear_vel.x, sp.linear_vel.y, sp.angular_vel];
    }

    // Compute predicted measurement mean
    let (wm, wc) = compute_weights(config);
    let mut z_mean = [0.0; 3];
    for i in 0..19 {
        for j in 0..3 {
            z_mean[j] += wm[i] * predicted_measurements[i][j];
        }
    }

    // Compute innovation covariance
    let mut pzz = [[0.0; 3]; 3];
    for i in 0..19 {
        for j in 0..3 {
            for k in 0..3 {
                let diff_j = predicted_measurements[i][j] - z_mean[j];
                let diff_k = predicted_measurements[i][k] - z_mean[k];
                pzz[j][k] += wc[i] * diff_j * diff_k;
            }
        }
    }

    // Add measurement noise
    for i in 0..3 {
        pzz[i][i] += config.measurement_noise_odom[i];
    }

    // Compute cross-covariance
    let mut pxz = [[0.0; 3]; 9];
    for i in 0..19 {
        let state_diff = sigma_points[i].sub(state);
        let state_diff_arr = state_diff.to_array();
        for j in 0..9 {
            for k in 0..3 {
                let meas_diff = predicted_measurements[i][k] - z_mean[k];
                pxz[j][k] += wc[i] * state_diff_arr[j] * meas_diff;
            }
        }
    }

    // Compute Kalman gain: K = Pxz * inv(Pzz)
    // Use simple 3x3 matrix inversion
    let inv_pzz = invert_3x3(&pzz);

    let mut kalman_gain = [[0.0; 3]; 9];
    for i in 0..9 {
        for j in 0..3 {
            for k in 0..3 {
                kalman_gain[i][j] += pxz[i][k] * inv_pzz[k][j];
            }
        }
    }

    // Innovation
    let innovation = [
        measurement[0] - z_mean[0],
        measurement[1] - z_mean[1],
        measurement[2] - z_mean[2],
    ];

    // Update state
    let mut state_arr = state.to_array();
    for i in 0..9 {
        for j in 0..3 {
            state_arr[i] += kalman_gain[i][j] * innovation[j];
        }
    }
    *state = State9::from_array(state_arr);

    // Update covariance: P = P - K * Pzz * K^T
    let mut k_pzz = [[0.0; 3]; 9];
    for i in 0..9 {
        for j in 0..3 {
            for k in 0..3 {
                k_pzz[i][j] += kalman_gain[i][k] * pzz[k][j];
            }
        }
    }

    for i in 0..9 {
        for j in 0..9 {
            for k in 0..3 {
                covariance.data[i][j] -= k_pzz[i][k] * kalman_gain[j][k];
            }
        }
    }
}

/// Update step for IMU measurements
pub fn update_imu(
    state: &mut State9,
    covariance: &mut Mat9x9,
    accel: Vec3,
    gyro: Vec3,
    config: &SlamConfig,
) {
    // Measurement model: IMU measures acceleration and gyro (with biases)
    // Simplified: we only use the z-component of acceleration and gyroscope
    let measurement = [accel.x, accel.y, gyro.z];

    // Generate sigma points
    let sigma_points = generate_sigma_points(state, covariance, config);

    // Transform sigma points through measurement model
    let mut predicted_measurements = [[0.0; 3]; 19];
    for (i, sp) in sigma_points.iter().enumerate() {
        // Predict accelerometer reading (with bias)
        predicted_measurements[i] = [
            sp.accel_bias.x,
            sp.accel_bias.y,
            sp.gyro_bias,
        ];
    }

    // Compute predicted measurement mean
    let (wm, wc) = compute_weights(config);
    let mut z_mean = [0.0; 3];
    for i in 0..19 {
        for j in 0..3 {
            z_mean[j] += wm[i] * predicted_measurements[i][j];
        }
    }

    // Compute innovation covariance
    let mut pzz = [[0.0; 3]; 3];
    for i in 0..19 {
        for j in 0..3 {
            for k in 0..3 {
                let diff_j = predicted_measurements[i][j] - z_mean[j];
                let diff_k = predicted_measurements[i][k] - z_mean[k];
                pzz[j][k] += wc[i] * diff_j * diff_k;
            }
        }
    }

    // Add measurement noise
    for i in 0..3 {
        pzz[i][i] += config.measurement_noise_imu[i];
    }

    // Compute cross-covariance
    let mut pxz = [[0.0; 3]; 9];
    for i in 0..19 {
        let state_diff = sigma_points[i].sub(state);
        let state_diff_arr = state_diff.to_array();
        for j in 0..9 {
            for k in 0..3 {
                let meas_diff = predicted_measurements[i][k] - z_mean[k];
                pxz[j][k] += wc[i] * state_diff_arr[j] * meas_diff;
            }
        }
    }

    // Compute Kalman gain
    let inv_pzz = invert_3x3(&pzz);

    let mut kalman_gain = [[0.0; 3]; 9];
    for i in 0..9 {
        for j in 0..3 {
            for k in 0..3 {
                kalman_gain[i][j] += pxz[i][k] * inv_pzz[k][j];
            }
        }
    }

    // Innovation
    let innovation = [
        measurement[0] - z_mean[0],
        measurement[1] - z_mean[1],
        measurement[2] - z_mean[2],
    ];

    // Update state
    let mut state_arr = state.to_array();
    for i in 0..9 {
        for j in 0..3 {
            state_arr[i] += kalman_gain[i][j] * innovation[j];
        }
    }
    *state = State9::from_array(state_arr);

    // Update covariance
    let mut k_pzz = [[0.0; 3]; 9];
    for i in 0..9 {
        for j in 0..3 {
            for k in 0..3 {
                k_pzz[i][j] += kalman_gain[i][k] * pzz[k][j];
            }
        }
    }

    for i in 0..9 {
        for j in 0..9 {
            for k in 0..3 {
                covariance.data[i][j] -= k_pzz[i][k] * kalman_gain[j][k];
            }
        }
    }
}

/// Update step for GPS measurements
pub fn update_gps(
    state: &mut State9,
    covariance: &mut Mat9x9,
    position: Vec2,
    accuracy: f32,
    config: &SlamConfig,
) {
    // Measurement model: GPS directly measures position
    let measurement = [position.x, position.y];

    // Generate sigma points
    let sigma_points = generate_sigma_points(state, covariance, config);

    // Transform sigma points through measurement model
    let mut predicted_measurements = [[0.0; 2]; 19];
    for (i, sp) in sigma_points.iter().enumerate() {
        predicted_measurements[i] = [sp.position.x, sp.position.y];
    }

    // Compute predicted measurement mean
    let (wm, wc) = compute_weights(config);
    let mut z_mean = [0.0; 2];
    for i in 0..19 {
        for j in 0..2 {
            z_mean[j] += wm[i] * predicted_measurements[i][j];
        }
    }

    // Compute innovation covariance
    let mut pzz = [[0.0; 2]; 2];
    for i in 0..19 {
        for j in 0..2 {
            for k in 0..2 {
                let diff_j = predicted_measurements[i][j] - z_mean[j];
                let diff_k = predicted_measurements[i][k] - z_mean[k];
                pzz[j][k] += wc[i] * diff_j * diff_k;
            }
        }
    }

    // Add measurement noise (scaled by accuracy)
    let noise_scale = (accuracy / 1.0).max(0.1); // Normalize around 1m accuracy
    for i in 0..2 {
        pzz[i][i] += config.measurement_noise_gps[i] * noise_scale;
    }

    // Compute cross-covariance
    let mut pxz = [[0.0; 2]; 9];
    for i in 0..19 {
        let state_diff = sigma_points[i].sub(state);
        let state_diff_arr = state_diff.to_array();
        for j in 0..9 {
            for k in 0..2 {
                let meas_diff = predicted_measurements[i][k] - z_mean[k];
                pxz[j][k] += wc[i] * state_diff_arr[j] * meas_diff;
            }
        }
    }

    // Compute Kalman gain: K = Pxz * inv(Pzz)
    let inv_pzz = invert_2x2(&pzz);

    let mut kalman_gain = [[0.0; 2]; 9];
    for i in 0..9 {
        for j in 0..2 {
            for k in 0..2 {
                kalman_gain[i][j] += pxz[i][k] * inv_pzz[k][j];
            }
        }
    }

    // Innovation
    let innovation = [measurement[0] - z_mean[0], measurement[1] - z_mean[1]];

    // Update state
    let mut state_arr = state.to_array();
    for i in 0..9 {
        for j in 0..2 {
            state_arr[i] += kalman_gain[i][j] * innovation[j];
        }
    }
    *state = State9::from_array(state_arr);

    // Update covariance: P = P - K * Pzz * K^T
    let mut k_pzz = [[0.0; 2]; 9];
    for i in 0..9 {
        for j in 0..2 {
            for k in 0..2 {
                k_pzz[i][j] += kalman_gain[i][k] * pzz[k][j];
            }
        }
    }

    for i in 0..9 {
        for j in 0..9 {
            for k in 0..2 {
                covariance.data[i][j] -= k_pzz[i][k] * kalman_gain[j][k];
            }
        }
    }
}

/// Helper: Invert a 2x2 matrix
fn invert_2x2(mat: &[[f32; 2]; 2]) -> [[f32; 2]; 2] {
    let det = mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0];
    if det.abs() < 1e-10 {
        // Return identity if matrix is singular
        return [[1.0, 0.0], [0.0, 1.0]];
    }
    let inv_det = 1.0 / det;
    [
        [mat[1][1] * inv_det, -mat[0][1] * inv_det],
        [-mat[1][0] * inv_det, mat[0][0] * inv_det],
    ]
}

/// Helper: Invert a 3x3 matrix using cofactor method
fn invert_3x3(mat: &[[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let det = mat[0][0] * (mat[1][1] * mat[2][2] - mat[1][2] * mat[2][1])
        - mat[0][1] * (mat[1][0] * mat[2][2] - mat[1][2] * mat[2][0])
        + mat[0][2] * (mat[1][0] * mat[2][1] - mat[1][1] * mat[2][0]);

    if det.abs() < 1e-10 {
        // Return identity if matrix is singular
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    }

    let inv_det = 1.0 / det;

    [
        [
            (mat[1][1] * mat[2][2] - mat[1][2] * mat[2][1]) * inv_det,
            (mat[0][2] * mat[2][1] - mat[0][1] * mat[2][2]) * inv_det,
            (mat[0][1] * mat[1][2] - mat[0][2] * mat[1][1]) * inv_det,
        ],
        [
            (mat[1][2] * mat[2][0] - mat[1][0] * mat[2][2]) * inv_det,
            (mat[0][0] * mat[2][2] - mat[0][2] * mat[2][0]) * inv_det,
            (mat[0][2] * mat[1][0] - mat[0][0] * mat[1][2]) * inv_det,
        ],
        [
            (mat[1][0] * mat[2][1] - mat[1][1] * mat[2][0]) * inv_det,
            (mat[0][1] * mat[2][0] - mat[0][0] * mat[2][1]) * inv_det,
            (mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]) * inv_det,
        ],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_process_model() {
        let state = State9 {
            position: Vec2::ZERO,
            heading: 0.0,
            linear_vel: Vec2::new(1.0, 0.0),
            angular_vel: 0.0,
            ..Default::default()
        };

        let next_state = process_model(&state, 1.0);

        // After 1 second at 1 m/s, position should be (1, 0)
        assert_relative_eq!(next_state.position.x, 1.0, epsilon = 0.01);
        assert_relative_eq!(next_state.position.y, 0.0, epsilon = 0.01);
    }

    #[test]
    fn test_predict() {
        let mut state = State9::default();
        state.linear_vel = Vec2::new(1.0, 0.0);
        let mut cov = Mat9x9::identity();
        let config = SlamConfig::default();

        predict(&mut state, &mut cov, 0.1, &config);

        // State should have moved
        assert!(state.position.x > 0.0);
    }

    #[test]
    fn test_update_odom() {
        let mut state = State9::default();
        let mut cov = Mat9x9::identity();
        let config = SlamConfig::default();

        update_odom(&mut state, &mut cov, 1.0, 0.0, 0.0, &config);

        // Velocity should be updated towards measurement
        assert!(state.linear_vel.x > 0.0);
    }

    #[test]
    fn test_update_gps() {
        let mut state = State9::default();
        let mut cov = Mat9x9::identity();
        let config = SlamConfig::default();

        let gps_pos = Vec2::new(10.0, 5.0);
        update_gps(&mut state, &mut cov, gps_pos, 1.0, &config);

        // Position should move towards GPS measurement
        assert!(state.position.x > 0.0);
        assert!(state.position.y > 0.0);
    }

    #[test]
    fn test_invert_2x2() {
        let mat = [[4.0, 2.0], [3.0, 1.0]];
        let inv = invert_2x2(&mat);

        // Check A * A^-1 ≈ I
        let i00 = mat[0][0] * inv[0][0] + mat[0][1] * inv[1][0];
        let i11 = mat[1][0] * inv[0][1] + mat[1][1] * inv[1][1];

        assert_relative_eq!(i00, 1.0, epsilon = 0.01);
        assert_relative_eq!(i11, 1.0, epsilon = 0.01);
    }

    #[test]
    fn test_invert_3x3() {
        let mat = [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]];
        let inv = invert_3x3(&mat);

        assert_relative_eq!(inv[0][0], 1.0, epsilon = 0.01);
        assert_relative_eq!(inv[1][1], 0.5, epsilon = 0.01);
        assert_relative_eq!(inv[2][2], 1.0 / 3.0, epsilon = 0.01);
    }
}

