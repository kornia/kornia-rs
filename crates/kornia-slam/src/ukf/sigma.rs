use crate::types::{Mat9x9, State9, SlamConfig};

/// Generate sigma points for UKF
pub fn generate_sigma_points(
    state: &State9,
    covariance: &Mat9x9,
    config: &SlamConfig,
) -> [State9; 19] {
    let n = 9.0;
    let alpha = config.ukf_alpha;
    let kappa = config.ukf_kappa;
    let lambda = alpha * alpha * (n + kappa) - n;

    // Compute matrix square root (Cholesky decomposition)
    let scale = n + lambda;
    let scaled_cov = covariance.scale(scale);
    
    let sqrt_cov = match scaled_cov.cholesky() {
        Some(l) => l,
        None => {
            // Fallback: use identity if Cholesky fails
            Mat9x9::identity().scale(scale.sqrt())
        }
    };

    let mut sigma_points = [State9::default(); 19];

    // First sigma point is the mean
    sigma_points[0] = *state;

    // Generate remaining 2n sigma points
    let state_arr = state.to_array();

    for i in 0..9 {
        // Extract column i from sqrt_cov
        let mut col = [0.0; 9];
        for j in 0..9 {
            col[j] = sqrt_cov.data[j][i];
        }

        // Positive sigma point
        let mut pos_arr = state_arr;
        for j in 0..9 {
            pos_arr[j] += col[j];
        }
        sigma_points[i + 1] = State9::from_array(pos_arr);

        // Negative sigma point
        let mut neg_arr = state_arr;
        for j in 0..9 {
            neg_arr[j] -= col[j];
        }
        sigma_points[i + 10] = State9::from_array(neg_arr);
    }

    sigma_points
}

/// Compute weights for sigma points
pub fn compute_weights(config: &SlamConfig) -> (Vec<f32>, Vec<f32>) {
    let n = 9.0;
    let alpha = config.ukf_alpha;
    let beta = config.ukf_beta;
    let kappa = config.ukf_kappa;
    let lambda = alpha * alpha * (n + kappa) - n;

    let mut wm = vec![0.0; 19];
    let mut wc = vec![0.0; 19];

    // Weight for mean sigma point
    wm[0] = lambda / (n + lambda);
    wc[0] = lambda / (n + lambda) + (1.0 - alpha * alpha + beta);

    // Weights for other sigma points
    let weight = 1.0 / (2.0 * (n + lambda));
    for i in 1..19 {
        wm[i] = weight;
        wc[i] = weight;
    }

    (wm, wc)
}

/// Compute weighted mean of sigma points
pub fn compute_mean(sigma_points: &[State9; 19], weights: &[f32]) -> State9 {
    let mut mean_arr = [0.0; 9];

    for (i, sp) in sigma_points.iter().enumerate() {
        let sp_arr = sp.to_array();
        for j in 0..9 {
            mean_arr[j] += weights[i] * sp_arr[j];
        }
    }

    State9::from_array(mean_arr)
}

/// Compute weighted covariance from sigma points
pub fn compute_covariance(
    sigma_points: &[State9; 19],
    mean: &State9,
    weights: &[f32],
) -> Mat9x9 {
    let mut cov = Mat9x9::zeros();

    for (i, sp) in sigma_points.iter().enumerate() {
        let diff = sp.sub(mean);
        let diff_arr = diff.to_array();

        for j in 0..9 {
            for k in 0..9 {
                cov.data[j][k] += weights[i] * diff_arr[j] * diff_arr[k];
            }
        }
    }

    cov
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use glam::Vec2;

    #[test]
    fn test_generate_sigma_points() {
        let state = State9::default();
        let cov = Mat9x9::identity();
        let config = SlamConfig::default();

        let sigma_points = generate_sigma_points(&state, &cov, &config);

        // Should have 19 sigma points (2n+1 where n=9)
        assert_eq!(sigma_points.len(), 19);

        // First sigma point should be the mean
        assert_relative_eq!(sigma_points[0].position.x, state.position.x);
    }

    #[test]
    fn test_compute_weights() {
        let config = SlamConfig::default();
        let (wm, wc) = compute_weights(&config);

        // Should have 19 weights
        assert_eq!(wm.len(), 19);
        assert_eq!(wc.len(), 19);

        // Mean weights should sum to 1 (for unbiased estimation)
        let sum_wm: f32 = wm.iter().sum();
        assert_relative_eq!(sum_wm, 1.0, epsilon = 1e-4);
        
        // Covariance weights don't need to sum to 1, but should be reasonable
        // (they account for higher order moments)
        let sum_wc: f32 = wc.iter().sum();
        assert!(sum_wc > 0.0);
        assert!(sum_wc < 10.0); // Sanity check
    }

    #[test]
    fn test_compute_mean() {
        let config = SlamConfig::default();
        let (wm, _) = compute_weights(&config);

        // Create simple sigma points around origin
        let mut sigma_points = [State9::default(); 19];
        sigma_points[0].position = Vec2::new(1.0, 1.0);
        for i in 1..19 {
            sigma_points[i].position = Vec2::new(1.0, 1.0);
        }

        let mean = compute_mean(&sigma_points, &wm);

        // Mean should be close to (1, 1)
        assert_relative_eq!(mean.position.x, 1.0, epsilon = 0.1);
        assert_relative_eq!(mean.position.y, 1.0, epsilon = 0.1);
    }

    #[test]
    fn test_compute_covariance() {
        let config = SlamConfig::default();
        let (_, wc) = compute_weights(&config);

        let mut sigma_points = [State9::default(); 19];
        let mean = State9::default();

        // Create sigma points with some spread
        for i in 1..10 {
            sigma_points[i].position = Vec2::new(0.1, 0.0);
        }
        for i in 10..19 {
            sigma_points[i].position = Vec2::new(-0.1, 0.0);
        }

        let cov = compute_covariance(&sigma_points, &mean, &wc);

        // Covariance should be non-zero
        assert!(cov.data[0][0] > 0.0);
    }

    #[test]
    fn test_sigma_point_mean_recovery() {
        let state = State9 {
            position: Vec2::new(5.0, 3.0),
            heading: 0.5,
            linear_vel: Vec2::new(1.0, 0.0),
            angular_vel: 0.1,
            accel_bias: Vec2::ZERO,
            gyro_bias: 0.0,
        };
        let cov = Mat9x9::identity();
        let config = SlamConfig::default();

        let sigma_points = generate_sigma_points(&state, &cov, &config);
        let (wm, _) = compute_weights(&config);
        let recovered_mean = compute_mean(&sigma_points, &wm);

        // Recovered mean should match original state
        assert_relative_eq!(recovered_mean.position.x, state.position.x, epsilon = 1e-3);
        assert_relative_eq!(recovered_mean.position.y, state.position.y, epsilon = 1e-3);
        assert_relative_eq!(recovered_mean.heading, state.heading, epsilon = 1e-3);
    }
}

