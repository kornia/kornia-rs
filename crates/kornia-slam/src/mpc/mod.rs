use crate::trajectory::Trajectory;
use crate::types::{SlamConfig, State9};
use glam::Vec2;

/// Compute control command using Model Predictive Control
pub fn compute_control(
    state: &State9,
    trajectory: &Trajectory,
    config: &SlamConfig,
) -> Option<(f32, f32)> {
    // Use shorter look-ahead for more precise tracking
    let current_speed = state.linear_vel.length();
    let base_look_ahead = config.mpc_dt * 3.0; // Reduced from 5.0 for tighter following
    let look_ahead = (base_look_ahead + current_speed * 0.3).max(0.5).min(5.0);
    
    let (ref_point, _ref_heading) = trajectory.get_reference_point(state.position, look_ahead)?;

    // Compute errors
    let position_error = ref_point - state.position;
    let distance_error = position_error.length();

    // Compute desired heading to target
    let desired_heading = position_error.y.atan2(position_error.x);
    let current_heading = state.heading;
    
    // Normalize heading error to [-pi, pi]
    let mut heading_to_target = desired_heading - current_heading;
    while heading_to_target > std::f32::consts::PI {
        heading_to_target -= 2.0 * std::f32::consts::PI;
    }
    while heading_to_target < -std::f32::consts::PI {
        heading_to_target += 2.0 * std::f32::consts::PI;
    }

    // Enhanced controller with better corner handling
    
    // Detect if we're approaching a corner (large heading error)
    let is_corner = heading_to_target.abs() > std::f32::consts::PI / 8.0; // > 22.5 degrees
    let is_sharp_corner = heading_to_target.abs() > std::f32::consts::PI / 4.0; // > 45 degrees
    
    // Velocity gain - balanced for accurate tracking
    let k_v = if is_sharp_corner {
        2.5 // Moderate for sharp corners to avoid overshoot
    } else if is_corner {
        2.8
    } else {
        3.0 // Standard tracking
    };
    
    // Heading-dependent speed factor
    let heading_factor = if is_sharp_corner {
        // At sharp corners, prioritize turning over speed
        0.15
    } else if is_corner {
        0.4
    } else {
        // Good alignment: allow higher speed
        (1.0 - heading_to_target.abs() / std::f32::consts::PI).max(0.6)
    };
    
    // Distance-based velocity
    let distance_factor = if distance_error < 0.5 {
        distance_error / 0.5  // Slow down very close to target
    } else if distance_error > 3.0 {
        1.0  // Full speed far from target
    } else {
        0.5 + 0.5 * (distance_error / 3.0)  // Gradual increase
    };
    
    let mut linear_vel = k_v * distance_error.min(3.0) * heading_factor * distance_factor;

    // Angular velocity: strong but not too aggressive
    let k_w = if is_sharp_corner {
        4.5  // Strong turning at sharp corners
    } else if is_corner {
        4.0
    } else {
        3.5
    };
    
    let mut angular_vel = k_w * heading_to_target;
    
    // Add derivative term for smoother control (reduced for tighter tracking)
    let angular_vel_error = angular_vel - state.angular_vel;
    angular_vel += 0.05 * angular_vel_error;

    // Apply constraints
    linear_vel = linear_vel.clamp(-config.max_linear_vel, config.max_linear_vel);
    angular_vel = angular_vel.clamp(-config.max_angular_vel, config.max_angular_vel);

    // At very sharp corners, allow pure rotation
    if is_sharp_corner && distance_error < 2.0 {
        linear_vel *= 0.1;  // Almost stop to rotate in place
    }
    
    // Ensure minimum forward progress when not at a corner
    if !is_corner && distance_error > 0.2 && linear_vel.abs() < 0.15 {
        linear_vel = 0.15;
    }

    Some((linear_vel, angular_vel))
}

/// Simplified optimization-based MPC (gradient descent)
#[allow(dead_code)]
fn optimize_trajectory(
    initial_state: &State9,
    reference_trajectory: &[Vec2],
    horizon: usize,
    dt: f32,
    q_weights: &[f32; 3],
    r_weights: &[f32; 2],
) -> Vec<(f32, f32)> {
    // Initialize control sequence with zeros
    let mut controls = vec![(0.0f32, 0.0f32); horizon];

    // Gradient descent iterations
    let num_iterations = 10;
    let learning_rate = 0.01;

    for _ in 0..num_iterations {
        // Forward simulate to get predicted trajectory
        let mut predicted_states = vec![*initial_state];
        for &(v, w) in &controls {
            let last_state = predicted_states.last().unwrap();
            let next_state = simulate_step(last_state, v, w, dt);
            predicted_states.push(next_state);
        }

        // Compute cost and gradients
        let mut grad = vec![(0.0f32, 0.0f32); horizon];

        for t in 0..horizon {
            let state = &predicted_states[t];
            let ref_pos = reference_trajectory
                .get(t)
                .unwrap_or(reference_trajectory.last().unwrap());

            // Position error gradient
            let pos_error = state.position - *ref_pos;
            let heading_error = state.heading;

            // Simplified gradient (would need proper backpropagation)
            let dv = pos_error.length() * q_weights[0] + controls[t].0 * r_weights[0];
            let dw = heading_error * q_weights[2] + controls[t].1 * r_weights[1];

            grad[t] = (dv, dw);
        }

        // Update controls
        for t in 0..horizon {
            controls[t].0 -= learning_rate * grad[t].0;
            controls[t].1 -= learning_rate * grad[t].1;

            // Apply constraints
            controls[t].0 = controls[t].0.clamp(-2.5, 2.5);
            controls[t].1 = controls[t].1.clamp(-std::f32::consts::PI, std::f32::consts::PI);
        }
    }

    controls
}

/// Simulate one step of robot dynamics
fn simulate_step(state: &State9, v: f32, w: f32, dt: f32) -> State9 {
    let mut next_state = *state;

    // Update heading
    next_state.heading += w * dt;

    // Update position based on current heading
    let cos_theta = next_state.heading.cos();
    let sin_theta = next_state.heading.sin();

    next_state.position.x += v * cos_theta * dt;
    next_state.position.y += v * sin_theta * dt;

    // Update velocities
    next_state.linear_vel = Vec2::new(v, 0.0);
    next_state.angular_vel = w;

    next_state
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_compute_control_straight_path() {
        let state = State9 {
            position: Vec2::new(0.0, 0.0),
            heading: 0.0,
            ..Default::default()
        };

        let waypoints = vec![Vec2::new(0.0, 0.0), Vec2::new(10.0, 0.0)];
        let trajectory = Trajectory::new(waypoints);
        let config = SlamConfig::default();

        let control = compute_control(&state, &trajectory, &config);
        assert!(control.is_some());

        let (linear_vel, angular_vel) = control.unwrap();
        // Should move forward
        assert!(linear_vel > 0.0);
        // Angular velocity should be small for straight path
        assert!(angular_vel.abs() < 0.5);
    }

    #[test]
    fn test_compute_control_turn() {
        let state = State9 {
            position: Vec2::new(0.0, 0.0),
            heading: 0.0, // Facing right
            ..Default::default()
        };

        // Target is behind and to the left
        let waypoints = vec![Vec2::new(0.0, 0.0), Vec2::new(-5.0, 5.0)];
        let trajectory = Trajectory::new(waypoints);
        let config = SlamConfig::default();

        let control = compute_control(&state, &trajectory, &config);
        assert!(control.is_some());

        let (_linear_vel, angular_vel) = control.unwrap();
        // Should have significant angular velocity to turn
        assert!(angular_vel.abs() > 0.1);
    }

    #[test]
    fn test_compute_control_respects_limits() {
        let state = State9 {
            position: Vec2::new(0.0, 0.0),
            heading: 0.0,
            ..Default::default()
        };

        let waypoints = vec![Vec2::new(0.0, 0.0), Vec2::new(100.0, 100.0)];
        let trajectory = Trajectory::new(waypoints);
        let config = SlamConfig::default();

        let control = compute_control(&state, &trajectory, &config);
        assert!(control.is_some());

        let (linear_vel, angular_vel) = control.unwrap();
        // Should respect velocity limits
        assert!(linear_vel.abs() <= config.max_linear_vel);
        assert!(angular_vel.abs() <= config.max_angular_vel);
    }

    #[test]
    fn test_simulate_step() {
        let state = State9 {
            position: Vec2::new(0.0, 0.0),
            heading: 0.0,
            ..Default::default()
        };

        let next = simulate_step(&state, 1.0, 0.0, 1.0);

        // After 1 second at 1 m/s forward, should be at (1, 0)
        assert_relative_eq!(next.position.x, 1.0, epsilon = 0.01);
        assert_relative_eq!(next.position.y, 0.0, epsilon = 0.01);
    }

    #[test]
    fn test_simulate_step_rotation() {
        let state = State9 {
            position: Vec2::new(0.0, 0.0),
            heading: 0.0,
            ..Default::default()
        };

        let omega = std::f32::consts::PI / 2.0; // 90 degrees per second
        let next = simulate_step(&state, 0.0, omega, 1.0);

        // After 1 second, heading should be pi/2
        assert_relative_eq!(next.heading, std::f32::consts::PI / 2.0, epsilon = 0.01);
    }

    #[test]
    fn test_optimize_trajectory_basic() {
        let state = State9 {
            position: Vec2::new(0.0, 0.0),
            heading: 0.0,
            ..Default::default()
        };

        let reference = vec![
            Vec2::new(0.5, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(1.5, 0.0),
        ];

        let controls = optimize_trajectory(
            &state,
            &reference,
            3,
            0.1,
            &[1.0, 1.0, 0.5],
            &[0.1, 0.1],
        );

        assert_eq!(controls.len(), 3);
        // All controls should be within limits
        for (v, w) in controls {
            assert!(v.abs() <= 2.5);
            assert!(w.abs() <= std::f32::consts::PI);
        }
    }
}

