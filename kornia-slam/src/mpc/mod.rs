use crate::trajectory::Trajectory;
use crate::types::{SlamConfig, State9};
use glam::Vec2;

/// Compute control command using Model Predictive Control
pub fn compute_control(
    state: &State9,
    trajectory: &Trajectory,
    config: &SlamConfig,
) -> Option<(f32, f32)> {
    // Get reference point with look-ahead distance
    let look_ahead = config.mpc_dt * 5.0; // Look ahead 5 time steps
    let (ref_point, ref_heading) = trajectory.get_reference_point(state.position, look_ahead)?;

    // Compute errors
    let position_error = ref_point - state.position;
    let distance_error = position_error.length();

    // Heading error (normalize to [-pi, pi])
    let current_heading = state.heading;
    let mut heading_error = ref_heading - current_heading;
    while heading_error > std::f32::consts::PI {
        heading_error -= 2.0 * std::f32::consts::PI;
    }
    while heading_error < -std::f32::consts::PI {
        heading_error += 2.0 * std::f32::consts::PI;
    }

    // Compute desired heading to target
    let desired_heading = position_error.y.atan2(position_error.x);
    let mut heading_to_target = desired_heading - current_heading;
    while heading_to_target > std::f32::consts::PI {
        heading_to_target -= 2.0 * std::f32::consts::PI;
    }
    while heading_to_target < -std::f32::consts::PI {
        heading_to_target += 2.0 * std::f32::consts::PI;
    }

    // Simple proportional controller for now (simplified MPC)
    // In a full MPC, we would solve an optimization problem over the horizon

    // Linear velocity: proportional to distance error, reduced when heading is off
    let k_v = 1.0; // Proportional gain for velocity
    let heading_factor = (1.0 - heading_to_target.abs() / std::f32::consts::PI).max(0.1);
    let mut linear_vel = k_v * distance_error * heading_factor;

    // Angular velocity: proportional to heading error
    let k_w = 2.0; // Proportional gain for angular velocity
    let mut angular_vel = k_w * heading_to_target;

    // Apply constraints
    linear_vel = linear_vel.clamp(-config.max_linear_vel, config.max_linear_vel);
    angular_vel = angular_vel.clamp(-config.max_angular_vel, config.max_angular_vel);

    // If very close to target, reduce velocity
    if distance_error < 0.5 {
        linear_vel *= distance_error / 0.5;
    }

    // If heading error is large, prioritize turning
    if heading_to_target.abs() > std::f32::consts::PI / 4.0 {
        linear_vel *= 0.3;
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

