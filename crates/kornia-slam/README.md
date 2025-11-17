# kornia-slam

A sensor fusion and trajectory following library for mobile robots using Unscented Kalman Filter (UKF) and Model Predictive Control (MPC).

## Features

- **Unscented Kalman Filter (UKF)** for sensor fusion
  - Odometry measurements (twist velocities)
  - IMU measurements (accelerometer and gyroscope)
  - GPS measurements (position)
  - Estimates 9D state: position, heading, velocities, and sensor biases

- **Model Predictive Control (MPC)** for trajectory following
  - Pure pursuit-style reference tracking
  - Velocity and heading control
  - Configurable constraints and weights

- **Pure Rust implementation** using only `glam` for math operations
  - No dependencies on heavy linear algebra libraries
  - Simple and efficient matrix operations

## Usage

```rust
use kornia_slam::{SlamFilter, SlamConfig, Vec2, Vec3};

// Create a SLAM filter with default configuration
let config = SlamConfig::default();
let mut slam = SlamFilter::new(config);

// Handle sensor measurements
slam.handle_odom(linear_x, linear_y, angular_z, dt);
slam.handle_imu(accel, gyro, dt);
slam.handle_gps(position, accuracy);

// Set a trajectory to follow
let waypoints = vec![
    Vec2::new(0.0, 0.0),
    Vec2::new(10.0, 0.0),
    Vec2::new(10.0, 10.0),
];
slam.set_trajectory(&waypoints);

// Compute control commands
if let Some((linear_vel, angular_vel)) = slam.compute_control() {
    // Send commands to robot actuators
    println!("v: {}, ω: {}", linear_vel, angular_vel);
}

// Get the estimated state
let state = slam.get_state();
println!("Position: {:?}", state.position);
println!("Heading: {}", state.heading);
```

## Configuration

The `SlamConfig` struct allows customization of:

- UKF parameters (alpha, beta, kappa)
- Process noise covariance
- Measurement noise covariances (odometry, IMU, GPS)
- MPC horizon and time step
- MPC cost function weights
- Velocity constraints

## Testing

Run the test suite:

```bash
cargo test
```

All tests include:
- Unit tests for UKF sigma point generation
- Unit tests for state prediction and measurement updates
- Unit tests for trajectory interpolation
- Unit tests for MPC control computation
- Integration tests for the complete SLAM filter

## Architecture

```
kornia-slam/
├── src/
│   ├── lib.rs              # Public API and SlamFilter
│   ├── types.rs            # State9, Mat9x9, SlamConfig
│   ├── ukf/
│   │   ├── mod.rs          # UKF predict/update
│   │   └── sigma.rs        # Sigma point generation
│   ├── mpc/
│   │   └── mod.rs          # MPC controller
│   └── trajectory.rs       # Path interpolation
```

## License

Apache-2.0

## Contributing

Contributions are welcome! This library is designed to be simple and focused on mobile robot navigation.

