# kornia-slam Examples

This document describes the realistic examples and test scenarios included with kornia-slam.

## Examples

### 1. Simple SLAM (`simple_slam.rs`)

Basic usage demonstration showing:
- Creating a SLAM filter
- Handling odometry, IMU, and GPS measurements
- Setting a trajectory
- Computing control commands

**Run:**
```bash
cargo run --example simple_slam
```

### 2. Realistic Navigation (`realistic_navigation.rs`)

Simulates navigating a 50m x 30m field boundary with:
- Realistic sensor noise simulation
- GPS updates at 2 Hz
- IMU updates at 10 Hz
- Odometry at 10 Hz
- Total mission time: ~120 seconds
- Distance: ~160m perimeter

**Features:**
- Simulated sensor noise
- Multi-leg navigation tracking
- Distance and speed metrics
- Mission completion evaluation

**Run:**
```bash
cargo run --example realistic_navigation --release
```

**Expected Output:**
- Position updates every 10 seconds
- Current navigation leg status
- Final position error and statistics

### 3. Figure-8 Trajectory (`figure_eight.rs`)

Demonstrates complex path following with:
- Two circular paths forming a figure-8
- 32 waypoints
- Radius: 5m per circle
- Total path: ~63m
- Tests tight turning capabilities

**Features:**
- Maximum heading rate tracking
- Trajectory smoothness analysis
- Circle detection (left vs right)

**Run:**
```bash
cargo run --example figure_eight --release
```

### 4. Parking Maneuver (`parking_maneuver.rs`)

Autonomous parking scenario:
- Parking spot: 3m x 5m
- Approach from lane
- Turn into spot
- Precise positioning
- Low speed control (0.5 m/s max)
- High-frequency control (20 Hz)

**Features:**
- Phase tracking (approach, turning, entering, parked)
- Distance to goal monitoring
- Automatic stop when parked
- Position accuracy evaluation

**Run:**
```bash
cargo run --example parking_maneuver --release
```

## Test Scenarios

Run all tests with:
```bash
cargo test
```

### Unit Tests (33 tests)

Located in `src/`, these test individual components:
- Sigma point generation
- UKF prediction and updates
- Matrix operations
- Trajectory interpolation
- MPC control computation

### Integration Tests (9 tests)

Located in `tests/realistic_scenarios.rs`:

1. **Straight Line Navigation** - Basic forward motion
2. **90-Degree Turn** - Sharp turning capability
3. **Square Trajectory** - Complete loop with four 90° turns
4. **Circular Path** - Continuous curved motion
5. **S-Curve** - Lane change maneuver
6. **Reverse and Forward** - Bidirectional motion
7. **GPS Corrections** - Drift prevention with GPS
8. **Field Boundary** - Agricultural robot perimeter navigation
9. **Obstacle Avoidance** - C-shaped path around obstacle

## Real-World Scenarios Modeled

### Agricultural Robotics
- **Field boundary navigation**: Perimeter tracking around crops
- **Row navigation**: Straight-line following between crop rows
- **GPS-guided operation**: RTK-GPS corrected positioning

### Autonomous Parking
- **Parallel parking**: Multi-step maneuver with precise positioning
- **Low-speed control**: Sub 0.5 m/s velocity control
- **Tight tolerances**: < 0.5m position accuracy

### Warehouse Navigation
- **Path following**: Predefined routes between stations
- **Sharp turns**: 90-degree corners in aisles
- **Obstacle detours**: C-shaped and S-curved paths

### Performance Testing
- **Figure-8 paths**: Continuous curved motion testing
- **Closed loops**: Return-to-start accuracy measurement
- **Long-distance**: Drift accumulation over 100m+ paths

## Typical Performance

Based on the included tests and examples:

| Metric | Value |
|--------|-------|
| Control frequency | 10-20 Hz |
| GPS update rate | 2-5 Hz |
| IMU update rate | 10 Hz |
| Max linear velocity | 1.5 m/s (configurable) |
| Max angular velocity | π rad/s (configurable) |
| Position accuracy | 0.5-2.0 m (with GPS) |
| Heading accuracy | ~5-10 degrees |

## Configuration Tips

For different scenarios, adjust `SlamConfig`:

**High-speed highway:**
```rust
config.max_linear_vel = 25.0;  // 90 km/h
config.max_angular_vel = 0.5;   // Gentle turns
config.mpc_q_weights = [10.0, 10.0, 1.0];  // Prioritize position
```

**Tight parking:**
```rust
config.max_linear_vel = 0.3;    // Very slow
config.max_angular_vel = PI/4.0; // 45 deg/s
config.measurement_noise_gps = [0.1, 0.1];  // High accuracy
```

**Agricultural field:**
```rust
config.max_linear_vel = 1.5;
config.measurement_noise_gps = [0.5, 0.5];  // RTK-GPS
config.mpc_horizon = 15;  // Longer horizon for smooth paths
```

## Adding Your Own Scenarios

To create custom scenarios:

1. Define waypoints as `Vec<Vec2>`
2. Create `SlamConfig` with appropriate parameters
3. Set up sensor update rates
4. Run simulation loop with control computation
5. Add assertions for success criteria

Example template:
```rust
let mut config = SlamConfig::default();
// Customize config...

let mut slam = SlamFilter::new(config);

let waypoints = vec![
    Vec2::new(0.0, 0.0),
    Vec2::new(10.0, 5.0),
    // More waypoints...
];
slam.set_trajectory(&waypoints);

for step in 0..total_steps {
    if let Some((v, omega)) = slam.compute_control() {
        slam.handle_odom(v, 0.0, omega, dt);
        // Add other sensor updates...
    }
}
```

