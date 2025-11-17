use kornia_slam::{SlamConfig, SlamFilter, Vec2, Vec3};

fn main() {
    println!("=== kornia-slam Example ===\n");

    // Create a SLAM filter with default configuration
    let config = SlamConfig::default();
    let mut slam = SlamFilter::new(config);

    println!("Initial state:");
    println!("  Position: {:?}", slam.get_state().position);
    println!("  Heading: {:.2} rad\n", slam.get_state().heading);

    // Simulate robot moving forward at 1 m/s for 1 second
    println!("Simulating forward motion...");
    for _ in 0..10 {
        slam.handle_odom(1.0, 0.0, 0.0, 0.1);
    }

    println!("After 1s of forward motion:");
    println!("  Position: {:?}", slam.get_state().position);
    println!("  Velocity: {:?}\n", slam.get_state().linear_vel);

    // Add IMU measurements (static, no rotation)
    println!("Adding IMU measurements...");
    let accel = Vec3::new(0.0, 0.0, 9.81); // Gravity
    let gyro = Vec3::new(0.0, 0.0, 0.0); // No rotation
    slam.handle_imu(accel, gyro, 0.1);

    // Add GPS measurement
    println!("Adding GPS measurement...");
    let gps_position = Vec2::new(1.0, 0.0);
    slam.handle_gps(gps_position, 1.0);

    println!("After GPS update:");
    println!("  Position: {:?}\n", slam.get_state().position);

    // Set a trajectory to follow
    println!("Setting trajectory...");
    let waypoints = vec![
        Vec2::new(0.0, 0.0),
        Vec2::new(10.0, 0.0),
        Vec2::new(10.0, 10.0),
        Vec2::new(0.0, 10.0),
    ];
    slam.set_trajectory(&waypoints);

    // Compute control commands
    println!("Computing control...");
    if let Some((linear_vel, angular_vel)) = slam.compute_control() {
        println!("  Linear velocity: {:.2} m/s", linear_vel);
        println!("  Angular velocity: {:.2} rad/s", angular_vel);
    }

    // Simulate following the trajectory
    println!("\nSimulating trajectory following for 5 steps:");
    for step in 0..5 {
        if let Some((v, omega)) = slam.compute_control() {
            // Apply control (simplified - no dynamics)
            slam.handle_odom(v, 0.0, omega, 0.1);

            let state = slam.get_state();
            println!(
                "  Step {}: pos=({:.2}, {:.2}), heading={:.2}, v={:.2}, ω={:.2}",
                step + 1,
                state.position.x,
                state.position.y,
                state.heading,
                v,
                omega
            );
        }
    }

    println!("\n=== Example complete ===");
}

