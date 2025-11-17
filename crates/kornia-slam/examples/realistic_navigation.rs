use kornia_slam::{SlamConfig, SlamFilter, Vec2, Vec3};
use std::f32::consts::PI;

/// Simulate realistic sensor noise
fn add_noise(value: f32, noise_std: f32) -> f32 {
    // Simple noise simulation using deterministic pattern
    let noise = (value * 123.456).sin() * noise_std;
    value + noise
}

fn main() {
    println!("=== Realistic Robot Navigation Scenario ===\n");

    // Create SLAM filter with slightly more realistic noise parameters
    let mut config = SlamConfig::default();
    config.measurement_noise_odom = [0.05, 0.05, 0.02]; // 5cm, 2deg noise
    config.measurement_noise_gps = [0.5, 0.5]; // 50cm GPS noise
    config.max_linear_vel = 1.5; // Typical Amiga max speed
    config.max_angular_vel = PI / 2.0; // 90 deg/s max turn rate
    
    let mut slam = SlamFilter::new(config);

    println!("Mission: Navigate through a field boundary");
    println!("Starting position: (0, 0)");
    println!("Target waypoints: Field perimeter\n");

    // Realistic field boundary trajectory (50m x 30m rectangular field)
    let waypoints = vec![
        Vec2::new(0.0, 0.0),      // Start at corner
        Vec2::new(50.0, 0.0),     // Drive along bottom edge
        Vec2::new(50.0, 30.0),    // Turn and go up right edge
        Vec2::new(0.0, 30.0),     // Drive along top edge
        Vec2::new(0.0, 0.0),      // Return to start
    ];
    
    slam.set_trajectory(&waypoints);
    
    println!("Field dimensions: 50m x 30m");
    println!("Total path length: ~160m");
    println!("Expected time: ~120 seconds at 1.5 m/s average\n");

    // Simulation parameters
    let dt = 0.1; // 10 Hz control loop
    let total_steps = 1200; // 120 seconds
    let gps_rate = 5; // GPS update every 0.5 seconds
    let imu_rate = 1; // IMU at 10 Hz
    
    let mut distance_traveled = 0.0;
    let mut prev_pos = Vec2::ZERO;

    println!("Starting navigation...\n");
    
    for step in 0..total_steps {
        let time = step as f32 * dt;
        
        // Compute control command
        if let Some((v_cmd, omega_cmd)) = slam.compute_control() {
            // Add realistic noise to velocity measurements (odometry)
            let v_measured = add_noise(v_cmd, 0.05);
            let omega_measured = add_noise(omega_cmd, 0.02);
            
            // Update state with odometry
            slam.handle_odom(v_measured, 0.0, omega_measured, dt);
            
            // IMU measurements at 10 Hz
            if step % imu_rate == 0 {
                let state = slam.get_state();
                
                // Simulate IMU reading with gravity and some noise
                let accel = Vec3::new(
                    add_noise(0.0, 0.1),
                    add_noise(0.0, 0.1),
                    add_noise(9.81, 0.2),
                );
                let gyro = Vec3::new(
                    add_noise(0.0, 0.01),
                    add_noise(0.0, 0.01),
                    add_noise(state.angular_vel, 0.02),
                );
                
                slam.handle_imu(accel, gyro, dt);
            }
            
            // GPS measurements at 2 Hz
            if step % gps_rate == 0 {
                let state = slam.get_state();
                let gps_pos = Vec2::new(
                    add_noise(state.position.x, 0.5),
                    add_noise(state.position.y, 0.5),
                );
                slam.handle_gps(gps_pos, 0.5);
            }
            
            // Track distance
            let current_pos = slam.get_state().position;
            distance_traveled += current_pos.distance(prev_pos);
            prev_pos = current_pos;
            
            // Print status every 10 seconds
            if step % 100 == 0 && step > 0 {
                let state = slam.get_state();
                println!("Time: {:.1}s", time);
                println!("  Position: ({:.2}, {:.2}) m", state.position.x, state.position.y);
                println!("  Heading: {:.2} rad ({:.1}°)", state.heading, state.heading.to_degrees());
                println!("  Velocity: {:.2} m/s", state.linear_vel.length());
                println!("  Distance: {:.1} m", distance_traveled);
                
                // Check which leg we're on
                if state.position.x < 25.0 && state.position.y < 5.0 {
                    println!("  Status: Navigating bottom edge (leg 1/4)");
                } else if state.position.x > 25.0 && state.position.y < 15.0 {
                    println!("  Status: Navigating right edge (leg 2/4)");
                } else if state.position.y > 15.0 && state.position.x > 25.0 {
                    println!("  Status: Navigating top edge (leg 3/4)");
                } else {
                    println!("  Status: Returning to start (leg 4/4)");
                }
                println!();
            }
        }
    }
    
    let final_state = slam.get_state();
    let final_error = final_state.position.distance(Vec2::ZERO);
    
    println!("\n=== Mission Complete ===");
    println!("Final position: ({:.2}, {:.2}) m", final_state.position.x, final_state.position.y);
    println!("Distance from start: {:.2} m", final_error);
    println!("Total distance traveled: {:.1} m", distance_traveled);
    println!("Average speed: {:.2} m/s", distance_traveled / (total_steps as f32 * dt));
    
    if final_error < 2.0 {
        println!("\n✓ Navigation successful! (< 2m position error)");
    } else {
        println!("\n⚠ Navigation completed with {:.2}m position error", final_error);
    }
}

