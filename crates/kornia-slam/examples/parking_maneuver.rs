use kornia_slam::{SlamConfig, SlamFilter, Vec2, Vec3};
use std::f32::consts::PI;

fn main() {
    println!("=== Autonomous Parking Maneuver ===\n");

    let mut config = SlamConfig::default();
    config.max_linear_vel = 0.5; // Slow speed for parking
    config.max_angular_vel = PI / 4.0; // 45 deg/s
    
    let mut slam = SlamFilter::new(config);

    // Parking scenario: approach from the side, turn into parking spot
    // Parking spot is 3m x 5m
    let waypoints = vec![
        Vec2::new(0.0, 0.0),      // Starting position (in the lane)
        Vec2::new(5.0, 0.0),      // Approach the parking area
        Vec2::new(8.0, 1.0),      // Start turning
        Vec2::new(10.0, 3.0),     // Mid turn
        Vec2::new(10.5, 5.0),     // Align with parking spot
        Vec2::new(10.5, 7.0),     // Enter parking spot
        Vec2::new(10.5, 9.0),     // Final position in spot
    ];
    
    println!("Parking spot location: (10.5, 9.0)");
    println!("Spot dimensions: 3m x 5m");
    println!("Approach speed: 0.5 m/s max\n");
    
    slam.set_trajectory(&waypoints);

    let dt = 0.05; // 20 Hz for smooth parking control
    let total_steps = 600; // 30 seconds
    
    println!("Initiating parking maneuver...\n");
    
    let mut phase = "approach";
    let mut min_speed: f32 = f32::MAX;
    let mut max_turn_rate: f32 = 0.0;
    
    for step in 0..total_steps {
        let time = step as f32 * dt;
        
        if let Some((v_cmd, omega_cmd)) = slam.compute_control() {
            // High-frequency odometry for precise parking
            slam.handle_odom(v_cmd, 0.0, omega_cmd, dt);
            
            // IMU at 20 Hz
            if step % 1 == 0 {
                let state = slam.get_state();
                let accel = Vec3::new(0.0, 0.0, 9.81);
                let gyro = Vec3::new(0.0, 0.0, omega_cmd);
                slam.handle_imu(accel, gyro, dt);
            }
            
            // GPS every 0.5s (less frequent, less accurate for parking)
            if step % 10 == 0 {
                let state = slam.get_state();
                slam.handle_gps(state.position, 2.0); // 2m accuracy
            }
            
            let state = slam.get_state();
            min_speed = min_speed.min(v_cmd);
            max_turn_rate = max_turn_rate.max(omega_cmd.abs());
            
            // Determine parking phase
            let dist_to_goal = state.position.distance(Vec2::new(10.5, 9.0));
            if dist_to_goal < 0.5 {
                phase = "parked";
            } else if state.position.y > 4.0 {
                phase = "entering";
            } else if state.position.x > 7.0 {
                phase = "turning";
            } else {
                phase = "approach";
            }
            
            // Print status every 5 seconds
            if step % 100 == 0 {
                println!("Time: {:.1}s", time);
                println!("  Position: ({:.2}, {:.2}) m", state.position.x, state.position.y);
                println!("  Heading: {:.1}°", state.heading.to_degrees());
                println!("  Speed: {:.2} m/s", v_cmd);
                println!("  Phase: {}", phase);
                println!("  Distance to spot: {:.2} m", dist_to_goal);
                println!();
            }
            
            // Stop when parked
            if phase == "parked" && dist_to_goal < 0.3 {
                println!("Time: {:.1}s - Parking complete!", time);
                println!("  Final position: ({:.2}, {:.2}) m", state.position.x, state.position.y);
                println!("  Final heading: {:.1}°", state.heading.to_degrees());
                println!("  Position error: {:.3} m", dist_to_goal);
                println!();
                break;
            }
        }
    }
    
    let final_state = slam.get_state();
    let final_error = final_state.position.distance(Vec2::new(10.5, 9.0));
    
    println!("=== Parking Maneuver Complete ===");
    println!("Final position: ({:.2}, {:.2}) m", final_state.position.x, final_state.position.y);
    println!("Target position: (10.50, 9.00) m");
    println!("Position error: {:.3} m", final_error);
    println!("Final heading: {:.1}° (target: 90°)", final_state.heading.to_degrees());
    println!("Max turn rate used: {:.2} rad/s ({:.1}°/s)", max_turn_rate, max_turn_rate.to_degrees());
    
    if final_error < 0.5 {
        println!("\n✓ Successfully parked! (< 0.5m error)");
    } else {
        println!("\n⚠ Parking alignment needs adjustment");
    }
}

