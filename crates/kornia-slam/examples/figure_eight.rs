use kornia_slam::{SlamConfig, SlamFilter, Vec2, Vec3};
use std::f32::consts::PI;

fn main() {
    println!("=== Figure-8 Trajectory Navigation ===\n");

    let mut config = SlamConfig::default();
    config.max_linear_vel = 1.0;
    config.max_angular_vel = PI / 3.0; // 60 deg/s for tighter turns
    
    let mut slam = SlamFilter::new(config);

    // Generate figure-8 trajectory (two circles)
    // Left circle centered at (-5, 0), right circle at (5, 0), radius 5m
    let mut waypoints = Vec::new();
    let radius = 5.0;
    let num_points = 32; // 16 points per circle
    
    // Right circle (clockwise from top)
    for i in 0..num_points/2 {
        let angle = (i as f32 / (num_points/2) as f32) * 2.0 * PI;
        waypoints.push(Vec2::new(
            radius + radius * angle.sin(),
            radius * angle.cos(),
        ));
    }
    
    // Left circle (counter-clockwise from bottom)
    for i in 0..num_points/2 {
        let angle = PI + (i as f32 / (num_points/2) as f32) * 2.0 * PI;
        waypoints.push(Vec2::new(
            -radius + radius * angle.sin(),
            -radius * angle.cos(),
        ));
    }
    
    // Close the loop
    waypoints.push(waypoints[0]);
    
    println!("Figure-8 path with {} waypoints", waypoints.len());
    println!("Each circle: radius = {}m", radius);
    println!("Total path length: ~63m\n");
    
    slam.set_trajectory(&waypoints);

    let dt = 0.1;
    let total_steps = 800; // 80 seconds
    
    println!("Starting navigation...\n");
    
    let mut max_heading_rate: f32 = 0.0;
    let mut positions: Vec<Vec2> = Vec::new();
    
    for step in 0..total_steps {
        if let Some((v_cmd, omega_cmd)) = slam.compute_control() {
            slam.handle_odom(v_cmd, 0.0, omega_cmd, dt);
            
            // Add IMU and GPS periodically
            if step % 5 == 0 {
                let accel = Vec3::new(0.0, 0.0, 9.81);
                let gyro = Vec3::new(0.0, 0.0, omega_cmd);
                slam.handle_imu(accel, gyro, dt);
                
                // GPS every second
                if step % 10 == 0 {
                    let pos = slam.get_state().position;
                    slam.handle_gps(pos, 1.0);
                }
            }
            
            let state = slam.get_state();
            max_heading_rate = max_heading_rate.max(state.angular_vel.abs());
            positions.push(state.position);
            
            if step % 100 == 0 && step > 0 {
                println!("Time: {:.1}s", step as f32 * dt);
                println!("  Position: ({:.2}, {:.2}) m", state.position.x, state.position.y);
                println!("  Heading rate: {:.3} rad/s ({:.1}°/s)", 
                    state.angular_vel, state.angular_vel.to_degrees());
                
                // Determine which circle
                if state.position.x > 0.0 {
                    println!("  Status: Right circle");
                } else {
                    println!("  Status: Left circle");
                }
                println!();
            }
        }
    }
    
    println!("=== Figure-8 Complete ===");
    println!("Maximum heading rate: {:.3} rad/s ({:.1}°/s)", 
        max_heading_rate, max_heading_rate.to_degrees());
    
    // Analyze trajectory smoothness
    if positions.len() > 1 {
        let mut total_deviation = 0.0;
        for i in 1..positions.len() {
            let movement = positions[i] - positions[i-1];
            let speed = movement.length() / dt;
            total_deviation += (speed - 1.0).abs();
        }
        let avg_deviation = total_deviation / positions.len() as f32;
        println!("Average speed deviation: {:.3} m/s", avg_deviation);
        
        if avg_deviation < 0.2 {
            println!("\n✓ Smooth trajectory following!");
        }
    }
}

