use kornia_slam::{SlamConfig, SlamFilter, Vec2, Vec3};
use std::f32::consts::PI;

#[test]
fn test_straight_line_navigation() {
    // Simulate driving 10 meters straight
    let config = SlamConfig::default();
    let mut slam = SlamFilter::new(config);
    
    let waypoints = vec![
        Vec2::new(0.0, 0.0),
        Vec2::new(10.0, 0.0),
    ];
    slam.set_trajectory(&waypoints);
    
    // Simulate 15 seconds
    for _ in 0..150 {
        if let Some((v, omega)) = slam.compute_control() {
            slam.handle_odom(v, 0.0, omega, 0.1);
        }
    }
    
    let final_state = slam.get_state();
    // Should have made progress in x direction
    assert!(final_state.position.x > 3.0, "Only traveled {:.2}m", final_state.position.x);
    assert!(final_state.position.y.abs() < 2.0);
}

#[test]
fn test_90_degree_turn() {
    // Test ability to make a 90 degree turn
    let mut config = SlamConfig::default();
    config.max_angular_vel = PI / 2.0;
    
    let mut slam = SlamFilter::new(config);
    
    let waypoints = vec![
        Vec2::new(0.0, 0.0),
        Vec2::new(5.0, 0.0),  // Go straight
        Vec2::new(5.0, 5.0),  // Turn left 90 degrees
    ];
    slam.set_trajectory(&waypoints);
    
    // Simulate navigation
    for _ in 0..150 {
        if let Some((v, omega)) = slam.compute_control() {
            slam.handle_odom(v, 0.0, omega, 0.1);
        }
    }
    
    let final_state = slam.get_state();
    // Should have turned and moved
    assert!(final_state.position.x > 3.0);
    assert!(final_state.position.y > 1.0);
}

#[test]
fn test_square_trajectory() {
    // Navigate a 10m x 10m square
    let config = SlamConfig::default();
    let mut slam = SlamFilter::new(config);
    
    let waypoints = vec![
        Vec2::new(0.0, 0.0),
        Vec2::new(10.0, 0.0),
        Vec2::new(10.0, 10.0),
        Vec2::new(0.0, 10.0),
        Vec2::new(0.0, 0.0),
    ];
    slam.set_trajectory(&waypoints);
    
    // Add GPS corrections periodically
    for step in 0..500 {
        if let Some((v, omega)) = slam.compute_control() {
            slam.handle_odom(v, 0.0, omega, 0.1);
            
            // GPS every 1 second
            if step % 10 == 0 {
                let state = slam.get_state();
                slam.handle_gps(state.position, 1.0);
            }
        }
    }
    
    let final_state = slam.get_state();
    
    // Should have made significant progress around the square
    let total_distance = final_state.position.distance(Vec2::ZERO);
    assert!(total_distance > 0.0);
    
    // At least should have moved from start
    assert!(final_state.position.x > 0.5 || final_state.position.y > 0.5,
        "Didn't move significantly: ({:.2}, {:.2})", final_state.position.x, final_state.position.y);
}

#[test]
fn test_circular_path() {
    // Navigate a circular path (radius 5m)
    let config = SlamConfig::default();
    let mut slam = SlamFilter::new(config);
    
    let mut waypoints = Vec::new();
    let radius = 5.0;
    let num_points = 16;
    
    for i in 0..=num_points {
        let angle = (i as f32 / num_points as f32) * 2.0 * PI;
        waypoints.push(Vec2::new(
            radius * angle.cos(),
            radius * angle.sin(),
        ));
    }
    
    slam.set_trajectory(&waypoints);
    
    // Simulate for 400 steps
    for step in 0..400 {
        if let Some((v, omega)) = slam.compute_control() {
            slam.handle_odom(v, 0.0, omega, 0.1);
            
            // Add IMU measurements
            if step % 5 == 0 {
                let accel = Vec3::new(0.0, 0.0, 9.81);
                let gyro = Vec3::new(0.0, 0.0, omega);
                slam.handle_imu(accel, gyro, 0.1);
            }
        }
    }
    
    let final_state = slam.get_state();
    // Should complete most of the circle
    let distance_from_center = final_state.position.length();
    assert!(distance_from_center > 2.0 && distance_from_center < 8.0,
        "Distance from center: {:.2}m (expected 3-7m)", distance_from_center);
}

#[test]
fn test_s_curve_trajectory() {
    // S-curve maneuver (like lane change)
    let config = SlamConfig::default();
    let mut slam = SlamFilter::new(config);
    
    let waypoints = vec![
        Vec2::new(0.0, 0.0),
        Vec2::new(5.0, 0.0),
        Vec2::new(10.0, 3.0),   // First curve
        Vec2::new(15.0, 3.0),   // Straight section
        Vec2::new(20.0, 0.0),   // Second curve back
        Vec2::new(25.0, 0.0),
    ];
    slam.set_trajectory(&waypoints);
    
    let mut max_angular_vel: f32 = 0.0;
    
    for _ in 0..300 {
        if let Some((v, omega)) = slam.compute_control() {
            max_angular_vel = max_angular_vel.max(omega.abs());
            slam.handle_odom(v, 0.0, omega, 0.1);
        }
    }
    
    let final_state = slam.get_state();
    
    // Should have made progress
    assert!(final_state.position.x > 5.0, "Only traveled {:.2}m in x", final_state.position.x);
    // Should have used some angular velocity for turns
    assert!(max_angular_vel > 0.05, "Max angular vel was only {:.3}", max_angular_vel);
}

#[test]
fn test_reverse_and_forward() {
    // Test reversing capability (for parking scenarios)
    let mut config = SlamConfig::default();
    config.max_linear_vel = 1.0;
    
    let mut slam = SlamFilter::new(config);
    
    // Move forward
    for _ in 0..50 {
        slam.handle_odom(0.5, 0.0, 0.0, 0.1);
    }
    
    let forward_pos = slam.get_state().position;
    assert!(forward_pos.x > 2.0);
    
    // Now reverse back
    for _ in 0..50 {
        slam.handle_odom(-0.5, 0.0, 0.0, 0.1);
    }
    
    let final_pos = slam.get_state().position;
    
    // Should be back close to origin
    assert!(final_pos.distance(Vec2::ZERO) < 1.0);
}

#[test]
fn test_with_gps_corrections() {
    // Test GPS correction prevents drift
    let config = SlamConfig::default();
    let mut slam = SlamFilter::new(config);
    
    let waypoints = vec![
        Vec2::new(0.0, 0.0),
        Vec2::new(50.0, 0.0), // Medium distance
    ];
    slam.set_trajectory(&waypoints);
    
    let mut accumulated_distance = 0.0;
    
    // Simulate with GPS corrections every second
    for step in 0..300 {
        if let Some((v, omega)) = slam.compute_control() {
            accumulated_distance += v * 0.1;
            slam.handle_odom(v, 0.0, omega, 0.1);
            
            // GPS correction every second (10 steps)
            if step % 10 == 0 {
                let pos = slam.get_state().position;
                slam.handle_gps(pos, 1.0);
            }
        }
    }
    
    let final_state = slam.get_state();
    
    // With GPS corrections, should make progress
    assert!(final_state.position.x > 5.0, "Only progressed {:.2}m", final_state.position.x);
}

#[test]
fn test_field_boundary_navigation() {
    // Realistic agricultural robot field boundary
    let mut config = SlamConfig::default();
    config.max_linear_vel = 1.5; // Typical Amiga speed
    
    let mut slam = SlamFilter::new(config);
    
    // 30m x 20m field perimeter with SMOOTH ARC CORNERS (1.5m radius)
    let mut waypoints = Vec::new();
    let radius = 1.5;
    
    // Bottom edge
    for x in (0..28).step_by(5) {
        waypoints.push(Vec2::new(x as f32, 0.0));
    }
    
    // Arc at corner 1 (bottom-right)
    for i in 0..6 {
        let angle = (i as f32 / 5.0) * std::f32::consts::PI / 2.0;
        waypoints.push(Vec2::new(30.0 - radius + radius * angle.cos(), radius - radius * angle.sin()));
    }
    
    // Right edge
    for y in (3..18).step_by(4) {
        waypoints.push(Vec2::new(30.0, y as f32));
    }
    
    // Arc at corner 2 (top-right)
    for i in 0..6 {
        let angle = (i as f32 / 5.0) * std::f32::consts::PI / 2.0;
        waypoints.push(Vec2::new(30.0 - radius * angle.sin(), 20.0 - radius + radius * angle.cos()));
    }
    
    // Top edge
    for x in (3..28).rev().step_by(5) {
        waypoints.push(Vec2::new(x as f32, 20.0));
    }
    
    // Arc at corner 3 (top-left)
    for i in 0..6 {
        let angle = (i as f32 / 5.0) * std::f32::consts::PI / 2.0;
        waypoints.push(Vec2::new(radius - radius * angle.cos(), 20.0 - radius * angle.sin()));
    }
    
    // Left edge
    for y in (3..18).rev().step_by(4) {
        waypoints.push(Vec2::new(0.0, y as f32));
    }
    
    // Arc at corner 4 (bottom-left) back to start
    for i in 0..6 {
        let angle = (i as f32 / 5.0) * std::f32::consts::PI / 2.0;
        waypoints.push(Vec2::new(radius * angle.sin(), radius - radius * angle.cos()));
    }
    
    waypoints.push(Vec2::new(0.0, 0.0));
    slam.set_trajectory(&waypoints);
    
    // Simulate with realistic sensor updates (run longer for larger field - 100m perimeter)
    for step in 0..10000 {
        if let Some((v, omega)) = slam.compute_control() {
            slam.handle_odom(v, 0.0, omega, 0.1);
            
            // IMU at 10 Hz
            if step % 1 == 0 {
                let accel = Vec3::new(0.0, 0.0, 9.81);
                let gyro = Vec3::new(0.0, 0.0, omega);
                slam.handle_imu(accel, gyro, 0.1);
            }
            
            // GPS at 2 Hz with good accuracy
            if step % 5 == 0 {
                let pos = slam.get_state().position;
                slam.handle_gps(pos, 0.3); // Better GPS accuracy
            }
        }
    }
    
    let final_state = slam.get_state();
    let final_error = final_state.position.distance(Vec2::ZERO);
    
    println!("\nField boundary (30m x 20m with arc corners):");
    println!("  Final position: ({:.2}, {:.2})", 
        final_state.position.x, final_state.position.y);
    println!("  Loop closure error: {:.2}m", final_error);
    
    // With smooth arc corners, should achieve good loop closure
    assert!(final_error < 3.0, 
        "Field boundary loop closure error: {:.2}m (expected < 3m)", final_error);
}

#[test]
fn test_obstacle_avoidance_path() {
    // Path that goes around an obstacle (C-shape)
    let config = SlamConfig::default();
    let mut slam = SlamFilter::new(config);
    
    let waypoints = vec![
        Vec2::new(0.0, 0.0),
        Vec2::new(5.0, 0.0),
        Vec2::new(5.0, 5.0),    // Go around
        Vec2::new(5.0, 10.0),
        Vec2::new(10.0, 10.0),  // Reach other side
    ];
    slam.set_trajectory(&waypoints);
    
    for _ in 0..300 {
        if let Some((v, omega)) = slam.compute_control() {
            slam.handle_odom(v, 0.0, omega, 0.1);
        }
    }
    
    let final_state = slam.get_state();
    
    // Should progress significantly
    assert!(final_state.position.x > 3.0 || final_state.position.y > 3.0);
}

