use kornia_slam::{SlamConfig, SlamFilter, Vec2, Vec3};
use std::f32::consts::PI;

/// Test that closed-loop trajectories return to start with low error
#[test]
fn test_square_loop_accuracy() {
    let config = SlamConfig::default();
    let mut slam = SlamFilter::new(config);
    
    // 10m x 10m square with SMOOTH ARC CORNERS (0.7m radius)
    let mut waypoints = Vec::new();
    let radius = 0.7;
    
    // Bottom edge
    for x in (0..9).step_by(2) {
        waypoints.push(Vec2::new(x as f32, 0.0));
    }
    
    // Arc at corner 1
    for i in 0..5 {
        let angle = (i as f32 / 4.0) * PI / 2.0;
        waypoints.push(Vec2::new(10.0 - radius + radius * angle.cos(), radius - radius * angle.sin()));
    }
    
    // Right edge
    for y in (2..9).step_by(2) {
        waypoints.push(Vec2::new(10.0, y as f32));
    }
    
    // Arc at corner 2
    for i in 0..5 {
        let angle = (i as f32 / 4.0) * PI / 2.0;
        waypoints.push(Vec2::new(10.0 - radius * angle.sin(), 10.0 - radius + radius * angle.cos()));
    }
    
    // Top edge
    for x in (2..9).rev().step_by(2) {
        waypoints.push(Vec2::new(x as f32, 10.0));
    }
    
    // Arc at corner 3
    for i in 0..5 {
        let angle = (i as f32 / 4.0) * PI / 2.0;
        waypoints.push(Vec2::new(radius - radius * angle.cos(), 10.0 - radius * angle.sin()));
    }
    
    // Left edge
    for y in (2..9).rev().step_by(2) {
        waypoints.push(Vec2::new(0.0, y as f32));
    }
    
    // Arc at corner 4
    for i in 0..5 {
        let angle = (i as f32 / 4.0) * PI / 2.0;
        waypoints.push(Vec2::new(radius * angle.sin(), radius - radius * angle.cos()));
    }
    
    waypoints.push(Vec2::new(0.0, 0.0));
    slam.set_trajectory(&waypoints);
    
    // Run longer to complete the loop
    for step in 0..5000 {
        if let Some((v, omega)) = slam.compute_control() {
            slam.handle_odom(v, 0.0, omega, 0.1);
            
            // Add GPS to prevent drift
            if step % 10 == 0 {
                let pos = slam.get_state().position;
                slam.handle_gps(pos, 0.5);
            }
        }
    }
    
    let final_state = slam.get_state();
    let final_error = final_state.position.distance(Vec2::ZERO);
    
    println!("Square loop (with arc corners) final position: ({:.2}, {:.2})", 
        final_state.position.x, final_state.position.y);
    println!("Final error from start: {:.2}m", final_error);
    
    // Should return very close to start with smooth corners
    assert!(final_error < 1.0, 
        "Square loop error too high: {:.2}m (expected < 1m with arc corners)", final_error);
}

#[test]
fn test_circle_loop_accuracy() {
    let config = SlamConfig::default();
    let mut slam = SlamFilter::new(config);
    
    // Generate circular trajectory
    let mut waypoints = Vec::new();
    let radius = 8.0;
    let num_points = 24;
    
    for i in 0..=num_points {
        let angle = (i as f32 / num_points as f32) * 2.0 * PI;
        waypoints.push(Vec2::new(
            radius * angle.cos(),
            radius * angle.sin(),
        ));
    }
    
    slam.set_trajectory(&waypoints);
    
    // Run simulation
    for step in 0..800 {
        if let Some((v, omega)) = slam.compute_control() {
            slam.handle_odom(v, 0.0, omega, 0.1);
            
            if step % 10 == 0 {
                let pos = slam.get_state().position;
                slam.handle_gps(pos, 0.5);
            }
        }
    }
    
    let final_state = slam.get_state();
    let start = waypoints[0];
    let final_error = final_state.position.distance(start);
    
    println!("Circle loop final position: ({:.2}, {:.2})", 
        final_state.position.x, final_state.position.y);
    println!("Final error from start: {:.2}m", final_error);
    
    assert!(final_error < 3.0, 
        "Circle loop error too high: {:.2}m (expected < 3m)", final_error);
}

#[test]
fn test_small_rectangle_accuracy() {
    let config = SlamConfig::default();
    let mut slam = SlamFilter::new(config);
    
    // Small 5m x 3m rectangle with SMOOTH ARC CORNERS (0.4m radius)
    let mut waypoints = Vec::new();
    let radius = 0.4;
    
    // Bottom edge
    waypoints.push(Vec2::new(0.0, 0.0));
    waypoints.push(Vec2::new(2.0, 0.0));
    waypoints.push(Vec2::new(4.0, 0.0));
    
    // Arc at corner 1
    for i in 0..4 {
        let angle = (i as f32 / 3.0) * PI / 2.0;
        waypoints.push(Vec2::new(5.0 - radius + radius * angle.cos(), radius - radius * angle.sin()));
    }
    
    // Right edge
    waypoints.push(Vec2::new(5.0, 1.5));
    waypoints.push(Vec2::new(5.0, 2.5));
    
    // Arc at corner 2
    for i in 0..4 {
        let angle = (i as f32 / 3.0) * PI / 2.0;
        waypoints.push(Vec2::new(5.0 - radius * angle.sin(), 3.0 - radius + radius * angle.cos()));
    }
    
    // Top edge
    waypoints.push(Vec2::new(3.5, 3.0));
    waypoints.push(Vec2::new(2.0, 3.0));
    waypoints.push(Vec2::new(0.5, 3.0));
    
    // Arc at corner 3
    for i in 0..4 {
        let angle = (i as f32 / 3.0) * PI / 2.0;
        waypoints.push(Vec2::new(radius - radius * angle.cos(), 3.0 - radius * angle.sin()));
    }
    
    // Left edge
    waypoints.push(Vec2::new(0.0, 2.0));
    waypoints.push(Vec2::new(0.0, 1.0));
    
    // Arc at corner 4
    for i in 0..4 {
        let angle = (i as f32 / 3.0) * PI / 2.0;
        waypoints.push(Vec2::new(radius * angle.sin(), radius - radius * angle.cos()));
    }
    
    waypoints.push(Vec2::new(0.0, 0.0));
    slam.set_trajectory(&waypoints);
    
    // Run simulation with frequent GPS updates
    for step in 0..4000 {
        if let Some((v, omega)) = slam.compute_control() {
            slam.handle_odom(v, 0.0, omega, 0.1);
            
            // Frequent GPS for accuracy
            if step % 5 == 0 {
                let pos = slam.get_state().position;
                slam.handle_gps(pos, 0.3);
            }
        }
    }
    
    let final_state = slam.get_state();
    let final_error = final_state.position.distance(Vec2::ZERO);
    
    println!("Small rectangle (with arc corners) final position: ({:.2}, {:.2})", 
        final_state.position.x, final_state.position.y);
    println!("Final error from start: {:.2}m", final_error);
    
    assert!(final_error < 1.0, 
        "Small rectangle error too high: {:.2}m (expected < 1m with arc corners)", final_error);
}

#[test]
fn test_trajectory_completion_detection() {
    let config = SlamConfig::default();
    let mut slam = SlamFilter::new(config);
    
    let waypoints = vec![
        Vec2::new(0.0, 0.0),
        Vec2::new(10.0, 0.0),
    ];
    slam.set_trajectory(&waypoints);
    
    let mut reached_goal = false;
    
    for _ in 0..300 {
        if let Some((v, omega)) = slam.compute_control() {
            slam.handle_odom(v, 0.0, omega, 0.1);
            
            let state = slam.get_state();
            let distance_to_goal = state.position.distance(Vec2::new(10.0, 0.0));
            
            if distance_to_goal < 1.0 {
                reached_goal = true;
                println!("Reached goal at position ({:.2}, {:.2})", 
                    state.position.x, state.position.y);
                break;
            }
        }
    }
    
    assert!(reached_goal, "Failed to reach goal");
}

#[test]
fn test_precise_waypoint_following() {
    let mut config = SlamConfig::default();
    config.max_linear_vel = 1.0;
    
    let mut slam = SlamFilter::new(config);
    
    let waypoints = vec![
        Vec2::new(0.0, 0.0),
        Vec2::new(5.0, 0.0),
        Vec2::new(5.0, 5.0),
    ];
    slam.set_trajectory(&waypoints);
    
    let mut min_distance_to_second = f32::MAX;
    
    for step in 0..300 {
        if let Some((v, omega)) = slam.compute_control() {
            slam.handle_odom(v, 0.0, omega, 0.1);
            
            if step % 10 == 0 {
                let pos = slam.get_state().position;
                slam.handle_gps(pos, 0.5);
            }
            
            let state = slam.get_state();
            let dist = state.position.distance(Vec2::new(5.0, 0.0));
            min_distance_to_second = min_distance_to_second.min(dist);
        }
    }
    
    println!("Minimum distance to waypoint (5,0): {:.2}m", min_distance_to_second);
    
    // Should pass close to the second waypoint
    assert!(min_distance_to_second < 2.0,
        "Never got close to waypoint, min distance: {:.2}m", min_distance_to_second);
}

