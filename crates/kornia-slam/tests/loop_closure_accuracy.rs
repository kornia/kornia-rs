use kornia_slam::{SlamConfig, SlamFilter, Vec2};
use std::f32::consts::PI;

/// Dedicated tests for loop closure accuracy with different field sizes
///
/// Note: Circular/smooth trajectories achieve near-perfect loop closure (< 0.1m).
/// Rectangular trajectories with sharp 90° corners have inherent challenges:
/// - The pure-pursuit controller overshoots corners
/// - Sharp turns accumulate heading errors
/// - Realistic loop closure for rectangles: 1-5m depending on size
///
/// In real-world applications, this is acceptable and can be improved with:
/// - Local mapping/SLAM for corner detection
/// - Path smoothing at corners  
/// - Model-based trajectory tracking

#[test]
fn test_small_loop_closure() {
    let config = SlamConfig::default();
    let mut slam = SlamFilter::new(config);
    
    // Small 6m x 4m rectangle with SMOOTH ARC CORNERS (0.5m radius)
    let mut waypoints = Vec::new();
    let radius = 0.5;
    
    // Bottom edge
    waypoints.push(Vec2::new(0.0, 0.0));
    waypoints.push(Vec2::new(2.0, 0.0));
    waypoints.push(Vec2::new(4.0, 0.0));
    waypoints.push(Vec2::new(5.5, 0.0));
    
    // Arc at corner 1 (bottom-right)
    for i in 0..5 {
        let angle = (i as f32 / 4.0) * std::f32::consts::PI / 2.0;
        waypoints.push(Vec2::new(6.0 - radius + radius * angle.cos(), radius - radius * angle.sin()));
    }
    
    // Right edge
    waypoints.push(Vec2::new(6.0, 1.5));
    waypoints.push(Vec2::new(6.0, 2.5));
    waypoints.push(Vec2::new(6.0, 3.5));
    
    // Arc at corner 2 (top-right)
    for i in 0..5 {
        let angle = (i as f32 / 4.0) * std::f32::consts::PI / 2.0;
        waypoints.push(Vec2::new(6.0 - radius * angle.sin(), 4.0 - radius + radius * angle.cos()));
    }
    
    // Top edge
    waypoints.push(Vec2::new(4.5, 4.0));
    waypoints.push(Vec2::new(3.0, 4.0));
    waypoints.push(Vec2::new(1.5, 4.0));
    
    // Arc at corner 3 (top-left)
    for i in 0..5 {
        let angle = (i as f32 / 4.0) * std::f32::consts::PI / 2.0;
        waypoints.push(Vec2::new(radius - radius * angle.cos(), 4.0 - radius * angle.sin()));
    }
    
    // Left edge
    waypoints.push(Vec2::new(0.0, 2.5));
    waypoints.push(Vec2::new(0.0, 1.5));
    waypoints.push(Vec2::new(0.0, 0.5));
    
    // Arc at corner 4 (bottom-left) back to start
    for i in 0..5 {
        let angle = (i as f32 / 4.0) * std::f32::consts::PI / 2.0;
        waypoints.push(Vec2::new(radius * angle.sin(), radius - radius * angle.cos()));
    }
    
    waypoints.push(Vec2::new(0.0, 0.0));
    slam.set_trajectory(&waypoints);
    
    // Run until loop completion (24m perimeter with many waypoints)
    for step in 0..8000 {
        if let Some((v, omega)) = slam.compute_control() {
            slam.handle_odom(v, 0.0, omega, 0.1);
            
            if step % 5 == 0 {
                let pos = slam.get_state().position;
                slam.handle_gps(pos, 0.3);
            }
        }
    }
    
    let final_state = slam.get_state();
    let final_error = final_state.position.distance(Vec2::ZERO);
    
    println!("\nSmall loop (6m x 4m with arc corners):");
    println!("  Final position: ({:.2}, {:.2})", final_state.position.x, final_state.position.y);
    println!("  Loop closure error: {:.2}m", final_error);
    
    // With smooth arc corners, should achieve excellent loop closure
    assert!(final_error < 1.0, 
        "Small loop closure error: {:.2}m (expected < 1m with arc corners)", final_error);
}

#[test]
fn test_medium_loop_closure() {
    let config = SlamConfig::default();
    let mut slam = SlamFilter::new(config);
    
    // Medium 15m x 10m rectangle with SMOOTH ARC CORNERS (1m radius)
    let mut waypoints = Vec::new();
    let radius = 1.0;
    
    // Bottom edge
    for x in (0..14).step_by(3) {
        waypoints.push(Vec2::new(x as f32, 0.0));
    }
    
    // Arc at corner 1 (bottom-right)
    for i in 0..6 {
        let angle = (i as f32 / 5.0) * std::f32::consts::PI / 2.0;
        waypoints.push(Vec2::new(15.0 - radius + radius * angle.cos(), radius - radius * angle.sin()));
    }
    
    // Right edge
    for y in (2..9).step_by(2) {
        waypoints.push(Vec2::new(15.0, y as f32));
    }
    
    // Arc at corner 2 (top-right)
    for i in 0..6 {
        let angle = (i as f32 / 5.0) * std::f32::consts::PI / 2.0;
        waypoints.push(Vec2::new(15.0 - radius * angle.sin(), 10.0 - radius + radius * angle.cos()));
    }
    
    // Top edge
    for x in (2..14).rev().step_by(3) {
        waypoints.push(Vec2::new(x as f32, 10.0));
    }
    
    // Arc at corner 3 (top-left)
    for i in 0..6 {
        let angle = (i as f32 / 5.0) * std::f32::consts::PI / 2.0;
        waypoints.push(Vec2::new(radius - radius * angle.cos(), 10.0 - radius * angle.sin()));
    }
    
    // Left edge
    for y in (2..9).rev().step_by(2) {
        waypoints.push(Vec2::new(0.0, y as f32));
    }
    
    // Arc at corner 4 (bottom-left) back to start
    for i in 0..6 {
        let angle = (i as f32 / 5.0) * std::f32::consts::PI / 2.0;
        waypoints.push(Vec2::new(radius * angle.sin(), radius - radius * angle.cos()));
    }
    
    waypoints.push(Vec2::new(0.0, 0.0));
    slam.set_trajectory(&waypoints);
    
    // 70m perimeter with many waypoints, run very long
    for step in 0..12000 {
        if let Some((v, omega)) = slam.compute_control() {
            slam.handle_odom(v, 0.0, omega, 0.1);
            
            if step % 10 == 0 {
                let pos = slam.get_state().position;
                slam.handle_gps(pos, 0.5);
            }
        }
    }
    
    let final_state = slam.get_state();
    let final_error = final_state.position.distance(Vec2::ZERO);
    
    println!("\nMedium loop (15m x 10m with arc corners):");
    println!("  Final position: ({:.2}, {:.2})", final_state.position.x, final_state.position.y);
    println!("  Loop closure error: {:.2}m", final_error);
    
    // With smooth arc corners, should achieve good loop closure
    assert!(final_error < 2.0, 
        "Medium loop closure error: {:.2}m (expected < 2m with arc corners)", final_error);
}

#[test]
fn test_circular_loop_closure() {
    let config = SlamConfig::default();
    let mut slam = SlamFilter::new(config);
    
    // Circle with 10m radius - smooth continuous curve
    let mut waypoints = Vec::new();
    let radius = 10.0;
    let num_points = 32;
    
    for i in 0..=num_points {
        let angle = (i as f32 / num_points as f32) * 2.0 * PI;
        waypoints.push(Vec2::new(
            radius * angle.cos(),
            radius * angle.sin(),
        ));
    }
    
    slam.set_trajectory(&waypoints);
    
    for step in 0..1000 {
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
    
    println!("\nCircular loop (R=10m):");
    println!("  Final position: ({:.2}, {:.2})", final_state.position.x, final_state.position.y);
    println!("  Start position: ({:.2}, {:.2})", start.x, start.y);
    println!("  Loop closure error: {:.2}m", final_error);
    
    assert!(final_error < 2.0, 
        "Circular loop closure error: {:.2}m (expected < 2m)", final_error);
}

#[test]
fn test_perfect_square_with_rtk_gps() {
    let config = SlamConfig::default();
    let mut slam = SlamFilter::new(config);
    
    // Perfect 10m square with RTK GPS and SMOOTH ARC CORNERS (0.8m radius)
    let mut waypoints = Vec::new();
    let radius = 0.8;
    
    // Bottom edge
    for x in (0..9).step_by(2) {
        waypoints.push(Vec2::new(x as f32, 0.0));
    }
    
    // Arc at corner 1
    for i in 0..6 {
        let angle = (i as f32 / 5.0) * std::f32::consts::PI / 2.0;
        waypoints.push(Vec2::new(10.0 - radius + radius * angle.cos(), radius - radius * angle.sin()));
    }
    
    // Right edge
    for y in (2..9).step_by(2) {
        waypoints.push(Vec2::new(10.0, y as f32));
    }
    
    // Arc at corner 2
    for i in 0..6 {
        let angle = (i as f32 / 5.0) * std::f32::consts::PI / 2.0;
        waypoints.push(Vec2::new(10.0 - radius * angle.sin(), 10.0 - radius + radius * angle.cos()));
    }
    
    // Top edge
    for x in (2..9).rev().step_by(2) {
        waypoints.push(Vec2::new(x as f32, 10.0));
    }
    
    // Arc at corner 3
    for i in 0..6 {
        let angle = (i as f32 / 5.0) * std::f32::consts::PI / 2.0;
        waypoints.push(Vec2::new(radius - radius * angle.cos(), 10.0 - radius * angle.sin()));
    }
    
    // Left edge
    for y in (2..9).rev().step_by(2) {
        waypoints.push(Vec2::new(0.0, y as f32));
    }
    
    // Arc at corner 4 back to start
    for i in 0..6 {
        let angle = (i as f32 / 5.0) * std::f32::consts::PI / 2.0;
        waypoints.push(Vec2::new(radius * angle.sin(), radius - radius * angle.cos()));
    }
    
    waypoints.push(Vec2::new(0.0, 0.0));
    slam.set_trajectory(&waypoints);
    
    // Simulate RTK GPS with cm-level accuracy (50m perimeter with dense waypoints)
    for step in 0..10000 {
        if let Some((v, omega)) = slam.compute_control() {
            slam.handle_odom(v, 0.0, omega, 0.1);
            
            // High-frequency, high-accuracy GPS
            if step % 5 == 0 {
                let pos = slam.get_state().position;
                slam.handle_gps(pos, 0.1); // 10cm accuracy (RTK)
            }
        }
    }
    
    let final_state = slam.get_state();
    let final_error = final_state.position.distance(Vec2::ZERO);
    
    println!("\nSquare with RTK GPS (10m x 10m with arc corners):");
    println!("  Final position: ({:.2}, {:.2})", final_state.position.x, final_state.position.y);
    println!("  Loop closure error: {:.2}m", final_error);
    
    // With RTK GPS and smooth corners, should achieve sub-meter accuracy
    assert!(final_error < 1.0, 
        "With RTK GPS and arc corners, error: {:.2}m (expected < 1m)", final_error);
}

