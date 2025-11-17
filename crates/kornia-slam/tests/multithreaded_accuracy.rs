use kornia_slam::{SlamConfig, SlamFilter, Vec2, Vec3};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

/// Test that multi-threaded sensor fusion with asynchronous data arrival
/// achieves near-zero loop closure error
#[test]
fn test_multithreaded_sensor_fusion_accuracy() {
    // Create SLAM filter wrapped in Arc<Mutex<>> for thread-safe access
    let config = SlamConfig::default();
    let slam = Arc::new(Mutex::new(SlamFilter::new(config)));

    // Create a 10m x 10m square trajectory with smooth arc corners
    let mut waypoints = Vec::new();
    let radius = 0.7;
    
    // Bottom edge
    for x in (0..9).step_by(2) {
        waypoints.push(Vec2::new(x as f32, 0.0));
    }
    
    // Arc at corner 1
    for i in 0..5 {
        let angle = (i as f32 / 4.0) * std::f32::consts::PI / 2.0;
        waypoints.push(Vec2::new(10.0 - radius + radius * angle.cos(), radius - radius * angle.sin()));
    }
    
    // Right edge
    for y in (2..9).step_by(2) {
        waypoints.push(Vec2::new(10.0, y as f32));
    }
    
    // Arc at corner 2
    for i in 0..5 {
        let angle = (i as f32 / 4.0) * std::f32::consts::PI / 2.0;
        waypoints.push(Vec2::new(10.0 - radius * angle.sin(), 10.0 - radius + radius * angle.cos()));
    }
    
    // Top edge
    for x in (2..9).rev().step_by(2) {
        waypoints.push(Vec2::new(x as f32, 10.0));
    }
    
    // Arc at corner 3
    for i in 0..5 {
        let angle = (i as f32 / 4.0) * std::f32::consts::PI / 2.0;
        waypoints.push(Vec2::new(radius - radius * angle.cos(), 10.0 - radius * angle.sin()));
    }
    
    // Left edge
    for y in (2..9).rev().step_by(2) {
        waypoints.push(Vec2::new(0.0, y as f32));
    }
    
    // Arc at corner 4
    for i in 0..5 {
        let angle = (i as f32 / 4.0) * std::f32::consts::PI / 2.0;
        waypoints.push(Vec2::new(radius * angle.sin(), radius - radius * angle.cos()));
    }
    
    waypoints.push(Vec2::new(0.0, 0.0));
    
    slam.lock().unwrap().set_trajectory(&waypoints);

    // Shared state for current velocity commands
    let current_velocity = Arc::new(Mutex::new((0.0f32, 0.0f32)));
    
    // Control variables
    let running = Arc::new(Mutex::new(true));
    let start_time = Instant::now();
    let max_duration_secs = 30.0; // Maximum runtime for test
    let completion_threshold = 1.5;

    // Spawn IMU thread (100 Hz)
    let slam_imu = Arc::clone(&slam);
    let velocity_imu = Arc::clone(&current_velocity);
    let running_imu = Arc::clone(&running);
    let imu_handle = thread::spawn(move || {
        let mut imu_count = 0;
        while *running_imu.lock().unwrap() {
            let (_, omega) = *velocity_imu.lock().unwrap();
            
            // Simulate IMU with small noise
            let accel = Vec3::new(
                (imu_count as f32 * 0.01).sin() * 0.1,
                (imu_count as f32 * 0.01).cos() * 0.1,
                9.81,
            );
            let gyro = Vec3::new(
                0.0,
                0.0,
                omega + (imu_count as f32 * 0.02).sin() * 0.05,
            );
            
            slam_imu.lock().unwrap().handle_imu(accel, gyro, 0.01);
            imu_count += 1;
            
            thread::sleep(Duration::from_millis(10)); // 100 Hz
        }
    });

    // Spawn Odometry thread (50 Hz)
    let slam_odom = Arc::clone(&slam);
    let velocity_odom = Arc::clone(&current_velocity);
    let running_odom = Arc::clone(&running);
    let odom_handle = thread::spawn(move || {
        let mut odom_count = 0;
        while *running_odom.lock().unwrap() {
            let (v, omega) = *velocity_odom.lock().unwrap();
            
            // Simulate odometry with noise
            let v_noisy = v + (odom_count as f32 * 0.03).sin() * 0.05;
            let omega_noisy = omega + (odom_count as f32 * 0.04).cos() * 0.02;
            
            slam_odom.lock().unwrap().handle_odom(v_noisy, 0.0, omega_noisy, 0.02);
            odom_count += 1;
            
            thread::sleep(Duration::from_millis(20)); // 50 Hz
        }
    });

    // Spawn GPS thread (5 Hz)
    let slam_gps = Arc::clone(&slam);
    let running_gps = Arc::clone(&running);
    let gps_handle = thread::spawn(move || {
        let mut gps_count = 0;
        while *running_gps.lock().unwrap() {
            let pos = slam_gps.lock().unwrap().get_state().position;
            
            // Simulate GPS with realistic noise (±0.5m accuracy)
            let gps_noise = Vec2::new(
                (gps_count as f32 * 0.1).sin() * 0.3,
                (gps_count as f32 * 0.15).cos() * 0.3,
            );
            let gps_pos = pos + gps_noise;
            
            slam_gps.lock().unwrap().handle_gps(gps_pos, 0.5);
            gps_count += 1;
            
            thread::sleep(Duration::from_millis(200)); // 5 Hz
        }
    });

    // Main control loop (10 Hz)
    let mut completed = false;
    
    while start_time.elapsed().as_secs_f32() < max_duration_secs {
        // Compute control command
        let (v, omega) = {
            let slam_locked = slam.lock().unwrap();
            slam_locked.compute_control().unwrap_or((0.0, 0.0))
        };
        
        // Update shared velocity
        *current_velocity.lock().unwrap() = (v, omega);
        
        // Check if we've completed the loop
        {
            let guard = slam.lock().unwrap();
            let state = guard.get_state();
            let distance_to_start = state.position.distance(Vec2::ZERO);
            
            // Consider complete if: we've traveled for at least 15s AND we're close to start
            if start_time.elapsed().as_secs() >= 15 && distance_to_start < completion_threshold {
                completed = true;
                break;
            }
        }
        
        thread::sleep(Duration::from_millis(100)); // 10 Hz
    }

    // Stop all threads
    *running.lock().unwrap() = false;
    
    // Wait for all threads to finish
    imu_handle.join().unwrap();
    odom_handle.join().unwrap();
    gps_handle.join().unwrap();

    // Final results
    let final_pos = {
        let guard = slam.lock().unwrap();
        let state = guard.get_state();
        state.position
    };
    let final_error = final_pos.distance(Vec2::ZERO);
    
    println!("\nMulti-threaded sensor fusion test:");
    println!("  Final position: ({:.3}, {:.3})", final_pos.x, final_pos.y);
    println!("  Loop closure error: {:.3}m", final_error);
    println!("  Completed: {}", completed);
    
    // Validate accuracy - with asynchronous sensor fusion should still achieve sub-meter accuracy
    assert!(completed, "Trajectory should complete within the time limit");
    assert!(final_error < 1.0, 
        "Multi-threaded loop closure error too high: {:.3}m (expected < 1m)", final_error);
}

/// Test that multi-threaded sensor fusion works with different sensor rates
#[test]
fn test_varying_sensor_frequencies() {
    let config = SlamConfig::default();
    let slam = Arc::new(Mutex::new(SlamFilter::new(config)));

    // Simple straight line trajectory
    let waypoints = vec![
        Vec2::new(0.0, 0.0),
        Vec2::new(5.0, 0.0),
        Vec2::new(10.0, 0.0),
    ];
    slam.lock().unwrap().set_trajectory(&waypoints);

    let current_velocity = Arc::new(Mutex::new((0.0f32, 0.0f32)));
    let running = Arc::new(Mutex::new(true));
    let start_time = Instant::now();
    let duration_secs = 10.0;

    // High-frequency IMU (200 Hz)
    let slam_imu = Arc::clone(&slam);
    let velocity_imu = Arc::clone(&current_velocity);
    let running_imu = Arc::clone(&running);
    let imu_handle = thread::spawn(move || {
        while *running_imu.lock().unwrap() {
            let (_, omega) = *velocity_imu.lock().unwrap();
            let accel = Vec3::new(0.0, 0.0, 9.81);
            let gyro = Vec3::new(0.0, 0.0, omega);
            slam_imu.lock().unwrap().handle_imu(accel, gyro, 0.005);
            thread::sleep(Duration::from_millis(5)); // 200 Hz
        }
    });

    // Low-frequency GPS (1 Hz)
    let slam_gps = Arc::clone(&slam);
    let running_gps = Arc::clone(&running);
    let gps_handle = thread::spawn(move || {
        while *running_gps.lock().unwrap() {
            let pos = slam_gps.lock().unwrap().get_state().position;
            slam_gps.lock().unwrap().handle_gps(pos, 1.0);
            thread::sleep(Duration::from_secs(1)); // 1 Hz
        }
    });

    // Control loop
    while start_time.elapsed().as_secs_f32() < duration_secs {
        let (v, omega) = {
            let slam_locked = slam.lock().unwrap();
            slam_locked.compute_control().unwrap_or((0.0, 0.0))
        };
        *current_velocity.lock().unwrap() = (v, omega);
        
        slam.lock().unwrap().handle_odom(v, 0.0, omega, 0.1);
        
        thread::sleep(Duration::from_millis(100));
    }

    *running.lock().unwrap() = false;
    imu_handle.join().unwrap();
    gps_handle.join().unwrap();

    let final_pos = {
        let guard = slam.lock().unwrap();
        let state = guard.get_state();
        state.position
    };
    
    println!("\nVarying sensor frequencies test:");
    println!("  Final position: ({:.3}, {:.3})", final_pos.x, final_pos.y);
    
    // Should have made significant forward progress
    assert!(final_pos.x > 5.0, "Should have traveled forward at least 5m, got {:.2}m", final_pos.x);
}

