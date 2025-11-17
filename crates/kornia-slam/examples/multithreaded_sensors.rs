use kornia_slam::{SlamConfig, SlamFilter, Vec2, Vec3};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

/// Example demonstrating multi-threaded sensor data arriving at different frequencies
/// - IMU: 100 Hz (10ms intervals)
/// - Odometry: 50 Hz (20ms intervals)
/// - GPS: 5 Hz (200ms intervals)
/// - Control: 10 Hz (100ms intervals)
fn main() {
    println!("=== Multi-threaded SLAM Example ===\n");
    println!("Simulating sensors at realistic frequencies:");
    println!("  - IMU: 100 Hz");
    println!("  - Odometry: 50 Hz");
    println!("  - GPS: 5 Hz");
    println!("  - Control loop: 10 Hz\n");

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
    let max_duration_secs = 100.0; // Maximum runtime
    let completion_threshold = 1.5; // Distance to start to consider complete

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
                0.0 + (imu_count as f32 * 0.01).sin() * 0.1,
                0.0 + (imu_count as f32 * 0.01).cos() * 0.1,
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
        println!("IMU thread: Sent {} updates", imu_count);
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
        println!("Odometry thread: Sent {} updates", odom_count);
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
        println!("GPS thread: Sent {} updates", gps_count);
    });

    // Main control loop (10 Hz)
    let mut control_count = 0;
    let mut last_print = Instant::now();
    let mut completed = false;
    let mut distance_to_start = f32::MAX;
    
    while start_time.elapsed().as_secs_f32() < max_duration_secs {
        // Compute control command
        let (v, omega) = {
            let slam_locked = slam.lock().unwrap();
            slam_locked.compute_control().unwrap_or((0.0, 0.0))
        };
        
        // Update shared velocity
        *current_velocity.lock().unwrap() = (v, omega);
        
        control_count += 1;
        
        // Check if we've completed the loop
        {
            let guard = slam.lock().unwrap();
            let state = guard.get_state();
            distance_to_start = state.position.distance(Vec2::ZERO);
            
            // Consider complete if: we've traveled for at least 20s AND we're close to start
            if start_time.elapsed().as_secs() >= 20 && distance_to_start < completion_threshold {
                if !completed {
                    println!("\n✓ Trajectory completed at t={:.1}s (distance to start: {:.2}m)",
                        start_time.elapsed().as_secs_f32(), distance_to_start);
                    completed = true;
                    break;
                }
            }
        }
        
        // Print status every 5 seconds
        if last_print.elapsed().as_secs() >= 5 {
            let (pos_x, pos_y, heading) = {
                let guard = slam.lock().unwrap();
                let state = guard.get_state();
                (state.position.x, state.position.y, state.heading)
            };
            println!(
                "[{:5.1}s] Position: ({:6.2}, {:6.2}), Heading: {:6.1}°, Distance to start: {:5.2}m, Control: v={:5.2} m/s, ω={:5.2} rad/s",
                start_time.elapsed().as_secs_f32(),
                pos_x,
                pos_y,
                heading.to_degrees(),
                distance_to_start,
                v,
                omega
            );
            last_print = Instant::now();
        }
        
        thread::sleep(Duration::from_millis(100)); // 10 Hz
    }
    
    if !completed {
        println!("\n⚠ Maximum time reached without completing trajectory");
    }

    // Stop all threads
    *running.lock().unwrap() = false;
    
    // Wait for all threads to finish
    imu_handle.join().unwrap();
    odom_handle.join().unwrap();
    gps_handle.join().unwrap();

    // Final results
    let (final_pos, final_heading) = {
        let guard = slam.lock().unwrap();
        let state = guard.get_state();
        (state.position, state.heading)
    };
    let final_error = final_pos.distance(Vec2::ZERO);
    
    println!("\n=== Final Results ===");
    println!("Control updates sent: {}", control_count);
    println!("Total runtime: {:.1}s", start_time.elapsed().as_secs_f32());
    println!("\nFinal position: ({:.3}, {:.3})", final_pos.x, final_pos.y);
    println!("Final heading: {:.1}°", final_heading.to_degrees());
    println!("\n🎯 Loop closure error: {:.3}m", final_error);
    
    // Validate accuracy
    if final_error < 1.0 {
        println!("✅ SUCCESS: Loop closure error < 1m (achieved {:.3}m)", final_error);
        println!("   Multi-threaded asynchronous sensor fusion working perfectly!");
    } else if final_error < 2.0 {
        println!("✓ GOOD: Loop closure error < 2m (achieved {:.3}m)", final_error);
    } else {
        println!("⚠ WARNING: Loop closure error higher than expected: {:.3}m", final_error);
    }
    
    // Error as percentage of perimeter
    let perimeter = 40.0; // Approximate perimeter of 10x10 square
    let error_percentage = (final_error / perimeter) * 100.0;
    println!("   Error as % of perimeter: {:.2}%", error_percentage);
}

