# kornia-pnp

Perspective-n-Point (PnP) solvers for kornia with camera distortion support.

## Overview

This crate provides efficient implementations of PnP solvers for estimating camera pose from 2D-3D point correspondences. The latest version includes comprehensive support for camera distortion parameters, making it suitable for real-world camera calibration scenarios.

## Features

- **EPnP Solver**: Efficient Perspective-n-Point solver based on the paper by Lepetit et al.
- **Camera Distortion Support**: Full support for Brown-Conrady distortion model
- **Multiple Interfaces**: Both traditional intrinsics matrix and modern camera model interfaces
- **Automatic Distortion Correction**: Seamless handling of distorted image points
- **Comprehensive Error Handling**: Detailed error types for debugging

## Camera Models

### CameraIntrinsics

Represents the intrinsic parameters of a pinhole camera:

```rust
use kornia_pnp::CameraIntrinsics;

let intrinsics = CameraIntrinsics::new(
    800.0,  // fx: focal length in x direction
    800.0,  // fy: focal length in y direction
    640.0,  // cx: principal point x coordinate
    480.0   // cy: principal point y coordinate
);

// Convert from 3x3 intrinsics matrix
let k = [[800.0, 0.0, 640.0], [0.0, 800.0, 480.0], [0.0, 0.0, 1.0]];
let intrinsics = CameraIntrinsics::from_matrix(&k)?;

// Convert back to matrix
let k_matrix = intrinsics.to_matrix();
```

### PolynomialDistortion

Represents distortion parameters using the Brown-Conrady model:

```rust
use kornia_pnp::PolynomialDistortion;

// No distortion
let no_distortion = PolynomialDistortion::none();

// Radial distortion only (k1, k2)
let radial_distortion = PolynomialDistortion::radial(0.1, 0.01);

// Radial and tangential distortion (k1, k2, p1, p2)
let full_distortion = PolynomialDistortion::radial_tangential(0.1, 0.01, 0.001, 0.001);

// Full Brown-Conrady model with all coefficients
let distortion = PolynomialDistortion {
    k1: 0.1, k2: 0.01, k3: 0.001, k4: 0.0, k5: 0.0, k6: 0.0,
    p1: 0.001, p2: 0.001,
};
```

### CameraModel

Combines intrinsics and distortion into a complete camera model:

```rust
use kornia_pnp::{CameraModel, CameraIntrinsics, PolynomialDistortion};

// Pinhole camera (no distortion)
let intrinsics = CameraIntrinsics::new(800.0, 800.0, 640.0, 480.0);
let pinhole_camera = CameraModel::pinhole(intrinsics);

// Camera with distortion
let distortion = PolynomialDistortion::radial(0.1, 0.01);
let camera_with_distortion = CameraModel::with_distortion(intrinsics, distortion);

// Check if camera has distortion
if camera_with_distortion.has_distortion() {
    println!("Camera has distortion parameters");
}
```

## PnP Solvers

### Basic Usage (No Distortion)

For cameras without distortion or when using pre-undistorted points:

```rust
use kornia_pnp::{solve_pnp, PnPMethod};

let world_points = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]];
let image_points = vec![[100.0, 100.0], [200.0, 100.0], [100.0, 200.0], [200.0, 200.0]];
let k = [[800.0, 0.0, 640.0], [0.0, 800.0, 480.0], [0.0, 0.0, 1.0]];

let result = solve_pnp(&world_points, &image_points, &k, PnPMethod::EPnPDefault)?;

println!("Rotation: {:?}", result.rotation);
println!("Translation: {:?}", result.translation);
println!("Reprojection RMSE: {:?} px", result.reproj_rmse);
```

### Advanced Usage (With Distortion)

For cameras with distortion parameters:

```rust
use kornia_pnp::{solve_pnp_with_camera, CameraModel, CameraIntrinsics, PolynomialDistortion, PnPMethod};

// Create camera model with distortion
let intrinsics = CameraIntrinsics::new(800.0, 800.0, 640.0, 480.0);
let distortion = PolynomialDistortion::radial(0.1, 0.01);
let camera = CameraModel::with_distortion(intrinsics, distortion);

// World points (3D coordinates)
let world_points = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]];

// Image points (distorted pixel coordinates)
let distorted_image_points = vec![[105.2, 98.7], [210.1, 95.3], [98.9, 215.6], [208.7, 212.1]];

// Solve PnP with automatic distortion correction
let result = solve_pnp_with_camera(
    &world_points, 
    &distorted_image_points, 
    &camera, 
    PnPMethod::EPnPDefault
)?;

println!("Estimated pose with distortion handling:");
println!("Rotation: {:?}", result.rotation);
println!("Translation: {:?}", result.translation);
println!("Reprojection RMSE: {:?} px", result.reproj_rmse);
```

### Convenience Function

For quick usage with distortion parameters:

```rust
use kornia_pnp::{solve_pnp_with_distortion, PolynomialDistortion, PnPMethod};

let k = [[800.0, 0.0, 640.0], [0.0, 800.0, 480.0], [0.0, 0.0, 1.0]];
let distortion = PolynomialDistortion::radial(0.1, 0.01);

let result = solve_pnp_with_distortion(
    &world_points, 
    &distorted_image_points, 
    &k, 
    &distortion, 
    PnPMethod::EPnPDefault
)?;
```

## Distortion Operations

### Applying Distortion

```rust
use kornia_pnp::{CameraModel, CameraIntrinsics, PolynomialDistortion};

let camera = CameraModel::with_distortion(
    CameraIntrinsics::new(800.0, 800.0, 640.0, 480.0),
    PolynomialDistortion::radial(0.1, 0.01)
);

// Undistorted point
let undistorted_point = [100.0, 200.0];

// Apply distortion
let (x_dist, y_dist) = camera.distort_point(undistorted_point[0], undistorted_point[1])?;
println!("Distorted point: ({}, {})", x_dist, y_dist);

// Distort multiple points
let undistorted_points = vec![[100.0, 200.0], [300.0, 400.0]];
let distorted_points = camera.distort_points(&undistorted_points)?;
```

### Removing Distortion

```rust
// Distorted point
let distorted_point = [105.2, 198.7];

// Remove distortion (iterative method)
let (x_undist, y_undist) = camera.undistort_point(distorted_point[0], distorted_point[1])?;
println!("Undistorted point: ({}, {})", x_undist, y_undist);

// Undistort multiple points
let distorted_points = vec![[105.2, 198.7], [310.1, 395.3]];
let undistorted_points = camera.undistort_points(&distorted_points)?;
```

## Error Handling

The crate provides comprehensive error handling:

```rust
use kornia_pnp::{PnPError, CameraError};

match solve_pnp_with_camera(&world_points, &image_points, &camera, method) {
    Ok(result) => {
        println!("Success: {:?}", result.translation);
    }
    Err(PnPError::InsufficientCorrespondences { required, actual }) => {
        println!("Need at least {} points, got {}", required, actual);
    }
    Err(PnPError::MismatchedArrayLengths { left_name, left_len, right_name, right_len }) => {
        println!("Array length mismatch: {} ({}) != {} ({})", left_name, left_len, right_name, right_len);
    }
    Err(PnPError::SvdFailed(msg)) => {
        println!("SVD computation failed: {}", msg);
    }
    Err(PnPError::CameraError(msg)) => {
        println!("Camera model error: {}", msg);
    }
}
```

## Examples

### Complete Example with Distortion

```rust
use kornia_pnp::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create camera model with distortion
    let intrinsics = CameraIntrinsics::new(800.0, 800.0, 640.0, 480.0);
    let distortion = PolynomialDistortion::radial(0.1, 0.01);
    let camera = CameraModel::with_distortion(intrinsics, distortion);

    // Define 3D world points (e.g., corners of a square)
    let world_points = vec![
        [0.0, 0.0, 0.0],   // bottom-left
        [1.0, 0.0, 0.0],   // bottom-right
        [1.0, 1.0, 0.0],   // top-right
        [0.0, 1.0, 0.0],   // top-left
    ];

    // Define corresponding 2D image points (distorted)
    let distorted_image_points = vec![
        [105.2, 98.7],     // distorted bottom-left
        [210.1, 95.3],     // distorted bottom-right
        [208.7, 212.1],    // distorted top-right
        [98.9, 215.6],     // distorted top-left
    ];

    // Solve PnP with automatic distortion handling
    let result = solve_pnp_with_camera(
        &world_points,
        &distorted_image_points,
        &camera,
        PnPMethod::EPnPDefault
    )?;

    println!("Estimated camera pose:");
    println!("Translation: {:?}", result.translation);
    println!("Rotation matrix:");
    for row in &result.rotation {
        println!("  {:?}", row);
    }
    if let Some(rmse) = result.reproj_rmse {
        println!("Reprojection RMSE: {:.3} pixels", rmse);
    }

    Ok(())
}
```

## Performance Considerations

- **Distortion Correction**: The iterative undistortion method typically converges in 5-10 iterations
- **Memory Usage**: The camera model approach may use slightly more memory but provides better type safety
- **Backward Compatibility**: All existing code using the intrinsics matrix interface continues to work unchanged

## Migration Guide

### From Intrinsics Matrix to Camera Model

**Before (old interface):**
```rust
let k = [[800.0, 0.0, 640.0], [0.0, 800.0, 480.0], [0.0, 0.0, 1.0]];
let result = solve_pnp(&world_points, &image_points, &k, method)?;
```

**After (new interface with distortion):**
```rust
let intrinsics = CameraIntrinsics::from_matrix(&k)?;
let distortion = PolynomialDistortion::radial(0.1, 0.01);
let camera = CameraModel::with_distortion(intrinsics, distortion);
let result = solve_pnp_with_camera(&world_points, &image_points, &camera, method)?;
```

**After (new interface without distortion):**
```rust
let intrinsics = CameraIntrinsics::from_matrix(&k)?;
let camera = CameraModel::pinhole(intrinsics);
let result = solve_pnp_with_camera(&world_points, &image_points, &camera, method)?;
```

## License

This crate is part of the kornia project and follows the same licensing terms.
