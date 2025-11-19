# kornia-algebra

**kornia-algebra** is a core library for geometric algebra in the [kornia-rs](https://github.com/kornia/kornia-rs) ecosystem. It provides a robust set of types for linear algebra and Lie theory, designed specifically for computer vision and robotics.

It is built on top of the excellent [glam](https://github.com/bitshifter/glam-rs) crate, providing strict newtypes and extending functionality for the aglebra types.

Most operations are accelerated using SIMD instructions thanks to `glam`'s optimized implementation.

## Features

- **Strict Type Definitions**: Newtype wrappers around `glam` types (`Vec3`, `Mat3`, `Quat`, etc.) to ensure API stability and allow for extension traits.
- **Lie Groups**: Comprehensive implementations of common Lie groups used in robotics and vision:
  - **SO(2)**: 2D Rotations.
  - **SE(2)**: 2D Rigid Body Transformations (Rotation + Translation).
  - **SO(3)**: 3D Rotations (unit quaternions).
  - **SE(3)**: 3D Rigid Body Transformations (Rotation + Translation).
- **Lie Algebra Operations**: Full support for manifold operations:
  - Exponential (`exp`) and Logarithmic (`log`) maps.
  - Adjoint representation (`adjoint`).
  - Hat (`hat`) and Vee (`vee`) operators.
  - Jacobians (left and right) for optimization.
- **Differentiation**: Analytical Jacobians for Lie group operations, essential for non-linear least squares and Kalman filtering.

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
kornia-algebra = "0.1"
```

### Basic Linear Algebra

```rust
use kornia_algebra::{Vec3, Mat3};

let v = Vec3::new(1.0, 2.0, 3.0);
let m = Mat3::IDENTITY;

let v_transformed = m * v;
assert_eq!(v_transformed, v);
```

### 3D Rotations (SO3)

```rust
use kornia_algebra::{SO3, Vec3};

// Create a rotation of 90 degrees around X axis
let rotation = SO3::exp(Vec3::new(std::f32::consts::FRAC_PI_2, 0.0, 0.0));

let point = Vec3::new(0.0, 1.0, 0.0);
let rotated_point = rotation * point;

// (0, 1, 0) rotated 90 deg around X becomes (0, 0, 1)
assert!((rotated_point - Vec3::new(0.0, 0.0, 1.0)).length() < 1e-6);
```

### Rigid Body Transformations (SE3)

```rust
use kornia_algebra::{SE3, SO3, Vec3};

// Rotation + Translation
let rotation = SO3::IDENTITY;
let translation = Vec3::new(1.0, 0.0, 0.0);
let pose = SE3::new(rotation, translation);

let point = Vec3::new(0.0, 0.0, 0.0);
let transformed = pose * point;

assert_eq!(transformed, Vec3::new(1.0, 0.0, 0.0));
```

### Lie Algebra Interpolation

```rust
use kornia_algebra::{SE3, Vec3};

let start = SE3::IDENTITY;
// Move 1 meter in X and rotate 90 degrees around Z
let end = SE3::exp(Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 0.0, std::f32::consts::FRAC_PI_2));

// Interpolate halfway
let t = 0.5;
let delta_log = start.rminus(&end); // Logarithmic difference
let interpolated = start.rplus(delta_log * t);
```

### 2D Rigid Body Transformations (SE2)

```rust
use kornia_algebra::{SE2, SO2, Vec2, Vec3};

// Create from angle (radians) and translation
let rotation = SO2::exp(std::f32::consts::FRAC_PI_2); // 90 deg
let translation = Vec2::new(10.0, 0.0);
let pose = SE2::new(rotation, translation);

// Transform a point
let point = Vec2::new(1.0, 0.0);
let transformed = pose * point;

// (1,0) rotated 90 deg -> (0,1) + (10,0) -> (10,1)
assert!((transformed - Vec2::new(10.0, 1.0)).length() < 1e-6);
```

### Optimization Tools (Jacobians)

```rust
use kornia_algebra::{SE3, Vec3};

let pose = SE3::from_random();
let point = Vec3::new(1.0, 0.0, 0.0);

// Compute Jacobians for optimization
// Right Jacobian maps tangent space variations to group variations
let twist = Vec3::new(0.1, 0.0, 0.0); // Small translation in x
// ...
```

### Data Conversions

```rust
use kornia_algebra::{SE3, SO3, Vec3};

// Create from array
let pose_arr = [0.0, 0.0, 0.0, 1.0, 10.0, 20.0, 30.0]; // qx, qy, qz, qw, tx, ty, tz
let pose = SE3::from_array(pose_arr);

// Convert back to array
let arr = pose.to_array();
assert_eq!(arr, pose_arr);

// Convert to 4x4 Matrix
let mat4 = pose.matrix();
```

## License

This crate is licensed under the Apache-2.0 License.
