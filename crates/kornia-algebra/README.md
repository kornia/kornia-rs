# Kornia: kornia-algebra

[![Crates.io](https://img.shields.io/crates/v/kornia-algebra.svg)](https://crates.io/crates/kornia-algebra)
[![Documentation](https://docs.rs/kornia-algebra/badge.svg)](https://docs.rs/kornia-algebra)
[![License](https://img.shields.io/crates/l/kornia-algebra.svg)](https://github.com/kornia/kornia/blob/main/LICENSE)

> **Geometric algebra and Lie theory for computer vision and robotics.**

## 🚀 Overview

`kornia-algebra` is the mathematical backbone of the Kornia ecosystem. It provides robust, strongly-typed implementations of core linear algebra, Lie groups, and (in the future) general numerical and non-linear optimization tools tailored for robotics and computer vision. Built on top of `glam` and its SIMD-accelerated primitives, it ensures high performance while offering a clean, mathematically correct API for 2D and 3D transformations and beyond.

## 🔑 Key Features

*   **Lie Groups:** Full implementations of **SO(2)**, **SE(2)**, **SO(3)**, and **SE(3)**.
*   **Manifold Operations:** Exp/Log maps, adjoints, and tangent space operations.
*   **Optimization Ready:** Analytical Jacobians for all group operations.
*   **Linear Algebra:** Typed vectors and matrices (`Vec3F32`, `Mat4F32`) with SIMD acceleration (via `glam`).
*   **Fixed-size f64 blocks:** `SMatF64<R, C>` / `SVecF64<N>` past 4x4 (`Mat6F64`, `Mat9F64`, `Mat15F64`), via `nalgebra`, for pose and IMU-state Jacobians and covariances.
*   **Quaternions:** Robust unit quaternion support for 3D rotations.

## 📦 Installation

Add the following to your `Cargo.toml`:

```toml
[dependencies]
kornia-algebra = "0.1.0"
```

## 🛠️ Usage

### 3D Rigid Body Transformation (SE3)

```rust
use kornia_algebra::{SE3F32, SO3F32, Vec3AF32};

fn main() {
    // 1. Define a rotation (90 deg around X)
    let rotation = SO3F32::exp(Vec3AF32::new(std::f32::consts::FRAC_PI_2, 0.0, 0.0));

    // 2. Define a translation (1m in Y)
    let translation = Vec3AF32::new(0.0, 1.0, 0.0);

    // 3. Create SE(3) pose
    let pose = SE3F32::new(rotation, translation);

    // 4. Transform a point (0, 0, 0)
    let point = Vec3AF32::ZERO;
    let transformed = pose * point;

    println!("Transformed point: {:?}", transformed);
}
```

### Checked Rotation Matrix Conversion (SO3)

When converting untrusted 3×3 matrices (e.g. from telemetry, external files, or intermediate floating-point operations) into a rotation, use `SO3F64::try_from_matrix` instead of the unchecked `from_matrix`. It checks that the matrix is orthonormal (`Rᵀ R ≈ I`) within the user-specified tolerance and has a positive determinant (`det R > 0`), ensuring proper handedness:

```rust
use kornia_algebra::{Mat3F64, SO3F64, RotationError};

// Construct a matrix from row-major flat data
let mat = Mat3F64::from_rows_array(&[
    0.0, -1.0, 0.0,
    1.0,  0.0, 0.0,
    0.0,  0.0, 1.0,
]);

// Checked conversion rejecting invalid matrices or reflections
let rot = SO3F64::try_from_matrix(&mat, 1e-6)?;
# Ok::<(), RotationError>(())
```

### Interpolation

```rust
use kornia_algebra::{SE3F32, SO3F32, Vec3AF32};

fn main() {
    let start = SE3F32::IDENTITY;
    let end = SE3F32::new(SO3F32::IDENTITY, Vec3AF32::new(10.0, 0.0, 0.0));

    // Interpolate halfway (t=0.5)
    let t = 0.5;
    let (delta_trans, delta_rot) = start.rminus(&end);
    let interpolated = start.rplus(delta_trans * t, delta_rot * t);
}
```

## 🧩 Modules

*   **`lie`**: Lie group implementations (SO2, SE2, SO3, SE3, RxSO3, Sim3).
*   **`vec`**: Vector types (`Vec2`, `Vec3`, `Vec4`, and dynamic `DVec`).
*   **`mat`**: Matrix types (`Mat2`, `Mat3`, `Mat4`, and dynamic `DMat`).
*   **`quat`**: Quaternion types.
*   **`linalg`**: General linear algebra helpers (`svd3`, `cholesky`, and Umeyama rigid/similarity alignment including the outlier-robust `umeyama_f64_robust`).
*   **`optim`**: Factor graph optimization module.
*   **`param`**: Parameterization utilities.

### Feature Flags

*   **`approx`**: Enable approximate equality comparisons for algebraic types.

## 💡 Related Examples

You can find comprehensive examples in the `examples` folder of the repository:

*   [`pnp_demo`](../../examples/pnp_demo): Uses algebraic types for PnP solving.
*   [`rotate`](../../examples/rotate): Image rotation using geometric transformations.

## 🤝 Contributing

Contributions are welcome! This crate is part of the Kornia workspace. Please refer to the main repository for contribution guidelines.

## 📄 License

This crate is licensed under the Apache-2.0 License.
