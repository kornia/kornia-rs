# Kornia: kornia-3d

[![Crates.io](https://img.shields.io/crates/v/kornia-3d.svg)](https://crates.io/crates/kornia-3d)
[![Documentation](https://docs.rs/kornia-3d/badge.svg)](https://docs.rs/kornia-3d)
[![License](https://img.shields.io/crates/l/kornia-3d.svg)](https://github.com/kornia/kornia/blob/main/LICENSE)

> **3D computer vision and geometry library.**

## 🚀 Overview

`kornia-3d` focuses on processing 3D data, including point clouds and geometric transformations, and aims to encompass everything 3D, with a roadmap toward richer 3D and map representations for large-scale environments. It provides tools for loading 3D data, performing rigid body transformations, and solving geometric problems like Perspective-n-Point (PnP), point cloud registration, and large-scale mapping

## 🔑 Key Features

*   **Point Cloud I/O:** Read and write support for standard 3D formats (e.g., PLY, PCD, XYZ).
*   **Geometric Transforms:** Apply rigid body transformations (rotation + translation) to point clouds.
*   **Lie Algebra Integration:** Built on `kornia-algebra` for robust SE(3) and SO(3) manipulations.
*   **Registration:** Algorithms like Iterative Closest Point (ICP) for aligning point clouds.
*   **Fundamental RANSAC:** Seven-point minimal hypotheses (every real solution that satisfies the oriented epipolar constraint on its sample, as in DEGENSAC), with eight-point inlier refinement. `pose::fundamental_7point` exposes the minimal solver; `ransac::estimators::Fundamental8PointEstimator` and `pose::ransac_fundamental_8point` retain the original sampling path for comparisons. The two-view builder also accepts `Fundamental7ptSolver`; its existing `Fundamental8ptSolver` default is preserved.
    `pose::RansacParams::confidence` accepts an explicit sampling target; `None` preserves each model family's default. Exhaustive struct literals must add `confidence: None` or use `..Default::default()`. Adaptive caps use the all-inlier probability for distinct sampling without replacement.
*   **PnP Solvers:** Solve for camera pose given 3D-2D point correspondences: EPnP, AP3P and SQPnP (`pnp::solve_sqpnp`, minimises the object-space error over all rotations), with RANSAC and optional LM refinement.

## 📦 Installation

Add the following to your `Cargo.toml`:

```toml
[dependencies]
kornia-3d = "0.1.0"
```

## 🛠️ Usage

### Reading a Point Cloud

```rust
use kornia_3d::io::ply::{read_ply_binary, PlyType};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Load point cloud
    // This expects a PLY file in the specified binary format
    let pc = read_ply_binary("tests/data/box_stack.ply", PlyType::XYZRgbNormals);

    match pc {
        Ok(pc) => {
             println!("Loaded {} points", pc.len());
             if let Some(p) = pc.points().first() {
                 println!("First point: {:?}", p);
             }
        }
        Err(e) => println!("Could not load point cloud: {}", e),
    }

    Ok(())
}
```

## 🧩 Modules

*   **`io`**: Input/Output for 3D file formats.
*   **`linalg`**: Linear algebra utilities.
*   **`ops`**: Operations on 3D data processing.
*   **`pointcloud`**: Point cloud data structures and operations.
*   **`pnp`**: Perspective-n-Point solvers.
*   **`pose`**: Pose estimation algorithms.
*   **`registration`**: Point cloud alignment (ICP).
*   **`transforms`**: Geometric transformations for 3D points.

## 💡 Related Examples

You can find comprehensive examples in the `examples` folder of the repository:

*   [`pnp_demo`](../../examples/pnp_demo): Example of solving the Perspective-n-Point problem.
*   [`icp_registration`](../../examples/icp_registration): Point cloud registration using ICP.

## 🤝 Contributing

Contributions are welcome! This crate is part of the Kornia workspace. Please refer to the main repository for contribution guidelines.

## 📄 License

This crate is licensed under the Apache-2.0 License.
