//! Times one full 320x180, 3-level projective point-to-plane ICP solve.
//!
//! Renders a synthetic corner-and-sphere scene from two nearby camera poses
//! (mirroring the in-crate test renderer), builds both pyramids, then times
//! `icp_projective_plane` alone. Run in release:
//!
//! ```sh
//! cargo build --release -p kornia-3d --example icp_bench -j1
//! ./target/release/examples/icp_bench
//! ```

use std::time::Instant;

use kornia_3d::registration::{
    icp_projective_plane, DepthIntrinsics, IcpPlaneCriteria, RgbdIcpError, RgbdPyramid,
};

/// Infinite plane `normal . p = d` and sphere, in world coordinates.
struct Plane {
    normal: [f64; 3],
    d: f64,
}

struct Sphere {
    center: [f64; 3],
    radius: f64,
}

fn dot(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Concave three-plane corner plus a sphere, raycast to a u16 millimetre
/// z-depth image from camera pose `T_world_cam`.
fn render_depth_mm(
    intr: &DepthIntrinsics,
    rot_world_cam: &[[f64; 3]; 3],
    t_world_cam: &[f64; 3],
) -> Vec<u16> {
    let planes = [
        Plane {
            normal: [0.0, 0.0, 1.0],
            d: 2.0,
        },
        Plane {
            normal: [1.0, 0.0, 0.0],
            d: 0.8,
        },
        Plane {
            normal: [0.0, 1.0, 0.0],
            d: 0.6,
        },
    ];
    let sphere = Sphere {
        center: [-0.2, -0.15, 1.2],
        radius: 0.3,
    };
    const NEAR_M: f64 = 0.05;

    let mut depth = vec![0u16; intr.width * intr.height];
    for v in 0..intr.height {
        for u in 0..intr.width {
            // ray with dir_cam.z = 1 so the ray parameter equals z-depth
            let dir_cam = [
                (u as f64 - intr.cx) / intr.fx,
                (v as f64 - intr.cy) / intr.fy,
                1.0,
            ];
            let dir = [
                dot(&rot_world_cam[0], &dir_cam),
                dot(&rot_world_cam[1], &dir_cam),
                dot(&rot_world_cam[2], &dir_cam),
            ];

            let mut best = f64::INFINITY;
            for plane in &planes {
                let denom = dot(&plane.normal, &dir);
                if denom.abs() < 1e-12 {
                    continue;
                }
                let t = (plane.d - dot(&plane.normal, t_world_cam)) / denom;
                if t > NEAR_M && t < best {
                    best = t;
                }
            }
            let oc = [
                t_world_cam[0] - sphere.center[0],
                t_world_cam[1] - sphere.center[1],
                t_world_cam[2] - sphere.center[2],
            ];
            let a = dot(&dir, &dir);
            let b = dot(&oc, &dir);
            let disc = b * b - a * (dot(&oc, &oc) - sphere.radius * sphere.radius);
            if disc >= 0.0 {
                for t in [(-b - disc.sqrt()) / a, (-b + disc.sqrt()) / a] {
                    if t > NEAR_M && t < best {
                        best = t;
                    }
                }
            }

            let mm = (best * 1000.0).round();
            if mm.is_finite() && mm >= 1.0 && mm <= u16::MAX as f64 {
                depth[v * intr.width + u] = mm as u16;
            }
        }
    }
    depth
}

/// Rotation of `angle` radians about the y axis.
fn rot_y(angle: f64) -> [[f64; 3]; 3] {
    let (s, c) = angle.sin_cos();
    [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]
}

fn main() -> Result<(), RgbdIcpError> {
    let intr = DepthIntrinsics {
        fx: 200.0,
        fy: 200.0,
        cx: 159.5,
        cy: 89.5,
        width: 320,
        height: 180,
    };
    let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    // target camera moved ~1.5 deg + ~2 cm
    let rot_wc = rot_y(1.5f64.to_radians());
    let t_wc = [0.02, 0.0, 0.01];

    let src_depth = render_depth_mm(&intr, &identity, &[0.0; 3]);
    let tgt_depth = render_depth_mm(&intr, &rot_wc, &t_wc);

    // warmup (first-touch allocations), then steady-state per-frame pyramid cost
    for _ in 0..2 {
        RgbdPyramid::from_depth_mm(&src_depth, &intr, 3)?;
    }
    const PYR_RUNS: usize = 10;
    let t0 = Instant::now();
    for _ in 0..PYR_RUNS {
        RgbdPyramid::from_depth_mm(&src_depth, &intr, 3)?;
    }
    let pyramid_ms = t0.elapsed().as_secs_f64() * 1e3 / PYR_RUNS as f64;

    let src_pyr = RgbdPyramid::from_depth_mm(&src_depth, &intr, 3)?;
    let tgt_pyr = RgbdPyramid::from_depth_mm(&tgt_depth, &intr, 3)?;

    // warmup
    for _ in 0..3 {
        icp_projective_plane(
            &src_pyr,
            &tgt_pyr,
            identity,
            [0.0; 3],
            IcpPlaneCriteria::default(),
        )?;
    }

    const RUNS: usize = 20;
    let mut total_ms = 0.0;
    let mut min_ms = f64::INFINITY;
    let mut last = None;
    for _ in 0..RUNS {
        let t = Instant::now();
        let result = icp_projective_plane(
            &src_pyr,
            &tgt_pyr,
            identity,
            [0.0; 3],
            IcpPlaneCriteria::default(),
        )?;
        let ms = t.elapsed().as_secs_f64() * 1e3;
        total_ms += ms;
        min_ms = min_ms.min(ms);
        last = Some(result);
    }

    let mean_ms = total_ms / RUNS as f64;
    println!("pyramid build (per frame): {pyramid_ms:.2} ms");
    println!(
        "icp_projective_plane 320x180 x3 levels: mean {mean_ms:.2} ms, min {min_ms:.2} ms over {RUNS} runs"
    );
    if let Some(result) = last {
        println!(
            "solution: iterations {}, rmse {:.4} mm, inliers {:.1} %",
            result.iterations,
            result.rmse * 1e3,
            result.inlier_fraction * 100.0
        );
    }

    // perf gate: a regression (debug-build measurement, O(N^2) slip) must FAIL, not just print
    const GATE_MS: f64 = 60.0;
    if mean_ms > GATE_MS {
        eprintln!("FAIL: mean {mean_ms:.2} ms/solve exceeds the {GATE_MS} ms gate");
        std::process::exit(1);
    }
    println!("PASS: mean {mean_ms:.2} ms/solve within the {GATE_MS} ms gate");
    Ok(())
}
