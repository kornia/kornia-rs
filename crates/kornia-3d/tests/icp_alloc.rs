//! A warmed real-time ICP call must not allocate: pyramid rebuild plus every level.
//!
//! Its own test binary, because it installs a counting global allocator. The count is per
//! thread so the harness's other threads cannot leak into it.

use kornia_3d::registration::{
    icp_projective_plane_with, Association, DepthIntrinsics, DepthNoise, IcpKernel,
    IcpPlaneCriteria, IcpWorkspace, PyramidScratch, RgbdPyramid,
};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

struct Counting;

thread_local! {
    static ALLOCS: Cell<usize> = const { Cell::new(0) };
}

// SAFETY: forwards every call to the system allocator unchanged; the counter is a plain
// thread-local `Cell` with const initialisation, so touching it never allocates or re-enters.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.with(|c| c.set(c.get() + 1));
        // SAFETY: same contract as the caller's.
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` came from `System.alloc` above with this layout.
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCS.with(|c| c.set(c.get() + 1));
        // SAFETY: same contract as the caller's.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

fn allocs() -> usize {
    ALLOCS.with(|c| c.get())
}

/// A wall corner and a sphere, z-depth in millimetres.
fn render(intr: &DepthIntrinsics, shift_x: f64) -> Vec<u16> {
    let planes = [
        ([0.0, 0.0, 1.0], 2.0),
        ([1.0, 0.0, 0.0], 0.8),
        ([0.0, 1.0, 0.0], 0.6),
    ];
    let (c, rad) = ([-0.2, -0.15, 1.2], 0.3);
    let o = [shift_x, 0.0, 0.0];
    let dot = |a: &[f64; 3], b: &[f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let mut out = vec![0u16; intr.width * intr.height];
    for v in 0..intr.height {
        for u in 0..intr.width {
            let d = [
                (u as f64 - intr.cx) / intr.fx,
                (v as f64 - intr.cy) / intr.fy,
                1.0,
            ];
            let mut best = f64::INFINITY;
            for (n, off) in &planes {
                let den = dot(n, &d);
                if den.abs() > 1e-12 {
                    let t = (off - dot(n, &o)) / den;
                    if t > 0.05 && t < best {
                        best = t;
                    }
                }
            }
            let oc = [o[0] - c[0], o[1] - c[1], o[2] - c[2]];
            let (a, b, cc) = (dot(&d, &d), dot(&oc, &d), dot(&oc, &oc) - rad * rad);
            let disc = b * b - a * cc;
            if disc >= 0.0 {
                let t = (-b - disc.sqrt()) / a;
                if t > 0.05 && t < best {
                    best = t;
                }
            }
            if best.is_finite() {
                out[v * intr.width + u] = (best * 1000.0).round() as u16;
            }
        }
    }
    out
}

#[test]
fn warmed_call_does_not_allocate() -> Result<(), Box<dyn std::error::Error>> {
    let intr = DepthIntrinsics {
        fx: 200.0,
        fy: 200.0,
        cx: 159.5,
        cy: 99.5,
        width: 320,
        height: 200,
    };
    let kf_depth = render(&intr, 0.0);
    let frames = [render(&intr, 0.01), render(&intr, 0.02)];
    const IDENTITY: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    for (kernel, association, decimation) in [
        (IcpKernel::Simd, Association::Nearest, 0),
        (IcpKernel::Simd, Association::Hybrid, 0),
        (IcpKernel::Simd, Association::Nearest, 1),
        (IcpKernel::Scalar, Association::Nearest, 0),
        (IcpKernel::Scalar, Association::Hybrid, 0),
    ] {
        let criteria = IcpPlaneCriteria {
            association,
            depth_noise: DepthNoise::Stereo {
                focal_px: 200.0,
                baseline_m: 0.075,
                sigma_disparity_px: 0.05,
            },
            ..Default::default()
        };
        let mut ws = IcpWorkspace::new(kernel);
        let mut scratch = PyramidScratch::default();
        let mut pyr = RgbdPyramid { levels: Vec::new() };
        let mut keyframe = RgbdPyramid { levels: Vec::new() };
        keyframe.rebuild_decimated_from_depth_mm(
            &kf_depth,
            &intr,
            decimation,
            3,
            kernel,
            &mut scratch,
        )?;
        // warm-up: buffers grow to this image size
        pyr.rebuild_decimated_from_depth_mm(
            &frames[0],
            &intr,
            decimation,
            3,
            kernel,
            &mut scratch,
        )?;
        icp_projective_plane_with(&mut ws, &pyr, &keyframe, IDENTITY, [0.0; 3], &criteria)?;

        let before = allocs();
        pyr.rebuild_decimated_from_depth_mm(
            &frames[1],
            &intr,
            decimation,
            3,
            kernel,
            &mut scratch,
        )?;
        let r = icp_projective_plane_with(&mut ws, &pyr, &keyframe, IDENTITY, [0.0; 3], &criteria)?;
        let n = allocs() - before;
        assert!(r.num_associated > 1000);
        assert_eq!(
            n, 0,
            "{kernel:?} {association:?} decimation {decimation}: {n} allocations in a \
                 warmed call"
        );
    }
    Ok(())
}
