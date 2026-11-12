//! A warmed ICP frame must not allocate: the in-place pyramid rebuild plus every level of the
//! solve.
//!
//! Its own test binary, because it installs a counting global allocator. The count is per
//! thread so the harness's other threads cannot leak into it.

use kornia_3d::camera::PinholeCamera;
use kornia_3d::registration::{
    icp_projective_plane_with, Association, DepthNoise, IcpPlaneCriteria, IcpWorkspace,
    PyramidScratch, RgbdPyramid,
};
use kornia_algebra::{Mat3F64, Vec3F64};
use kornia_image::{Image, ImageSize};
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
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOCS.with(|c| c.set(c.get() + 1));
        // SAFETY: same contract as the caller's.
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` came from `System` above with this layout.
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

const SIZE: ImageSize = ImageSize {
    width: 320,
    height: 200,
};

fn camera() -> PinholeCamera {
    PinholeCamera {
        fx: 200.0,
        fy: 200.0,
        cx: 159.5,
        cy: 99.5,
        ..PinholeCamera::IDENTITY
    }
}

/// A wall corner and a sphere seen from a camera at `(shift_x, 0, 0)`, z-depth in millimetres.
/// With `sparse`, only the left quarter of the columns keeps its depth.
fn render(shift_x: f64, sparse: bool) -> Result<Image<u16, 1>, Box<dyn std::error::Error>> {
    let planes = [
        (Vec3F64::new(0.0, 0.0, 1.0), 2.0),
        (Vec3F64::new(1.0, 0.0, 0.0), 0.8),
        (Vec3F64::new(0.0, 1.0, 0.0), 0.6),
    ];
    let (centre, radius) = (Vec3F64::new(-0.2, -0.15, 1.2), 0.3);
    let origin = Vec3F64::new(shift_x, 0.0, 0.0);
    let cam = camera();
    let mut out = Image::<u16, 1>::from_size_val(SIZE, 0)?;
    for (idx, px) in out.as_slice_mut().iter_mut().enumerate() {
        let (u, v) = (idx % SIZE.width, idx / SIZE.width);
        if sparse && u >= SIZE.width / 4 {
            continue;
        }
        let d = Vec3F64::new(
            (u as f64 - cam.cx) / cam.fx,
            (v as f64 - cam.cy) / cam.fy,
            1.0,
        );
        let mut best = f64::INFINITY;
        for (n, off) in &planes {
            let den = n.dot(d);
            if den.abs() > 1e-12 {
                let t = (off - n.dot(origin)) / den;
                if t > 0.05 && t < best {
                    best = t;
                }
            }
        }
        let oc = origin - centre;
        let (a, b, c) = (d.dot(d), oc.dot(d), oc.dot(oc) - radius * radius);
        let disc = b * b - a * c;
        if disc >= 0.0 {
            let t = (-b - disc.sqrt()) / a;
            if t > 0.05 && t < best {
                best = t;
            }
        }
        if best.is_finite() {
            *px = (best * 1000.0).round() as u16;
        }
    }
    Ok(out)
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "about 45 s in a debug build; runs in the release test job"
)]
fn warmed_call_does_not_allocate() -> Result<(), Box<dyn std::error::Error>> {
    let cam = camera();
    let keyframe = RgbdPyramid::from_depth_mm(&render(0.0, false)?, &cam, 3)?;
    let full = [render(0.01, false)?, render(0.02, false)?];
    let sparse = render(0.01, true)?;
    let stereo = DepthNoise::Stereo {
        focal_px: 200.0,
        baseline_m: 0.075,
        sigma_disparity_px: 0.05,
    };

    let constant = IcpPlaneCriteria::default().depth_noise;

    // (association, warm-up frame is sparse, depth noise); a sparse warm-up then a full frame
    // is the case where buffers sized by the inlier count, not the image, regrow
    let (near, bilinear, hybrid) = (
        Association::Nearest,
        Association::Bilinear,
        Association::Hybrid,
    );
    for (association, is_sparse, noise) in [
        (near, false, constant),
        (near, false, stereo),
        (near, true, constant),
        (near, true, stereo),
        (bilinear, true, constant),
        (bilinear, true, stereo),
        (hybrid, false, stereo),
        (hybrid, true, constant),
        (hybrid, true, stereo),
    ] {
        let warm = if is_sparse { &sparse } else { &full[0] };
        let mut criteria = IcpPlaneCriteria::default();
        criteria.depth_noise = noise;
        criteria.association = association;
        let (r0, t0) = (Mat3F64::IDENTITY, Vec3F64::ZERO);
        let mut ws = IcpWorkspace::default();
        let mut scratch = PyramidScratch::default();
        let mut pyr = RgbdPyramid::from_depth_mm(warm, &cam, 3)?;
        pyr.rebuild_from_depth_mm(warm, &cam, 3, &mut scratch)?;
        let warm_up = icp_projective_plane_with(&mut ws, &pyr, &keyframe, &r0, &t0, &criteria)?;

        let before = allocs();
        pyr.rebuild_from_depth_mm(&full[1], &cam, 3, &mut scratch)?;
        let r = icp_projective_plane_with(&mut ws, &pyr, &keyframe, &r0, &t0, &criteria)?;
        let n = allocs() - before;
        if is_sparse {
            assert!(
                r.num_inliers > 2 * warm_up.num_inliers,
                "{}",
                warm_up.num_inliers
            );
        }
        assert!(r.num_associated > 1000);
        assert_eq!(
            n, 0,
            "{association:?}, sparse warm-up {is_sparse}, {noise:?}: {n} allocations in a warmed call"
        );
    }
    Ok(())
}
