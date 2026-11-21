//! Pins that a solve's allocation count does not grow with the number of points. A counting
//! allocator applies to the whole process, so this test needs its own binary.

use kornia_3d::{
    ba_affine::{bundle_adjust_affine, AffineBaParams, AffineObservation},
    camera::AffineCamera,
};
use kornia_algebra::Vec3F64;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

struct CountingAllocator;

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

// SAFETY: every method forwards its arguments unchanged to `System`, which upholds the
// `GlobalAlloc` contract; the counter does not touch the allocated memory.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        System.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        System.alloc_zeroed(layout)
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

const CAMERAS: usize = 4;
const MAX_ITERATIONS: usize = 3;
/// Iterations and rejected damping trials allocate too, and rounding can change how many a
/// scene has. Setup growth is far above this: 3,612 allocations at these sizes before the index.
const PER_ITERATION_MARGIN: usize = 16;

/// Allocations in a solve of `n_points` noisy points, each seen by every camera.
fn solve_allocations(n_points: usize) -> usize {
    let truth: Vec<_> = (0..n_points)
        .map(|i| {
            let t = i as f64;
            Vec3F64::new(
                (t * 1.7).sin() * 4.,
                (t * 0.9).cos() * 3.,
                (t * 0.4).sin() * 2.,
            )
        })
        .collect();
    let cameras: Vec<_> = (0..CAMERAS)
        .map(|i| {
            let a = i as f64 * 1.5;
            AffineCamera {
                matrix: [
                    [a.cos() * 3., a.sin() * 3., 0., 100.],
                    [-a.sin() * 2., a.cos() * 2., -2., 200.],
                ],
            }
        })
        .collect();
    let observations: Vec<_> = cameras
        .iter()
        .enumerate()
        .flat_map(|(c, camera)| {
            truth.iter().enumerate().map(move |(p, x)| {
                let mut pixel = camera.project(x);
                pixel.x += 0.3 * ((c * 7 + p * 13) as f64).sin();
                pixel.y -= 0.2 * ((c * 11 + p * 5) as f64).cos();
                AffineObservation {
                    camera_index: c,
                    point_index: p,
                    pixel,
                }
            })
        })
        .collect();
    let fixed: Vec<_> = (0..n_points).map(|p| p < 4).collect();
    let mut start = truth;
    for (i, x) in start.iter_mut().enumerate().skip(4) {
        x.x += 0.1 * (i as f64).sin();
        x.z -= 0.05;
    }
    let params = AffineBaParams {
        max_iterations: MAX_ITERATIONS,
        ..Default::default()
    };
    let before = ALLOCATIONS.load(Ordering::Relaxed);
    bundle_adjust_affine(&cameras, &start, &observations, &fixed, &params).unwrap();
    ALLOCATIONS.load(Ordering::Relaxed) - before
}

#[test]
fn allocations_do_not_scale_with_points() {
    let small = solve_allocations(200);
    let large = solve_allocations(2_000);
    assert!(
        large <= small + PER_ITERATION_MARGIN,
        "{small} allocations at 200 points, {large} at 2000"
    );
}
