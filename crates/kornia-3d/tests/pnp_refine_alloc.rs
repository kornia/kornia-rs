//! Heap-allocation count of `refine_pose_lm` on a fixed PnP problem.
//!
//! Lives in its own integration-test binary because `#[global_allocator]`
//! replaces the allocator for the whole binary; inside the crate's unit tests
//! it would also count allocations from every other test running in parallel.

use kornia_3d::pnp::{
    refine_pose_lm, solve_pnp_ransac, EPnPParams, LMRefineParams, PnPMethod, RansacParams,
};
use kornia_algebra::{Mat3AF32, Vec2F32, Vec3AF32};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Number of `alloc` and `realloc` calls since the last reset.
static ALLOCS: AtomicUsize = AtomicUsize::new(0);

/// Allocator that counts allocation requests and forwards them to [`System`].
struct CountingAlloc;

// SAFETY: every method forwards its arguments unchanged to `System`, which
// upholds the `GlobalAlloc` contract; the only extra work is an atomic counter
// increment, which never allocates and never touches the returned memory.
unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::SeqCst);
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::SeqCst);
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

// Camera Matrix K
fn k_default() -> Mat3AF32 {
    Mat3AF32::from_cols(
        Vec3AF32::new(800.0, 0.0, 0.0),
        Vec3AF32::new(0.0, 800.0, 0.0),
        Vec3AF32::new(640.0, 480.0, 1.0),
    )
}

fn synthetic_problem(n: usize) -> (Vec<Vec3AF32>, Vec<Vec2F32>) {
    let mut world = Vec::with_capacity(n);
    let mut image = Vec::with_capacity(n);

    for i in 0..n {
        let col = (i % 10) as f32;
        let row = ((i / 10) % 10) as f32;

        let x = (col - 4.5) / 4.5;
        let y = (row - 4.5) / 4.5;
        let z = 2.0 + 0.25 * (i % 7) as f32;
        world.push(Vec3AF32::new(x, y, z));
        let u = 800.0 * x / z + 640.0;
        let v = 800.0 * y / z + 480.0;
        image.push(Vec2F32::new(u, v));
    }

    (world, image)
}

/// Allocations made by `f`, and its result.
fn count_allocs<T>(f: impl FnOnce() -> T) -> (usize, T) {
    ALLOCS.store(0, Ordering::SeqCst);
    let out = f();
    (ALLOCS.load(Ordering::SeqCst), out)
}

// One test only: the counter is global, so a second test running in parallel would add to it.
#[test]
fn pnp_refine_allocations() -> Result<(), Box<dyn std::error::Error>> {
    let k = k_default();
    let params = LMRefineParams::default();
    // True pose is identity / zero; start LM from a slightly wrong translation.
    let initial_rotation = Mat3AF32::IDENTITY;
    let initial_translation = Vec3AF32::new(0.05, -0.03, 0.1);

    let mut refine_counts = Vec::new();
    for n in [10, 100] {
        let (world, image) = synthetic_problem(n);
        let (count, result) = count_allocs(|| {
            refine_pose_lm(
                &world,
                &image,
                &k,
                &initial_rotation,
                &initial_translation,
                None,
                &params,
            )
        });
        let result = result?;
        println!(
            "refine_pose_lm      n={n:>3}  allocations={count:>6}  iterations={:?}  rmse={:?}",
            result.num_iterations, result.reproj_rmse
        );
        assert!(
            result.translation.length() < 1e-3,
            "translation {:?}",
            result.translation
        );
        assert!(result.reproj_rmse.unwrap_or(f32::INFINITY) < 1e-2);
        refine_counts.push(count);
    }

    // The VIO path: RANSAC with an LM-refined EPnP final fit, against the same call without LM.
    // Same seed, so both runs draw the same samples; the difference is what the LM refine costs.
    let (world, image) = synthetic_problem(100);
    let ransac = RansacParams {
        random_seed: Some(0),
        ..Default::default()
    };
    let epnp = |refine_lm| {
        PnPMethod::EPnP(EPnPParams {
            refine_lm,
            ..Default::default()
        })
    };
    let (with_lm, result) = count_allocs(|| {
        solve_pnp_ransac(
            &world,
            &image,
            &k,
            None,
            epnp(Some(params.clone())),
            &ransac,
        )
    });
    result?;
    let (without_lm, result) =
        count_allocs(|| solve_pnp_ransac(&world, &image, &k, None, epnp(None), &ransac));
    result?;
    println!("solve_pnp_ransac    n=100  allocations with LM={with_lm}  without LM={without_lm}");

    // Two per refine call, whatever the size: the work-frame median buffer and the factor Vec.
    for count in refine_counts {
        assert!(count <= 2, "refine_pose_lm allocated {count} times");
    }
    // On this clean data RANSAC stops after one sample, so LM runs twice (the minimal-sample fit
    // and the final fit both use the base method), at two allocations each.
    let lm_cost = with_lm.saturating_sub(without_lm);
    assert!(
        lm_cost <= 4,
        "the LM refine added {lm_cost} allocations to solve_pnp_ransac"
    );
    Ok(())
}
