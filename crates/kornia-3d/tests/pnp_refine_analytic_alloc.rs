//! `refine_pose_lm_analytic` must not allocate on a successful solve: it runs once per
//! tracked frame.
//!
//! Its own test binary, because it installs a counting global allocator. The count is per
//! thread so the harness's other threads cannot leak into it.

use kornia_3d::pnp::{refine_pose_lm_analytic, LMRefineParams};
use kornia_3d::ransac::RobustKernelKind;
use kornia_algebra::{Mat3AF32, Vec2F32, Vec3AF32};
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

#[test]
fn refine_pose_lm_analytic_does_not_allocate() {
    let k = Mat3AF32::from_cols(
        Vec3AF32::new(450.0, 0.0, 0.0),
        Vec3AF32::new(0.0, 450.0, 0.0),
        Vec3AF32::new(376.0, 240.0, 1.0),
    );
    let mut world = Vec::new();
    let mut image = Vec::new();
    for i in 0..60 {
        let p = Vec3AF32::new(
            ((i * 37) % 17) as f32 * 0.1 - 0.8,
            ((i * 11) % 13) as f32 * 0.08 - 0.5,
            1.0 + ((i * 7) % 5) as f32 * 0.3,
        );
        // Seen from a camera at (-0.05, 0.02, -0.1) with no rotation.
        let pc = p + Vec3AF32::new(0.05, -0.02, 0.1);
        world.push(p);
        image.push(Vec2F32::new(
            450.0 * pc.x / pc.z + 376.0,
            450.0 * pc.y / pc.z + 240.0,
        ));
    }
    let params = LMRefineParams {
        robust: RobustKernelKind::Huber,
        robust_scale_sq: 25.0,
        ..LMRefineParams::default()
    };

    let before = allocs();
    let result = refine_pose_lm_analytic(
        &world,
        &image,
        &k,
        &Mat3AF32::IDENTITY,
        &Vec3AF32::ZERO,
        &params,
        false,
    );
    let used = allocs() - before;
    assert!(result.is_ok());
    assert_eq!(used, 0, "refine_pose_lm_analytic allocated {used} times");
}
