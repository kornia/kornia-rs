//! A warmed `DenseLlt` must not allocate: it exists to keep the factor, its scratch and the
//! solve column across the iterations of a solver.
//!
//! Its own test binary, because it installs a counting global allocator. The count is per
//! thread so the harness's other threads cannot leak into it.

use faer::{Mat, Par};
use kornia_3d::linalg::dense_llt::DenseLlt;
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

const N: usize = 40;

#[test]
fn a_warmed_factor_and_its_solves_do_not_allocate() {
    // Diagonally dominant, so symmetric positive definite.
    let a = Mat::<f64>::from_fn(N, N, |i, j| {
        if i == j {
            N as f64
        } else {
            1.0 / (1.0 + i.abs_diff(j) as f64)
        }
    });
    let b: Vec<f64> = (0..N).map(|i| (i as f64 * 0.37).sin()).collect();
    let mut rhs = Mat::<f64>::from_fn(N, 3, |i, j| (i + j) as f64);
    let mut llt = DenseLlt::new(N, Par::Seq);
    // Warm-up: the first calls may touch lazily initialised state.
    llt.factor(&a).unwrap();
    llt.solve_column(&b).unwrap();
    llt.solve_in_place(&mut rhs).unwrap();

    let before = allocs();
    for _ in 0..3 {
        llt.factor(&a).unwrap();
        llt.solve_column(&b).unwrap();
        llt.solve_in_place(&mut rhs).unwrap();
    }
    let used = allocs() - before;
    assert_eq!(used, 0, "DenseLlt allocated {used} times");
}
