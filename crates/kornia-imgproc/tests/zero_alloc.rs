//! Steady-state calls of the frontend kernels make no heap allocation.
//!
//! A counting global allocator for this test binary only; everything runs in
//! one `#[test]` so no other test allocates concurrently.

use kornia_image::{Image, ImageSize};
use kornia_imgproc::filter::scharr_spatial_gradient_separable_f32;
use kornia_imgproc::pyramid::{pyrdown_separable_f32, pyrdown_u8};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

struct Counting;

static ALLOCS: AtomicUsize = AtomicUsize::new(0);

// SAFETY: forwards every call to `System` unchanged; only counts.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        System.alloc(layout)
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        System.alloc_zeroed(layout)
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        System.realloc(ptr, layout, new_size)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// Heap allocations made by the second call of `f` (the first warms up).
fn steady_allocs(mut f: impl FnMut()) -> usize {
    f();
    let before = ALLOCS.load(Ordering::Relaxed);
    f();
    ALLOCS.load(Ordering::Relaxed) - before
}

fn lcg(n: usize) -> Vec<u8> {
    let mut x = 7u64;
    (0..n)
        .map(|_| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (x >> 56) as u8
        })
        .collect()
}

#[test]
fn frontend_kernels_do_not_allocate_in_steady_state() -> Result<(), Box<dyn std::error::Error>> {
    let size = ImageSize {
        width: 752,
        height: 480,
    };
    let half = ImageSize {
        width: 376,
        height: 240,
    };
    let src_u8 = Image::<u8, 1>::new(size, lcg(size.width * size.height))?;
    let src_f32 = Image::<f32, 1>::new(
        size,
        src_u8.as_slice().iter().map(|&v| f32::from(v)).collect(),
    )?;
    let mut half_u8 = Image::<u8, 1>::from_size_val(half, 0)?;
    let mut half_f32 = Image::<f32, 1>::from_size_val(half, 0.0)?;
    let mut dx = Image::<f32, 1>::from_size_val(size, 0.0)?;
    let mut dy = Image::<f32, 1>::from_size_val(size, 0.0)?;

    let n = steady_allocs(|| pyrdown_u8(&src_u8, &mut half_u8).unwrap());
    assert_eq!(n, 0, "pyrdown_u8");
    let n = steady_allocs(|| pyrdown_separable_f32(&src_f32, &mut half_f32).unwrap());
    assert_eq!(n, 0, "pyrdown_separable_f32");
    let n = steady_allocs(|| {
        scharr_spatial_gradient_separable_f32(&src_f32, &mut dx, &mut dy).unwrap()
    });
    assert_eq!(n, 0, "scharr_spatial_gradient_separable_f32");
    Ok(())
}
