//! Optional worker-thread pinning for latency-sensitive deployments.
//!
//! `extract()` issues ~38 short Rayon parallel sections per frame. When the OS
//! migrates workers between cores mid-frame, each migration costs warm-cache
//! refill and adds wakeup-latency variance to every dispatch barrier. Pinning
//! Rayon worker *i* to core *i* removes that variance: measured on a 6-core
//! Jetson Orin it tightens the `extract()` median latency by ~1.5-2 ms with
//! identical min latency.
//!
//! Pinning is **opt-in** and process-global (it configures Rayon's global
//! pool), so the *application* decides — a library must not silently grab the
//! global pool. Call [`init_pinned_threadpool`] once at startup, before any
//! Rayon use:
//!
//! ```no_run
//! // Pin one worker to each of cores 0-5 (6 workers).
//! kornia_xfeat::affinity::init_pinned_threadpool(0, 5).unwrap();
//! ```
//!
//! Implementation note: on aarch64-linux the pin is a raw `sched_setaffinity`
//! syscall (stable inline asm, no libc dependency — consistent with the
//! crate's existing FCVTL/FMLA asm helpers). On other targets pinning is a
//! no-op but the pool is still sized to the core range.

/// Pin the *calling thread* to the CPU set in `mask` (bit *i* = core *i*).
///
/// Returns `false` (with a `warn:` line on stderr) if the syscall fails;
/// callers can treat failure as "no pinning" and continue.
#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
pub fn pin_current_thread(mask: u64) -> bool {
    unsafe {
        let m: u64 = mask;
        let ret: i64;
        std::arch::asm!(
            "svc 0",
            in("x8") 122u64,                    // __NR_sched_setaffinity
            inout("x0") 0u64 => ret,            // pid 0 = current thread
            in("x1") 8u64,                      // cpusetsize (bytes)
            in("x2") &m as *const u64 as u64,
            options(nostack),
        );
        if ret < 0 {
            eprintln!("warn: sched_setaffinity(mask={mask:#x}) failed: {ret}");
        }
        ret >= 0
    }
}

/// No-op on targets without the raw-syscall pin path.
#[cfg(not(all(target_arch = "aarch64", target_os = "linux")))]
pub fn pin_current_thread(_mask: u64) -> bool {
    false
}

/// Build the **global** Rayon pool with one worker pinned to each core of
/// `first_core..=last_core` (inclusive), and pin the calling thread to the
/// same range.
///
/// Must be called once, before any Rayon use in the process; returns an error
/// if the global pool already exists. Worker *i* is pinned to core
/// `first_core + (i % n_cores)`.
pub fn init_pinned_threadpool(
    first_core: u32,
    last_core: u32,
) -> Result<(), rayon::ThreadPoolBuildError> {
    assert!(
        first_core <= last_core && last_core < 64,
        "core range {first_core}-{last_core} invalid (need first <= last < 64)"
    );
    let n_cores = (last_core - first_core + 1) as usize;
    // The caller (main thread) participates in Rayon sections — keep it inside
    // the allowed range so it never competes with a pinned worker's core.
    let range_mask: u64 = (first_core..=last_core).fold(0u64, |m, c| m | (1u64 << c));
    pin_current_thread(range_mask);
    rayon::ThreadPoolBuilder::new()
        .num_threads(n_cores)
        .start_handler(move |i| {
            let core = first_core + (i % n_cores) as u32;
            pin_current_thread(1u64 << core);
        })
        .build_global()
}
