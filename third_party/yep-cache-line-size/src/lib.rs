//! A library to quickly get the size and line size of your CPU caches.
//!
//! Currently this crate only supports x86 CPUs, since it relies on the `CPUID` instruction, via
//! the [`raw_cpuid`](raw_cpuid) crate. It is a goal to support other architectures; PRs are
//! welcome!
//!
//! Check the [Intel 64 and IA-32 Architectures Software Developers Manual](https://software.intel.com/sites/default/files/managed/39/c5/325462-sdm-vol-1-2abcd-3abcd.pdf)
//! for more information on the `CPUID` instruction.

mod types;
pub use types::{CacheInfoError, CacheLevel, CacheType};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use x86::*;

#[cfg(all(
    any(target_arch = "arm", target_arch = "aarch64"),
    not(all(target_arch = "aarch64", target_os = "macos"))
))]
mod arm;
#[cfg(all(
    any(target_arch = "arm", target_arch = "aarch64"),
    not(all(target_arch = "aarch64", target_os = "macos"))
))]
pub use arm::*;

#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
mod appl_arm;
#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
pub use appl_arm::*;
