use crate::{CacheInfoError, CacheLevel, CacheType};
use libc::{c_void, size_t};

const SYSCTL_NAME: &[u8] = b"hw.cachelinesize\0";

#[inline]
fn read_cache_line_size() -> Result<usize, CacheInfoError> {
    let mut value: libc::size_t = 0;
    let mut len: size_t = core::mem::size_of::<libc::size_t>() as size_t;
    let ret = unsafe {
        libc::sysctlbyname(
            SYSCTL_NAME.as_ptr() as *const _,
            &mut value as *mut _ as *mut c_void,
            &mut len,
            core::ptr::null_mut(),
            0,
        )
    };

    let expected = core::mem::size_of::<libc::size_t>() as size_t;
    if ret == 0 && len > 0 && len <= expected {
        Ok(value as usize)
    } else {
        Err(CacheInfoError::Unsupported)
    }
}

/// Returns the line size in bytes of the L1 cache for Apple Silicon.
/// Other cache levels are not exposed on macOS user space.
#[inline]
pub fn get_cache_line_size(
    level: CacheLevel,
    cache_type: CacheType,
) -> Result<usize, CacheInfoError> {
    match level {
        CacheLevel::L1 => match cache_type {
            CacheType::Instruction | CacheType::Data | CacheType::Unified => read_cache_line_size(),
        },
        _ => Err(CacheInfoError::Unsupported),
    }
}
