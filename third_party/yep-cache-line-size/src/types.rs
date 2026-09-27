use core::fmt;

/// Common data structures representing CPU cache metadata.
///
/// These enums are shared across all architecture specific implementations.

/// Type of cache being described.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum CacheType {
    /// Data cache
    Data = 1,
    /// Instruction cache
    Instruction = 2,
    /// Data and Instruction cache
    Unified = 3,
}

/// Identifier for the cache hierarchy level being queried.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheLevel {
    /// Level 1 cache.
    L1,
    /// Level 2 cache.
    L2,
    /// Level 3 cache.
    L3,
    /// Any other cache level that is not explicitly represented above.
    Other(u8),
}

impl CacheLevel {
    /// Creates a [`CacheLevel`] from its numeric identifier as reported by CPUID.
    #[inline]
    pub fn from_u8(level: u8) -> Self {
        match level {
            1 => CacheLevel::L1,
            2 => CacheLevel::L2,
            3 => CacheLevel::L3,
            other => CacheLevel::Other(other),
        }
    }

    /// Returns the numeric identifier associated with the cache level.
    #[inline]
    pub fn as_u8(self) -> u8 {
        match self {
            CacheLevel::L1 => 1,
            CacheLevel::L2 => 2,
            CacheLevel::L3 => 3,
            CacheLevel::Other(value) => value,
        }
    }
}

impl From<u8> for CacheLevel {
    #[inline]
    fn from(level: u8) -> Self {
        CacheLevel::from_u8(level)
    }
}

impl From<CacheLevel> for u8 {
    #[inline]
    fn from(level: CacheLevel) -> Self {
        level.as_u8()
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl From<CacheType> for raw_cpuid::CacheType {
    #[inline]
    fn from(cache_type: CacheType) -> Self {
        match cache_type {
            CacheType::Data => raw_cpuid::CacheType::Data,
            CacheType::Instruction => raw_cpuid::CacheType::Instruction,
            CacheType::Unified => raw_cpuid::CacheType::Unified,
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl From<raw_cpuid::CacheType> for CacheType {
    #[inline]
    fn from(cache_type: raw_cpuid::CacheType) -> Self {
        match cache_type {
            raw_cpuid::CacheType::Data => CacheType::Data,
            raw_cpuid::CacheType::Instruction => CacheType::Instruction,
            raw_cpuid::CacheType::Unified => CacheType::Unified,
            _ => panic!("Unsupported cache type {} returned from CPUID", cache_type),
        }
    }
}

/// Errors that can occur while querying cache metadata.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheInfoError {
    /// The system does not expose cache metadata for the requested query.
    Unsupported,
    /// The requested cache level/type combination is not present.
    NotPresent,
    /// The CPU reported an invalid or out-of-range value.
    InvalidValue,
}

impl fmt::Display for CacheInfoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CacheInfoError::Unsupported => {
                f.write_str("retrieving cache metadata is not supported on this system")
            }
            CacheInfoError::NotPresent => {
                f.write_str("the requested cache level/type combination is not present")
            }
            CacheInfoError::InvalidValue => f.write_str("the CPU reported invalid cache metadata"),
        }
    }
}

impl std::error::Error for CacheInfoError {}
