use crate::{CacheInfoError, CacheLevel, CacheType};
use raw_cpuid::{self, CpuId, CpuIdReaderNative};

/// Uses the CPUID family info to detect Zen architecture CPUs.
///
/// Data pulled from https://en.wikichip.org/wiki/amd/cpuid.
#[inline]
fn amd_is_zen(cpuid: &CpuId<CpuIdReaderNative>) -> Result<bool, CacheInfoError> {
    let info = cpuid
        .get_feature_info()
        .ok_or(CacheInfoError::Unsupported)?;
    Ok(matches!(
        (info.base_family_id(), info.extended_family_id()),
        (0xF, 0x8..=0xA)
    ))
}

/// Uses cache parameters to get cache line size at a given level with the provided cache type.
#[inline]
fn generic_cache_line_size(
    cpuid: CpuId<CpuIdReaderNative>,
    level: CacheLevel,
    cache_type: CacheType,
) -> Result<usize, CacheInfoError> {
    let level_id: u8 = level.into();
    let cache_kind: raw_cpuid::CacheType = cache_type.into();
    cpuid
        .get_cache_parameters()
        .ok_or(CacheInfoError::Unsupported)?
        .filter(|cparams| cparams.level() == level_id && cparams.cache_type() == cache_kind)
        .map(|cparams| cparams.coherency_line_size())
        .min()
        .ok_or(CacheInfoError::NotPresent)
}

/// This is computed using tlb info. Instruction and data cache line sizes
/// are available separately for the L1 cache, but only unified is available for L2 and L3 caches.
#[inline]
fn amd_cache_line_size(
    cpuid: CpuId<CpuIdReaderNative>,
    level: CacheLevel,
    cache_type: CacheType,
) -> Result<usize, CacheInfoError> {
    match (level, cache_type) {
        (CacheLevel::L1, CacheType::Instruction) => cpuid
            .get_l1_cache_and_tlb_info()
            .map(|i| i.icache_line_size() as usize)
            .ok_or(CacheInfoError::Unsupported),
        (CacheLevel::L1, CacheType::Data) => cpuid
            .get_l1_cache_and_tlb_info()
            .map(|i| i.dcache_line_size() as usize)
            .ok_or(CacheInfoError::Unsupported),
        (CacheLevel::L2, CacheType::Unified) => cpuid
            .get_l2_l3_cache_and_tlb_info()
            .map(|i| i.l2cache_line_size() as usize)
            .ok_or(CacheInfoError::Unsupported),
        (CacheLevel::L3, CacheType::Unified) => cpuid
            .get_l2_l3_cache_and_tlb_info()
            .map(|i| i.l3cache_line_size() as usize)
            .ok_or(CacheInfoError::Unsupported),
        _ => Err(CacheInfoError::NotPresent),
    }
}

/// Returns the line size in bytes of `level` cache with type `cache_type`.
///
/// The only possibilities for this returning an [`Err`] are if the system does not support cache
/// parameters, in which case we will return `CacheInfoError::NotPresent`, or if the CPU
/// reported an invalid value, in which case we will return `CacheInfoError::InvalidValue`.
///
/// Example:
/// ```
/// use yep_cache_line_size::{get_cache_line_size, CacheLevel, CacheType};
/// let l1d_line_size = get_cache_line_size(CacheLevel::L1, CacheType::Data).unwrap();
/// println!("L1 Data Cache Line Size: {} bytes", l1d_line_size);
/// ```
#[inline]
pub fn get_cache_line_size(
    level: CacheLevel,
    cache_type: CacheType,
) -> Result<usize, CacheInfoError> {
    let cpuid = CpuId::new();
    let vendor = cpuid.get_vendor_info().ok_or(CacheInfoError::Unsupported)?;

    if vendor.as_str() == "AuthenticAMD" && amd_is_zen(&cpuid)? {
        amd_cache_line_size(cpuid, level, cache_type)
    } else {
        generic_cache_line_size(cpuid, level, cache_type)
    }
}
