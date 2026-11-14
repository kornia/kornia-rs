//! Device impl of the TSDF core's float ops: inline PTX, so no libdevice and
//! sm_75-safe. `fma.rn`/`sqrt.rn` are correctly rounded, the same bits as the
//! host's `f32::mul_add`/`f32::sqrt` (kornia-3d TSDF contract, section 2).

use crate::tsdf_core::F32Ops;
use cuda_device::ptx_asm;

/// `F32Ops` through `fma.rn.f32` / `sqrt.rn.f32`.
pub struct PtxOps;

impl F32Ops for PtxOps {
    #[inline(always)]
    fn fma(a: f32, b: f32, c: f32) -> f32 {
        let y: f32;
        // SAFETY: register-only arithmetic, no memory or side effects.
        unsafe {
            ptx_asm!("fma.rn.f32 %0, %1, %2, %3;", out("=f") y, in("f") a, in("f") b, in("f") c, options(register_only));
        }
        y
    }

    #[inline(always)]
    fn sqrt(x: f32) -> f32 {
        let y: f32;
        // SAFETY: register-only arithmetic, no memory or side effects.
        unsafe {
            ptx_asm!("sqrt.rn.f32 %0, %1;", out("=f") y, in("f") x, options(register_only));
        }
        y
    }
}
