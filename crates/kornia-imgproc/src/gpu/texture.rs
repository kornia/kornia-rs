//! Thin safe wrapper around `cuTexObjectCreate` / `cuTexObjectDestroy`.
//!
//! Exposes a 1-channel pitch-2D texture built from an existing device
//! allocation.  Three channels of interleaved RGB data are accessed by
//! setting `width = src_w * 3` so each logical RGB pixel occupies three
//! consecutive single-channel texels.
//!
//! `CU_TR_ADDRESS_MODE_BORDER` is always used: out-of-bounds fetches return
//! 0.0, giving `BORDER_CONSTANT = 0` behavior without an explicit bounds-check
//! in the kernel — the primary source of warp-affine branch divergence.

use cudarc::driver::sys::{
    CUaddress_mode, CUarray_format, CUfilter_mode, CUresourcetype, CUtexObject,
    CUDA_RESOURCE_DESC_st, CUDA_RESOURCE_DESC_st__bindgen_ty_1,
    CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_4, CUDA_TEXTURE_DESC_st,
};

/// RAII guard for a CUDA texture object handle.
pub(super) struct CudaTexObject(CUtexObject);

impl CudaTexObject {
    /// Create a 1-channel pitch-2D float texture with border addressing.
    ///
    /// `dev_ptr` must be the `CUdeviceptr` of an `f32` device allocation.
    /// `width` is the number of `f32` texels per row (pass `src_w * 3` for
    /// interleaved RGB).  `height` is the number of rows (`src_h`).
    pub(super) fn new_pitch2d_border(
        dev_ptr: u64,
        width: usize,
        height: usize,
    ) -> Result<Self, String> {
        let res_desc = CUDA_RESOURCE_DESC_st {
            resType: CUresourcetype::CU_RESOURCE_TYPE_PITCH2D,
            res: CUDA_RESOURCE_DESC_st__bindgen_ty_1 {
                pitch2D: CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_4 {
                    devPtr: dev_ptr,
                    format: CUarray_format::CU_AD_FORMAT_FLOAT,
                    numChannels: 1,
                    width,
                    height,
                    pitchInBytes: width * core::mem::size_of::<f32>(),
                },
            },
            flags: 0,
        };
        let tex_desc = CUDA_TEXTURE_DESC_st {
            addressMode: [
                CUaddress_mode::CU_TR_ADDRESS_MODE_BORDER,
                CUaddress_mode::CU_TR_ADDRESS_MODE_BORDER,
                CUaddress_mode::CU_TR_ADDRESS_MODE_BORDER,
            ],
            filterMode: CUfilter_mode::CU_TR_FILTER_MODE_POINT,
            flags: 0, // unnormalized coordinates
            maxAnisotropy: 0,
            mipmapFilterMode: CUfilter_mode::CU_TR_FILTER_MODE_POINT,
            mipmapLevelBias: 0.0,
            minMipmapLevelClamp: 0.0,
            maxMipmapLevelClamp: 0.0,
            borderColor: [0.0; 4],
            reserved: [0; 12],
        };

        let mut handle: CUtexObject = 0;
        let rc = unsafe {
            cudarc::driver::sys::cuTexObjectCreate(
                &mut handle,
                &res_desc,
                &tex_desc,
                core::ptr::null(),
            )
        };
        // cudaError_enum::CUDA_SUCCESS == 0
        if rc as u32 != 0 {
            return Err(format!("cuTexObjectCreate failed: {rc:?}"));
        }
        Ok(Self(handle))
    }

    /// Raw `unsigned long long` handle to pass as a kernel argument.
    #[inline]
    pub(super) fn handle(&self) -> u64 {
        self.0
    }
}

impl Drop for CudaTexObject {
    fn drop(&mut self) {
        // Ignore errors on drop — best effort.
        unsafe { cudarc::driver::sys::cuTexObjectDestroy(self.0) };
    }
}
