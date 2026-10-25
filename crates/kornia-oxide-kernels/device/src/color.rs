//! Color-conversion kernels; each is byte-exact with its CPU path in
//! `crates/kornia-imgproc/src/color/`.

use cuda_device::{kernel, thread};

// OpenCV Q14 BT.601 weights; mirror `RW_Q14`/`GW_Q14`/`BW_Q14`/`Q14_GRAY_HALF`
// in `crates/kornia-imgproc/src/color/gray/kernels.rs`.
const WR: u32 = 4899;
const WG: u32 = 9617;
const WB: u32 = 1868;
const HALF: u32 = 8192;

#[inline(always)]
fn gray_q14(r: u32, g: u32, b: u32) -> u32 {
    (WR * r + WG * g + WB * b + HALF) >> 14
}

/// RGB8 -> Gray8, 4 px/thread: three u32 loads (12 RGB bytes), one u32 store
/// (4 grays); the final partial quad falls back to bytes.
///
/// # Safety
///
/// `src` holds `3 * npixels` bytes and `dst` `npixels`, both 4-byte aligned
/// (cudarc allocations are 256-byte aligned), and `3 * npixels <= u32::MAX`.
/// `launch_map` checks the lengths and the range.
#[kernel]
pub unsafe fn gray_from_rgb_u8(src: *const u8, dst: *mut u8, npixels: u32) {
    let q = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
    let nquads = (npixels + 3) / 4;
    if q >= nquads {
        return;
    }

    let p = q * 4;
    // SAFETY: `p + 4 <= npixels` bounds the word path and `i < npixels` the
    // byte tail, within the lengths and 4-byte alignment launch_map checks.
    unsafe {
        if p + 4 <= npixels {
            let s32 = src as *const u32;
            let w0 = *s32.add((q * 3) as usize);
            let w1 = *s32.add((q * 3 + 1) as usize);
            let w2 = *s32.add((q * 3 + 2) as usize);
            // w0 = r0 g0 b0 r1 | w1 = g1 b1 r2 g2 | w2 = b2 r3 g3 b3
            let g0 = gray_q14(w0 & 0xFF, (w0 >> 8) & 0xFF, (w0 >> 16) & 0xFF);
            let g1 = gray_q14(w0 >> 24, w1 & 0xFF, (w1 >> 8) & 0xFF);
            let g2 = gray_q14((w1 >> 16) & 0xFF, w1 >> 24, w2 & 0xFF);
            let g3 = gray_q14((w2 >> 8) & 0xFF, (w2 >> 16) & 0xFF, w2 >> 24);
            *(dst as *mut u32).add(q as usize) = g0 | (g1 << 8) | (g2 << 16) | (g3 << 24);
        } else {
            for i in p..npixels {
                let b = (i * 3) as usize;
                let r = *src.add(b) as u32;
                let g = *src.add(b + 1) as u32;
                let bl = *src.add(b + 2) as u32;
                *dst.add(i as usize) = gray_q14(r, g, bl) as u8;
            }
        }
    }
}
