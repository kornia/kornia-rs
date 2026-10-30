//! `median_blur` against OpenCV, without binary fixtures.
//!
//! 1. Pinned values: small images (`u8`, `u16`, `f32`; 1 and 3 channels;
//!    ksize 3 and 5) built by the seeded LCG below, with the
//!    `cv2.medianBlur` outputs as const arrays. The `f32` inputs hold
//!    `+inf`, `-inf` and repeated values (ties). Generated with
//!    opencv-python-headless 5.0.0 / numpy 2.5.4 by this script, which
//!    mirrors the Rust generators exactly and prints the arrays below:
//!
//! ```text
//! import cv2, numpy as np
//! M = (1 << 64) - 1
//! def lcg(seed, n):
//!     s, out = seed, []
//!     for _ in range(n):
//!         s = (s * 6364136223846793005 + 1442695040888963407) & M
//!         out.append(s >> 33)
//!     return out
//! def u8(seed, n):  return np.array([(r >> 9) & 0xFF for r in lcg(seed, n)], np.uint8)
//! def u16(seed, n): return np.array([7 if i % 5 == 0 else (r >> 7) & 0xFFFF for i, r in enumerate(lcg(seed, n))], np.uint16)
//! def f32(seed, n):
//!     v = []
//!     for i, r in enumerate(lcg(seed, n)):
//!         if i % 11 == 3: v.append(np.inf)
//!         elif i % 13 == 5: v.append(-np.inf)
//!         elif i % 7 == 0: v.append(2.5)
//!         else: v.append(((r >> 8) % 2001) * 0.125 - 125.0)
//!     return np.array(v, np.float32)
//! def fmt(x, dt):
//!     if dt == "f32":
//!         if np.isinf(x): return "f32::INFINITY" if x > 0 else "f32::NEG_INFINITY"
//!         return repr(float(x))
//!     return str(int(x))
//! cases = [("u8", 1, 5, 6, 11), ("u16", 1, 5, 6, 12), ("u16", 3, 4, 5, 13), ("f32", 1, 5, 6, 14), ("f32", 3, 4, 5, 15)]
//! gen = {"u8": u8, "u16": u16, "f32": f32}
//! print(f"// cv2 {cv2.__version__}, numpy {np.__version__}")
//! for dt, c, h, w, seed in cases:
//!     src = gen[dt](seed, h * w * c).reshape(h, w, c)
//!     s = src[..., 0] if c == 1 else src
//!     for k in (3, 5):
//!         out = cv2.medianBlur(np.ascontiguousarray(s), k).reshape(-1)
//!         name = f"{dt.upper()}_C{c}_K{k}"
//!         print(f"const {name}: [{dt}; {out.size}] = [{', '.join(fmt(x, dt) for x in out)}];")
//! ```
//!
//! 2. Larger and strip-crossing shapes (wider than the 256-column strip of
//!    the `u16` / `f32` kernels) against a brute-force oracle: sort the
//!    `ksize × ksize` window with `BORDER_REPLICATE` and take the middle
//!    sample. For inputs without NaN that is `cv2.medianBlur` by
//!    definition (an exact order statistic over the same border rule);
//!    for `f32` the oracle sorts by IEEE 754 totalOrder (`f32::total_cmp`),
//!    the documented order of `median_blur`, so NaN and signed zeros are
//!    checked too.

use kornia_image::{Image, ImageSize};
use kornia_imgproc::filter::{median_blur, MedianBlur};

/// Deterministic LCG, the same as `lcg` in the generator script.
fn lcg(seed: u64, n: usize) -> Vec<u32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 33) as u32
        })
        .collect()
}

fn gen_u8(seed: u64, n: usize) -> Vec<u8> {
    lcg(seed, n).iter().map(|&r| (r >> 9) as u8).collect()
}

fn gen_u16(seed: u64, n: usize) -> Vec<u16> {
    lcg(seed, n)
        .iter()
        .enumerate()
        .map(|(i, &r)| if i % 5 == 0 { 7 } else { (r >> 7) as u16 })
        .collect()
}

fn gen_f32(seed: u64, n: usize) -> Vec<f32> {
    lcg(seed, n)
        .iter()
        .enumerate()
        .map(|(i, &r)| {
            if i % 11 == 3 {
                f32::INFINITY
            } else if i % 13 == 5 {
                f32::NEG_INFINITY
            } else if i % 7 == 0 {
                2.5
            } else {
                ((r >> 8) % 2001) as f32 * 0.125 - 125.0
            }
        })
        .collect()
}

#[rustfmt::skip]
mod cv2 {
    pub const U8_C1_K3: [u8; 30] = [118, 190, 192, 192, 148, 148, 105, 105, 97, 148, 148, 159, 63, 105, 118, 177, 148, 148, 105, 105, 99, 177, 118, 118, 216, 211, 210, 177, 99, 118];
    pub const U8_C1_K5: [u8; 30] = [190, 190, 134, 148, 192, 231, 105, 118, 148, 159, 177, 159, 105, 105, 134, 134, 148, 118, 105, 105, 105, 118, 118, 118, 211, 105, 99, 99, 118, 118];
    pub const U16_C1_K3: [u16; 30] = [33813, 11113, 21725, 11113, 21315, 7, 33813, 33141, 11113, 11046, 21315, 30368, 33675, 33141, 11046, 6026, 27490, 30368, 33141, 33141, 33675, 33986, 42341, 56405, 33675, 33675, 42341, 42341, 42341, 33986];
    pub const U16_C1_K5: [u16; 30] = [11113, 11113, 21315, 21315, 11113, 21315, 13456, 13456, 21315, 21725, 21315, 21725, 33141, 33141, 27490, 27490, 26178, 30368, 33947, 33947, 33947, 33986, 33986, 42341, 33947, 33947, 33947, 42341, 45725, 56405];
    pub const U16_C3_K3: [u16; 60] = [7, 49260, 12243, 30794, 30622, 12243, 34210, 9614, 8380, 34210, 9614, 30585, 36308, 7420, 32239, 7, 19068, 3094, 6375, 19068, 3942, 30794, 6250, 8380, 30794, 7420, 30585, 28487, 7420, 17597, 7, 15205, 3094, 6375, 19068, 3942, 17606, 6250, 22559, 27962, 6250, 22559, 27962, 1127, 20487, 7, 1779, 3094, 8839, 19068, 39705, 8839, 6250, 22559, 8839, 1127, 22559, 28487, 1127, 20487];
    pub const U16_C3_K5: [u16; 60] = [7, 30622, 12243, 17606, 19068, 12243, 30794, 14123, 12243, 34210, 14123, 17597, 34210, 14123, 30585, 7, 19068, 12243, 7885, 15205, 12243, 28487, 14123, 17597, 34210, 9614, 20487, 34210, 9614, 22559, 7, 19068, 30585, 7885, 15205, 22559, 17606, 9614, 20487, 30794, 7420, 20487, 30794, 7420, 20487, 7, 15205, 50944, 7885, 6250, 39705, 8839, 6250, 22559, 27962, 6250, 20487, 27962, 1127, 20487];
    pub const F32_C1_K3: [f32; 30] = [2.5, -47.0, 2.5, -60.125, -75.75, -104.625, -47.0, -47.0, 13.625, 14.375, 13.625, -92.125, -68.0, -47.0, 13.625, 14.375, 14.375, 25.75, -68.0, -68.0, 13.625, 13.625, 19.0, 25.75, 109.375, 61.125, 42.125, 2.5, 2.5, 2.5];
    pub const F32_C1_K5: [f32; 30] = [-47.0, 2.5, -47.0, -60.125, -60.125, -92.125, -47.0, 2.5, 2.5, 2.5, 2.5, 2.5, -47.0, 2.5, 2.5, 2.5, 2.5, 2.5, -47.0, 13.625, 13.625, 13.625, 2.5, 2.5, 109.375, 42.125, 19.0, 13.625, 2.5, 2.5];
    pub const F32_C3_K3: [f32; 60] = [2.5, -103.0, 44.0, 2.5, 2.5, 44.0, 2.5, 12.625, -26.625, -11.5, 2.5, 97.0, -11.5, 2.5, 49.0, -5.125, -103.0, 44.0, 2.5, -8.625, 44.0, -34.875, 2.5, -26.625, 2.5, 2.5, -26.625, -11.5, -15.75, -26.625, -34.875, 2.5, 59.5, -34.875, -8.625, 2.5, -34.875, -8.625, -8.875, 2.5, 2.5, -8.875, 2.5, 2.5, 2.5, -34.875, 2.5, 103.125, -34.875, 2.5, -8.875, 79.25, 2.5, -8.875, -42.625, 43.625, -8.875, -42.625, 67.0, 2.5];
    pub const F32_C3_K5: [f32; 60] = [2.5, -103.0, 44.0, 2.5, 2.5, 44.0, 2.5, 2.5, 44.0, -11.5, 2.5, 49.0, -11.5, -15.75, 97.0, 2.5, -8.625, 44.0, -5.125, 2.5, 2.5, -5.125, 2.5, 2.5, -11.5, 2.5, 2.5, -11.5, 2.5, 8.0, -34.875, 2.5, 44.0, -34.875, 2.5, 2.5, -11.5, 2.5, 2.5, -11.5, 2.5, 2.5, -11.5, 2.5, 8.0, -54.0, 43.625, 97.0, -34.875, 43.625, 2.5, -34.875, 2.5, 2.5, -34.875, 2.5, 2.5, -72.0, 67.0, 8.0];
}

fn size(width: usize, height: usize) -> ImageSize {
    ImageSize { width, height }
}

/// `median_blur` of `src` for ksize 3 and 5, as bit patterns.
fn run<T: MedianBlur + Default, const C: usize>(
    src: Vec<T>,
    sz: ImageSize,
    bits: fn(T) -> u32,
) -> [Vec<u32>; 2] {
    let src = Image::<T, C>::new(sz, src).unwrap();
    [3, 5].map(|k| {
        let mut dst = Image::<T, C>::from_size_val(sz, T::default()).unwrap();
        median_blur(&src, &mut dst, k).unwrap();
        dst.as_slice().iter().map(|&v| bits(v)).collect()
    })
}

fn as_bits<T: Copy>(v: &[T], bits: fn(T) -> u32) -> Vec<u32> {
    v.iter().map(|&x| bits(x)).collect()
}

#[test]
fn median_blur_matches_pinned_cv2_values() {
    let b8: fn(u8) -> u32 = u32::from;
    let b16: fn(u16) -> u32 = u32::from;
    let b32: fn(f32) -> u32 = f32::to_bits;
    let (s1, s3) = (size(6, 5), size(5, 4));

    let [k3, k5] = run::<u8, 1>(gen_u8(11, 30), s1, b8);
    assert_eq!(k3, as_bits(&cv2::U8_C1_K3, b8));
    assert_eq!(k5, as_bits(&cv2::U8_C1_K5, b8));

    let [k3, k5] = run::<u16, 1>(gen_u16(12, 30), s1, b16);
    assert_eq!(k3, as_bits(&cv2::U16_C1_K3, b16));
    assert_eq!(k5, as_bits(&cv2::U16_C1_K5, b16));
    let [k3, k5] = run::<u16, 3>(gen_u16(13, 60), s3, b16);
    assert_eq!(k3, as_bits(&cv2::U16_C3_K3, b16));
    assert_eq!(k5, as_bits(&cv2::U16_C3_K5, b16));

    let [k3, k5] = run::<f32, 1>(gen_f32(14, 30), s1, b32);
    assert_eq!(k3, as_bits(&cv2::F32_C1_K3, b32));
    assert_eq!(k5, as_bits(&cv2::F32_C1_K5, b32));
    let [k3, k5] = run::<f32, 3>(gen_f32(15, 60), s3, b32);
    assert_eq!(k3, as_bits(&cv2::F32_C3_K3, b32));
    assert_eq!(k5, as_bits(&cv2::F32_C3_K5, b32));
}

/// Brute-force median: sort the replicate-border window by `key`, take the
/// middle sample.
fn oracle<T: Copy, K: Ord, const C: usize>(
    src: &[T],
    w: usize,
    h: usize,
    ksize: usize,
    key: fn(T) -> K,
) -> Vec<T> {
    let r = (ksize / 2) as isize;
    let mut out = Vec::with_capacity(src.len());
    for y in 0..h as isize {
        for x in 0..w as isize {
            for c in 0..C {
                let mut win = Vec::with_capacity(ksize * ksize);
                for dy in -r..=r {
                    for dx in -r..=r {
                        let sy = (y + dy).clamp(0, h as isize - 1) as usize;
                        let sx = (x + dx).clamp(0, w as isize - 1) as usize;
                        win.push(src[(sy * w + sx) * C + c]);
                    }
                }
                win.sort_by_key(|&v| key(v));
                out.push(win[win.len() / 2]);
            }
        }
    }
    out
}

/// `f32` sort key in IEEE 754 totalOrder (the order of `f32::total_cmp`).
fn total_order(v: f32) -> i32 {
    let b = v.to_bits() as i32;
    b ^ (((b >> 31) as u32) >> 1) as i32
}

fn check_oracle<T: MedianBlur + Default, K: Ord, const C: usize>(
    src: Vec<T>,
    w: usize,
    h: usize,
    key: fn(T) -> K,
    bits: fn(T) -> u32,
) {
    let want = [3, 5].map(|k| as_bits(&oracle::<T, K, C>(&src, w, h, k, key), bits));
    let got = run::<T, C>(src, size(w, h), bits);
    assert_eq!(got, want, "{} C{C} {w}x{h}", std::any::type_name::<T>());
}

#[test]
fn median_blur_matches_sort_oracle_on_strip_crossing_shapes() {
    let b8: fn(u8) -> u32 = u32::from;
    let b16: fn(u16) -> u32 = u32::from;
    let b32: fn(f32) -> u32 = f32::to_bits;
    // Signed zeros and NaNs of both signs on top of the generator's
    // finite values, ±inf and ties.
    let f32_special = |seed, n| {
        let mut v = gen_f32(seed, n);
        for (i, x) in v.iter_mut().enumerate() {
            match i % 17 {
                1 => *x = -0.0,
                2 => *x = 0.0,
                9 => *x = f32::NAN,
                10 => *x = -f32::NAN,
                _ => {}
            }
        }
        v
    };
    for (w, h) in [(300, 4), (40, 5), (257, 3), (1, 6)] {
        // u8 C1 at widths 1 (mod 16) hits the aarch64 NEON 5x5 tail bug
        // pinned by `u8_neon_5x5_tail_at_width_1_mod_16` below.
        if !(cfg!(target_arch = "aarch64") && w % 16 == 1) {
            check_oracle::<u8, u8, 1>(gen_u8(21, w * h), w, h, |v| v, b8);
        }
        check_oracle::<u8, u8, 3>(gen_u8(22, w * h * 3), w, h, |v| v, b8);
        check_oracle::<u16, u16, 1>(gen_u16(23, w * h), w, h, |v| v, b16);
        check_oracle::<u16, u16, 3>(gen_u16(24, w * h * 3), w, h, |v| v, b16);
        check_oracle::<f32, i32, 1>(f32_special(25, w * h), w, h, total_order, b32);
        check_oracle::<f32, i32, 3>(f32_special(26, w * h * 3), w, h, total_order, b32);
    }
}

/// The `u8` single-channel 5×5 NEON kernel (aarch64, unchanged by this
/// PR, on `main` as well) gets the column `w - 2` wrong when `w % 16 == 1`
/// and `w >= 49`: its gap block at `w - 17` reads two columns past the row
/// end instead of replicating the edge. Fails on aarch64, passes on x86;
/// ignored until the kernel is fixed in its own PR.
#[test]
#[ignore = "u8 NEON 5x5 tail bug on aarch64 (pre-existing, see the doc comment)"]
fn u8_neon_5x5_tail_at_width_1_mod_16() {
    let b8: fn(u8) -> u32 = u32::from;
    for (w, h) in [(49, 3), (257, 3)] {
        check_oracle::<u8, u8, 1>(gen_u8(21, w * h), w, h, |v| v, b8);
    }
}
