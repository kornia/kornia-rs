/// RVL (Run-Length Variable) depth codec.
///
/// Implements the algorithm from "Real-Time Compression of Kinect Depth Streams"
/// (Tang et al., CVPR 2017). Two-phase pipeline:
///
///   Phase 1 — delta + zigzag preprocessing (SIMD-accelerated):
///     `delta[i]  = pixels[i] − pixels[i−1]`  (signed 16-bit, wrapping)
///     `zigzag[i] = (delta << 1) ^ (delta >> 15)` (maps signed → non-negative u16)
///
///   Phase 2 — variable-length nibble packing (scalar, inherently sequential):
///     Each zigzag value is packed as groups of 3-bit value + 1-bit continuation flag.
///     Nibbles are stored 2-per-byte, LSB first.
///
/// Wire format (12-byte header):
///   `[4 bytes: magic b"RVL1"][4 bytes: width u32 LE][4 bytes: height u32 LE]`
///
/// Compression ratio: ~3–5× over raw u16 for typical depth frames. Zeros (background /
/// missing depth) compress to a single nibble (0x00) — very efficient for sparse scenes.
use crate::error::IoError;
use kornia_image::{Image, ImageSize};
use std::{fs, path::Path};

const MAGIC: &[u8; 4] = b"RVL1";
const HEADER_LEN: usize = 12; // magic(4) + width(4) + height(4)

/// Sanity ceiling on decoded image pixels. `decode_image_rvl` takes the image dimensions from an
/// untrusted 12-byte header and allocates `width * height` up front, before reading any pixel
/// data. A tiny payload can declare a huge image (e.g. 65535x65535), so without a bound a corrupt
/// or hostile buffer drives a multi-gigabyte allocation — an OOM/abort instead of a clean error.
/// 8192x8192 covers any real frame with wide margin; anything larger is rejected.
const MAX_PIXELS: usize = 8192 * 8192;

// ── NibbleWriter ──────────────────────────────────────────────────────────────

struct NibbleWriter {
    buf: Vec<u8>,
    pending: Option<u8>,
}

impl NibbleWriter {
    fn with_capacity(cap: usize) -> Self {
        Self {
            buf: Vec::with_capacity(cap),
            pending: None,
        }
    }

    #[inline(always)]
    fn write_nibble(&mut self, n: u8) {
        match self.pending.take() {
            None => self.pending = Some(n & 0xF),
            Some(lo) => self.buf.push(lo | ((n & 0xF) << 4)),
        }
    }

    fn finish(mut self) -> Vec<u8> {
        if let Some(lo) = self.pending {
            self.buf.push(lo);
        }
        self.buf
    }
}

// ── NibbleReader ─────────────────────────────────────────────────────────────

struct NibbleReader<'a> {
    data: &'a [u8],
    pos: usize,
    hi: bool,
}

impl<'a> NibbleReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            hi: false,
        }
    }

    #[inline(always)]
    fn next_nibble(&mut self) -> Option<u8> {
        let byte = *self.data.get(self.pos)?;
        let nibble = if self.hi {
            (byte >> 4) & 0xF
        } else {
            byte & 0xF
        };
        self.hi = !self.hi;
        if self.hi {
            // consumed lo nibble, don't advance yet
        } else {
            // consumed hi nibble, advance to next byte
            self.pos += 1;
        }
        Some(nibble)
    }
}

// ── VLE encode / decode ───────────────────────────────────────────────────────

#[inline(always)]
fn encode_vle(writer: &mut NibbleWriter, mut val: u32) {
    loop {
        let low3 = (val & 0x7) as u8;
        val >>= 3;
        writer.write_nibble(if val != 0 { low3 | 0x8 } else { low3 });
        if val == 0 {
            break;
        }
    }
}

#[inline(always)]
fn decode_vle(reader: &mut NibbleReader) -> Option<u32> {
    let mut val = 0u32;
    let mut shift = 0u32;
    loop {
        let nibble = reader.next_nibble()?;
        val |= ((nibble & 0x7) as u32) << shift;
        shift += 3;
        if nibble & 0x8 == 0 {
            break;
        }
    }
    Some(val)
}

// ── SIMD-accelerated zigzag preprocessing ────────────────────────────────────
//
// Goal: compute zigzag[i] = ((pixels[i] as i16 - pixels[i-1] as i16) << 1)
//                           ^ ((pixels[i] as i16 - pixels[i-1] as i16) >> 15)
// then immediately VLE-encode into the NibbleWriter.
//
// NEON (aarch64): 8 u16 per iteration, always available.
// AVX2 (x86_64):  16 u16 per iteration, runtime-detected.
// Scalar fallback for all other targets (or remainder pixels).

/// Encode a flat pixel slice into the nibble writer.
fn encode_pixels(pixels: &[u16], writer: &mut NibbleWriter) {
    // Returns the number of pixels handled by the SIMD path; scalar covers the rest.
    #[cfg(target_arch = "aarch64")]
    let simd_end = encode_pixels_neon(pixels, writer);

    #[cfg(target_arch = "x86_64")]
    let simd_end = if std::is_x86_feature_detected!("avx2") {
        unsafe { encode_pixels_avx2(pixels, writer) }
    } else {
        0
    };

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    let simd_end = 0usize;

    let mut prev: i16 = if simd_end > 0 {
        pixels[simd_end - 1] as i16
    } else {
        0
    };
    for &p in &pixels[simd_end..] {
        let delta = (p as i16).wrapping_sub(prev);
        let zigzag = ((delta << 1) ^ (delta >> 15)) as u16;
        encode_vle(writer, zigzag as u32);
        prev = p as i16;
    }
}

#[cfg(target_arch = "aarch64")]
fn encode_pixels_neon(pixels: &[u16], writer: &mut NibbleWriter) -> usize {
    use std::arch::aarch64::*;

    let n = pixels.len();
    let mut i = 0;
    let mut prev_scalar: i16 = 0;

    // Safety: vld1q_s16 / vextq_s16 / vsubq_s16 / vshlq_n_s16 / vshrq_n_s16 /
    // veorq_s16 / vreinterpretq_u16_s16 / vst1q_u16 are all safe to call on aarch64.
    unsafe {
        while i + 8 <= n {
            let curr = vld1q_s16(pixels[i..].as_ptr() as *const i16);

            // prev_vec = [prev_scalar, pixels[i], ..., pixels[i+6]]
            // vextq_s16(a, b, 7) = [a[7], b[0], ..., b[6]]
            let prev_broadcast = vdupq_n_s16(prev_scalar);
            let prev_vec = vextq_s16(prev_broadcast, curr, 7);

            // delta = curr - prev_vec (signed, wrapping at 16-bit boundary)
            let delta = vsubq_s16(curr, prev_vec);

            // zigzag = (delta << 1) ^ (delta >> 15)
            let shl = vshlq_n_s16(delta, 1);
            let shr = vshrq_n_s16(delta, 15); // arithmetic: all-1s or all-0s
            let zigzag = vreinterpretq_u16_s16(veorq_s16(shl, shr));

            let mut tmp = [0u16; 8];
            vst1q_u16(tmp.as_mut_ptr(), zigzag);
            for z in tmp {
                encode_vle(writer, z as u32);
            }

            prev_scalar = pixels[i + 7] as i16;
            i += 8;
        }
    }

    i
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn encode_pixels_avx2(pixels: &[u16], writer: &mut NibbleWriter) -> usize {
    use std::arch::x86_64::*;

    let n = pixels.len();
    let mut i = 0;
    let mut prev_scalar: i16 = 0;

    while i + 16 <= n {
        let curr = _mm256_loadu_si256(pixels[i..].as_ptr() as *const __m256i);

        // Split curr into two 128-bit halves.
        let curr_lo = _mm256_castsi256_si128(curr); // pixels[i..i+8]
        let curr_hi = _mm256_extracti128_si256(curr, 1); // pixels[i+8..i+16]

        // Build prev_vec (256-bit): [prev_scalar, pixels[i..i+15]]
        //   prev_vec low  = [prev_scalar, pixels[i..i+7]]
        //   prev_vec high = [pixels[i+7], pixels[i+8..i+15]]
        let prev_128 = _mm_set1_epi16(prev_scalar);
        // _mm_alignr_epi8(a, b, n): concat [b|a], extract bytes [n..n+16]
        let prev_vec_lo = _mm_alignr_epi8(curr_lo, prev_128, 14); // [prev_128[7], curr_lo[0..6]]
        let prev_vec_hi = _mm_alignr_epi8(curr_hi, curr_lo, 14); // [curr_lo[7], curr_hi[0..6]]
        let prev_vec = _mm256_set_m128i(prev_vec_hi, prev_vec_lo);

        let delta = _mm256_sub_epi16(curr, prev_vec);
        let shl = _mm256_slli_epi16(delta, 1);
        let shr = _mm256_srai_epi16(delta, 15); // arithmetic
        let zigzag = _mm256_xor_si256(shl, shr);

        let mut tmp = [0u16; 16];
        _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, zigzag);
        for z in tmp {
            encode_vle(writer, z as u32);
        }

        prev_scalar = pixels[i + 15] as i16;
        i += 16;
    }

    i
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Encodes a single-channel 16-bit depth image to RVL bytes.
///
/// The first 12 bytes are a header: `b"RVL1"`, width (u32 LE), height (u32 LE).
///
/// # Example
///
/// ```rust
/// use kornia_io::rvl::{encode_image_rvl, decode_image_rvl};
/// use kornia_image::{Image, ImageSize};
///
/// let size = ImageSize { width: 4, height: 2 };
/// let data = vec![1000u16, 1001, 1002, 1003, 0, 500, 500, 500];
/// let img = Image::<u16, 1>::new(size, data).unwrap();
///
/// let compressed = encode_image_rvl(&img).unwrap();
/// let decoded = decode_image_rvl(&compressed).unwrap();
/// assert_eq!(decoded.as_slice(), img.as_slice());
/// ```
pub fn encode_image_rvl(image: &Image<u16, 1>) -> Result<Vec<u8>, IoError> {
    let pixels = image.as_slice();
    let w = image.width() as u32;
    let h = image.height() as u32;

    // Worst-case: 6 nibbles = 3 bytes per pixel (for delta = ±32768)
    let mut header = Vec::with_capacity(HEADER_LEN);
    header.extend_from_slice(MAGIC);
    header.extend_from_slice(&w.to_le_bytes());
    header.extend_from_slice(&h.to_le_bytes());

    let mut writer = NibbleWriter::with_capacity(pixels.len() * 3);
    encode_pixels(pixels, &mut writer);
    let nibble_buf = writer.finish();

    let mut out = header;
    out.extend_from_slice(&nibble_buf);
    Ok(out)
}

/// Decodes RVL-compressed bytes back to a single-channel 16-bit depth image.
///
/// Reads the 12-byte header produced by [`encode_image_rvl`] to recover
/// the image dimensions, then decodes the variable-length nibble stream.
pub fn decode_image_rvl(src: &[u8]) -> Result<Image<u16, 1>, IoError> {
    if src.len() < HEADER_LEN {
        return Err(IoError::RvlDecodeError(
            "buffer too short for 12-byte RVL header".into(),
        ));
    }
    if &src[..4] != MAGIC {
        return Err(IoError::RvlDecodeError(
            "invalid magic bytes — expected b\"RVL1\"".into(),
        ));
    }
    let width = u32::from_le_bytes(src[4..8].try_into().unwrap()) as usize;
    let height = u32::from_le_bytes(src[8..12].try_into().unwrap()) as usize;
    let n_pixels = width
        .checked_mul(height)
        .ok_or_else(|| IoError::RvlDecodeError("image dimensions overflow".into()))?;
    if n_pixels > MAX_PIXELS {
        return Err(IoError::RvlDecodeError(format!(
            "image {width}x{height} exceeds max {MAX_PIXELS} pixels"
        )));
    }

    let mut pixels = vec![0u16; n_pixels];
    let mut reader = NibbleReader::new(&src[HEADER_LEN..]);
    let mut prev: i16 = 0;

    for p in pixels.iter_mut() {
        let zigzag = decode_vle(&mut reader)
            .ok_or_else(|| IoError::RvlDecodeError("unexpected end of nibble stream".into()))?;
        let zigzag_u16 = zigzag as u16;
        // Inverse zigzag: delta = (z >> 1) ^ -(z & 1)
        let delta = ((zigzag_u16 >> 1) as i16) ^ -((zigzag_u16 & 1) as i16);
        *p = prev.wrapping_add(delta) as u16;
        prev = *p as i16;
    }

    let size = ImageSize { width, height };
    Ok(Image::new(size, pixels)?)
}

/// Writes a single-channel 16-bit depth image to an RVL file.
pub fn write_image_rvl(file_path: impl AsRef<Path>, image: &Image<u16, 1>) -> Result<(), IoError> {
    let bytes = encode_image_rvl(image)?;
    fs::write(file_path, bytes)?;
    Ok(())
}

/// Reads an RVL file into a single-channel 16-bit depth image.
pub fn read_image_rvl(file_path: impl AsRef<Path>) -> Result<Image<u16, 1>, IoError> {
    let bytes = fs::read(file_path)?;
    decode_image_rvl(&bytes)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_image(data: Vec<u16>, w: usize, h: usize) -> Image<u16, 1> {
        Image::new(
            ImageSize {
                width: w,
                height: h,
            },
            data,
        )
        .unwrap()
    }

    #[test]
    fn decode_rejects_oversized_dimensions() {
        // A tiny payload declaring a 65535x65535 image (~4.29e9 px) must be rejected, not drive a
        // multi-gigabyte allocation. Magic + width + height, empty stream.
        let mut data = MAGIC.to_vec();
        data.extend_from_slice(&0xFFFF_u32.to_le_bytes());
        data.extend_from_slice(&0xFFFF_u32.to_le_bytes());
        assert!(decode_image_rvl(&data).is_err());
    }

    #[test]
    fn roundtrip_zeros() {
        let img = make_image(vec![0u16; 64], 8, 8);
        let enc = encode_image_rvl(&img).unwrap();
        let dec = decode_image_rvl(&enc).unwrap();
        assert_eq!(dec.as_slice(), img.as_slice());
    }

    #[test]
    fn roundtrip_constant() {
        let img = make_image(vec![1000u16; 100], 10, 10);
        let enc = encode_image_rvl(&img).unwrap();
        let dec = decode_image_rvl(&enc).unwrap();
        assert_eq!(dec.as_slice(), img.as_slice());
    }

    #[test]
    fn roundtrip_ramp() {
        let data: Vec<u16> = (0..1024).map(|x| (x * 64) as u16).collect();
        let img = make_image(data, 32, 32);
        let enc = encode_image_rvl(&img).unwrap();
        let dec = decode_image_rvl(&enc).unwrap();
        assert_eq!(dec.as_slice(), img.as_slice());
    }

    #[test]
    fn roundtrip_max_delta() {
        // Alternating 0 and 65535 — maximum delta at every pixel
        let data: Vec<u16> = (0..64)
            .map(|i: usize| if i.is_multiple_of(2) { 0 } else { 65535 })
            .collect();
        let img = make_image(data, 8, 8);
        let enc = encode_image_rvl(&img).unwrap();
        let dec = decode_image_rvl(&enc).unwrap();
        assert_eq!(dec.as_slice(), img.as_slice());
    }

    #[test]
    fn roundtrip_hd_frame() {
        // Simulate a 1280×720 depth frame with realistic values (500–5000 mm)
        let data: Vec<u16> = (0..1280 * 720)
            .map(|i| ((i as u32 * 7 + i as u32 / 100) % 4500 + 500) as u16)
            .collect();
        let img = make_image(data, 1280, 720);
        let enc = encode_image_rvl(&img).unwrap();
        let dec = decode_image_rvl(&enc).unwrap();
        assert_eq!(dec.as_slice(), img.as_slice());
        // Sanity: compressed size < raw size (2 bytes × 921600 = 1843200)
        assert!(enc.len() < 1_843_200, "compressed={}", enc.len());
    }

    #[test]
    fn header_magic_validated() {
        let mut bad = b"PNG\x89".to_vec();
        bad.extend_from_slice(&[0u8; 8]);
        assert!(decode_image_rvl(&bad).is_err());
    }

    #[test]
    fn header_too_short() {
        assert!(decode_image_rvl(b"RVL").is_err());
    }

    #[test]
    fn zigzag_inverse_identity() {
        // Verify our zigzag ↔ inverse-zigzag math via a trivial scalar encode+decode
        let vals: Vec<i16> = vec![-32768, -1, 0, 1, 32767];
        for &d in &vals {
            let z = ((d << 1) ^ (d >> 15)) as u16;
            let recovered = ((z >> 1) as i16) ^ -((z & 1) as i16);
            assert_eq!(recovered, d, "zigzag failed for delta={d}");
        }
    }

    #[test]
    fn compression_ratio_zeros() {
        // All-zero frame: every delta is 0 → 1 nibble per pixel
        // 1 nibble = 0.5 bytes/pixel vs 2 bytes/pixel raw = 4× compression
        let img = make_image(vec![0u16; 640 * 480], 640, 480);
        let enc = encode_image_rvl(&img).unwrap();
        let raw = 640 * 480 * 2;
        assert!(
            enc.len() < raw / 3,
            "expected >3× on zeros, got {}",
            enc.len()
        );
    }
}
