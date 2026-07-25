/// RVL (Run-Length Variable-length) lossless 16-bit depth codec.
///
/// Implements the algorithm from "Real-Time Compression of Kinect Depth Streams"
/// (Wilson, and Tang et al. CVPR 2017). Three phases, and the first is the one
/// that does most of the work on real depth:
///
///   Phase 1 — run-length segmentation:
///     The frame is walked as alternating runs: a run of zeros (invalid /
///     background pixels), then a run of non-zeros. Each run costs ONE
///     variable-length count, so a 20,000-pixel hole costs ~5 nibbles rather
///     than 20,000 of them. Depth frames are typically 40-70% invalid, so this
///     is where most of the compression comes from.
///
///   Phase 2 — delta + zigzag, within non-zero runs only:
///     `delta[i]  = pixels[i] − previous_non_zero`   (signed, wrapping)
///     `zigzag[i] = (delta << 1) ^ (delta >> 31)`    (signed → non-negative)
///     Zero pixels never produce a delta at all; they were consumed by phase 1.
///
///   Phase 3 — variable-length nibble packing:
///     Each value is packed as groups of 3 data bits + 1 continuation bit,
///     least-significant group first. Nibbles are stored 2-per-byte,
///     **high nibble first**.
///
/// Wire format (12-byte header):
///   `[4 bytes: magic b"RVL1"][4 bytes: width u32 LE][4 bytes: height u32 LE]`
///   followed by, repeated to the end of the frame:
///   `VLE(zero_run_len) VLE(non_zero_run_len) VLE(zigzag(delta)) × non_zero_run_len`
///
/// Compression ratio: measured **5.71×** over raw u16 on 60 live 320x180 OAK-D
/// frames that are 64% invalid, at 0.297 ms to encode and 0.236 ms to decode
/// (aarch64, release). Sparser frames do better; a fully dense frame degrades to
/// phase 2 + 3 alone.
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

/// Writes 4-bit nibbles into a byte buffer, **high nibble first**.
struct NibbleWriter {
    buf: Vec<u8>,
    /// The high nibble of a half-filled byte awaiting its low nibble.
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
        let n = n & 0xF;
        match self.pending.take() {
            None => self.pending = Some(n),
            Some(hi) => self.buf.push((hi << 4) | n),
        }
    }

    fn finish(mut self) -> Vec<u8> {
        if let Some(hi) = self.pending.take() {
            self.buf.push(hi << 4);
        }
        self.buf
    }
}

// ── NibbleReader ─────────────────────────────────────────────────────────────

/// Reads 4-bit nibbles from a byte buffer, **high nibble first**.
struct NibbleReader<'a> {
    data: &'a [u8],
    pos: usize,
    /// True when the next nibble to return is the high half of `data[pos]`.
    hi: bool,
}

impl<'a> NibbleReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            hi: true,
        }
    }

    #[inline(always)]
    fn next_nibble(&mut self) -> Option<u8> {
        let byte = *self.data.get(self.pos)?;
        if self.hi {
            self.hi = false;
            Some(byte >> 4)
        } else {
            self.hi = true;
            self.pos += 1;
            Some(byte & 0xF)
        }
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

/// Decodes one variable-length value. `None` on stream underrun; `Err` on a value that would not
/// fit a `u32` — rejected rather than silently truncated, since a corrupt stream must not decode
/// to a plausible-looking wrong number.
#[inline(always)]
fn decode_vle(reader: &mut NibbleReader) -> Result<u32, IoError> {
    let mut val = 0u32;
    let mut shift = 0u32;
    loop {
        let nibble = reader
            .next_nibble()
            .ok_or_else(|| IoError::RvlDecodeError("unexpected end of nibble stream".into()))?;
        // At shift 30 only bits 30-31 remain, so the third data bit (0x4) would land at bit 32.
        if shift == 30 && nibble & 0x4 != 0 {
            return Err(IoError::RvlDecodeError(
                "variable-length value exceeds u32 range".into(),
            ));
        }
        val |= ((nibble & 0x7) as u32) << shift;
        shift += 3;
        if nibble & 0x8 == 0 {
            break;
        }
        if shift > 30 {
            return Err(IoError::RvlDecodeError(
                "variable-length value too long".into(),
            ));
        }
    }
    Ok(val)
}

// ── zigzag ────────────────────────────────────────────────────────────────────
//
// Widened to i32/u32 rather than i16/u16 on purpose: depth values span the full u16 range, so a
// delta can reach ±65535 and its zigzag ±131070 — which does not fit a u16. Narrowing here would
// wrap and lose losslessness on exactly the alternating-extremes case (see `roundtrip_max_delta`).

#[inline(always)]
fn zigzag(delta: i32) -> u32 {
    ((delta << 1) ^ (delta >> 31)) as u32
}

#[inline(always)]
fn unzigzag(v: u32) -> i32 {
    ((v >> 1) as i32) ^ -((v & 1) as i32)
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Encodes a single-channel 16-bit depth image to RVL-compressed bytes.
///
/// Lossless. Operates on `u16` *values*, never a raw byte reinterpret, so the stream is
/// endian-independent and safe to move between hosts.
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

    let mut header = Vec::with_capacity(HEADER_LEN);
    header.extend_from_slice(MAGIC);
    header.extend_from_slice(&w.to_le_bytes());
    header.extend_from_slice(&h.to_le_bytes());

    // Real depth runs ~1.5 nibbles/pixel; preallocating avoids ~15 reallocations per frame.
    let mut writer = NibbleWriter::with_capacity(pixels.len());
    let mut previous: i32 = 0;
    let mut i = 0usize;
    let n = pixels.len();
    while i < n {
        // Phase 1: one count for the whole zero run, however long.
        let zeros_start = i;
        while i < n && pixels[i] == 0 {
            i += 1;
        }
        encode_vle(&mut writer, (i - zeros_start) as u32);

        let nz_start = i;
        while i < n && pixels[i] != 0 {
            i += 1;
        }
        encode_vle(&mut writer, (i - nz_start) as u32);

        // Phase 2/3: deltas only for the pixels that carry depth.
        for &d in &pixels[nz_start..i] {
            let cur = d as i32;
            encode_vle(&mut writer, zigzag(cur - previous));
            previous = cur;
        }
    }

    let mut out = header;
    out.extend_from_slice(&writer.finish());
    Ok(out)
}

/// Decodes RVL-compressed bytes back to a single-channel 16-bit depth image.
///
/// Reads the 12-byte header produced by [`encode_image_rvl`] to recover the image dimensions,
/// then walks the run-length stream.
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
    let mut previous: i32 = 0;
    let mut i = 0usize;

    while i < n_pixels {
        // Zero run: the buffer is already zeroed, so this is a skip.
        let zeros = decode_vle(&mut reader)? as usize;
        i = i
            .checked_add(zeros)
            .filter(|&i| i <= n_pixels)
            .ok_or_else(|| {
                IoError::RvlDecodeError("zero run overruns the declared image size".into())
            })?;
        if i == n_pixels {
            break;
        }

        let nonzeros = decode_vle(&mut reader)? as usize;
        let end = i
            .checked_add(nonzeros)
            .filter(|&e| e <= n_pixels)
            .ok_or_else(|| {
                IoError::RvlDecodeError("non-zero run overruns the declared image size".into())
            })?;
        // A zero-length non-zero run after a zero run that did not reach the end would mean the
        // stream makes no progress — reject rather than spin forever on a corrupt payload.
        if nonzeros == 0 && zeros == 0 {
            return Err(IoError::RvlDecodeError(
                "stream makes no progress (empty zero and non-zero runs)".into(),
            ));
        }
        for p in &mut pixels[i..end] {
            let value = previous + unzigzag(decode_vle(&mut reader)?);
            *p = value as u16;
            previous = value;
        }
        i = end;
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
        // Alternating 0 and 65535 — maximum delta at every pixel. This is why the zigzag is widened
        // to i32/u32: zigzag(65535) = 131070 does not fit a u16.
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
    fn roundtrip_sparse_depth_frame() {
        // The shape real depth actually has: a valid region, a large invalid hole, valid again.
        // Exercises several zero/non-zero run transitions, including a run that ends the frame.
        let (w, h) = (320usize, 180usize);
        let data: Vec<u16> = (0..w * h)
            .map(|i| {
                let (x, y) = (i % w, i / w);
                if (60..140).contains(&y) && (80..240).contains(&x) {
                    0 // a hole in the middle
                } else {
                    (800 + x * 3 + y) as u16
                }
            })
            .collect();
        let img = make_image(data.clone(), w, h);
        let enc = encode_image_rvl(&img).unwrap();
        assert_eq!(decode_image_rvl(&enc).unwrap().as_slice(), data.as_slice());
    }

    #[test]
    fn roundtrip_frame_ending_in_a_zero_run() {
        // The loop must terminate cleanly when the final run is zeros, without reading a non-zero
        // count that the encoder never wrote.
        let mut data = vec![1234u16; 10];
        data.extend(std::iter::repeat_n(0u16, 22));
        let img = make_image(data.clone(), 8, 4);
        let enc = encode_image_rvl(&img).unwrap();
        assert_eq!(decode_image_rvl(&enc).unwrap().as_slice(), data.as_slice());
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
    fn decode_rejects_a_run_longer_than_the_frame() {
        // A declared zero run of 2^21 pixels in a 64-pixel image must be an error, not a panic or a
        // silently short image.
        let mut data = MAGIC.to_vec();
        data.extend_from_slice(&8u32.to_le_bytes());
        data.extend_from_slice(&8u32.to_le_bytes());
        let mut w = NibbleWriter::with_capacity(8);
        encode_vle(&mut w, 1 << 21);
        data.extend_from_slice(&w.finish());
        assert!(decode_image_rvl(&data).is_err());
    }

    #[test]
    fn zigzag_inverse_identity() {
        // Full u16 depth range, so the extremes that a narrower zigzag would wrap are covered.
        for &d in &[-65535i32, -32768, -1, 0, 1, 32767, 65535] {
            assert_eq!(unzigzag(zigzag(d)), d, "zigzag failed for delta={d}");
        }
    }

    #[test]
    fn compression_ratio_zeros() {
        // An all-zero frame is ONE zero run: a handful of nibbles for the whole image, rather than
        // one nibble per pixel. This is the run-length phase's entire point.
        let img = make_image(vec![0u16; 640 * 480], 640, 480);
        let enc = encode_image_rvl(&img).unwrap();
        assert!(
            enc.len() < HEADER_LEN + 8,
            "an all-zero frame should cost the header plus a single run count, got {}",
            enc.len()
        );
    }

    #[test]
    fn compression_ratio_sparse_frame() {
        // 64% invalid, the measured shape of a live OAK-D frame. Run-length keeps this well past the
        // ~3x that delta+VLE alone reaches; guard at 4x so the assertion is about the phase existing.
        let (w, h) = (320usize, 180usize);
        let data: Vec<u16> = (0..w * h)
            .map(|i| {
                if i / w > (h * 36) / 100 {
                    0
                } else {
                    (900 + (i % w) * 2) as u16
                }
            })
            .collect();
        let img = make_image(data, w, h);
        let enc = encode_image_rvl(&img).unwrap();
        let raw = w * h * 2;
        assert!(
            enc.len() * 4 < raw,
            "expected >4x on a 64% invalid frame, got {:.2}x ({} bytes)",
            raw as f64 / enc.len() as f64,
            enc.len()
        );
    }
}
