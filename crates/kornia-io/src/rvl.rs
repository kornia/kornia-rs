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
///   `[4 bytes: magic][4 bytes: width u32 LE][4 bytes: height u32 LE]`
///   followed by, repeated to the end of the frame:
///   `VLE(zero_run_len) VLE(non_zero_run_len) VLE(zigzag(delta)) × non_zero_run_len`
///
/// Two magics share that layout byte-for-byte, differing only in what the values mean:
///   - `RVL1` — absolute depth in millimetres. Self-contained; decode with `decode_image_rvl`.
///   - `RVLD` — a temporal delta against the previous frame, so an unchanged pixel becomes 0 and
///     phase 1 collapses it. Needs that reference frame to decode, so it goes through
///     `decode_image_rvl_delta` instead. See `encode_image_rvl_delta` for why the gain on real
///     (noisy) depth is far smaller than a static test scene suggests.
///
/// Compression ratio: measured **5.71×** over raw u16 on 60 live 320x180 OAK-D
/// frames that are 64% invalid, at 0.297 ms to encode and 0.236 ms to decode
/// (aarch64, release). Sparser frames do better; a fully dense frame degrades to
/// phase 2 + 3 alone.
use crate::error::IoError;
use kornia_image::{Image, ImageSize};
use std::{fs, path::Path};

const MAGIC: &[u8; 4] = b"RVL1";
/// Temporal-delta variant. The stream layout is byte-for-byte identical to `RVL1`; only the meaning
/// of the values changes — each is the zigzagged delta against the *previous frame* rather than an
/// absolute depth. A decoder must be handed that reference frame, which is why
/// [`decode_image_rvl`] deliberately does not accept this magic: it has no way to supply one.
const MAGIC_DELTA: &[u8; 4] = b"RVLD";
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
    Ok(encode_rvl_stream(
        image.as_slice(),
        image.width() as u32,
        image.height() as u32,
        MAGIC,
    ))
}

/// Encodes depth straight from a `&[u16]` slice, without requiring an owning [`Image`].
///
/// Same bytes as [`encode_image_rvl`]. This exists because a producer typically holds only a
/// *borrow* of the frame — out of a shared buffer, an `Arc`, or a driver's mapped memory — and
/// [`Image`] can only be built from an owned `Vec`. Going through the image type would mean
/// copying the whole frame just to hand it straight to the encoder, which reads it once and
/// discards it. At 320x180 that is 115 KB per frame, per camera; at 720p, 1.8 MB.
///
/// # Example
///
/// ```rust
/// use kornia_io::rvl::{encode_image_rvl_slice, decode_image_rvl};
///
/// let pixels = [1000u16, 1001, 1002, 1003, 0, 500, 500, 500];
/// let compressed = encode_image_rvl_slice(&pixels, 4, 2).unwrap();
/// assert_eq!(decode_image_rvl(&compressed).unwrap().as_slice(), &pixels);
/// ```
pub fn encode_image_rvl_slice(
    pixels: &[u16],
    width: usize,
    height: usize,
) -> Result<Vec<u8>, IoError> {
    let expected = width.checked_mul(height).ok_or_else(|| {
        IoError::RvlEncodeError(format!("image dimensions {width}x{height} overflow"))
    })?;
    if pixels.len() != expected {
        return Err(IoError::RvlEncodeError(format!(
            "{width}x{height} needs {expected} values, got {}",
            pixels.len()
        )));
    }
    Ok(encode_rvl_stream(
        pixels,
        width as u32,
        height as u32,
        MAGIC,
    ))
}

/// Encodes `image` as a temporal delta against `previous` (`RVLD`).
///
/// Each pixel becomes `zigzag(cur - prev)`, so an **unchanged pixel maps to 0** and RVL's
/// run-length phase collapses it. The caller owns keyframe policy: a decoder can only apply this
/// against the exact frame it was encoded from, so a dropped payload poisons every delta after it
/// until the next keyframe ([`encode_image_rvl`]).
///
/// **Expect far less than a synthetic static scene suggests.** Measured on live OAK-D frames
/// (320x180, 63% invalid) a delta came out only **11% smaller** than a keyframe — 18.0 vs 20.2 KB —
/// because per-pixel sensor noise jitters every valid reading by a few millimetres, and RVL
/// collapses *runs*, not small values. An identical frame compresses spectacularly; a real one
/// barely moves. Worth it only where that 11% matters and the transport is reliable and in-order.
///
/// # Errors
///
/// Returns an error if the two frames differ in size, or if any `cur - prev` falls outside
/// `-32768..=32767`. That interval is asymmetric because the zigzag is: it maps exactly that range
/// onto `0..=65535`, so `-32768` fits and `+32768` does not. Depth is `u16` with no guaranteed
/// upstream clamp, so a hole (0) adjacent to a saturated or sentinel reading can reach it. The
/// encoder already visits every pixel, so the check is free — and without it the value would wrap
/// and reconstruction would be silently lossy.
pub fn encode_image_rvl_delta(
    image: &Image<u16, 1>,
    previous: &Image<u16, 1>,
) -> Result<Vec<u8>, IoError> {
    let (pixels, prev) = (image.as_slice(), previous.as_slice());
    if pixels.len() != prev.len() {
        return Err(IoError::RvlEncodeError(format!(
            "delta reference is {} values, frame is {}",
            prev.len(),
            pixels.len()
        )));
    }
    let mut deltas = Vec::with_capacity(pixels.len());
    for (i, (&cur, &prev)) in pixels.iter().zip(prev).enumerate() {
        let delta = cur as i32 - prev as i32;
        // The `as u16` below is the whole reason for this bound: zigzag(32768) = 65536 overflows it.
        if !(-32768..32768).contains(&delta) {
            return Err(IoError::RvlEncodeError(format!(
                "delta {delta} at pixel {i} ({prev} -> {cur}) is outside -32768..=32767; the \
                 zigzag would wrap and the payload would not be lossless — send a keyframe \
                 (`encode_image_rvl`) for that frame instead"
            )));
        }
        deltas.push(zigzag(delta) as u16);
    }
    Ok(encode_rvl_stream(
        &deltas,
        image.width() as u32,
        image.height() as u32,
        MAGIC_DELTA,
    ))
}

/// The run-length + zigzag + nibble core, shared by the absolute (`RVL1`) and delta (`RVLD`)
/// entry points — they differ only in the four magic bytes and in what the values *mean*.
/// Pure transform over whatever `u16`s it is handed; never fails.
fn encode_rvl_stream(pixels: &[u16], width: u32, height: u32, magic: &[u8; 4]) -> Vec<u8> {
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

    let stream = writer.finish();
    let mut out = Vec::with_capacity(HEADER_LEN + stream.len());
    out.extend_from_slice(magic);
    out.extend_from_slice(&width.to_le_bytes());
    out.extend_from_slice(&height.to_le_bytes());
    out.extend_from_slice(&stream);
    out
}

/// Decodes RVL-compressed bytes back to a single-channel 16-bit depth image.
///
/// Reads the 12-byte header produced by [`encode_image_rvl`] to recover the image dimensions,
/// then walks the run-length stream.
pub fn decode_image_rvl(src: &[u8]) -> Result<Image<u16, 1>, IoError> {
    let s = decode_rvl_stream(src, MAGIC, "RVL1")?;
    let size = ImageSize {
        width: s.width,
        height: s.height,
    };
    Ok(Image::new(size, s.values)?)
}

/// Decodes an `RVLD` temporal delta and applies it to `previous`, reconstructing the depth image.
///
/// `previous` must be the exact frame the delta was encoded against — see
/// [`encode_image_rvl_delta`]. Reconstruction wraps at 16 bits, mirroring the encoder's `as u16`
/// narrowing, so a **mismatched reference yields wrong depth rather than an error**: the payload
/// carries no fingerprint of the frame it was built from. Sequencing is the caller's job.
///
/// # Example
///
/// ```rust
/// use kornia_io::rvl::{encode_image_rvl_delta, decode_image_rvl_delta};
/// use kornia_image::{Image, ImageSize};
///
/// let size = ImageSize { width: 4, height: 1 };
/// let prev = Image::<u16, 1>::new(size, vec![1000u16, 1000, 0, 500]).unwrap();
/// let cur = Image::<u16, 1>::new(size, vec![1000u16, 1002, 0, 495]).unwrap();
///
/// let delta = encode_image_rvl_delta(&cur, &prev).unwrap();
/// let decoded = decode_image_rvl_delta(&delta, &prev).unwrap();
/// assert_eq!(decoded.as_slice(), cur.as_slice());
/// ```
pub fn decode_image_rvl_delta(
    src: &[u8],
    previous: &Image<u16, 1>,
) -> Result<Image<u16, 1>, IoError> {
    let s = decode_rvl_stream(src, MAGIC_DELTA, "RVLD")?;
    let prev = previous.as_slice();
    if s.values.len() != prev.len() {
        return Err(IoError::RvlDecodeError(format!(
            "delta reference is {} values, payload is {}",
            prev.len(),
            s.values.len()
        )));
    }
    let pixels = s
        .values
        .iter()
        .zip(prev)
        .map(|(&zz, &p)| (p as i32 + unzigzag(zz as u32)) as u16)
        .collect();
    let size = ImageSize {
        width: s.width,
        height: s.height,
    };
    Ok(Image::new(size, pixels)?)
}

/// The raw contents of a decoded RVL stream: `width * height` values plus the header dims.
///
/// Deliberately *not* an [`Image`]: for an `RVLD` payload these are zigzagged **deltas**, not depth
/// in millimetres, and typing them as an image is exactly what would let one be displayed or
/// published as if it were a frame.
struct RvlStream {
    values: Vec<u16>,
    width: usize,
    height: usize,
}

fn decode_rvl_stream(src: &[u8], magic: &[u8; 4], what: &str) -> Result<RvlStream, IoError> {
    if src.len() < HEADER_LEN {
        return Err(IoError::RvlDecodeError(
            "buffer too short for 12-byte RVL header".into(),
        ));
    }
    if &src[..4] != magic {
        return Err(IoError::RvlDecodeError(format!(
            "invalid magic bytes — expected a {what} payload"
        )));
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

    Ok(RvlStream {
        values: pixels,
        width,
        height,
    })
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

    // ── Slice encoder ─────────────────────────────────────────────────────────

    #[test]
    fn slice_encoder_matches_the_image_encoder_byte_for_byte() {
        // The point of the slice entry point is to skip building an Image, so the one thing that
        // must hold is that it does not change the bytes.
        let data = vec![0u16, 0, 1000, 1002, 0, 65535, 1, 0];
        let img = make_image(data.clone(), 4, 2);
        assert_eq!(
            encode_image_rvl_slice(&data, 4, 2).unwrap(),
            encode_image_rvl(&img).unwrap()
        );
    }

    #[test]
    fn slice_encoder_rejects_dims_that_disagree_with_the_buffer() {
        // Without this the header would claim a size the stream cannot fill, and the error would
        // surface in some other process's decoder instead of here.
        let data = vec![0u16; 8];
        assert!(encode_image_rvl_slice(&data, 4, 3).is_err());
        assert!(encode_image_rvl_slice(&data, 4, 2).is_ok());
    }

    // ── Temporal delta (RVLD) ─────────────────────────────────────────────────

    #[test]
    fn roundtrip_delta_against_its_reference() {
        let size = (4usize, 2usize);
        let prev = make_image(vec![1000u16, 1000, 0, 500, 0, 0, 300, 301], size.0, size.1);
        let cur = make_image(vec![1000u16, 1002, 0, 495, 0, 7, 300, 299], size.0, size.1);
        let enc = encode_image_rvl_delta(&cur, &prev).unwrap();
        assert_eq!(&enc[..4], MAGIC_DELTA, "delta must carry the RVLD magic");
        let dec = decode_image_rvl_delta(&enc, &prev).unwrap();
        assert_eq!(dec.as_slice(), cur.as_slice());
    }

    #[test]
    fn an_unchanged_frame_collapses_to_almost_nothing() {
        // This is the entire reason RVLD exists: every delta is 0, so phase 1 sees one giant zero
        // run. It is also why the doc warns that real (noisy) depth gets nowhere near this.
        let (w, h) = (320usize, 180usize);
        let data: Vec<u16> = (0..w * h).map(|i| (900 + i % 500) as u16).collect();
        let img = make_image(data, w, h);
        let enc = encode_image_rvl_delta(&img, &img).unwrap();
        assert!(
            enc.len() < 64,
            "an identical frame should cost a handful of bytes, got {}",
            enc.len()
        );
    }

    #[test]
    fn delta_rejects_a_swing_the_zigzag_cannot_represent() {
        // Asymmetric on purpose: the zigzag maps -32768..=32767 onto 0..=65535, so the negative
        // edge fits and the positive one does not. A hole (0) next to a saturated reading reaches
        // exactly this, so both edges are pinned.
        let one = |v: u16| make_image(vec![v], 1, 1);
        assert!(
            encode_image_rvl_delta(&one(0), &one(32768)).is_ok(),
            "-32768 is representable and must encode"
        );
        assert!(
            encode_image_rvl_delta(&one(32768), &one(0)).is_err(),
            "+32768 would wrap the zigzag and must be refused, not silently truncated"
        );
    }

    #[test]
    fn delta_rejects_a_mismatched_reference_size() {
        let cur = make_image(vec![1u16; 8], 4, 2);
        let prev = make_image(vec![1u16; 6], 3, 2);
        assert!(encode_image_rvl_delta(&cur, &prev).is_err());

        let enc = encode_image_rvl_delta(&cur, &cur).unwrap();
        assert!(decode_image_rvl_delta(&enc, &prev).is_err());
    }

    #[test]
    fn the_two_magics_do_not_decode_as_each_other() {
        // An RVLD payload decoded as absolute depth would be a frame of near-zero "depth" rather
        // than an error — the layouts are identical, so only the magic can catch it.
        let img = make_image(vec![1000u16, 1001, 0, 500], 4, 1);
        let keyframe = encode_image_rvl(&img).unwrap();
        let delta = encode_image_rvl_delta(&img, &img).unwrap();

        assert!(decode_image_rvl(&delta).is_err());
        assert!(decode_image_rvl_delta(&keyframe, &img).is_err());
    }
}
