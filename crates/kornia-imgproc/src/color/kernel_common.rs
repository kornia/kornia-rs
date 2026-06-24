//! Shared SIMD-kernel plumbing for color conversions.
//!
//! Every color conversion follows the same 4-layer dispatch (public Image entry → rayon
//! strip split → `#[inline]` cfg dispatcher → `_neon`/`_avx2`/`_scalar` leaves). This
//! module holds the pieces shared across all of them so the per-conversion kernels stay
//! focused on the math.

/// Return `Err(ImageError::InvalidImageSize)` when `src` and `dst` sizes differ.
#[inline]
pub(crate) fn check_size<T, U, const C1: usize, const C2: usize, A1, A2>(
    src: &kornia_image::Image<T, C1, A1>,
    dst: &kornia_image::Image<U, C2, A2>,
) -> Result<(), kornia_image::ImageError>
where
    A1: kornia_image::allocator::ImageAllocator,
    A2: kornia_image::allocator::ImageAllocator,
{
    if src.size() != dst.size() {
        return Err(kornia_image::ImageError::InvalidImageSize(
            src.cols(),
            src.rows(),
            dst.cols(),
            dst.rows(),
        ));
    }
    Ok(())
}

/// Sealed-trait plumbing shared across all color-conversion families.
pub(crate) mod sealed {
    /// Marker trait — prevents external crates from implementing
    /// the sealed color-dispatch traits.
    pub trait Sealed {}

    impl Sealed for u8 {}
    impl Sealed for u16 {}
    impl Sealed for i32 {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// Below this pixel count rayon spawn cost exceeds the compute budget; above it
/// (e.g. 1080p ≈ 2M px) strip-splitting across available threads pays off.
pub(crate) const PAR_THRESHOLD: usize = 1024 * 1024;

/// Split `src`/`dst` into strips and run `kernel` on each strip in parallel.
///
/// `channels` is the number of source elements per output pixel (e.g. 3 for RGB).
/// `align` is the SIMD loop width (pixels) so strip boundaries never cut a bulk
/// iteration in half.
///
/// `dst_channels` lets the source and destination have different channel counts
/// (e.g. RGBA→RGB is 4→3); the strip is sized by destination pixels and the source
/// slice is offset by `channels` (source elements per pixel).
pub(crate) fn par_strip_dispatch_nm<S, D>(
    src: &[S],
    dst: &mut [D],
    npixels: usize,
    src_channels: usize,
    dst_channels: usize,
    align: usize,
    kernel: impl Fn(&[S], &mut [D], usize) + Send + Sync,
) where
    S: Sync,
    D: Send,
{
    if npixels < PAR_THRESHOLD {
        kernel(src, dst, npixels);
        return;
    }
    use rayon::prelude::*;
    let nthreads = rayon::current_num_threads().max(1);
    let strip = npixels.div_ceil(nthreads).next_multiple_of(align);
    dst.par_chunks_mut(strip * dst_channels)
        .enumerate()
        .for_each(|(i, dchunk)| {
            let start = i * strip;
            let n = dchunk.len() / dst_channels;
            let schunk = &src[start * src_channels..start * src_channels + n * src_channels];
            kernel(schunk, dchunk, n);
        });
}

/// Strip-dispatch for the common case where source and destination have the same
/// per-pixel element count (`channels` in, `channels` out), e.g. RGB↔HSV (3→3).
pub(crate) fn par_strip_dispatch<S, D>(
    src: &[S],
    dst: &mut [D],
    npixels: usize,
    channels: usize,
    align: usize,
    kernel: impl Fn(&[S], &mut [D], usize) + Send + Sync,
) where
    S: Sync,
    D: Send,
{
    par_strip_dispatch_nm(src, dst, npixels, channels, channels, align, kernel)
}
