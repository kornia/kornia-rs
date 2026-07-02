//! CUDA kernels for RGB ↔ HSV and RGB ↔ HLS (f32, kornia conventions).
//!
//! Mirrors the scalar oracles in `color/hsv/kernels.rs` and
//! `color/hls/kernels.rs` operation-for-operation: channels in `[0, 255]`,
//! hue in degrees `[0, 360)` scaled to `[0, 255]`, HLS channel order
//! `[H, L, S]`. `fmodf` has the same sign-of-dividend semantics as Rust's
//! `%` on f32, so the hue sextant math matches.

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream};

use super::{check_len, get_kernel_suite, CudaColorError, KernelSuiteCell};

static HSV_HLS_F32_SRC: &str = r#"
#define INV_255     (1.0f / 255.0f)
#define DEG_TO_BYTE (255.0f / 360.0f)
#define BYTE_TO_DEG (360.0f / 255.0f)

// Hue from the max-channel sextant (shared by HSV and HLS forward paths).
__device__ __forceinline__ float hue_deg(
    float r, float g, float b, float maxv, float delta)
{
    float h;
    if (maxv == r)      h = 60.0f * fmodf((g - b) / delta, 6.0f);
    else if (maxv == g) h = 60.0f * (((b - r) / delta) + 2.0f);
    else                h = 60.0f * (((r - g) / delta) + 4.0f);
    if (h < 0.0f) h += 360.0f;
    return h;
}

extern "C" __global__ void hsv_from_rgb_f32(
    const float* __restrict__ src, float* __restrict__ dst, unsigned int npixels)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= npixels) return;
    unsigned int si = i * 3u;
    float r = __ldg(&src[si]) * INV_255;
    float g = __ldg(&src[si + 1u]) * INV_255;
    float b = __ldg(&src[si + 2u]) * INV_255;
    float maxv = fmaxf(fmaxf(r, g), b);
    float minv = fminf(fminf(r, g), b);
    float delta = maxv - minv;
    float h = (delta == 0.0f) ? 0.0f : hue_deg(r, g, b, maxv, delta);
    float s = (maxv == 0.0f) ? 0.0f : (delta / maxv) * 255.0f;
    dst[si]      = h * DEG_TO_BYTE;
    dst[si + 1u] = s;
    dst[si + 2u] = maxv * 255.0f;
}

extern "C" __global__ void rgb_from_hsv_f32(
    const float* __restrict__ src, float* __restrict__ dst, unsigned int npixels)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= npixels) return;
    unsigned int si = i * 3u;
    float s = __ldg(&src[si + 1u]) * INV_255;
    float v = __ldg(&src[si + 2u]) * INV_255;
    float hh = __ldg(&src[si]) * (BYTE_TO_DEG / 60.0f);   // [0, 6)
    float c = v * s;
    float hmod2 = hh - 2.0f * floorf(hh * 0.5f);
    float x = c * (1.0f - fabsf(hmod2 - 1.0f));
    float m = v - c;
    int sext = (int)floorf(hh);
    float r1, g1, b1;
    switch (sext) {
        case 0:  r1 = c;    g1 = x;    b1 = 0.0f; break;
        case 1:  r1 = x;    g1 = c;    b1 = 0.0f; break;
        case 2:  r1 = 0.0f; g1 = c;    b1 = x;    break;
        case 3:  r1 = 0.0f; g1 = x;    b1 = c;    break;
        case 4:  r1 = x;    g1 = 0.0f; b1 = c;    break;
        default: r1 = c;    g1 = 0.0f; b1 = x;    break;
    }
    dst[si]      = (r1 + m) * 255.0f;
    dst[si + 1u] = (g1 + m) * 255.0f;
    dst[si + 2u] = (b1 + m) * 255.0f;
}

extern "C" __global__ void hls_from_rgb_f32(
    const float* __restrict__ src, float* __restrict__ dst, unsigned int npixels)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= npixels) return;
    unsigned int si = i * 3u;
    float r = __ldg(&src[si]) * INV_255;
    float g = __ldg(&src[si + 1u]) * INV_255;
    float b = __ldg(&src[si + 2u]) * INV_255;
    float maxv = fmaxf(fmaxf(r, g), b);
    float minv = fminf(fminf(r, g), b);
    float diff = maxv - minv;
    float sum = maxv + minv;
    float l = sum * 0.5f;
    float h = 0.0f, s = 0.0f;
    if (diff != 0.0f) {
        s = (l <= 0.5f) ? (diff / sum) : (diff / (2.0f - sum));
        h = hue_deg(r, g, b, maxv, diff);
    }
    dst[si]      = h * DEG_TO_BYTE;
    dst[si + 1u] = l * 255.0f;
    dst[si + 2u] = s * 255.0f;
}

// Matches color/hls/kernels.rs::hue2rgb_scalar.
__device__ __forceinline__ float hue2rgb(float p, float q, float t)
{
    if (t < 0.0f) t += 1.0f;
    if (t > 1.0f) t -= 1.0f;
    if (t < 1.0f / 6.0f) return p + (q - p) * 6.0f * t;
    if (t < 0.5f)        return q;
    if (t < 2.0f / 3.0f) return p + (q - p) * (2.0f / 3.0f - t) * 6.0f;
    return p;
}

extern "C" __global__ void rgb_from_hls_f32(
    const float* __restrict__ src, float* __restrict__ dst, unsigned int npixels)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= npixels) return;
    unsigned int si = i * 3u;
    float l = __ldg(&src[si + 1u]) * INV_255;
    float s = __ldg(&src[si + 2u]) * INV_255;
    if (s == 0.0f) {
        float v = l * 255.0f;
        dst[si] = v; dst[si + 1u] = v; dst[si + 2u] = v;
        return;
    }
    float h_deg = __ldg(&src[si]) * BYTE_TO_DEG;
    float q = (l < 0.5f) ? (l * (1.0f + s)) : (l + s - l * s);
    float p = 2.0f * l - q;
    float hk = h_deg / 360.0f;
    dst[si]      = hue2rgb(p, q, hk + 1.0f / 3.0f) * 255.0f;
    dst[si + 1u] = hue2rgb(p, q, hk) * 255.0f;
    dst[si + 2u] = hue2rgb(p, q, hk - 1.0f / 3.0f) * 255.0f;
}
"#;
const HSV_HLS_F32_FNS: &[&str] = &[
    "hsv_from_rgb_f32",
    "rgb_from_hsv_f32",
    "hls_from_rgb_f32",
    "rgb_from_hls_f32",
];

static HSV_HLS_F32: KernelSuiteCell = KernelSuiteCell::new();

fn launch_entry(
    index: usize,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<f32>,
    dst: &mut CudaSlice<f32>,
    npixels: usize,
) -> Result<(), CudaColorError> {
    check_len("src", src.len(), npixels * 3)?;
    check_len("dst", dst.len(), npixels * 3)?;
    let kernel = get_kernel_suite(
        &HSV_HLS_F32,
        stream,
        HSV_HLS_F32_SRC,
        HSV_HLS_F32_FNS,
        index,
    )?;
    let n = npixels as u32;
    kernel
        .launch_builder(stream)
        .arg(src)
        .arg(dst)
        .arg(&n)
        .launch_1d(n)?;
    Ok(())
}

/// Launch RGB f32 → HSV f32 (channels `[0,255]`, H scaled from `[0,360)`).
pub fn launch_hsv_from_rgb_f32(
    stream: &Arc<CudaStream>,
    src: &CudaSlice<f32>,
    dst: &mut CudaSlice<f32>,
    npixels: usize,
) -> Result<(), CudaColorError> {
    launch_entry(0, stream, src, dst, npixels)
}

/// Launch HSV f32 → RGB f32.
pub fn launch_rgb_from_hsv_f32(
    stream: &Arc<CudaStream>,
    src: &CudaSlice<f32>,
    dst: &mut CudaSlice<f32>,
    npixels: usize,
) -> Result<(), CudaColorError> {
    launch_entry(1, stream, src, dst, npixels)
}

/// Launch RGB f32 → HLS f32 (channel order `[H, L, S]`).
pub fn launch_hls_from_rgb_f32(
    stream: &Arc<CudaStream>,
    src: &CudaSlice<f32>,
    dst: &mut CudaSlice<f32>,
    npixels: usize,
) -> Result<(), CudaColorError> {
    launch_entry(2, stream, src, dst, npixels)
}

/// Launch HLS f32 → RGB f32 (channel order `[H, L, S]`).
pub fn launch_rgb_from_hls_f32(
    stream: &Arc<CudaStream>,
    src: &CudaSlice<f32>,
    dst: &mut CudaSlice<f32>,
    npixels: usize,
) -> Result<(), CudaColorError> {
    launch_entry(3, stream, src, dst, npixels)
}

#[cfg(all(test, feature = "gpu-cuda"))]
mod tests {
    use super::*;
    use crate::gpu::color_cuda::test_utils::{default_stream, pattern_u8};

    /// [0,255]-scaled f32 RGB pattern with gray pixels (delta == 0), saturated
    /// primaries (hue sextant boundaries), and LCG noise.
    fn rgb_pattern(npixels: usize) -> Vec<f32> {
        let mut v = vec![
            0.0, 0.0, 0.0, // black
            255.0, 255.0, 255.0, // white (delta == 0)
            128.0, 128.0, 128.0, // gray
            255.0, 0.0, 0.0, // saturated R
            0.0, 255.0, 0.0, // saturated G
            0.0, 0.0, 255.0, // saturated B
            255.0, 255.0, 0.0, // yellow (max tie r/g)
            0.0, 255.0, 255.0, // cyan
            255.0, 0.0, 255.0, // magenta (hue wrap region)
        ];
        v.extend(
            pattern_u8(npixels * 3 - v.len())
                .into_iter()
                .map(|b| b as f32),
        );
        v
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max)
    }

    #[test]
    fn hsv_hls_roundtrip_close_to_cpu() {
        let stream = default_stream();
        let n = 37 * 23;
        let rgb = rgb_pattern(n);

        type CpuFn = fn(&[f32], &mut [f32], usize);
        type Launch = fn(
            &Arc<CudaStream>,
            &CudaSlice<f32>,
            &mut CudaSlice<f32>,
            usize,
        ) -> Result<(), CudaColorError>;
        let cases: &[(&str, CpuFn, Launch)] = &[
            (
                "hsv_from_rgb",
                crate::color::hsv::kernels::hsv_from_rgb_f32,
                launch_hsv_from_rgb_f32,
            ),
            (
                "hls_from_rgb",
                crate::color::hls::kernels::hls_from_rgb_f32,
                launch_hls_from_rgb_f32,
            ),
        ];
        for (name, cpu_fn, launch) in cases {
            let mut cpu = vec![0f32; n * 3];
            cpu_fn(&rgb, &mut cpu, n);
            let d_src = stream.clone_htod(&rgb).unwrap();
            let mut d_dst = stream.alloc_zeros::<f32>(n * 3).unwrap();
            launch(&stream, &d_src, &mut d_dst, n).unwrap();
            let gpu: Vec<f32> = stream.clone_dtoh(&d_dst).unwrap();
            stream.synchronize().unwrap();
            let diff = max_abs_diff(&gpu, &cpu);
            assert!(diff <= 1e-3, "{name} max diff {diff} > 1e-3");
        }

        // Inverse: feed CPU-forward output through the CUDA inverse and compare
        // against the CPU inverse.
        let inverse_cases: &[(&str, CpuFn, CpuFn, Launch)] = &[
            (
                "rgb_from_hsv",
                crate::color::hsv::kernels::hsv_from_rgb_f32,
                crate::color::hsv::kernels::rgb_from_hsv_f32,
                launch_rgb_from_hsv_f32,
            ),
            (
                "rgb_from_hls",
                crate::color::hls::kernels::hls_from_rgb_f32,
                crate::color::hls::kernels::rgb_from_hls_f32,
                launch_rgb_from_hls_f32,
            ),
        ];
        for (name, fwd, inv, launch) in inverse_cases {
            let mut mid = vec![0f32; n * 3];
            fwd(&rgb, &mut mid, n);
            let mut cpu = vec![0f32; n * 3];
            inv(&mid, &mut cpu, n);
            let d_src = stream.clone_htod(&mid).unwrap();
            let mut d_dst = stream.alloc_zeros::<f32>(n * 3).unwrap();
            launch(&stream, &d_src, &mut d_dst, n).unwrap();
            let gpu: Vec<f32> = stream.clone_dtoh(&d_dst).unwrap();
            stream.synchronize().unwrap();
            let diff = max_abs_diff(&gpu, &cpu);
            assert!(diff <= 1e-3, "{name} max diff {diff} > 1e-3");
        }
    }
}
