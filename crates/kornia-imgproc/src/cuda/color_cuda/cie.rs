//! CUDA kernels for the CIE family: RGB ↔ linear-RGB / XYZ / Lab / Luv (f32).
//!
//! Mirrors `color/cie/kernels.rs` (f32 px oracles) and
//! `color/cie/transfer.rs` (sRGB transfer constants): RGB in `[0, 1]`,
//! `L ∈ [0, 100]`, D65 white point, canonical OpenCV coefficient digits.
//! `powf`/`cbrtf` ULP differences vs Rust libm keep the tests at a 1e-3
//! absolute tolerance rather than bit-exact.

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream};

use super::{get_kernel_suite, launch_map, CudaColorError, KernelSuiteCell, PxPerThread};

static CIE_F32_SRC: &str = r#"
// sRGB transfer — constants from color/cie/transfer.rs.
#define SRGB_THRESH     0.04045f
#define SRGB_INV_THRESH 0.0031308f
#define SRGB_A          0.055f
#define SRGB_INV_1055   (1.0f / 1.055f)
#define SRGB_INV_1292   (1.0f / 12.92f)
#define SRGB_1292       12.92f
#define SRGB_1055       1.055f
#define SRGB_GAMMA      2.4f
#define SRGB_INV_GAMMA  (1.0f / 2.4f)

// D65 white point + Lab/Luv constants from color/cie/kernels.rs.
#define XN 0.950456f
#define YN 1.0f
#define ZN 1.088754f
#define INV_XN (1.0f / 0.950456f)
#define INV_ZN (1.0f / 1.088754f)
#define LAB_DELTA       0.008856f
#define LAB_F_SLOPE     (1.0f / 0.12841855f)
#define LAB_F_OFFSET    0.13793103f
#define LAB_FINV_THRESH 0.20689655f
#define LAB_FINV_SLOPE  0.12841855f
#define LUV_UN    0.19793943f
#define LUV_VN    0.46831096f
#define LUV_KAPPA 903.3f

// __powf (hardware exp2/log2) instead of powf: ~4x cheaper, and its relative
// error (~1e-5 on this domain) sits well inside the f32 pipeline tolerance —
// verified against the f64 oracle in the tests.
__device__ __forceinline__ float srgb_to_linear(float x) {
    x = fmaxf(x, 0.0f);
    return (x <= SRGB_THRESH)
        ? x * SRGB_INV_1292
        : __powf((x + SRGB_A) * SRGB_INV_1055, SRGB_GAMMA);
}

__device__ __forceinline__ float linear_to_srgb(float l) {
    l = fmaxf(l, 0.0f);
    return (l <= SRGB_INV_THRESH)
        ? l * SRGB_1292
        : SRGB_1055 * __powf(l, SRGB_INV_GAMMA) - SRGB_A;
}

// linear-RGB -> XYZ (row-major M_RGB2XYZ) and inverse — same MAC order as matvec32.
__device__ __forceinline__ void xyz_mat(
    float r, float g, float b, float* x, float* y, float* z)
{
    *x = 0.412453f * r + 0.357580f * g + 0.180423f * b;
    *y = 0.212671f * r + 0.715160f * g + 0.072169f * b;
    *z = 0.019334f * r + 0.119193f * g + 0.950227f * b;
}

__device__ __forceinline__ void rgb_mat(
    float x, float y, float z, float* r, float* g, float* b)
{
    *r =  3.240479f * x + -1.537150f * y + -0.498535f * z;
    *g = -0.969256f * x +  1.875991f * y +  0.041556f * z;
    *b =  0.055648f * x + -0.204043f * y +  1.057311f * z;
}

__device__ __forceinline__ float lab_f(float t) {
    return (t > LAB_DELTA) ? cbrtf(t) : (t * LAB_F_SLOPE + LAB_F_OFFSET);
}

__device__ __forceinline__ float lab_finv(float f) {
    return (f > LAB_FINV_THRESH) ? (f * f * f) : (LAB_FINV_SLOPE * (f - LAB_F_OFFSET));
}

#define PIXEL_LOOP_HEAD \
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; \
    if (i >= npixels) return; \
    unsigned int si = i * 3u; \
    float c0 = __ldg(&src[si]); \
    float c1 = __ldg(&src[si + 1u]); \
    float c2 = __ldg(&src[si + 2u]);

extern "C" __global__ void linear_rgb_from_rgb_f32(
    const float* __restrict__ src, float* __restrict__ dst, unsigned int npixels)
{
    PIXEL_LOOP_HEAD
    dst[si]      = srgb_to_linear(c0);
    dst[si + 1u] = srgb_to_linear(c1);
    dst[si + 2u] = srgb_to_linear(c2);
}

extern "C" __global__ void rgb_from_linear_rgb_f32(
    const float* __restrict__ src, float* __restrict__ dst, unsigned int npixels)
{
    PIXEL_LOOP_HEAD
    dst[si]      = linear_to_srgb(c0);
    dst[si + 1u] = linear_to_srgb(c1);
    dst[si + 2u] = linear_to_srgb(c2);
}

extern "C" __global__ void xyz_from_rgb_f32(
    const float* __restrict__ src, float* __restrict__ dst, unsigned int npixels)
{
    PIXEL_LOOP_HEAD
    float x, y, z;
    xyz_mat(c0, c1, c2, &x, &y, &z);
    dst[si] = x; dst[si + 1u] = y; dst[si + 2u] = z;
}

extern "C" __global__ void rgb_from_xyz_f32(
    const float* __restrict__ src, float* __restrict__ dst, unsigned int npixels)
{
    PIXEL_LOOP_HEAD
    float r, g, b;
    rgb_mat(c0, c1, c2, &r, &g, &b);
    dst[si] = r; dst[si + 1u] = g; dst[si + 2u] = b;
}

extern "C" __global__ void lab_from_rgb_f32(
    const float* __restrict__ src, float* __restrict__ dst, unsigned int npixels)
{
    PIXEL_LOOP_HEAD
    float x, y, z;
    xyz_mat(srgb_to_linear(c0), srgb_to_linear(c1), srgb_to_linear(c2), &x, &y, &z);
    float fx = lab_f(x * INV_XN);
    float fy = lab_f(y);            // YN == 1
    float fz = lab_f(z * INV_ZN);
    dst[si]      = 116.0f * fy - 16.0f;
    dst[si + 1u] = 500.0f * (fx - fy);
    dst[si + 2u] = 200.0f * (fy - fz);
}

extern "C" __global__ void rgb_from_lab_f32(
    const float* __restrict__ src, float* __restrict__ dst, unsigned int npixels)
{
    PIXEL_LOOP_HEAD
    float fy = (c0 + 16.0f) / 116.0f;
    float fx = fy + c1 / 500.0f;
    float fz = fy - c2 / 200.0f;
    float r, g, b;
    rgb_mat(XN * lab_finv(fx), YN * lab_finv(fy), ZN * lab_finv(fz), &r, &g, &b);
    dst[si]      = linear_to_srgb(r);
    dst[si + 1u] = linear_to_srgb(g);
    dst[si + 2u] = linear_to_srgb(b);
}

extern "C" __global__ void luv_from_rgb_f32(
    const float* __restrict__ src, float* __restrict__ dst, unsigned int npixels)
{
    PIXEL_LOOP_HEAD
    float x, y, z;
    xyz_mat(srgb_to_linear(c0), srgb_to_linear(c1), srgb_to_linear(c2), &x, &y, &z);
    float l = (y > LAB_DELTA) ? (116.0f * cbrtf(y) - 16.0f) : (LUV_KAPPA * y);
    float d = x + 15.0f * y + 3.0f * z;
    float up = 0.0f, vp = 0.0f;
    if (d != 0.0f) {
        up = 4.0f * x / d;
        vp = 9.0f * y / d;
    }
    dst[si]      = l;
    dst[si + 1u] = 13.0f * l * (up - LUV_UN);
    dst[si + 2u] = 13.0f * l * (vp - LUV_VN);
}

extern "C" __global__ void rgb_from_luv_f32(
    const float* __restrict__ src, float* __restrict__ dst, unsigned int npixels)
{
    PIXEL_LOOP_HEAD
    float x = 0.0f, y = 0.0f, z = 0.0f;
    if (c0 > 0.0f) {
        if (c0 > 8.0f) {
            float t = (c0 + 16.0f) / 116.0f;
            y = YN * t * t * t;
        } else {
            y = YN * c0 / LUV_KAPPA;
        }
        float inv13l = 1.0f / (13.0f * c0);
        float up = c1 * inv13l + LUV_UN;
        float vp = c2 * inv13l + LUV_VN;
        x = y * 9.0f * up / (4.0f * vp);
        z = y * (12.0f - 3.0f * up - 20.0f * vp) / (4.0f * vp);
    }
    float r, g, b;
    rgb_mat(x, y, z, &r, &g, &b);
    dst[si]      = linear_to_srgb(r);
    dst[si + 1u] = linear_to_srgb(g);
    dst[si + 2u] = linear_to_srgb(b);
}
"#;
const CIE_F32_FNS: &[&str] = &[
    "linear_rgb_from_rgb_f32",
    "rgb_from_linear_rgb_f32",
    "xyz_from_rgb_f32",
    "rgb_from_xyz_f32",
    "lab_from_rgb_f32",
    "rgb_from_lab_f32",
    "luv_from_rgb_f32",
    "rgb_from_luv_f32",
];

// f64 twins. Constants are float literals widened to double — the CPU f64
// oracle also derives them from the shared f32 constants (`X as f64`), so
// this reproduces the exact same values.
static CIE_F64_SRC: &str = r#"
#define D(x) ((double)(x))
#define SRGB_THRESH_D     D(0.04045f)
#define SRGB_INV_THRESH_D D(0.0031308f)
#define SRGB_A_D          D(0.055f)
#define XN_D D(0.950456f)
#define ZN_D D(1.088754f)
#define LAB_DELTA_D       D(0.008856f)
#define LAB_F_SLOPE_D     (1.0 / D(0.12841855f))
#define LAB_F_OFFSET_D    D(0.13793103f)
#define LAB_FINV_THRESH_D D(0.20689655f)
#define LAB_FINV_SLOPE_D  D(0.12841855f)
#define LUV_UN_D    D(0.19793943f)
#define LUV_VN_D    D(0.46831096f)
#define LUV_KAPPA_D D(903.3f)

// Transfer uses EXACT double literals — the CPU f64 oracle
// (linear_from_srgb_scalar64) writes 12.92/1.055/0.055/2.4 as doubles, while
// only the branch thresholds come from widened f32 constants.
__device__ __forceinline__ double srgb_to_linear_d(double x) {
    x = fmax(x, 0.0);
    return (x <= SRGB_THRESH_D)
        ? x / 12.92
        : pow((x + 0.055) / 1.055, 2.4);
}

__device__ __forceinline__ double linear_to_srgb_d(double l) {
    l = fmax(l, 0.0);
    return (l <= SRGB_INV_THRESH_D)
        ? 12.92 * l
        : 1.055 * pow(l, 1.0 / 2.4) - 0.055;
}

__device__ __forceinline__ void xyz_mat_d(
    double r, double g, double b, double* x, double* y, double* z)
{
    *x = D(0.412453f) * r + D(0.357580f) * g + D(0.180423f) * b;
    *y = D(0.212671f) * r + D(0.715160f) * g + D(0.072169f) * b;
    *z = D(0.019334f) * r + D(0.119193f) * g + D(0.950227f) * b;
}

__device__ __forceinline__ void rgb_mat_d(
    double x, double y, double z, double* r, double* g, double* b)
{
    *r =  D(3.240479f) * x + D(-1.537150f) * y + D(-0.498535f) * z;
    *g = D(-0.969256f) * x +  D(1.875991f) * y +  D(0.041556f) * z;
    *b =  D(0.055648f) * x + D(-0.204043f) * y +  D(1.057311f) * z;
}

__device__ __forceinline__ double lab_f_d(double t) {
    return (t > LAB_DELTA_D) ? cbrt(t) : (t * LAB_F_SLOPE_D + LAB_F_OFFSET_D);
}

__device__ __forceinline__ double lab_finv_d(double f) {
    return (f > LAB_FINV_THRESH_D) ? (f * f * f) : (LAB_FINV_SLOPE_D * (f - LAB_F_OFFSET_D));
}

#define PIXEL_HEAD_D     unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;     if (i >= npixels) return;     unsigned int si = i * 3u;     double c0 = __ldg(&src[si]);     double c1 = __ldg(&src[si + 1u]);     double c2 = __ldg(&src[si + 2u]);

extern "C" __global__ void linear_rgb_from_rgb_f64(
    const double* __restrict__ src, double* __restrict__ dst, unsigned int npixels)
{
    PIXEL_HEAD_D
    dst[si] = srgb_to_linear_d(c0);
    dst[si + 1u] = srgb_to_linear_d(c1);
    dst[si + 2u] = srgb_to_linear_d(c2);
}

extern "C" __global__ void rgb_from_linear_rgb_f64(
    const double* __restrict__ src, double* __restrict__ dst, unsigned int npixels)
{
    PIXEL_HEAD_D
    dst[si] = linear_to_srgb_d(c0);
    dst[si + 1u] = linear_to_srgb_d(c1);
    dst[si + 2u] = linear_to_srgb_d(c2);
}

extern "C" __global__ void xyz_from_rgb_f64(
    const double* __restrict__ src, double* __restrict__ dst, unsigned int npixels)
{
    PIXEL_HEAD_D
    double x, y, z;
    xyz_mat_d(c0, c1, c2, &x, &y, &z);
    dst[si] = x; dst[si + 1u] = y; dst[si + 2u] = z;
}

extern "C" __global__ void rgb_from_xyz_f64(
    const double* __restrict__ src, double* __restrict__ dst, unsigned int npixels)
{
    PIXEL_HEAD_D
    double r, g, b;
    rgb_mat_d(c0, c1, c2, &r, &g, &b);
    dst[si] = r; dst[si + 1u] = g; dst[si + 2u] = b;
}

extern "C" __global__ void lab_from_rgb_f64(
    const double* __restrict__ src, double* __restrict__ dst, unsigned int npixels)
{
    PIXEL_HEAD_D
    double x, y, z;
    xyz_mat_d(srgb_to_linear_d(c0), srgb_to_linear_d(c1), srgb_to_linear_d(c2), &x, &y, &z);
    double fx = lab_f_d(x / XN_D);
    double fy = lab_f_d(y);
    double fz = lab_f_d(z / ZN_D);
    dst[si]      = 116.0 * fy - 16.0;
    dst[si + 1u] = 500.0 * (fx - fy);
    dst[si + 2u] = 200.0 * (fy - fz);
}

extern "C" __global__ void rgb_from_lab_f64(
    const double* __restrict__ src, double* __restrict__ dst, unsigned int npixels)
{
    PIXEL_HEAD_D
    double fy = (c0 + 16.0) / 116.0;
    double fx = fy + c1 / 500.0;
    double fz = fy - c2 / 200.0;
    double r, g, b;
    rgb_mat_d(XN_D * lab_finv_d(fx), lab_finv_d(fy), ZN_D * lab_finv_d(fz), &r, &g, &b);
    dst[si]      = linear_to_srgb_d(r);
    dst[si + 1u] = linear_to_srgb_d(g);
    dst[si + 2u] = linear_to_srgb_d(b);
}

extern "C" __global__ void luv_from_rgb_f64(
    const double* __restrict__ src, double* __restrict__ dst, unsigned int npixels)
{
    PIXEL_HEAD_D
    double x, y, z;
    xyz_mat_d(srgb_to_linear_d(c0), srgb_to_linear_d(c1), srgb_to_linear_d(c2), &x, &y, &z);
    double l = (y > LAB_DELTA_D) ? (116.0 * cbrt(y) - 16.0) : (LUV_KAPPA_D * y);
    double d = x + 15.0 * y + 3.0 * z;
    double up = 0.0, vp = 0.0;
    if (d != 0.0) {
        up = 4.0 * x / d;
        vp = 9.0 * y / d;
    }
    dst[si]      = l;
    dst[si + 1u] = 13.0 * l * (up - LUV_UN_D);
    dst[si + 2u] = 13.0 * l * (vp - LUV_VN_D);
}

extern "C" __global__ void rgb_from_luv_f64(
    const double* __restrict__ src, double* __restrict__ dst, unsigned int npixels)
{
    PIXEL_HEAD_D
    double x = 0.0, y = 0.0, z = 0.0;
    if (c0 > 0.0) {
        if (c0 > 8.0) {
            double t = (c0 + 16.0) / 116.0;
            y = t * t * t;
        } else {
            y = c0 / LUV_KAPPA_D;
        }
        double inv13l = 1.0 / (13.0 * c0);
        double up = c1 * inv13l + LUV_UN_D;
        double vp = c2 * inv13l + LUV_VN_D;
        x = y * 9.0 * up / (4.0 * vp);
        z = y * (12.0 - 3.0 * up - 20.0 * vp) / (4.0 * vp);
    }
    double r, g, b;
    rgb_mat_d(x, y, z, &r, &g, &b);
    dst[si]      = linear_to_srgb_d(r);
    dst[si + 1u] = linear_to_srgb_d(g);
    dst[si + 2u] = linear_to_srgb_d(b);
}
"#;
const CIE_F64_FNS: &[&str] = &[
    "linear_rgb_from_rgb_f64",
    "rgb_from_linear_rgb_f64",
    "xyz_from_rgb_f64",
    "rgb_from_xyz_f64",
    "lab_from_rgb_f64",
    "rgb_from_lab_f64",
    "luv_from_rgb_f64",
    "rgb_from_luv_f64",
];
static CIE_F64: KernelSuiteCell = KernelSuiteCell::new();

static CIE_F32: KernelSuiteCell = KernelSuiteCell::new();

fn launch_entry(
    index: usize,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<f32>,
    dst: &mut CudaSlice<f32>,
    npixels: usize,
) -> Result<(), CudaColorError> {
    let kernel = get_kernel_suite(&CIE_F32, stream, CIE_F32_SRC, CIE_F32_FNS, index)?;
    launch_map(kernel, stream, src, dst, npixels, 3, 3, PxPerThread::One)
}

fn launch_entry_f64(
    index: usize,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<f64>,
    dst: &mut CudaSlice<f64>,
    npixels: usize,
) -> Result<(), CudaColorError> {
    let kernel = get_kernel_suite(&CIE_F64, stream, CIE_F64_SRC, CIE_F64_FNS, index)?;
    launch_map(kernel, stream, src, dst, npixels, 3, 3, PxPerThread::One)
}

macro_rules! cie_launcher_f64 {
    ($(#[$meta:meta])* $name:ident, $index:expr) => {
        $(#[$meta])*
        pub fn $name(
            stream: &Arc<CudaStream>,
            src: &CudaSlice<f64>,
            dst: &mut CudaSlice<f64>,
            npixels: usize,
        ) -> Result<(), CudaColorError> {
            launch_entry_f64($index, stream, src, dst, npixels)
        }
    };
}

cie_launcher_f64!(
    /// Launch sRGB f64 → linear-RGB f64.
    launch_linear_rgb_from_rgb_f64, 0
);
cie_launcher_f64!(
    /// Launch linear-RGB f64 → sRGB f64.
    launch_rgb_from_linear_rgb_f64, 1
);
cie_launcher_f64!(
    /// Launch RGB f64 → XYZ f64.
    launch_xyz_from_rgb_f64, 2
);
cie_launcher_f64!(
    /// Launch XYZ f64 → RGB f64.
    launch_rgb_from_xyz_f64, 3
);
cie_launcher_f64!(
    /// Launch RGB f64 → Lab f64.
    launch_lab_from_rgb_f64, 4
);
cie_launcher_f64!(
    /// Launch Lab f64 → RGB f64.
    launch_rgb_from_lab_f64, 5
);
cie_launcher_f64!(
    /// Launch RGB f64 → Luv f64.
    launch_luv_from_rgb_f64, 6
);
cie_launcher_f64!(
    /// Launch Luv f64 → RGB f64.
    launch_rgb_from_luv_f64, 7
);

macro_rules! cie_launcher {
    ($(#[$meta:meta])* $name:ident, $index:expr) => {
        $(#[$meta])*
        pub fn $name(
            stream: &Arc<CudaStream>,
            src: &CudaSlice<f32>,
            dst: &mut CudaSlice<f32>,
            npixels: usize,
        ) -> Result<(), CudaColorError> {
            launch_entry($index, stream, src, dst, npixels)
        }
    };
}

cie_launcher!(
    /// Launch sRGB → linear-RGB (per-channel transfer, RGB in `[0,1]`).
    launch_linear_rgb_from_rgb_f32, 0
);
cie_launcher!(
    /// Launch linear-RGB → sRGB (per-channel transfer).
    launch_rgb_from_linear_rgb_f32, 1
);
cie_launcher!(
    /// Launch RGB → XYZ (matrix only, matches OpenCV semantics).
    launch_xyz_from_rgb_f32, 2
);
cie_launcher!(
    /// Launch XYZ → RGB (matrix only).
    launch_rgb_from_xyz_f32, 3
);
cie_launcher!(
    /// Launch RGB → Lab (gamma-aware, D65, `L ∈ [0,100]`).
    launch_lab_from_rgb_f32, 4
);
cie_launcher!(
    /// Launch Lab → RGB.
    launch_rgb_from_lab_f32, 5
);
cie_launcher!(
    /// Launch RGB → Luv (gamma-aware, D65).
    launch_luv_from_rgb_f32, 6
);
cie_launcher!(
    /// Launch Luv → RGB.
    launch_rgb_from_luv_f32, 7
);

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use kornia_image::{Image, ImageSize};

    use super::*;
    use crate::cuda::color_cuda::test_utils::{default_stream, pattern_f32};

    type CpuF64Fn = fn(&Image<f64, 3>, &mut Image<f64, 3>) -> Result<(), kornia_image::ImageError>;
    type Launch = fn(
        &Arc<CudaStream>,
        &CudaSlice<f32>,
        &mut CudaSlice<f32>,
        usize,
    ) -> Result<(), CudaColorError>;

    const SIZE: ImageSize = ImageSize {
        width: 37,
        height: 23,
    };

    /// Run the f64 CPU path (the crate's exact oracle — the f32 NEON path
    /// approximates pow/cbrt with polynomials, so it is NOT a reference).
    fn oracle_f64(f: CpuF64Fn, input: &[f32]) -> Vec<f32> {
        let src = Image::<f64, 3>::new(SIZE, input.iter().map(|&v| v as f64).collect()).unwrap();
        let mut dst = Image::<f64, 3>::from_size_val(SIZE, 0.0).unwrap();
        f(&src, &mut dst).unwrap();
        dst.as_slice().iter().map(|&v| v as f32).collect()
    }

    fn gpu_run(launch: Launch, input: &[f32], n: usize) -> Vec<f32> {
        let stream = default_stream();
        let d_src = stream.clone_htod(input).unwrap();
        let mut d_dst = stream.alloc_zeros::<f32>(n * 3).unwrap();
        launch(&stream, &d_src, &mut d_dst, n).unwrap();
        let out: Vec<f32> = stream.clone_dtoh(&d_dst).unwrap();
        stream.synchronize().unwrap();
        out
    }

    fn assert_close(name: &str, gpu: &[f32], oracle: &[f32], tol: f32) {
        let max_diff = crate::cuda::color_cuda::test_utils::max_abs_diff_f32(gpu, oracle);
        assert!(max_diff <= tol, "{name} max diff {max_diff} > {tol}");
    }

    #[test]
    fn cie_family_close_to_f64_oracle() {
        let n = SIZE.width * SIZE.height;
        // RGB in [0,1] with exact 0.0/1.0 (black/white) in the prefix.
        let rgb = pattern_f32(n * 3);

        // Tolerances: unit-range ops at 1e-4 (f32 rounding); Lab/Luv span
        // [0,100] with ±200 chroma → 1e-3 absolute.
        let cases: &[(&str, CpuF64Fn, Launch, f32)] = &[
            (
                "linear_rgb_from_rgb",
                crate::color::linear_rgb_from_rgb,
                launch_linear_rgb_from_rgb_f32,
                1e-4,
            ),
            (
                "rgb_from_linear_rgb",
                crate::color::rgb_from_linear_rgb,
                launch_rgb_from_linear_rgb_f32,
                1e-4,
            ),
            (
                "xyz_from_rgb",
                crate::color::xyz_from_rgb,
                launch_xyz_from_rgb_f32,
                1e-4,
            ),
            (
                "rgb_from_xyz",
                crate::color::rgb_from_xyz,
                launch_rgb_from_xyz_f32,
                1e-4,
            ),
            (
                "lab_from_rgb",
                crate::color::lab_from_rgb,
                launch_lab_from_rgb_f32,
                1e-3,
            ),
            (
                "luv_from_rgb",
                crate::color::luv_from_rgb,
                launch_luv_from_rgb_f32,
                1e-3,
            ),
        ];
        for (name, cpu64, launch, tol) in cases {
            let truth = oracle_f64(*cpu64, &rgb);
            let gpu = gpu_run(*launch, &rgb, n);
            assert_close(name, &gpu, &truth, *tol);
        }
    }

    #[test]
    fn lab_luv_inverse_close_to_f64_oracle() {
        let n = SIZE.width * SIZE.height;
        let rgb = pattern_f32(n * 3);

        let cases: &[(&str, CpuF64Fn, CpuF64Fn, Launch)] = &[
            (
                "rgb_from_lab",
                crate::color::lab_from_rgb,
                crate::color::rgb_from_lab,
                launch_rgb_from_lab_f32,
            ),
            (
                "rgb_from_luv",
                crate::color::luv_from_rgb,
                crate::color::rgb_from_luv,
                launch_rgb_from_luv_f32,
            ),
        ];
        for (name, fwd64, inv64, launch) in cases {
            let mid = oracle_f64(*fwd64, &rgb);
            let truth = oracle_f64(*inv64, &mid);
            let gpu = gpu_run(*launch, &mid, n);
            // Output is RGB in [0,1]; inverse pipelines round-trip through the
            // transfer, so allow 1e-3.
            assert_close(name, &gpu, &truth, 1e-3);
        }
    }
}
