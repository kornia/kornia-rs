//! Bit-exact CUDA ports of the two scalar-math primitives the orientation and
//! descriptor stages depend on.
//!
//! Neither can be replaced by its obvious CUDA counterpart:
//!
//! * `atan2f` is not the reference's angle function. On this platform the
//!   reference's angle helper is overridden by an accelerated backend whose
//!   implementation divides with a **NEON reciprocal estimate**
//!   (`vrecpe` + exactly two Newton-Raphson steps), not an exact division, and
//!   evaluates its polynomial with plain multiply/add. Measured natively on
//!   aarch64: 0 mismatches over 4096 samples at two Newton steps, 2540 at one
//!   and 262 at three — more refinement is *worse*, because the target is the
//!   approximation, not `1/x`.
//! * `expf` is not the reference's exponential either: it uses a 64-entry table
//!   plus exponent injection plus a 4-term FMA polynomial.
//!
//! Both are reproduced here bit-for-bit, verified against the reference's own
//! entry points (never against a language binding — a binding may resolve to a
//! different accelerated backend and give different bits).

use super::kernels::f32_lit;

/// ARM `FRECPE` reciprocal-estimate table.
///
/// For a positive normal input the estimate depends only on the top 8 fraction
/// bits, so the whole instruction reduces to an integer construction plus this
/// 256-entry lookup:
///
/// ```text
/// e = (bits >> 23) & 0xFF;  f = bits & 0x7FFFFF;
/// q = 256 + (f >> 15);                  // floor(((1.m)/2) * 512)
/// s = TABLE[q - 256];                   // floor(512/(q+0.5) * 256 + 0.5)
/// result = ((253 - e) << 23) | ((s - 256) << 15);
/// ```
pub fn vrecpe_table() -> [u16; 256] {
    let mut t = [0u16; 256];
    for (i, slot) in t.iter_mut().enumerate() {
        let q = 256.0 + i as f64;
        let r = 512.0 / (q + 0.5);
        *slot = (r * 256.0 + 0.5).floor() as u16;
    }
    t
}

/// ARM `FRSQRTE` reciprocal-square-root estimate tables.
///
/// Like `FRECPE` this is table-driven over the top fraction bits, but with an
/// **exponent-parity split**: the biased exponent's low bit selects which of two
/// 128-entry tables to use. Note the bias 127 is odd, so an *odd* biased
/// exponent means an *even* unbiased one, which is the `a in [0.25,0.5)` branch.
/// Getting that backwards yields results off by exactly sqrt(2) on every input.
pub fn vrsqrte_tables() -> ([u16; 128], [u16; 128]) {
    let mut lo = [0u16; 128]; // a in [0.25, 0.5)
    let mut hi = [0u16; 128]; // a in [0.5, 1.0)
    for i in 0..128 {
        let q = 128.5 + i as f64;
        lo[i] = (256.0 / (q / 512.0).sqrt() + 0.5).floor() as u16;
        hi[i] = (256.0 / (q / 256.0).sqrt() + 0.5).floor() as u16;
    }
    (lo, hi)
}

/// Table of `2^(i/64) * A0` used by the reference's exponential, where
/// `A0` is the constant its polynomial coefficients are divided by (the factor
/// cancels, but the intermediate rounding does not, so it must be kept).
pub fn exp_table_f32() -> [f32; 64] {
    const A0: f64 = 0.009_670_371_139_572_338;
    let mut t = [0.0f32; 64];
    for (i, slot) in t.iter_mut().enumerate() {
        *slot = (2f64.powf(i as f64 / 64.0) * A0) as f32;
    }
    t
}

/// Device-side source for both primitives, with all tables baked as literals.
pub(crate) fn hal_device_src() -> String {
    let recp = vrecpe_table()
        .iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    // Bit patterns, not `__int_as_float(...)` calls: a `__device__` array must
    // have a constant initialiser, and the intrinsic does not qualify.
    let exp_tab = exp_table_f32()
        .iter()
        .map(|&v| format!("0x{:08x}u", v.to_bits()))
        .collect::<Vec<_>>()
        .join(", ");

    const A0: f64 = 0.009_670_371_139_572_338;
    let a4 = f32_lit((1.0_f64 / A0) as f32);
    let a3 = f32_lit((0.693_147_180_552_144_8 / A0) as f32);
    let a2 = f32_lit((0.240_226_510_951_330_15 / A0) as f32);
    let a1 = f32_lit((0.055_503_393_667_531_25 / A0) as f32);

    let prescale = std::f64::consts::LOG2_E * 64.0;
    let exp_max = 3000.0 * 64.0;
    let minval = f32_lit((-exp_max / prescale) as f32);
    let maxval = f32_lit((exp_max / prescale) as f32);
    let pres = f32_lit(prescale as f32);
    let posts = f32_lit((1.0 / 64.0) as f32);

    // Angle polynomial constants: products evaluated in DOUBLE then narrowed.
    // The other variant of the reference computes them as float-literal
    // products, which differs in the last bit of p1 and is wrong for this path.
    let d = 180.0f64 / std::f64::consts::PI;
    let p1 = f32_lit((0.999_787_841_279_480_7 * d) as f32);
    let p3 = f32_lit((-0.325_808_397_464_097_5 * d) as f32);
    let p5 = f32_lit((0.155_578_651_846_328_1 * d) as f32);
    let p7 = f32_lit((-0.044_326_555_547_921_28 * d) as f32);
    let eps = f32_lit(f64::EPSILON as f32);
    let (rs_lo_t, rs_hi_t) = vrsqrte_tables();
    let rs_lo = rs_lo_t
        .iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let rs_hi = rs_hi_t
        .iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(", ");

    format!(
        r#"
// The estimate tables live at FILE scope, not inside the functions that read
// them. A `const` array declared inside a `__device__` function and indexed by a
// runtime value cannot live in registers, so nvcc gives it a per-thread local
// allocation and re-materialises all of it on every call: ptxas reported a
// 768-byte stack frame per thread, and the resulting local-memory traffic made
// the descriptor stage ~85% of the whole pipeline. At file scope they are read
// once into L1 and shared by every thread.
__device__ const unsigned short KORNIA_SIFT_RECP_TAB[256] = {{ {recp} }};
__device__ const unsigned short KORNIA_SIFT_RSQRT_LO[128] = {{ {rs_lo} }};
__device__ const unsigned short KORNIA_SIFT_RSQRT_HI[128] = {{ {rs_hi} }};
__device__ const unsigned int KORNIA_SIFT_EXP_TAB[64] = {{ {exp_tab} }};

__device__ __forceinline__ float sift_vrecpe(float v) {{
    // `q` indexes the baked estimate table by the top 8 fraction bits.
    const unsigned short* tab = KORNIA_SIFT_RECP_TAB;
    const unsigned int b = __float_as_uint(v);
    const unsigned int e = (b >> 23) & 0xFFu;
    const unsigned int q = (b >> 15) & 0xFFu;
    const unsigned int s = (unsigned int)tab[q];
    return __uint_as_float(((253u - e) << 23) | ((s - 256u) << 15));
}}

// FRECPS: fused 2 - a*b.
__device__ __forceinline__ float sift_vrecps(float a, float b) {{
    return __fmaf_rn(-a, b, 2.0f);
}}

// Reciprocal with exactly two Newton-Raphson steps.
__device__ __forceinline__ float sift_recip(float v) {{
    float r = sift_vrecpe(v);
    r = sift_vrecps(v, r) * r;
    r = sift_vrecps(v, r) * r;
    return r;
}}

// Raw polynomial value, before the quadrant fixups. Test probe only.
__device__ __forceinline__ float sift_atan2_raw(float y, float x) {{
    const float ax = fabsf(x), ay = fabsf(y);
    const float tmin = fminf(ax, ay), tmax = fmaxf(ax, ay);
    const float c = tmin * sift_recip(tmax + {eps});
    const float c2 = c * c;
    float a = c2 * {p7};
    a = (a + {p5}) * c2;
    a = (a + {p3}) * c2;
    a = (a + {p1}) * c;
    return a;
}}

// Angle in degrees, matching the accelerated backend bit for bit.
__device__ __forceinline__ float sift_atan2_deg(float y, float x) {{
    const float ax = fabsf(x), ay = fabsf(y);
    const float tmin = fminf(ax, ay), tmax = fmaxf(ax, ay);
    const float c = tmin * sift_recip(tmax + {eps});
    const float c2 = c * c;
    // Horner with FUSED multiply-adds. The separate-mul/add form differs by one
    // ULP and was the source of a long-standing residual: only the fused Horner
    // chain reproduces the reference.
    float a = __fmaf_rn(c2, {p7}, {p5});
    a = __fmaf_rn(a, c2, {p3});
    a = __fmaf_rn(a, c2, {p1});
    a = a * c;
    if (!(ax >= ay)) a = 90.0f - a;
    if (x < 0.0f)    a = 180.0f - a;
    if (y < 0.0f)    a = 360.0f - a;
    return a;
}}


__device__ __forceinline__ float sift_rsqrte(float v) {{
    const unsigned short* lo = KORNIA_SIFT_RSQRT_LO;
    const unsigned short* hi = KORNIA_SIFT_RSQRT_HI;
    const unsigned int b = __float_as_uint(v);
    const unsigned int e = (b >> 23) & 0xFFu;
    const unsigned int idx = (b >> 16) & 0x7Fu;
    const unsigned int s = (e & 1u) ? (unsigned int)lo[idx] : (unsigned int)hi[idx];
    return __uint_as_float((((380u - e) >> 1) << 23) | ((s - 256u) << 15));
}}

// FRSQRTS: fused (3 - a*b) / 2.
__device__ __forceinline__ float sift_rsqrts(float a, float b) {{
    return __fmaf_rn(-a, b, 3.0f) * 0.5f;
}}

// Reciprocal square root with exactly two refinement steps.
__device__ __forceinline__ float sift_rsqrt(float v) {{
    float e = sift_rsqrte(v);
    e = sift_rsqrts(e * e, v) * e;
    e = sift_rsqrts(e * e, v) * e;
    return e;
}}

// The accelerated backend computes sqrt as recip(rsqrt) — it never issues a
// square-root instruction, so `sqrtf` does NOT reproduce it.
__device__ __forceinline__ float sift_magnitude(float x, float y) {{
    // The accelerated backend's build contracts `x*x + y*y` into a single FMA;
    // computing it with two roundings disagrees on ~7% of inputs.
    const float s = __fmaf_rn(x, x, y * y);
    return sift_recip(sift_rsqrt(s));
}}

__device__ __forceinline__ float sift_exp(float x) {{
    const unsigned int* tab = KORNIA_SIFT_EXP_TAB;
    x = fminf(fmaxf(x, {minval}), {maxval});
    x = x * {pres};
    const int xi = __float2int_rn(x);           // round-to-nearest-even
    const float xf = (x - (float)xi) * {posts};
    float yf = __uint_as_float(tab[xi & 63]);
    int t = (xi >> 6) + 127;
    t = min(max(t, 0), 255);
    yf = yf * __uint_as_float((unsigned int)t << 23);
    float z = xf + {a1};
    z = __fmaf_rn(z, xf, {a2});
    z = __fmaf_rn(z, xf, {a3});
    z = __fmaf_rn(z, xf, {a4});
    return z * yf;
}}
"#
    )
}

/// Probe kernel used only by tests: evaluates both primitives elementwise so
/// they can be compared against reference vectors captured from the reference's
/// own entry points.
#[cfg(test)]
pub(crate) fn hal_probe_src() -> String {
    format!(
        r#"{}{}
extern "C" __global__ void sift_hal_probe(
    const float* __restrict__ x, const float* __restrict__ y,
    float* __restrict__ out_exp, float* __restrict__ out_atan, int n)
{{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out_exp[i] = sift_exp(x[i]);
    out_atan[i] = sift_atan2_deg(y[i], x[i]);
}}
"#,
        hal_device_src(),
        super::detect::exp2f_device_src()
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::color::test_utils::default_stream;
    use crate::cuda::make_config;
    use crate::cuda::sift::kernels::get_or_compile;

    /// `(x, y, exp(x), atan2(y, x))` reference vectors.
    type HalRef = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);

    /// Reference vectors captured from the reference implementation's own
    /// entry points: `[i32 n][n f32 x][n f32 y][n f32 exp][n f32 atan2]`.
    fn load_halref() -> Option<HalRef> {
        let path = std::env::var("KORNIA_SIFT_HALREF").ok()?;
        let b = std::fs::read(path).ok()?;
        let n = i32::from_le_bytes(b[0..4].try_into().unwrap()) as usize;
        let rd = |off: usize| -> Vec<f32> {
            b[off..off + 4 * n]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect()
        };
        Some((rd(4), rd(4 + 4 * n), rd(4 + 8 * n), rd(4 + 12 * n)))
    }

    #[test]
    fn hal_primitives_match_reference_bitwise() {
        let Some((x, y, want_exp, want_atan)) = load_halref() else {
            eprintln!("KORNIA_SIFT_HALREF unset; skipping");
            return;
        };
        let n = x.len();
        let stream = default_stream();
        let ctx = &stream.context();

        let kernel = get_or_compile(ctx, "sift_hal_probe", hal_probe_src, "sift_hal_probe")
            .expect("compile probe");
        let d_x = stream.clone_htod(&x).unwrap();
        let d_y = stream.clone_htod(&y).unwrap();
        let mut d_e = stream.alloc_zeros::<f32>(n).unwrap();
        let mut d_a = stream.alloc_zeros::<f32>(n).unwrap();
        let n_i = n as i32;
        kernel
            .launch_builder(&stream)
            .arg(&d_x)
            .arg(&d_y)
            .arg(&mut d_e)
            .arg(&mut d_a)
            .arg(&n_i)
            .launch_2d(n as u32, 1, make_config(n as u32, 1, None))
            .expect("launch probe");

        let got_e: Vec<f32> = stream.clone_dtoh(&d_e).unwrap();
        let got_a: Vec<f32> = stream.clone_dtoh(&d_a).unwrap();

        let bad_e = got_e
            .iter()
            .zip(want_exp.iter())
            .filter(|(g, w)| g.to_bits() != w.to_bits())
            .count();
        let bad_a = got_a
            .iter()
            .zip(want_atan.iter())
            .filter(|(g, w)| g.to_bits() != w.to_bits())
            .count();
        if bad_a > 0 {
            let first = (0..n).find(|&i| got_a[i].to_bits() != want_atan[i].to_bits());
            eprintln!("  first mismatching index: {first:?}");
            let mut shown = 0;
            for i in 0..n {
                if got_a[i].to_bits() != want_atan[i].to_bits() && shown < 8 {
                    let (ax, ay) = (x[i].abs(), y[i].abs());
                    eprintln!(
                        "  x={:e} y={:e} ax>=ay={} |ax-ay|={:e} got={:?}({:#010x}) want={:?}({:#010x})",
                        x[i], y[i], ax >= ay, (ax - ay).abs(),
                        got_a[i], got_a[i].to_bits(), want_atan[i], want_atan[i].to_bits()
                    );
                    shown += 1;
                }
            }
        }
        assert_eq!(bad_e, 0, "exp: {bad_e} of {n} differ");
        assert_eq!(bad_a, 0, "atan2: {bad_a} of {n} differ");
    }

    /// Writes the generated device source so the emitted text can be read
    /// directly. Reasoning about what codegen *should* emit has been wrong
    /// before in this module; read the actual output.
    #[test]
    fn dump_generated_source() {
        if let Ok(path) = std::env::var("KORNIA_SIFT_DUMP_SRC") {
            std::fs::write(&path, hal_probe_src()).expect("write source dump");
            eprintln!("wrote generated source to {path}");
        }
    }

    #[test]
    fn vrecpe_table_endpoints() {
        let t = vrecpe_table();
        // q = 256 -> 512/256.5 = 1.99610 -> floor(511.00 + 0.5) = 511
        assert_eq!(t[0], 511);
        // q = 511 -> 512/511.5 = 1.000977 -> floor(256.25 + 0.5) = 256
        assert_eq!(t[255], 256);
        assert!(t.iter().all(|&s| (256..=511).contains(&s)));
    }
}
