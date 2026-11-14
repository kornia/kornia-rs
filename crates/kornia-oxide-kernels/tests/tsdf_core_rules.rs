//! Source rules of the shared TSDF core (`device/src/tsdf_core.rs`): the same file is compiled
//! by the host (CPU path of kornia-3d) and by cuda-oxide, so it may only use what both produce
//! the same bits for. The clippy `disallowed-methods` in `clippy.toml` covers the typed cases;
//! this scan also covers what clippy cannot resolve in a `no_std` build.

const CORE: &str = include_str!("../device/src/tsdf_core.rs");

/// Token and the reason it is forbidden in per-element code.
const FORBIDDEN: &[(&str, &str)] = &[
    (
        ".min(",
        "float min/max differ between CPU and device on +-0 and NaN: compare+select",
    ),
    (
        ".max(",
        "float min/max differ between CPU and device on +-0 and NaN: compare+select",
    ),
    (".clamp(", "lowers to min/max: compare+select"),
    ("f32::min", "compare+select"),
    ("f32::max", "compare+select"),
    ("f32::clamp", "compare+select"),
    (".mul_add(", "use F32Ops::fma"),
    ("f32::mul_add", "use F32Ops::fma"),
    (".sqrt(", "use F32Ops::sqrt"),
    ("f32::sqrt", "use F32Ops::sqrt"),
    (".floor(", "use floor_i32"),
    ("f32::floor", "use floor_i32"),
    (".ceil(", "libdevice on the device"),
    (".round(", "libdevice on the device"),
    ("libm::", "host tables only, never per element"),
    ("F32x4", "SIMD helper fuses only on some targets"),
    ("F32x8", "SIMD helper fuses only on some targets"),
    ("vmlaq_f32", "unfused despite the name"),
    ("vmlsq_f32", "unfused despite the name"),
    ("vrecpe", "approximate"),
    ("vrecps", "approximate"),
    ("vrsqrte", "approximate"),
    ("vrsqrts", "approximate"),
    ("_fast(", "fast-math intrinsic"),
    ("_algebraic(", "fast-math intrinsic"),
    ("std::", "the core is no_std"),
    ("alloc::", "the core has no allocator"),
    ("unsafe", "the core is safe code"),
];

/// The file without `//` comments (doc comments list the forbidden names).
fn code() -> String {
    CORE.lines()
        .map(|l| l.split("//").next().unwrap_or(""))
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn tsdf_core_uses_no_forbidden_float_op() {
    let code = code();
    let hits: Vec<String> = FORBIDDEN
        .iter()
        .filter(|(tok, _)| code.contains(tok))
        .map(|(tok, why)| format!("`{tok}`: {why}"))
        .collect();
    assert!(hits.is_empty(), "tsdf_core.rs uses:\n{}", hits.join("\n"));
}

#[test]
fn tsdf_core_scan_sees_a_planted_violation() {
    // the scan is not vacuous: the same rule flags a planted `max`
    let planted = format!("{}\nfn bad(a: f32, b: f32) -> f32 {{ a.max(b) }}", code());
    assert!(FORBIDDEN.iter().any(|(tok, _)| planted.contains(tok)));
    assert!(code().contains("F32Ops"), "the scan reads the right file");
}
