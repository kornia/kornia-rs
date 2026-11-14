//! PTX gate for the cuda-oxide kernels. Shared by `build.rs` and
//! `tests/ptx_gate.rs`.

/// PTX target: JITs on any Turing-or-newer GPU.
pub const ARCH: &str = "sm_75";
/// PTX ISA: a newer one fails to JIT on older drivers (CUDA 12.6 on JetPack 6).
pub const PTX_VERSION: &str = "6.3";
/// Entries allowed to emit `fma`: only kernels whose CPU twin uses an explicit
/// `mul_add`/`fmaf` (the TSDF core's `F32Ops::fma`, contract G4). Everything
/// else must be contraction-free for byte parity.
pub const FMA_ALLOWED: &[&str] = &[
    "tsdf_integrate_pinhole",
    "tsdf_integrate_spherical",
    "tsdf_mc_emit",
    "tsdf_range_from_points",
    "tsdf_raycast",
    "tsdf_touch_pinhole",
    "tsdf_touch_spherical",
];

/// Gate the module and every function in it.
///
/// Module: `.version` is `PTX_VERSION`, `.target` is `ARCH`, no `.extern` other than `.extern .shared`
/// (dynamic shared memory; anything else is a link-time symbol such as
/// libdevice), and the `.entry` set equals `scanned`. Per `.entry` and `.func`:
/// every `ld`/`ldu`/`st`/`atom`/`red` names a state space (generic access means
/// address-space inference was lost), no `__nv_` (libdevice) symbol anywhere,
/// and no `fma` or float `mad` unless it is an entry listed in `FMA_ALLOWED`
/// (CPU byte parity). Float rules (kornia-3d TSDF contract, section 18):
/// - G1: every float `add`/`sub`/`mul` has a rounding modifier (a bare one
///   is fusable by ptxas or the driver JIT);
/// - G2: no `.ftz`, `.approx`, `div.full`, `rsqrt`/`sin`/`cos`/`ex2`/`lg2`/
///   `tanh`, float `min`/`max`, or `.sat` on a float op (`rcp.rn` is IEEE);
/// - G3: no float `atom`/`red`;
/// - G4: every `fma` (and float `mad`) of an `FMA_ALLOWED` entry sits between the inline-asm
///   markers, i.e. comes from an explicit `fma.rn` (the shared core's
///   `F32Ops`), never from contraction.
///
/// # Arguments
///
/// * `ptx` - The PTX module text.
/// * `scanned` - The sorted `#[kernel]` names found in the kernel sources.
///
/// # Errors
///
/// Returns one message per violation.
pub fn gate(ptx: &str, scanned: &[String]) -> Result<(), Vec<String>> {
    let mut errs = Vec::new();
    if !ptx.lines().any(|l| l.trim() == format!(".target {ARCH}")) {
        errs.push(format!("gate: module: .target is not {ARCH}"));
    }
    if !ptx
        .lines()
        .any(|l| l.trim() == format!(".version {PTX_VERSION}"))
    {
        errs.push(format!("gate: module: .version is not {PTX_VERSION}"));
    }
    // (is_entry, name); `None` until the first function header.
    let mut func: Option<(bool, String)> = None;
    let mut names = Vec::new();
    // inside a `// begin inline asm` .. `// end inline asm` pair
    let mut in_asm = false;
    for line in ptx.lines() {
        match line.trim() {
            "// begin inline asm" => in_asm = true,
            "// end inline asm" => in_asm = false,
            _ => {}
        }
        let code = line.split("//").next().unwrap_or("").trim();
        if code.starts_with(".extern") {
            if code.split_whitespace().nth(1) != Some(".shared") {
                errs.push(format!("gate: module: external symbol `{code}`"));
            }
            continue;
        }
        if let Some((is_entry, name)) = header(code) {
            if name.is_empty() || name.starts_with(|c: char| c.is_ascii_digit()) {
                errs.push(format!("gate: function name `{name}` is not an identifier"));
            }
            if name.starts_with("__nv_") {
                errs.push(format!("gate: module: libdevice function `{name}`"));
            }
            if is_entry {
                names.push(name.clone());
            }
            func = Some((is_entry, name));
            continue;
        }
        let Some((is_entry, name)) = &func else {
            continue;
        };
        // Checked on every body line: a multi-line `call` puts the callee on a
        // line of its own.
        if code.contains("__nv_") {
            errs.push(format!("gate: {name}: libdevice symbol `{code}`"));
        }
        let mut toks = code.split_whitespace();
        let Some(op) = (match toks.next() {
            Some(t) if t.starts_with('@') => toks.next(),
            t => t,
        }) else {
            continue;
        };
        const MEM_OPS: &[&str] = &["ld", "ldu", "st", "atom", "red"];
        const SPACES: &[&str] = &["global", "param", "shared", "local", "const"];
        let mut parts = op.split('.');
        // The space may follow other qualifiers (`ld.volatile.global`,
        // `ld.relaxed.gpu.global`).
        if parts.next().is_some_and(|o| MEM_OPS.contains(&o))
            && op.contains('.')
            && !parts.any(|q| SPACES.contains(&q))
        {
            errs.push(format!("gate: {name}: generic memory op `{op}`"));
        }
        let float = is_float(op);
        let float_mad = op.starts_with("mad.") && float;
        let fma_ok = *is_entry && FMA_ALLOWED.contains(&name.as_str());
        if (op.starts_with("fma.") || float_mad) && !fma_ok {
            errs.push(format!(
                "gate: {name}: contracted `{op}` (not an entry in FMA_ALLOWED)"
            ));
        } else if (op.starts_with("fma.") || float_mad) && !in_asm {
            errs.push(format!(
                "gate: {name}: `{op}` outside inline asm (contracted, not an explicit fma.rn)"
            ));
        }
        if let Some(why) = float_violation(op, float) {
            errs.push(format!("gate: {name}: `{op}` ({why})"));
        }
    }
    names.sort();
    if names != scanned {
        errs.push(format!(
            "gate: module: PTX entries {names:?} != #[kernel] items {scanned:?}"
        ));
    }
    if errs.is_empty() {
        Ok(())
    } else {
        Err(errs)
    }
}

/// Whether opcode `op` works on floats.
fn is_float(op: &str) -> bool {
    op.split('.')
        .skip(1)
        .any(|q| matches!(q, "f16" | "f16x2" | "bf16" | "bf16x2" | "f32" | "f64"))
}

/// Why float rules G1-G3 reject opcode `op`, if they do.
fn float_violation(op: &str, float: bool) -> Option<&'static str> {
    let mut parts = op.split('.');
    let base = parts.next().unwrap_or("");
    let quals: Vec<&str> = parts.collect();
    let has = |q: &str| quals.contains(&q);
    if has("ftz") {
        return Some("G2: flushes subnormals");
    }
    if has("approx") || (base == "div" && has("full")) {
        return Some("G2: approximate");
    }
    if matches!(base, "rsqrt" | "sin" | "cos" | "ex2" | "lg2" | "tanh") {
        return Some("G2: approximate transcendental");
    }
    if !float {
        return None;
    }
    if matches!(base, "min" | "max") {
        return Some("G2: float min/max differ from the CPU on +-0 and NaN");
    }
    if has("sat") {
        return Some("G2: saturating float op");
    }
    if matches!(base, "atom" | "red") {
        return Some("G3: float atomic");
    }
    let rounded = ["rn", "rz", "rm", "rp"].iter().any(|r| has(r));
    if matches!(base, "add" | "sub" | "mul") && !rounded {
        return Some("G1: no rounding modifier, fusable by ptxas");
    }
    None
}

/// `(is_entry, name)` if `code` opens an `.entry` or `.func`.
fn header(code: &str) -> Option<(bool, String)> {
    let mut toks = code.split_whitespace();
    let (is_entry, rest) = loop {
        match toks.next()? {
            ".visible" | ".weak" => continue,
            ".entry" => break (true, code.split_once(".entry")?.1),
            ".func" => break (false, code.split_once(".func")?.1),
            _ => return None,
        }
    };
    let mut rest = rest.trim_start();
    // `.func (.param .b32 ret) name(`: skip the return parameter list.
    if rest.starts_with('(') {
        rest = rest.split_once(')').map_or("", |(_, r)| r).trim_start();
    }
    let name = rest
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '$' | '.'))
        .collect();
    Some((is_entry, name))
}
