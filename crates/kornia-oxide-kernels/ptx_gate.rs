//! PTX gate for the cuda-oxide kernels. Shared by `build.rs` and
//! `tests/ptx_gate.rs`.

/// PTX target: JITs on any Turing-or-newer GPU.
pub const ARCH: &str = "sm_75";
/// PTX ISA: a newer one fails to JIT on older drivers (CUDA 12.6 on JetPack 6).
pub const PTX_VERSION: &str = "6.3";
/// Entries allowed to emit `fma`: only kernels whose CPU twin uses an explicit
/// `mul_add`/`fmaf`. Everything else must be contraction-free for byte parity.
pub const FMA_ALLOWED: &[&str] = &[];

/// Gate the module and every function in it.
///
/// Module: `.version` is `PTX_VERSION`, `.target` is `ARCH`, no `.extern` other than `.extern .shared`
/// (dynamic shared memory; anything else is a link-time symbol such as
/// libdevice), and the `.entry` set equals `scanned`. Per `.entry` and `.func`:
/// every `ld`/`ldu`/`st`/`atom`/`red` names a state space (generic access means
/// address-space inference was lost), no `__nv_` (libdevice) symbol anywhere,
/// and no `fma` or float `mad` unless it is an entry listed in `FMA_ALLOWED`
/// (CPU byte parity).
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
    for line in ptx.lines() {
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
        let float_mad = op.starts_with("mad.") && (op.contains(".f32") || op.contains(".f64"));
        let fma_ok = *is_entry && FMA_ALLOWED.contains(&name.as_str());
        if (op.starts_with("fma.") || float_mad) && !fma_ok {
            errs.push(format!(
                "gate: {name}: contracted `{op}` (not an entry in FMA_ALLOWED)"
            ));
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
