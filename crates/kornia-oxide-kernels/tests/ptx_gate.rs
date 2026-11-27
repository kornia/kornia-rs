//! The cuda-oxide PTX gate (`ptx_gate.rs`, run by build.rs). Needs no GPU or
//! toolchain.

#[path = "../ptx_gate.rs"]
mod ptx_gate;
use ptx_gate::{gate, FMA_ALLOWED, MAXNTID};

const HEAD: &str = ".version 6.3\n.target sm_75\n.address_size 64\n";

fn entry(name: &str, body: &str) -> String {
    bounded_entry(name, ".maxntid 256, 1, 1\n", body)
}

/// An entry whose header ends with `bounds` (directive lines).
fn bounded_entry(name: &str, bounds: &str, body: &str) -> String {
    format!(".visible .entry {name}(\n\t.param .u64 .ptr .align 1 {name}_param_0\n)\n{bounds}{{\n{body}\tret;\n}}\n")
}

fn helper(body: &str) -> String {
    format!(".func  (.param .b32 func_retval0) _RNvCs0_4kern6helper(\n\t.param .b32 p0\n)\n{{\n{body}\tret;\n}}\n")
}

fn k() -> Vec<String> {
    vec!["k".to_string()]
}

#[test]
fn clean_module_passes() {
    let body = "\tld.param.b64 %rd1, [k_param_0];\n\tld.global.b32 %r1, [%rd1];\n\tmad.lo.s32 %r2, %r1, 3, %r1;\n\tst.global.b32 [%rd1], %r2;\n";
    let ptx = format!(
        "{HEAD}{}{}",
        helper("\tadd.s32 %r1, %r1, 1;\n"),
        entry("k", body)
    );
    assert_eq!(gate(&ptx, &k()), Ok(()));
}

#[test]
fn entry_violations() {
    for op in [
        "fma.rn.f32 %f1, %f2, %f3, %f4;",
        "mad.rn.f32 %f1, %f2, %f3, %f4;",
        "ld.u32 %r1, [%rd1];",
        "@%p1 st.b8 [%rd1], %rs1;",
        "call.uni __nv_expf, (%f1);",
        "ldu.u32 %r1, [%rd1];",
        "atom.add.u32 %r1, [%rd1], 1;",
        "red.add.u32 [%rd1], 1;",
        "ld.volatile.u8 %rs1, [%rd1];",
    ] {
        let ptx = format!("{HEAD}{}", entry("k", &format!("\t{op}\n")));
        assert!(gate(&ptx, &k()).is_err(), "`{op}` passed the gate");
    }
}

/// Legitimate spellings must pass: a gate that rejects them blocks the next
/// kernel with a misleading error.
#[test]
fn legitimate_ops_pass() {
    for op in [
        "ld.volatile.global.u8 %rs1, [%rd1];",
        "ld.relaxed.gpu.global.u32 %r1, [%rd1];",
        "ld.global.nc.u32 %r1, [%rd1];",
        "atom.global.add.u32 %r1, [%rd1], 1;",
        "red.shared.add.u32 [%rd1], 1;",
        "mad.wide.u32 %rd2, %r1, 4, %rd1;",
    ] {
        let ptx = format!("{HEAD}{}", entry("k", &format!("\t{op}\n")));
        assert_eq!(gate(&ptx, &k()), Ok(()), "`{op}`");
    }
    let dyn_smem = format!(
        "{HEAD}.extern .shared .align 16 .b8 smem[];\n{}",
        entry("k", "")
    );
    assert_eq!(gate(&dyn_smem, &k()), Ok(()));
}

/// libdevice is caught on a multi-line call and when it is linked in as a
/// defined `.func` (no `.extern` then).
#[test]
fn libdevice_violations() {
    let multi_line = entry("k", "\tcall.uni (retval0),\n\t__nv_expf,\n\t(param0);\n");
    let defined = format!(
        "{HEAD}.func  (.param .b32 r) __nv_expf(\n\t.param .b32 x\n)\n{{\n\tret;\n}}\n{}",
        entry("k", "")
    );
    for ptx in [format!("{HEAD}{multi_line}"), defined] {
        let errs = gate(&ptx, &k()).expect_err(&ptx);
        assert!(errs.iter().any(|e| e.contains("libdevice")), "{errs:?}");
    }
}

/// A non-inlined helper is gated wherever LLVM places it.
#[test]
fn helper_violations() {
    for op in ["fma.rn.f32 %f1, %f2, %f3, %f4;", "ld.u32 %r1, [%rd1];"] {
        let h = helper(&format!("\t{op}\n"));
        for ptx in [
            format!("{HEAD}{h}{}", entry("k", "")),
            format!("{HEAD}{}{h}", entry("k", "")),
        ] {
            let errs = gate(&ptx, &k()).expect_err(op);
            assert!(errs.iter().any(|e| e.contains("helper")), "{errs:?}");
        }
    }
}

#[test]
fn module_violations() {
    // Wrong target or ISA, extern symbol, entry set mismatch.
    let wrong_target = format!("{}{}", HEAD.replace("sm_75", "sm_80"), entry("k", ""));
    let wrong_version = format!("{}{}", HEAD.replace("6.3", "8.0"), entry("k", ""));
    let extern_sym = format!(
        "{HEAD}.extern .func __nv_sinf(.param .b32 x);\n{}",
        entry("k", "")
    );
    let extra_entry = format!("{HEAD}{}{}", entry("k", ""), entry("j", ""));
    let commented = format!("{HEAD}// .visible .entry j(\n{}", entry("k", ""));
    for ptx in [&wrong_target, &wrong_version, &extern_sym, &extra_entry] {
        assert!(gate(ptx, &k()).is_err(), "{ptx}");
    }
    assert_eq!(gate(&commented, &k()), Ok(()));
    assert!(gate(&format!("{HEAD}{}", entry("k", "")), &[]).is_err());
}

fn asm(op: &str) -> String {
    format!("\t// begin inline asm\n\t{op}\n\t// end inline asm\n")
}

/// Float rules G1-G3 (kornia-3d TSDF contract, section 18): each rejected
/// opcode fails in an entry and in a helper, even inside inline asm.
#[test]
fn float_violations() {
    for op in [
        // G1: no rounding modifier
        "add.f32 %f1, %f2, %f3;",
        "sub.f32 %f1, %f2, %f3;",
        "mul.f32 %f1, %f2, %f3;",
        "add.f64 %fd1, %fd2, %fd3;",
        "mul.f64 %fd1, %fd2, %fd3;",
        // G2: flush, approximations, transcendentals, min/max, saturation
        "add.rn.ftz.f32 %f1, %f2, %f3;",
        "mul.rn.ftz.f32 %f1, %f2, %f3;",
        "div.approx.f32 %f1, %f2, %f3;",
        "div.full.f32 %f1, %f2, %f3;",
        "rcp.approx.f32 %f1, %f2;",
        "rcp.approx.ftz.f64 %fd1, %fd2;",
        "sqrt.approx.f32 %f1, %f2;",
        "rsqrt.approx.f32 %f1, %f2;",
        "sin.approx.f32 %f1, %f2;",
        "cos.approx.f32 %f1, %f2;",
        "ex2.approx.f32 %f1, %f2;",
        "lg2.approx.f32 %f1, %f2;",
        "tanh.approx.f32 %f1, %f2;",
        "min.f32 %f1, %f2, %f3;",
        "max.f32 %f1, %f2, %f3;",
        "max.NaN.f32 %f1, %f2, %f3;",
        "min.f64 %fd1, %fd2, %fd3;",
        "add.rn.sat.f32 %f1, %f2, %f3;",
        "cvt.sat.f32.f32 %f1, %f2;",
        // G3: float atomics
        "atom.global.add.f32 %f1, [%rd1], %f2;",
        "red.global.add.f32 [%rd1], %f2;",
        "atom.global.add.f64 %fd1, [%rd1], %fd2;",
    ] {
        for ptx in [
            format!("{HEAD}{}", entry("k", &format!("\t{op}\n"))),
            format!("{HEAD}{}", entry("k", &asm(op))),
            format!("{HEAD}{}{}", helper(&format!("\t{op}\n")), entry("k", "")),
        ] {
            assert!(gate(&ptx, &k()).is_err(), "`{op}` passed the gate:\n{ptx}");
        }
    }
}

/// IEEE float ops and integer look-alikes pass (`rcp.rn` is how LLVM
/// lowers `1.0 / x`).
#[test]
fn float_ops_pass() {
    for op in [
        "add.rn.f32 %f1, %f2, %f3;",
        "sub.rn.f32 %f1, %f2, %f3;",
        "mul.rn.f32 %f1, %f2, %f3;",
        "add.rz.f64 %fd1, %fd2, %fd3;",
        "div.rn.f32 %f1, %f2, %f3;",
        "rcp.rn.f32 %f1, %f2;",
        "sqrt.rn.f32 %f1, %f2;",
        "neg.f32 %f1, %f2;",
        "abs.f32 %f1, %f2;",
        "cvt.rzi.s32.f32 %r1, %f1;",
        "cvt.rmi.f32.f32 %f1, %f2;",
        "setp.lt.f32 %p1, %f1, %f2;",
        "selp.f32 %f1, %f2, %f3, %p1;",
        "min.u32 %r1, %r2, %r3;",
        "max.s32 %r1, %r2, %r3;",
        "add.s32 %r1, %r2, %r3;",
        "mul.lo.s32 %r1, %r2, %r3;",
        "mul.wide.u32 %rd1, %r2, %r3;",
        "atom.global.add.u64 %rd1, [%rd2], 1;",
        "atom.global.cas.b64 %rd1, [%rd2], %rd3, %rd4;",
        "red.global.or.b64 [%rd1], %rd2;",
        "ld.global.f32 %f1, [%rd1];",
    ] {
        let ptx = format!("{HEAD}{}", entry("k", &format!("\t{op}\n")));
        assert_eq!(gate(&ptx, &k()), Ok(()), "`{op}`");
    }
}

/// G4: an allowed entry's fma must come from inline asm (an explicit
/// `fma.rn`); a contracted one, or one in a helper, fails.
#[test]
fn fma_only_from_inline_asm() {
    let fma = "fma.rn.f32 %f1, %f2, %f3, %f4;";
    let name = FMA_ALLOWED[0];
    let names = vec![name.to_string()];
    let ok = format!("{HEAD}{}", entry(name, &asm(fma)));
    assert_eq!(gate(&ok, &names), Ok(()));
    let bare = format!("{HEAD}{}", entry(name, &format!("\t{fma}\n")));
    let errs = gate(&bare, &names).expect_err("bare fma passed");
    assert!(errs.iter().any(|e| e.contains("inline asm")), "{errs:?}");
    // the markers are scoped: an fma after `end inline asm` is bare
    let after = format!(
        "{HEAD}{}",
        entry(name, &format!("{}\t{fma}\n", asm("mov.f32 %f1, %f2;")))
    );
    assert!(gate(&after, &names).is_err());
    // a float `mad` is the same fused op: bare in an allowed entry fails
    let mad = "mad.rn.f32 %f1, %f2, %f3, %f4;";
    let bare_mad = format!("{HEAD}{}", entry(name, &format!("\t{mad}\n")));
    let errs = gate(&bare_mad, &names).expect_err("bare float mad passed");
    assert!(errs.iter().any(|e| e.contains("inline asm")), "{errs:?}");
    // an integer `mad` is not an fma
    let int_mad = "mad.lo.s32 %r1, %r2, 3, %r2;";
    let ok_int = format!("{HEAD}{}", entry(name, &format!("\t{int_mad}\n")));
    assert_eq!(gate(&ok_int, &names), Ok(()));
    // never in a helper, even from asm
    let helper_asm = format!("{HEAD}{}{}", helper(&asm(fma)), entry(name, ""));
    assert!(gate(&helper_asm, &names).is_err());
}

/// The contract names exactly these entries (G4).
#[test]
fn fma_allowed_is_the_contract_set() {
    let mut got = FMA_ALLOWED.to_vec();
    got.sort_unstable();
    assert_eq!(
        got,
        [
            "tsdf_integrate_pinhole",
            "tsdf_integrate_spherical",
            "tsdf_mc_emit",
            "tsdf_range_from_points",
            "tsdf_raycast",
            "tsdf_touch_pinhole",
            "tsdf_touch_spherical",
        ]
    );
}

/// A fixed-block (`MAXNTID`) entry must declare exactly that `.maxntid`;
/// other entries need none.
#[test]
fn maxntid_required_on_fixed_block_entries() {
    let (prefix, want) = MAXNTID[0];
    let name = format!("{prefix}k");
    let names = vec![name.clone()];
    let ok = format!("{HEAD}{}", entry(&name, ""));
    assert_eq!(gate(&ok, &names), Ok(()));
    let commented = format!(".maxntid {want}                      // @{name}\n");
    let ok_commented = format!("{HEAD}{}", bounded_entry(&name, &commented, ""));
    assert_eq!(gate(&ok_commented, &names), Ok(()));
    for bounds in ["", ".maxntid 1024, 1, 1\n", ".reqntid 256, 1, 1\n"] {
        let ptx = format!("{HEAD}{}", bounded_entry(&name, bounds, ""));
        let errs = gate(&ptx, &names).expect_err(bounds);
        assert!(errs.iter().any(|e| e.contains(".maxntid")), "{errs:?}");
    }
    // a `.maxntid` in a helper does not count for the entry after it
    let h = format!(".func _RNvCs0_4kern6helper()\n.maxntid {want}\n{{\n\tret;\n}}\n");
    let ptx = format!("{HEAD}{h}{}", bounded_entry(&name, "", ""));
    assert!(gate(&ptx, &names).is_err());
    let free = format!("{HEAD}{}", bounded_entry("k", "", ""));
    assert_eq!(gate(&free, &k()), Ok(()));
}
