//! The cuda-oxide PTX gate (`ptx_gate.rs`, run by build.rs). Needs no GPU or
//! toolchain.

#[path = "../ptx_gate.rs"]
mod ptx_gate;
use ptx_gate::gate;

const HEAD: &str = ".version 6.3\n.target sm_75\n.address_size 64\n";

fn entry(name: &str, body: &str) -> String {
    format!(".visible .entry {name}(\n\t.param .u64 .ptr .align 1 {name}_param_0\n)\n{{\n{body}\tret;\n}}\n")
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
    ] {
        let ptx = format!("{HEAD}{}", entry("k", &format!("\t{op}\n")));
        assert!(gate(&ptx, &k()).is_err(), "`{op}` passed the gate");
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
    // Wrong target, extern symbol, entry set mismatch.
    let wrong_target = format!("{}{}", HEAD.replace("sm_75", "sm_80"), entry("k", ""));
    let extern_sym = format!(
        "{HEAD}.extern .func __nv_sinf(.param .b32 x);\n{}",
        entry("k", "")
    );
    let extra_entry = format!("{HEAD}{}{}", entry("k", ""), entry("j", ""));
    let commented = format!("{HEAD}// .visible .entry j(\n{}", entry("k", ""));
    for ptx in [&wrong_target, &extern_sym, &extra_entry] {
        assert!(gate(ptx, &k()).is_err(), "{ptx}");
    }
    assert_eq!(gate(&commented, &k()), Ok(()));
    assert!(gate(&format!("{HEAD}{}", entry("k", "")), &[]).is_err());
}
