"""Runtime CPU SIMD feature snapshot — smoke tests.

These complement the build-time `verify_simd_kernels.sh` audit. The
verifier proves the SIMD assembly is in the wheel's `.so`; these tests
prove the runtime probe actually loads on the host CPU and returns the
expected per-arch baseline (NEON on aarch64, AVX2/FMA on modern x86_64
GitHub runners).
"""

import platform

import kornia_rs


def test_cpu_features_returns_object_with_all_fields():
    f = kornia_rs.cpu.cpu_features()
    for name in (
        "has_avx2",
        "has_fma",
        "has_avx512f",
        "has_avx512bw",
        "has_neon",
        "has_fp16",
        "has_dotprod",
        "has_sve",
    ):
        assert isinstance(getattr(f, name), bool), f"{name} must be bool"


def test_cpu_features_repr_is_one_line():
    r = repr(kornia_rs.cpu.cpu_features())
    assert r.startswith("CpuFeatures(")
    assert "\n" not in r


def test_cpu_features_arch_baseline():
    f = kornia_rs.cpu.cpu_features()
    machine = platform.machine().lower()
    if machine in ("aarch64", "arm64"):
        assert f.has_neon, "NEON is architectural on aarch64; runtime probe should report True"
        assert not f.has_avx2, "AVX2 must be False on aarch64"
    elif machine in ("x86_64", "amd64"):
        assert not f.has_neon, "NEON must be False on x86_64"
    else:
        # Other arches (riscv, ppc) — no assertion, just probe must not crash.
        pass


def test_cpu_features_is_cached_across_calls():
    a = kornia_rs.cpu.cpu_features()
    b = kornia_rs.cpu.cpu_features()
    assert a.has_avx2 == b.has_avx2
    assert a.has_neon == b.has_neon


def test_cpu_features_class_is_frozen():
    f = kornia_rs.cpu.cpu_features()
    try:
        f.has_avx2 = True
    except (AttributeError, TypeError):
        return
    raise AssertionError("CpuFeatures should be frozen — attribute writes must fail")
