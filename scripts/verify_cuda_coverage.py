#!/usr/bin/env python3
"""verify_cuda_coverage.py — Verify CUDA coverage audit against codebase reality.

Cross-references `docs/CUDA_COVERAGE_AUDIT.md` with:
  1. Rust CUDA implementations in `crates/kornia-imgproc/src/cuda/`
  2. Python bindings in `kornia-py/src/cuda_ext/`
  3. Python CUDA tests in `kornia-py/tests/test_cuda*.py`

Usage:
    python scripts/verify_cuda_coverage.py [--check]

Exits with code 0 on success. If --check is supplied, exits with code 1 on discrepancies.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Ensure UTF-8 output on all platforms (avoid Windows cp1252 encode errors)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent.parent
DOC_PATH = ROOT / "docs" / "CUDA_COVERAGE_AUDIT.md"
CUDA_SRC = ROOT / "crates" / "kornia-imgproc" / "src" / "cuda"
CUDA_EXT = ROOT / "kornia-py" / "src" / "cuda_ext"
CUDA_TESTS = ROOT / "kornia-py" / "tests"


# Map audit operations to expected source artifacts and identification symbols in crates/kornia-imgproc/src/cuda
OP_SRC_MAP: dict[str, tuple[list[str], str]] = {
    # Filters (Chapter 1)
    "box_blur": (["filter.rs"], r"separable_(?:filter|blur)|box"),
    "box_blur_u8": (["filter.rs"], r"separable_blur_u8|box"),
    "box_blur_fast": ([], ""),
    "gaussian_blur": (["filter.rs"], r"gaussian|binomial|separable"),
    "gaussian_blur_u8": (["filter.rs"], r"gaussian|binomial|separable"),
    "sobel": (["filter.rs"], r"sobel|gradient"),
    "scharr": (["filter.rs"], r"scharr|gradient"),
    "spatial_gradient": ([], ""),
    "laplacian_u8": (["filter/laplacian/cuda.rs", "filter.rs"], r"laplacian"),
    "bilateral_filter": (["bilateral.rs"], r"bilateral"),
    "median_blur": (["median.rs"], r"median"),
    "integral_image": (["filter/integral/cuda.rs", "filter.rs"], r"integral"),
    # Geometry (Chapter 2)
    "resize": (["resize.rs", "resize_u8.rs"], r"resize"),
    "warp_affine": (["warp_affine.rs", "warp_affine_u8.rs"], r"warp_affine"),
    "warp_perspective": (["warp_perspective.rs", "warp_perspective_u8.rs"], r"warp_perspective"),
    "remap": (["remap.rs"], r"remap"),
    "crop": ([], ""),
    "pad": ([], ""),
    "flip": ([], ""),
    # Color, Hist & CLAHE (Chapter 3)
    "gray_from_rgb": (["color/gray.rs"], r"(?:launch_)?gray_from_rgb(?:_|$)"),
    "rgb_from_gray": (["color/gray.rs"], r"(?:launch_)?rgb_from_gray(?:_|$)"),
    "bgr_from_rgb": (["color/swizzle.rs"], r"(?:launch_)?bgr_from_rgb(?:_|$)"),
    "rgba_from_rgb": (["color/swizzle.rs"], r"(?:launch_)?rgba_from_rgb(?:_|$)"),
    "bgra_from_rgb": (["color/swizzle.rs"], r"(?:launch_)?bgra_from_rgb(?:_|$)"),
    "hsv": (["color/hsv_hls.rs"], r"hsv"),
    "hls": (["color/hsv_hls.rs"], r"hls"),
    "linear_rgb": (["color/cie.rs"], r"linear_rgb"),
    "xyz": (["color/cie.rs"], r"xyz"),
    "lab": (["color/cie.rs"], r"lab"),
    "luv": (["color/cie.rs"], r"luv"),
    "yuv": (["color/yuv.rs"], r"ycc|yuv"),
    "ycbcr": (["color/video.rs", "color/yuv.rs"], r"ycc|yuyv|nv12|ycbcr"),
    "rgb_from_bayer": (["color/bayer.rs"], r"bayer"),
    "compute_histogram": (["histogram.rs"], r"histogram"),
    "equalize_hist": (["histogram.rs"], r"equalize|histogram"),
    "clahe": (["clahe.rs"], r"clahe"),
    "apply_colormap": (["color/misc.rs"], r"colormap"),
    "transform_color": ([], ""),
    "threshold_binary": ([], ""),
    "truncate": ([], ""),
    "otsu": ([], ""),
    # Features (Chapter 4)
    "sift": (["sift/descriptor.rs", "sift/detect.rs", "sift/matcher.rs"], r"sift"),
    "fast": ([], ""),
    "orb": ([], ""),
    "responses": ([], ""),
    "match": ([], ""),
    "cells": ([], ""),
}

# The complete set of declared CUDA modules in crates/kornia-imgproc/src/cuda/mod.rs
EXPECTED_CUDA_MODULES = {
    "resize",
    "filter",
    "resize_u8",
    "warp_affine",
    "warp_affine_u8",
    "warp_perspective",
    "warp_perspective_u8",
    "remap",
    "color",
    "sift",
    "bilateral",
    "canny",
    "ccl",
    "clahe",
    "histogram",
    "median",
    "morphology",
    "pyramid",
    "fusion",
}


def parse_audit_table(content: str) -> dict[str, str]:
    """Parse markdown tables to extract operation -> status mapping via header column detection."""
    status_map: dict[str, str] = {}
    current_op_col: int | None = None
    current_status_col: int | None = None

    for line in content.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            current_op_col = None
            current_status_col = None
            continue

        # Split on unescaped pipe '|' to preserve escaped pipes in cell content
        raw_parts = re.split(r"(?<!\\)\|", line)[1:-1]
        parts = [p.replace(r"\|", "|").strip() for p in raw_parts]
        if not parts:
            continue

        lower_parts = [p.lower() for p in parts]

        # Detect table header row by matching column titles
        if any("status" in p for p in lower_parts) and (
            any("operation" in p for p in lower_parts) or any("module" in p for p in lower_parts)
        ):
            current_op_col = next(
                (i for i, p in enumerate(lower_parts) if "operation" in p or "module" in p),
                None,
            )
            current_status_col = next(
                (i for i, p in enumerate(lower_parts) if "status" in p),
                None,
            )
            continue

        # Skip markdown table divider row (e.g. |---|:---:|)
        if all(re.match(r"^:?-+:?$", p) for p in parts):
            continue

        # Parse data row using detected header column indices
        if (
            current_op_col is not None
            and current_status_col is not None
            and current_op_col < len(parts)
            and current_status_col < len(parts)
        ):
            status_match = re.search(r"[✅🟡❌]", parts[current_status_col])
            if status_match:
                status = status_match.group(0)
                ops = re.findall(r"`([a-zA-Z0-9_]+)`", parts[current_op_col])
                for op in ops:
                    status_map[op] = status

    return status_map


def find_implemented_cuda_artifacts(op: str) -> list[str]:
    """Heuristically scan crates/kornia-imgproc/src/cuda/ for unexpected implementations of an op."""
    found: list[str] = []

    # 1. Check for filename matches (e.g., flip.rs, flip_u8.rs)
    for pattern in (f"{op}.rs", f"{op}_*.rs", f"*_{op}.rs"):
        for path in CUDA_SRC.glob(f"**/{pattern}"):
            found.append(str(path.relative_to(CUDA_SRC)))

    # 2. Check for public functions/modules including prefixed launchers (e.g. launch_flip, flip_cuda)
    fn_pattern = re.compile(rf"\bpub\s+(?:fn|mod)\s+(?:launch_)?{re.escape(op)}(?:_[a-z0-9_]+)?\b")
    for rs_file in CUDA_SRC.glob("**/*.rs"):
        # For generic 'match', ignore sift/matcher.rs which houses the SIFT-specific matcher
        if op == "match" and "sift" in rs_file.parts:
            continue

        code = rs_file.read_text(encoding="utf-8")
        if fn_pattern.search(code):
            rel = str(rs_file.relative_to(CUDA_SRC))
            if rel not in found:
                found.append(f"{rel}::{op}")

    return found


def verify_coverage(strict: bool = False) -> int:
    if not DOC_PATH.exists():
        print(f"Error: {DOC_PATH} does not exist", file=sys.stderr)
        return 1

    content = DOC_PATH.read_text(encoding="utf-8")
    status_map = parse_audit_table(content)

    print(f"Found {len(status_map)} audited operations in {DOC_PATH.name}")

    errors: list[str] = []

    # 1. Verify every mapped operation exists in the audit document and check implementations
    for op, (expected_files, op_symbol_pattern) in OP_SRC_MAP.items():
        status = status_map.get(op)
        if not status:
            errors.append(f"Operation '{op}' is defined in mapper but missing from audit document")
            continue

        if status in ("✅", "🟡"):
            if not expected_files:
                errors.append(f"Operation '{op}' marked as {status} but has no CUDA kernel mapped")
                continue

            found = False
            for rel_file in expected_files:
                candidate = CUDA_SRC / rel_file
                if not candidate.exists():
                    alt_candidate = ROOT / "crates" / "kornia-imgproc" / "src" / rel_file
                    if alt_candidate.exists():
                        candidate = alt_candidate

                if candidate.exists():
                    # Verify declared CUDA symbol (pub fn / __global__ void) exists in candidate file
                    code = candidate.read_text(encoding="utf-8")
                    declared_symbols = re.findall(
                        r'(?:pub(?:\s*\([^)]*\))?\s+fn|extern\s+"C"\s+__global__\s+void)\s+([a-zA-Z0-9_]+)',
                        code,
                    )
                    if not op_symbol_pattern or any(
                        re.search(op_symbol_pattern, sym, re.IGNORECASE) for sym in declared_symbols
                    ):
                        found = True
                        break

            if not found:
                errors.append(
                    f"Status {status} for '{op}': expected declared symbol matching '{op_symbol_pattern}' in one of {expected_files}"
                )
        elif status == "❌":
            # Assert no expected files mapped
            if expected_files:
                errors.append(
                    f"Operation '{op}' marked as ❌ (missing) in audit, but mapped to implemented files {expected_files}"
                )
            # Scan crates/kornia-imgproc/src/cuda/ to enforce that it is indeed not implemented
            unexpected = find_implemented_cuda_artifacts(op)
            if unexpected:
                errors.append(
                    f"Operation '{op}' marked as ❌ in audit, but unexpected CUDA artifacts were found: {unexpected}"
                )

    # 2. Check that all declared operations in audit are registered in OP_SRC_MAP (PERF401)
    errors.extend(
        f"Audited operation '{audited_op}' is not registered in OP_SRC_MAP"
        for audited_op in status_map
        if audited_op not in OP_SRC_MAP
    )

    # 3. Check that top-level CUDA modules in src/cuda/mod.rs match EXPECTED_CUDA_MODULES and affect --check
    cuda_mod_path = CUDA_SRC / "mod.rs"
    if cuda_mod_path.exists():
        cuda_mod_text = cuda_mod_path.read_text(encoding="utf-8")
        declared_modules = set(re.findall(r"^pub\s+mod\s+([a-zA-Z0-9_]+);", cuda_mod_text, re.MULTILINE))
        print(f"Detected {len(declared_modules)} CUDA modules in {cuda_mod_path.name}")

        missing_modules = EXPECTED_CUDA_MODULES - declared_modules
        if missing_modules:
            errors.append(f"Expected CUDA modules missing from mod.rs: {sorted(missing_modules)}")

        extra_modules = declared_modules - EXPECTED_CUDA_MODULES
        if extra_modules:
            errors.append(f"New undeclared CUDA modules in mod.rs not registered in verifier: {sorted(extra_modules)}")

    # 4. Check test coverage presence
    cuda_tests = list(CUDA_TESTS.glob("test_cuda*.py"))
    print(f"Detected {len(cuda_tests)} Python CUDA test suites in {CUDA_TESTS.name}")

    if errors:
        print(f"\nDiscrepancies found ({len(errors)}):")
        for err in errors:
            print(f"  [FAIL] {err}")
        if strict:
            return 1
    else:
        print("\n[OK] All audited operations, symbols, and declared modules match codebase reality!")

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify CUDA coverage audit against codebase")
    parser.add_argument("--check", action="store_true", help="Exit with non-zero code on discrepancy")
    args = parser.parse_args()

    exit_code = verify_coverage(strict=args.check)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
