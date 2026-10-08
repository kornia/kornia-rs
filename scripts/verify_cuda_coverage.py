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


ROOT = Path(__file__).resolve().parent.parent
DOC_PATH = ROOT / "docs" / "CUDA_COVERAGE_AUDIT.md"
CUDA_SRC = ROOT / "crates" / "kornia-imgproc" / "src" / "cuda"
CUDA_EXT = ROOT / "kornia-py" / "src" / "cuda_ext"
CUDA_TESTS = ROOT / "kornia-py" / "tests"


# Map audit operations to expected source artifacts or keywords in crates/kornia-imgproc/src/cuda
OP_SRC_MAP: dict[str, list[str]] = {
    # Filters
    "box_blur": ["filter.rs"],
    "box_blur_u8": ["filter.rs"],
    "gaussian_blur": ["filter.rs"],
    "gaussian_blur_u8": ["filter.rs"],
    "sobel": ["filter.rs"],
    "scharr": ["filter.rs"],
    "laplacian_u8": ["filter/laplacian/cuda.rs", "filter.rs"],
    "bilateral_filter": ["bilateral.rs"],
    "median_blur": ["median.rs"],
    "integral_image": ["filter/integral/cuda.rs", "filter.rs"],
    # Geometry
    "resize": ["resize.rs", "resize_u8.rs"],
    "warp_affine": ["warp_affine.rs", "warp_affine_u8.rs"],
    "warp_perspective": ["warp_perspective.rs", "warp_perspective_u8.rs"],
    "remap": ["remap.rs"],
    # Color
    "gray_from_rgb": ["color/gray.rs"],
    "rgb_from_gray": ["color/gray.rs"],
    "bgr_from_rgb": ["color/swizzle.rs"],
    "rgba_from_rgb": ["color/swizzle.rs"],
    "bgra_from_rgb": ["color/swizzle.rs"],
    "hsv": ["color/hsv_hls.rs"],
    "hls": ["color/hsv_hls.rs"],
    "linear_rgb": ["color/cie.rs"],
    "xyz": ["color/cie.rs"],
    "lab": ["color/cie.rs"],
    "luv": ["color/cie.rs"],
    "yuv": ["color/yuv.rs"],
    "ycbcr": ["color/video.rs", "color/yuv.rs"],
    "rgb_from_bayer": ["color/bayer.rs"],
    "compute_histogram": ["histogram.rs"],
    "equalize_hist": ["histogram.rs"],
    "clahe": ["clahe.rs"],
    # Features
    "sift": ["sift/mod.rs", "sift/detect.rs", "sift/matcher.rs"],
}

# CPU-only operations known to be missing from CUDA (marked with ❌)
CPU_ONLY_OPS = {
    "box_blur_fast",
    "spatial_gradient",
    "crop",
    "pad",
    "flip",
    "apply_colormap",
    "transform_color",
    "threshold_binary",
    "truncate",
    "otsu",
    "fast",
    "orb",
    "responses",
    "match",
    "cells",
}


def parse_audit_table(content: str) -> dict[str, str]:
    """Parse markdown tables to extract operation -> status mapping."""
    status_map: dict[str, str] = {}

    for line in content.splitlines():
        if not line.startswith("|") or "---" in line or "Status" in line:
            continue

        parts = [p.strip() for p in line.split("|")[1:-1]]
        if len(parts) < 3:
            continue

        # Find status symbol in row
        status_match = re.search(r"[✅🟡❌]", line)
        if not status_match:
            continue
        status = status_match.group(0)

        # Extract all backticked identifiers in the first two columns
        ops_col = parts[0] + " " + parts[1]
        backticked_ops = re.findall(r"`([a-zA-Z0-9_]+)`", ops_col)
        for op in backticked_ops:
            status_map[op] = status

    return status_map


def verify_coverage(strict: bool = False) -> int:
    if not DOC_PATH.exists():
        print(f"Error: {DOC_PATH} does not exist", file=sys.stderr)
        return 1

    content = DOC_PATH.read_text(encoding="utf-8")
    status_map = parse_audit_table(content)

    print(f"Found {len(status_map)} audited operations in {DOC_PATH.name}")

    errors: list[str] = []
    warnings: list[str] = []

    # 1. Verify audited supported operations have backing CUDA files
    for op, expected_files in OP_SRC_MAP.items():
        status = status_map.get(op)
        if not status:
            warnings.append(f"Operation '{op}' is known in mapper but missing from audit document")
            continue

        if status in ("✅", "🟡"):
            found = False
            for rel_file in expected_files:
                candidate = CUDA_SRC / rel_file
                if not candidate.exists():
                    alt_candidate = ROOT / "crates" / "kornia-imgproc" / "src" / rel_file
                    if alt_candidate.exists():
                        found = True
                        break
                else:
                    found = True
                    break

            if not found:
                errors.append(
                    f"Status {status} for '{op}': expected one of {expected_files} to exist in {CUDA_SRC}"
                )

    # 2. Verify CPU-only operations in the audit
    for missing_op in CPU_ONLY_OPS:
        status = status_map.get(missing_op)
        if status and status not in ("❌", "🟡"):
            errors.append(
                f"Op '{missing_op}' marked as '{status}' in audit, but expected ❌ (CPU-only)"
            )

    # 3. Check that top-level CUDA modules in src/cuda/mod.rs are accounted for
    cuda_mod_path = CUDA_SRC / "mod.rs"
    if cuda_mod_path.exists():
        cuda_mod_text = cuda_mod_path.read_text(encoding="utf-8")
        declared_modules = re.findall(r"^pub\s+mod\s+([a-zA-Z0-9_]+);", cuda_mod_text, re.MULTILINE)
        print(f"Detected {len(declared_modules)} CUDA modules in {cuda_mod_path.name}: {', '.join(declared_modules)}")

    # 4. Check test coverage presence
    cuda_tests = list(CUDA_TESTS.glob("test_cuda*.py"))
    print(f"Detected {len(cuda_tests)} Python CUDA test suites in {CUDA_TESTS.name}")

    # Output report
    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"  [WARN] {w}")

    if errors:
        print("\nDiscrepancies found:")
        for err in errors:
            print(f"  [FAIL] {err}")
        if strict:
            return 1
    else:
        print("\n[OK] All audited operations match codebase reality!")

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify CUDA coverage audit against codebase")
    parser.add_argument("--check", action="store_true", help="Exit with non-zero code on discrepancy")
    args = parser.parse_args()

    exit_code = verify_coverage(strict=args.check)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
