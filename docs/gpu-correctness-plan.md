# GPU Correctness Plan

Status: Draft working plan for Phase 1a CUDA correctness validation.

This is one workstream within Phase 1. It covers correctness tooling and
external-reference validation. It does not yet cover GPU-kernel execution in
CI or contributor documentation for the `MemoryDomain` and allocator model.

## Goal

Validate GPU operations at three levels where the semantics are comparable:

1. Kornia CPU output.
2. Kornia CUDA output.
3. An independent implementation such as OpenCV, NVIDIA VPI, or PyTorch.

The CPU and CUDA paths must agree for every supported GPU operation. An
independent-library comparison is added when that library implements the same
operation, interpolation, border behavior, coordinate convention, dtype, and
rounding contract.

## Current Coverage

### Kornia CPU versus CUDA

Rust tests already cover substantial CUDA parity for color conversion, filters,
morphology, resize, remap, warps, connected components, CLAHE, histogram, and
SIFT-related paths. These tests should remain next to the implementation and
should compare exact bytes or exact floating-point bit patterns whenever the
kernel is designed for that contract.

The public `remap_u8` path already has CPU/CUDA parity tests covering:

- bilinear and nearest interpolation;
- fractional and out-of-bounds maps;
- multiple image sizes; and
- one-, three-, and four-channel images.

### CUDA versus independent libraries

The current external Python checks are narrower:

- `examples/check_correctness_cuda.py` checks CUDA resize and warp-affine
  output against OpenCV;
- `examples/check_correctness_vpi.py` checks selected resize, warp-affine, and
  warp-perspective cases against NVIDIA VPI;
- other operations currently rely on internal Rust CPU/CUDA parity tests.

The external checks currently run manually on a machine with the required
CUDA, OpenCV, or VPI dependencies. Standard CI compiles and lints CUDA-gated
code but does not execute kernels. Making these checks automated, for example
in a scheduled or hardware-specific job, is a follow-up scope item rather
than an assumption of this Phase 1a workstream.

## Oracle Rules

Each operation must declare its comparison contract before adding an external
check:

| Contract | Required assertion |
| --- | --- |
| Byte-exact | CPU, CUDA, and external output are identical byte-for-byte |
| Bit-exact float | CPU and CUDA have identical `to_bits()` values; external output follows the documented ULP bound |
| Numeric tolerance | Report max/mean error and enforce a documented tolerance |
| Semantic equivalence | Compare a suitable invariant, such as feature matches or labels, rather than array order |
| Report-only | Use when the external algorithm intentionally differs; do not treat it as a pass/fail oracle |

Border modes, interpolation kernels, map representation, channel layout,
rounding, and library versions must be recorded for every external comparison.
OpenCV and VPI results must not be called byte-exact unless those parameters
are known to match the Kornia contract.

## Work Items

### 1. Keep CPU/CUDA parity complete

- Audit each public CUDA-dispatched operation for a public API parity test.
- Prefer exact byte or bit comparisons where the implementation is designed
  to mirror the CPU path.
- Add regression cases for dimensions, channel counts, borders, fractional
  coordinates, and unsupported dtype or residency combinations.

### 2. Extend the existing OpenCV checker

Do not create one Python checker per operation. Extend the existing external
correctness tooling with shared input generation, output reporting, and
comparison helpers.

The initial operation order is:

1. Morphology, where u8 min/max and border semantics can be made byte-exact.
2. Median filtering, where the existing implementation already documents
   OpenCV compatibility.
3. Color conversion, for conversions with established OpenCV formulas.
4. Remap, beginning with nearest-neighbor and then bilinear after measuring
   interpolation coefficient and border differences.
5. Warp perspective, using the same contract documentation as warp affine.
6. Remaining filters and feature operations where OpenCV semantics are
   comparable.

### 3. Extend optional VPI checks

Add VPI comparisons only for operations supported by the installed VPI
version. Keep these checks optional because VPI is normally available on Jetson
or specially provisioned NVIDIA systems, not standard CI runners.

### 4. Make results reproducible

Every external check should report:

- operation and mode;
- input dimensions, dtype, and channel count;
- border and interpolation settings;
- reference library and version;
- maximum and mean error;
- mismatch count; and
- accepted tolerance or exactness contract.

## First Implementation Slice

Begin with the first operation in the ordered list above: morphology. It has a
clear u8 min/max contract and existing CUDA dispatch, making it a useful
template for shared input generation, output reporting, and byte-exact OpenCV
comparison. Continue through the same ordered list; add remap to the shared
framework rather than creating a standalone checker.

The existing `remap_u8` CPU/CUDA tests are the internal parity foundation. When
the shared checker reaches remap, first establish whether OpenCV's nearest and
bilinear coordinate and coefficient rules match exactly before enforcing a
byte-for-byte assertion.

## Progress Log

- 2026-09-10: Confirmed broad in-process CPU/CUDA coverage.
- 2026-09-10: Confirmed external Python coverage is currently limited to
  selected resize and warp operations.
- 2026-09-10: Rejected a separate `check_correctness_remap_u8.py`; future
  external checks should extend shared tooling.
- 2026-09-10: Added morphology to the shared CUDA checker. 3x3 u8 dilation
  and erosion are byte-identical across kornia CPU, kornia CUDA, and OpenCV
  with `BORDER_REPLICATE`.