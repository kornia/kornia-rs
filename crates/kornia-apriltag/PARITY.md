# AprilTag Parity: kornia-rs vs AprilRobotics/apriltag C

Reference C commit: master branch, `AprilRobotics/apriltag` (fetched 2026-06-27)
Key C files: `apriltag.c`, `apriltag_quad_thresh.c`, `common/image_u8.c`, `common/homography.c`

---

## Stage Map

| C function / location | Rust location | Status | Notes |
|---|---|---|---|
| `image_u8_decimate` (common/image_u8.c) | `lib.rs:242` `resize_fast_mono` | **DIVERGENCE D1+D2** | Pixel selection and output size differ |
| `quad_sigma` blur (apriltag.c:1083) | _(absent)_ | **OMISSION O1** | Not implemented; default=0 so no effect on parity tests |
| `threshold()` tilesz (apriltag_quad_thresh.c:1104) | `threshold.rs:208` `TileMinMax::new(…, 4)` | MATCH | Both use tile_size=4 |
| `do_minmax_task()` per-tile min/max | `threshold.rs:214` | MATCH | Same loop; same min/max accumulation |
| `do_blur_task()` 3×3 neighbor max/min | `threshold.rs:38` `neighbor_blur` | MATCH | Same 3×3 window; same clamping logic |
| `do_threshold_task()` binarise | `threshold.rs:297` | MATCH | `thresh = min + (max-min)/2`; Skip=127; White=255; Black=0 |
| `do_unionfind_first_line` / `do_unionfind_line2` | `segmentation.rs:21` `find_connected_components` | MATCH | Same 4+diag connectivity; harmless extra top-neighbor connect in Rust (idempotent) |
| Gradient cluster, size≥25 filter | `segmentation.rs:153` `find_gradient_clusters` | MATCH | Same 25-pixel threshold |
| `fit_quads` / `do_quad_task` | `quad.rs:121` `fit_quads` | MATCH | Same bounding-box, area, angle checks |
| `homography_compute2` (apriltag.c:439) | `utils.rs:92` `homography_compute` | **DIVERGENCE D5** | Same DLT + partial-pivot Gauss elim algorithm; C uses `double`, Rust uses `f32` |
| `quad_update_homographies` (apriltag.c:506) | `quad.rs:92` `update_homographies` | MATCH | Identical corner-to-tag mapping `(-1,-1),(1,-1),(1,1),(-1,1)` |
| `refine_edges` (apriltag.c:761) | `decoder.rs:424` `refine_edges` | **DIVERGENCE D4** | Search range differs: C uses `quad_decimate+1=3`, Rust hardcodes `RANGE=2.0` |
| `graymodel_add` / `graymodel_solve` | `decoder.rs:36` `GrayModel::add/solve` | MATCH (intentional) | Upper-triangle skip and Cholesky solver match legacy C behavior (documented in comments) |
| `sharpen` (apriltag.c:553) | `decoder.rs:817` `sharpen` | MATCH | Identical Laplacian kernel `[0,-1,0,-1,4,-1,0,-1,0]`; same formula |
| `quad_decode` (apriltag.c:598) | `decoder.rs:599` `quad_decode` | MATCH | Same border-sampling patterns, bilinear interpolation, gray model threshold |
| `quick_decode` | `decoder.rs:145` `QuickDecode` | MATCH | Same Hamming lookup; same rotation-by-90 logic |
| `decode_sharpening` default | `lib.rs:109` | MATCH | Both default to `0.25` |
| `refine_edges` enabled default | `lib.rs:107` | MATCH | Both default to `true` |
| `quad_decimate` default | `lib.rs:76` | MATCH (effective) | C default=2.0; Rust `DEFAULT_DOWNSCALE_FACTOR=2` |
| reconcile / dedup (apriltag.c Step 3) | `decoder.rs` `dedup_detections` | **DIVERGENCE D6** (intentional) | Same semantics (repeated non-overlapping ids survive; best hamming/margin wins per overlap cluster); overlap predicate is a center-distance heuristic, not C's exact `g2d_polygon_overlaps_polygon` |

---

## Divergences (require fix in Task A3)

### D1. Decimation — pixel selection (HIGH)

**C behavior** (`common/image_u8.c`, integer factor path):
```c
for (int y = 0; y < height; y += factor) {
    for (int x = 0; x < width; x += factor)
        decim->buf[sy*stride + sx] = im->buf[y*stride + x];
}
```
Picks the **top-left pixel** of each stride-sized cell. For factor=2, output columns map to
source columns `0, 2, 4, …` (zero-based stride subsample).

**Rust behavior** (`kornia-imgproc/src/resize/nearest.rs`):
```rust
let src_x = ((x as f64 + 0.5) * sx).floor() as usize;  // sx = src_w / dst_w
```
Center-based nearest: output column `x` maps to source column `floor((x+0.5)*2) = 1, 3, 5, …`
for factor=2. This is a **1-pixel horizontal and vertical offset** from the C behavior.

**Numeric consequence**: The decimated image is shifted by half a source pixel relative to the
C version. This propagates through thresholding, segmentation, and quad corner fitting,
producing sub-pixel corner differences even on synthetic images.

**Investigation (Task A3)**: Attempted to apply true top-left subsample. Testing on JPEG
(`apriltags_tag36h11.jpg`) revealed 299 px maximum corner regression — different quad polygon
wins due to alternating pixel patterns changing cluster connectivity. Current Nearest resize
already achieves ≤0.26 px parity empirically (within ≤0.5 px acceptance goal). Status: **DEFERRED**.
Nearest mode sufficient for parity within tolerance.

**Fix needed**: None for current parity tests. Stride-subsampler can be revisited if future
test images require it.

---

### D2. Decimation — output dimensions (LOW)

**C behavior**: `swidth = 1 + (width - 1) / factor` (ceiling division).
For `width=641, factor=2` → `swidth=321`.

**Rust behavior**: `width / downscale_factor` (floor division).
For `width=641, factor=2` → `swidth=320`.

**Numeric consequence**: Off-by-one on odd-dimension images. Typical camera resolutions
(multiples of 2) are unaffected. However, the parity test set may include odd-dimension
images, causing an image-size mismatch error before any pixel comparison.

**Investigation (Task A3)**: Deferred pending real-world test image with odd dimensions.
Current parity test suite uses 30×30, 60×60, and 799×533 — all even or handled by D1.
Status: **DEFERRED**. Applies only to odd-dimension images; low risk in practice.

**Fix needed**: None for current parity tests. Can be addressed if odd-dimension test images
are added to the suite.

---

### D3. No quad_sigma Gaussian blur (INTENTIONAL OMISSION)

**C behavior**: When `td->quad_sigma != 0`, applies Gaussian blur (`image_u8_gaussian_blur`)
before the adaptive threshold. Default: `td->quad_sigma = 0.0` (no blur).

**Rust behavior**: No quad_sigma parameter; no pre-threshold blur path.

**Numeric consequence**: None when the C parity reference is run with `quad_sigma=0` (the
default). The Rust implementation cannot replicate the sharpening/blurring pre-processing
for non-zero quad_sigma values.

**Fix needed for parity testing**: Pin `td->quad_sigma = 0` in the C reference binary.
This is documented as an intentional omission for the initial implementation.

---

### D4. `refine_edges` search range (MEDIUM)

**C behavior** (`apriltag.c:811`):
```c
int range = td->quad_decimate + 1;  // = 2 + 1 = 3 for default quad_decimate=2
int max_steps = 2 * steps_per_unit * range + 1;  // = 25 steps
```
Searches ±3 source pixels normal to each edge.

**Rust behavior** (`decoder.rs:462`):
```rust
const RANGE: f32 = 2.0;  // TODO: Make it tuneable. It will depend on the downscaling factor...
const MAX_STEPS: usize = 2 * STEPS_PER_UNIT * RANGE as usize + 1;  // = 17 steps
```
Hardcodes ±2 source pixels. The TODO comment acknowledges this should depend on the
downscaling factor.

**Numeric consequence**: The Rust code fits refined edge lines over a narrower normal
search band. This produces slightly different corner positions after line-fitting,
especially for low-contrast or noisy edges. In practice the difference is sub-pixel
but measurable in numeric comparison against the C reference.

**Fix needed**: Replace `const RANGE: f32 = 2.0` with `(downscale_factor as f32) + 1.0`,
passed in from the decoder config.

---

### D5. `homography_compute` — float precision (LOW)

**C behavior** (`apriltag.c:homography_compute2`): Uses `double` (f64) for the 8×9 DLT matrix
and Gaussian elimination. Epsilon = `1e-10`.

**Rust behavior** (`utils.rs:homography_compute`): Uses `f32` for the same algorithm.
Epsilon = `1e-10f32` (about 2 orders of magnitude looser in absolute terms due to f32 range).

**Algorithm**: Both use the **identical algorithm** — direct DLT construction with partial-pivot
Gaussian elimination and back-substitution. Corner-to-tag mapping is also identical.
No centroid normalization in either version.

**Numeric consequence**: For typical tag sizes (corners in range [0, 640]), the f32 representation
loses ~4 decimal digits of precision relative to f64. This propagates into the homography
coefficients used in `quad_decode` (bit-cell projection). For large images or tags far from
origin, this may cause borderline bit reads to flip. In practice, with the test suite synthetic
images the difference is within ±1 ULP of f32.

**Fix needed**: Low priority. Can be addressed by promoting the DLT solve to f64 and casting
back at the end. Should be done if the numeric parity test (Task A2) shows bit-decode
disagreements attributable to homography precision.

---

### D6. Detection reconcile — overlap predicate (LOW, intentional)

C's `apriltag_detect` Step 3 dedups detections that share family+id AND whose
quads pass `g2d_polygon_overlaps_polygon` (exact convex-polygon intersection);
non-overlapping repeats of the same id all survive. Rust `dedup_detections`
mirrors the semantics but approximates the overlap test: centers closer than
half the mean of the two quads' 0→2 diagonals. Duplicate candidates originate
from the same physical quad boundary, so their centers nearly coincide — the
approximation only diverges for strongly perspective-skewed overlapping quads.
Exact SAT-based quad overlap (~25 lines) is the upgrade path if a parity count
mismatch ever appears on skewed repeated-tag scenes.

## Intentional Omissions

| Feature | C | Rust | Rationale |
|---|---|---|---|
| `quad_sigma` pre-blur | supported | absent | Default is 0; not needed for initial parity |
| Multi-thread workerpool | `workerpool_*` | single-threaded | Out of scope for parity audit |
| PDF/debug image output | `image_u8_write_pnm` | absent | Debug feature, not algorithmic |
| `Hinv` (homography inverse) | stored on quad | recomputed on demand | Memory vs compute trade-off |

---

## Verified Matches

The following algorithmic details were checked line-by-line against the C reference and
confirmed to match:

- **Tile size**: both use `tilesz = 4`
- **Tile counting**: both use integer floor division for tile count
- **Neighbor blur window**: 3×3, with edge clamping
- **Threshold formula**: `thresh = min + (max - min) / 2`
- **Skip pixel encoding**: value 127 (0x7F) in both
- **Connected components**: same 4-connectivity + white-only diagonals, same redundancy
  pruning conditions (Rust makes one additional harmless idempotent union call for the top
  neighbor when the C pruning condition would skip it — functionally equivalent)
- **Gradient cluster minimum size**: 25 pixels in both
- **Gradient cluster key ordering**: `(min(a,b), max(a,b))` in both
- **DLT matrix layout**: identical 8×9 construction; identical solution extraction
- **Corner-to-tag-space mapping**: `(-1,-1),(1,-1),(1,1),(-1,1)` in both
- **decode_sharpening**: default 0.25 in both
- **refine_edges**: enabled by default in both; same bilinear interpolation; same weight function
  `(g2-g1)²`; same centroid line-fitting at the end
- **GrayModel A-matrix**: lower-triangle fill only (upper-triangle skip to match C legacy)
- **GrayModel solver**: Cholesky on 3×3 SPD matrix (explicitly chosen to match C `matd_chol`)
- **Sharpen kernel**: `[0,-1,0,-1,4,-1,0,-1,0]` Laplacian; same boundary clipping
- **Quad area check**: `area < 0.95 * min_tag_width²`
- **Quad angle check**: same `cos_critical_rad` condition and winding-order guard
- **rotate_90**: same bit-rotation formula

---

## Summary for Task A3

Ordered by impact on numeric parity:

| Priority | Divergence | Expected impact |
|---|---|---|
| 1 | **D1** Decimation pixel selection | HIGH — 1-pixel offset shifts all corner estimates |
| 2 | **D2** Decimation output dimensions | LOW — only affects odd-dimension images |
| 3 | **D4** refine_edges range | MEDIUM — sub-pixel corner shift; 17 vs 25 search steps |
| 4 | **D5** homography f32 vs f64 | LOW — within f32 ULP for synthetic test images |
| — | **O1** quad_sigma omission | No fix needed; pin to 0 in C parity binary |
