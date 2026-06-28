# xfeat-regen-fixtures

Regenerates the parity fixtures in `tests/fixtures/v1/` from the upstream
XFeat PyTorch reference. Lives outside the Rust build deliberately — CI
never runs this. The fixtures are committed; bumping is a deliberate act.

## When to run

- Pinning to a newer upstream XFeat commit.
- Switching test image (e.g. a different EuRoC frame).
- Investigating a parity regression (cross-check current outputs against the upstream).

## When NOT to run

- "The parity test fails, let me regenerate the baseline." The whole point of
  the fixture is that it doesn't move silently. Investigate the diff first.

## Usage

```bash
pip install -r requirements.txt
python regen.py --output ../../tests/fixtures/v1
```

Outputs:

- `input_preprocessed.safetensors` — the f32 gray tensor exactly as the model
  consumes it after `align_to_32` + `F.interpolate`. We dump this instead of
  trusting that two PIL/imread paths produce bitwise identical bytes.
- `expected_dense.safetensors` — `keypoint_logits`, `descriptor` (pre and post
  L2-norm), and `reliability` dense maps from one forward pass.
- `expected_sparse.json` — post-NMS top-K with descriptor and score per kp.
- Updated `MANIFEST.toml` with the new upstream commit + sha.

## Pinned versions

`requirements.txt` pins PyTorch and numpy. Don't update on a whim — different
PyTorch versions can produce non-trivial f32 reduction-order drift on the
dense maps.
