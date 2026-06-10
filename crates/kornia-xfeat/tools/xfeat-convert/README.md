# xfeat-convert

Converts the upstream XFeat PyTorch checkpoint to kornia-xfeat's packed `.safetensors` format.

## What it does

1. Loads `xfeat.pt` (upstream PyTorch state dict, NCHW float32).
2. Folds every `BatchNorm2d(affine=False)` into its preceding `Conv2d`:
   - `W_eff[co, ...] = W[co, ...] / sqrt(var[co] + 1e-5)`
   - `b_eff[co]      = -mean[co]  / sqrt(var[co] + 1e-5)`
3. Transposes all conv weights to NHWC `[c_out, k_h, k_w, c_in]`
   (or `[c_out, c_in]` for 1×1 convs).
4. Writes a `.safetensors` file with named tensors matching `model.rs`.

## Usage

```bash
pip install -r requirements.txt

# From a local checkpoint:
python convert.py --checkpoint /path/to/xfeat.pt --output xfeat_packed.safetensors

# Download automatically from Hugging Face:
python convert.py --hf-hub --output xfeat_packed.safetensors
```

Copy the resulting file to `crates/kornia-xfeat/assets/xfeat_packed.safetensors` and
update `PACKED_WEIGHTS_SHA256` in `src/weights.rs` with the printed SHA-256.

## Output tensor names

| Key                  | Shape              | Source                         |
|----------------------|--------------------|--------------------------------|
| `block1.N.weight`    | `[C_out,3,3,C_in]` | BasicLayer N of block1, BN-fold|
| `block1.N.bias`      | `[C_out]`          | BN-fold effective bias         |
| `skip1.weight`       | `[24, 1]`          | skip1 conv1×1, no BN           |
| `skip1.bias`         | `[24]`             | zeros (bias=False in PyTorch)  |
| `block5.3.weight`    | `[64, 128]`        | plain conv1×1, no BN           |
| `heatmap_head.2.*`   | `[65, 64]` / `[65]`| 65-channel keypoint conv       |
| … (see convert.py)   | …                  | …                              |
