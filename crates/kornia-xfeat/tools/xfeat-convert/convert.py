#!/usr/bin/env python3
"""
xfeat-convert — fold BatchNorm into the preceding conv and repack all weights
to kornia-xfeat's NHWC safetensors format.

Usage
-----
    # Install deps once:
    #   pip install torch safetensors numpy

    # Load the upstream XFeat checkpoint and convert:
    python convert.py --checkpoint xfeat.pt --output xfeat_packed.safetensors

    # Or download the upstream checkpoint on the fly:
    python convert.py --hf-hub --output xfeat_packed.safetensors

Input
-----
The upstream PyTorch checkpoint (`xfeat.pt`) as downloaded from:
  https://github.com/verlab/accelerated_features/releases

Output format
-------------
A safetensors file where every tensor is:
  - dtype f32
  - weights for 3x3 convs: NHWC layout [c_out, k_h, k_w, c_in]
  - weights for 1x1 convs: [c_out, c_in]
  - bias tensors: [c_out]

BatchNorm fold (affine=False → gamma=1, beta=0):
  W_eff[co, ...] = W[co, ...] / sqrt(var[co] + eps)
  b_eff[co]      = -mean[co] / sqrt(var[co] + eps)

Layer naming in output file
---------------------------
block1.{0..3}.weight / .bias     — Block-1 BasicLayers (BN folded)
skip1.weight / .bias             — Skip1 conv1x1 (no BN; bias = zeros)
block2.{0..1}.weight / .bias
block3.{0..2}.weight / .bias
block4.{0..2}.weight / .bias
block5.{0..2}.weight / .bias     — block5 BasicLayers (BN folded)
block5.3.weight / .bias          — block5 plain conv1x1 (no BN; bias = zeros)
block_fusion.{0..1}.weight/.bias
block_fusion.2.weight / .bias    — plain conv1x1
heatmap_head.{0..1}.weight/.bias
heatmap_head.2.weight / .bias    — conv1x1(64→65)
keypoint_head.{0..1}.weight/.bias
keypoint_head.2.weight / .bias
"""

import argparse
import sys
import hashlib
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:
    sys.exit("torch is required: pip install torch")

try:
    from safetensors.numpy import save_file
except ImportError:
    sys.exit("safetensors is required: pip install safetensors")

BN_EPS = 1e-5


def fold_bn(conv_w: np.ndarray, bn_mean: np.ndarray, bn_var: np.ndarray) -> tuple:
    """
    Fold affine=False BatchNorm into the preceding conv.

    Returns (W_eff, b_eff) in NCHW layout (caller transposes to NHWC).
    """
    inv_std = 1.0 / np.sqrt(bn_var + BN_EPS)          # [c_out]
    W_eff = conv_w * inv_std[:, None, None, None]       # broadcast over c_in, kH, kW
    b_eff = -bn_mean * inv_std
    return W_eff, b_eff


def nchw_to_nhwc_3x3(W: np.ndarray) -> np.ndarray:
    """[c_out, c_in, 3, 3] → [c_out, 3, 3, c_in]"""
    return W.transpose(0, 2, 3, 1).astype(np.float32)


def nchw_to_nhwc_1x1(W: np.ndarray) -> np.ndarray:
    """[c_out, c_in, 1, 1] → [c_out, c_in]"""
    return W.squeeze(-1).squeeze(-1).astype(np.float32)


def to_np(t) -> np.ndarray:
    return t.detach().cpu().float().numpy()


def add_basic_layer(out: dict, sd: dict, pt_prefix: str, our_name: str):
    """
    Read a BasicLayer (Sequential[Conv2d, BN(affine=False), ReLU]) from the
    PyTorch state dict and write a BN-folded NHWC tensor pair to `out`.

    PyTorch keys:  {pt_prefix}.0.weight  +  {pt_prefix}.1.running_mean/var
    Output keys:   {our_name}.weight  /  {our_name}.bias
    """
    W = to_np(sd[f"{pt_prefix}.0.weight"])          # NCHW [c_out,c_in,3,3]
    mean = to_np(sd[f"{pt_prefix}.1.running_mean"])
    var  = to_np(sd[f"{pt_prefix}.1.running_var"])
    W_eff, b_eff = fold_bn(W, mean, var)
    out[f"{our_name}.weight"] = nchw_to_nhwc_3x3(W_eff)
    out[f"{our_name}.bias"]   = b_eff.astype(np.float32)


def add_plain_conv1x1(out: dict, sd: dict, pt_key: str, our_name: str):
    """
    Read a plain Conv2d(bias=False) 1×1 from the state dict and write
    an NHWC tensor pair (with zero bias) to `out`.

    PyTorch key:  {pt_key}    (shape [c_out, c_in, 1, 1])
    Output keys:  {our_name}.weight  /  {our_name}.bias  (zeros)
    """
    W = to_np(sd[pt_key])         # [c_out, c_in, 1, 1]
    c_out = W.shape[0]
    out[f"{our_name}.weight"] = nchw_to_nhwc_1x1(W)
    out[f"{our_name}.bias"]   = np.zeros(c_out, dtype=np.float32)


def convert(state_dict: dict) -> dict:
    """Convert a loaded XFeat state dict to kornia-xfeat packed format."""
    sd = state_dict
    out: dict = {}

    # ── Block 1 (4 × BasicLayer) ──────────────────────────────────────────
    for i in range(4):
        add_basic_layer(out, sd, f"block1.{i}", f"block1.{i}")

    # ── Skip1 (AvgPool — no params — + Conv1×1 without BN) ────────────────
    # PyTorch key:  skip1.1.weight  (the second element; index 0 = AvgPool)
    add_plain_conv1x1(out, sd, "skip1.1.weight", "skip1")

    # ── Block 2 (2 × BasicLayer) ──────────────────────────────────────────
    for i in range(2):
        add_basic_layer(out, sd, f"block2.{i}", f"block2.{i}")

    # ── Block 3 (3 × BasicLayer) ──────────────────────────────────────────
    for i in range(3):
        add_basic_layer(out, sd, f"block3.{i}", f"block3.{i}")

    # ── Block 4 (3 × BasicLayer) ──────────────────────────────────────────
    for i in range(3):
        add_basic_layer(out, sd, f"block4.{i}", f"block4.{i}")

    # ── Block 5 (3 × BasicLayer + 1 plain Conv1×1) ────────────────────────
    for i in range(3):
        add_basic_layer(out, sd, f"block5.{i}", f"block5.{i}")
    add_plain_conv1x1(out, sd, "block5.3.weight", "block5.3")

    # ── block_fusion (2 × BasicLayer + 1 plain Conv1×1) ──────────────────
    for i in range(2):
        add_basic_layer(out, sd, f"block_fusion.{i}", f"block_fusion.{i}")
    add_plain_conv1x1(out, sd, "block_fusion.2.weight", "block_fusion.2")

    # ── heatmap_head (2 × BasicLayer + 1 plain Conv1×1 → 65ch) ──────────
    for i in range(2):
        add_basic_layer(out, sd, f"heatmap_head.{i}", f"heatmap_head.{i}")
    add_plain_conv1x1(out, sd, "heatmap_head.2.weight", "heatmap_head.2")

    # ── keypoint_head (2 × BasicLayer + 1 plain Conv1×1) ─────────────────
    for i in range(2):
        add_basic_layer(out, sd, f"keypoint_head.{i}", f"keypoint_head.{i}")
    add_plain_conv1x1(out, sd, "keypoint_head.2.weight", "keypoint_head.2")

    return out


def load_checkpoint(path: Path) -> dict:
    """Load a .pt checkpoint and extract the model state dict."""
    ckpt = torch.load(str(path), map_location="cpu")
    if isinstance(ckpt, dict):
        # Prefer 'model' or 'state_dict' keys; otherwise assume the dict IS the SD
        for key in ("model", "state_dict", "net"):
            if key in ckpt:
                return ckpt[key]
        return ckpt
    # Might be the model object itself
    return getattr(ckpt, "state_dict", lambda: ckpt)()


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description="Convert XFeat weights to kornia-xfeat packed format")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", type=Path, help="Path to upstream xfeat.pt checkpoint")
    src.add_argument("--hf-hub", action="store_true", help="Download from Hugging Face hub")
    ap.add_argument("--output", type=Path, required=True, help="Output .safetensors path")
    args = ap.parse_args()

    if args.hf_hub:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            sys.exit("huggingface_hub is required for --hf-hub: pip install huggingface_hub")
        print("Downloading XFeat weights from Hugging Face …")
        ckpt_path = Path(hf_hub_download(
            repo_id="verlab/accelerated_features",
            filename="xfeat.pt",
        ))
    else:
        ckpt_path = args.checkpoint
        if not ckpt_path.exists():
            sys.exit(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading {ckpt_path} …")
    sd = load_checkpoint(ckpt_path)

    print(f"Converting {len(sd)} state-dict tensors …")
    packed = convert(sd)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(packed, str(args.output))

    sha = sha256_of_file(args.output)
    print(f"Written {args.output}  ({args.output.stat().st_size // 1024} KiB)")
    print(f"SHA-256: {sha}")
    print()
    print("Update PACKED_WEIGHTS_SHA256 in src/weights.rs with the SHA above.")
    print("Commit the packed weights to the assets/ directory for CI.")


if __name__ == "__main__":
    main()
