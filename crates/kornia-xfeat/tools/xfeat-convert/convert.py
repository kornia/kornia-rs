#!/usr/bin/env python3
"""
xfeat-convert — fold BatchNorm into the preceding conv and repack all weights
to kornia-xfeat's NHWC safetensors format.

Usage
-----
    pip install torch safetensors numpy

    python convert.py --checkpoint xfeat.pt --output xfeat_packed.safetensors
    python convert.py --hf-hub --output xfeat_packed.safetensors

Architecture (from upstream modules/model.py)
---------------------------------------------
All BasicLayers store: {prefix}.layer.0.weight (conv) + {prefix}.layer.1.running_mean/var (BN).
BN is always affine=False → gamma=1, beta=0, so fold simplifies to:
    W_eff = W / sqrt(var + eps),  b_eff = -mean / sqrt(var + eps)

Layer key mapping (PyTorch state dict → packed output)
-------------------------------------------------------
BasicLayer (3×3 or 1×1 + BN, affine=False):
    {block}.{idx}.layer.0.weight  +  .layer.1.running_{mean,var}
    → packed: {block}.{idx}.weight / .bias   [NHWC, BN-folded]

Plain Conv2d with real bias (no BN):
    {key}.weight + {key}.bias
    → packed: {key}.weight / .bias   [NHWC, bias as-is]

Output shapes (packed NHWC):
    3×3 BasicLayer: [c_out, 3, 3, c_in]
    1×1 BasicLayer: [c_out, c_in]
    1×1 plain:      [c_out, c_in]
    bias:           [c_out]
"""

import argparse
import hashlib
import sys
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


def to_np(t) -> np.ndarray:
    return t.detach().cpu().float().numpy()


def nchw_to_nhwc_3x3(W: np.ndarray) -> np.ndarray:
    """[c_out, c_in, 3, 3] → [c_out, 3, 3, c_in]"""
    return W.transpose(0, 2, 3, 1).astype(np.float32)


def nchw_to_nhwc_1x1(W: np.ndarray) -> np.ndarray:
    """[c_out, c_in, 1, 1] or [c_out, c_in, H, W] → [c_out, c_in]"""
    return W.squeeze(-1).squeeze(-1).astype(np.float32)


def fold_basic_layer(sd: dict, pt_prefix: str) -> tuple:
    """
    Fold BN(affine=False) into the preceding conv for a BasicLayer.

    State dict keys:
      {pt_prefix}.layer.0.weight         → conv weight [c_out, c_in, kH, kW]
      {pt_prefix}.layer.1.running_mean   → BN mean     [c_out]
      {pt_prefix}.layer.1.running_var    → BN var      [c_out]

    Returns (W_nhwc, bias_f32) where:
      W_nhwc is [c_out, kH, kW, c_in] for 3×3 or [c_out, c_in] for 1×1
      bias_f32 is [c_out] (the BN-folded effective bias)
    """
    W = to_np(sd[f"{pt_prefix}.layer.0.weight"])   # [c_out, c_in, kH, kW]
    mean = to_np(sd[f"{pt_prefix}.layer.1.running_mean"])
    var  = to_np(sd[f"{pt_prefix}.layer.1.running_var"])
    inv_std = 1.0 / np.sqrt(var + BN_EPS)
    W_eff   = W * inv_std[:, None, None, None]
    b_eff   = (-mean * inv_std).astype(np.float32)

    kH, kW = W.shape[2], W.shape[3]
    if kH == 3 and kW == 3:
        return nchw_to_nhwc_3x3(W_eff), b_eff
    else:  # 1×1 (or any non-3×3)
        return nchw_to_nhwc_1x1(W_eff), b_eff


def add_basic_layer(out: dict, sd: dict, pt_prefix: str, our_name: str) -> None:
    W, b = fold_basic_layer(sd, pt_prefix)
    out[f"{our_name}.weight"] = W
    out[f"{our_name}.bias"]   = b


def add_plain_conv(out: dict, sd: dict, pt_key: str, our_name: str) -> None:
    """
    Plain Conv2d with real bias (no BN), any kernel size.
    PyTorch keys: {pt_key}.weight  /  {pt_key}.bias
    """
    W = to_np(sd[f"{pt_key}.weight"])
    b = to_np(sd[f"{pt_key}.bias"])
    kH, kW = W.shape[2], W.shape[3]
    if kH == 3 and kW == 3:
        out[f"{our_name}.weight"] = nchw_to_nhwc_3x3(W)
    else:
        out[f"{our_name}.weight"] = nchw_to_nhwc_1x1(W)
    out[f"{our_name}.bias"] = b.astype(np.float32)


def add_skip1(out: dict, sd: dict) -> None:
    """
    skip1 is nn.Sequential(AvgPool2d, Conv2d(1, 24, 1, bias=True)).
    The conv has real bias; no BN.
    PyTorch key: skip1.1.weight  /  skip1.1.bias
    """
    W = to_np(sd["skip1.1.weight"])   # [24, 1, 1, 1]
    b = to_np(sd["skip1.1.bias"])     # [24]
    out["skip1.weight"] = nchw_to_nhwc_1x1(W)
    out["skip1.bias"]   = b.astype(np.float32)


def convert(state_dict: dict) -> dict:
    """Convert an XFeat state dict to kornia-xfeat packed format."""
    sd = state_dict
    out: dict = {}

    # ── Block 1 (4 × BasicLayer, 3×3) ─────────────────────────────────────
    for i in range(4):
        add_basic_layer(out, sd, f"block1.{i}", f"block1.{i}")

    # ── Skip1 (AvgPool + Conv2d with bias) ────────────────────────────────
    add_skip1(out, sd)

    # ── Block 2 (2 × BasicLayer, 3×3) ─────────────────────────────────────
    for i in range(2):
        add_basic_layer(out, sd, f"block2.{i}", f"block2.{i}")

    # ── Block 3 (2 × 3×3 BasicLayer + 1 × 1×1 BasicLayer) ────────────────
    for i in range(2):
        add_basic_layer(out, sd, f"block3.{i}", f"block3.{i}")
    add_basic_layer(out, sd, "block3.2", "block3.2")   # 1×1, BN-fold, relu

    # ── Block 4 (3 × BasicLayer, 3×3) ─────────────────────────────────────
    for i in range(3):
        add_basic_layer(out, sd, f"block4.{i}", f"block4.{i}")

    # ── Block 5 (3 × 3×3 BasicLayer + 1 × 1×1 BasicLayer) ────────────────
    for i in range(3):
        add_basic_layer(out, sd, f"block5.{i}", f"block5.{i}")
    add_basic_layer(out, sd, "block5.3", "block5.3")   # 1×1, BN-fold, relu

    # ── block_fusion (2 × 3×3 BasicLayer + 1 × 1×1 plain conv with bias) ─
    for i in range(2):
        add_basic_layer(out, sd, f"block_fusion.{i}", f"block_fusion.{i}")
    add_plain_conv(out, sd, "block_fusion.2", "block_fusion.2")

    # ── heatmap_head (2 × 1×1 BasicLayer + 1 × 1×1 plain conv with bias) ─
    # Outputs 1 channel; activation is Sigmoid (applied in forward pass).
    for i in range(2):
        add_basic_layer(out, sd, f"heatmap_head.{i}", f"heatmap_head.{i}")
    add_plain_conv(out, sd, "heatmap_head.2", "heatmap_head.2")

    # ── keypoint_head (3 × 1×1 BasicLayer + 1 × 1×1 plain conv with bias) ─
    # Takes unfold_8x8(norm_gray) as input; outputs 65 channels (dustbin).
    for i in range(3):
        add_basic_layer(out, sd, f"keypoint_head.{i}", f"keypoint_head.{i}")
    add_plain_conv(out, sd, "keypoint_head.3", "keypoint_head.3")

    return out


def load_checkpoint(path: Path) -> dict:
    ckpt = torch.load(str(path), map_location="cpu")
    if isinstance(ckpt, dict):
        for key in ("model", "state_dict", "net"):
            if key in ckpt:
                return ckpt[key]
        return ckpt
    return getattr(ckpt, "state_dict", lambda: ckpt)()


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(
        description="Convert XFeat weights to kornia-xfeat packed safetensors format"
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", type=Path, help="Path to upstream xfeat.pt")
    src.add_argument("--hf-hub", action="store_true",
                     help="Auto-download from Hugging Face hub")
    ap.add_argument("--output", type=Path, required=True,
                    help="Output .safetensors path")
    args = ap.parse_args()

    if args.hf_hub:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            sys.exit("huggingface_hub required for --hf-hub: pip install huggingface_hub")
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
    size_kb = args.output.stat().st_size // 1024
    print(f"\nWritten {args.output}  ({size_kb} KiB)")
    print(f"SHA-256: {sha}")
    print(f"\nUpdate PACKED_WEIGHTS_SHA256 in src/weights.rs with the SHA above.")
    print(f"Copy {args.output} to crates/kornia-xfeat/assets/xfeat_packed.safetensors")


if __name__ == "__main__":
    main()
