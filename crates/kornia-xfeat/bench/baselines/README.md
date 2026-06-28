# Baseline drivers

External baselines we benchmark against. The Rust harness in
`bench/harness/run-all.rs` reads JSON results from this directory and
compares them against the current `kornia-xfeat` measurement.

## Driver contract

Each baseline driver is a standalone script that:

1. Loads `tests/fixtures/v1/input.png` (the same input the parity tests use).
2. Runs the model to completion 1000 times after a 100-iter warmup.
3. Writes `bench/results/<platform>/<baseline>.json` with the schema:

```json
{
  "baseline": "pytorch_eager",
  "platform": "jetson-orin",
  "median_ms": 215.4,
  "p95_ms":    231.2,
  "p99_ms":    260.0,
  "iterations": 1000,
  "warmup":    100,
  "rss_mb":    430,
  "env": {
    "torch":   "2.4.1",
    "python":  "3.10.12",
    "cpu":     "Cortex-A78AE",
    "kernel":  "5.15.148-tegra"
  }
}
```

The Rust harness never invokes these scripts itself — they run in a separate
weekly CI workflow (or manually when bumping competitor pins). Day-to-day
PR CI just diffs the committed snapshots in `bench/results/`.

## Drivers

| Script | Runtime | Device | Notes |
|--------|---------|--------|-------|
| `pytorch_eager.py` | PyTorch eager | CPU | Reference baseline |
| `pytorch_compile.py` | `torch.compile` | CPU | JIT-compiled variant |
| `onnxruntime.py` | ONNX Runtime CPU EP | CPU | ONNX forward-pass only |
| `pytorch_cuda.py` | PyTorch eager | CUDA GPU | Requires `torch.cuda.is_available()`; exits with code 2 if not |
| `onnxruntime_cuda.py` | ONNX Runtime CUDA EP | CUDA GPU | Requires pre-exported `.onnx`; exits with code 2 if CUDA EP absent |

All drivers share `requirements.txt` (pinned versions live here, not in
the regen-fixtures tool's requirements, since their purposes diverge —
fixture regen uses one torch version; baselines may track different ones).

### CUDA driver prerequisites

`pytorch_cuda.py` — no extra setup beyond a CUDA-enabled PyTorch wheel.

`onnxruntime_cuda.py` — the ONNX model must be exported once before
running the benchmark:

```bash
python3 - <<'EOF'
import sys, torch
sys.path.insert(0, '/path/to/accelerated_features')
from modules.xfeat import XFeat
model = XFeat(weights='/path/to/accelerated_features/weights/xfeat.pt').eval()
dummy = torch.zeros(1, 1, 480, 640)
torch.onnx.export(
    model, (dummy,),
    'xfeat_480x640.onnx',
    opset_version=17,                          # aten::unfold requires opset 17+
    input_names=['image'],
    output_names=['feats', 'keypoints', 'reliability'],
    dynamic_axes={'image': {2: 'H', 3: 'W'}},
)
EOF
```

On Jetson (JetPack 5.x) use the `onnxruntime-gpu` wheel from the NVIDIA
feed (built against CUDA 11.4+), not the stock PyPI wheel.

### Exit-code convention for CUDA drivers

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Unhandled exception / crash |
| 2 | CUDA / CUDA EP not available — harness skips this baseline |
