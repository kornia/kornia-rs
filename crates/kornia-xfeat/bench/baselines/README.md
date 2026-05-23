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

- `pytorch_eager.py` — upstream XFeat unmodified, CPU device.
- `pytorch_compile.py` — same model under `torch.compile`.
- `onnxruntime.py` — upstream ONNX export through onnxruntime CPU EP.

All three drivers share `requirements.txt` (pinned versions live here, not in
the regen-fixtures tool's requirements, since their purposes diverge —
fixture regen uses one torch version; baselines may track different ones).
