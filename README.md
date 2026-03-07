# Kornia: kornia-py

[![License](https://img.shields.io/crates/l/kornia-py.svg)](https://github.com/kornia/kornia/blob/main/LICENSE)

> **Python bindings for the Kornia Rust computer‑vision library.**

## 🚀 Overview
`kornia-py` provides Python bindings for the core Kornia Rust ecosystem using [PyO3](https://github.com/PyO3/pyo3). It allows Python developers to leverage the high‑performance, safe Rust implementations of Kornia's vision algorithms directly from Python.

## 🔑 Key Features
- Seamless integration with NumPy and PyTorch arrays
- Pythonic API for Rust‑backed tensor and image operations
- High‑performance execution bypassing Python's GIL where possible
- Easy installation via `pip`

## 📦 Installation
```bash
pip install kornia-rs
```
*(Note: the published package is often named `kornia-rs` on PyPI, while building locally uses `maturin`)*

To build from source:
```bash
pip install maturin
maturin develop --release
```

## 🛠️ Usage
```python
import kornia_rs as K
import numpy as np

# Example: create a tensor from numpy and apply an operation
arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).reshape(2, 2)
tensor = K.Tensor(arr)
# Access Rust-backed Kornia functions
```

## 🤝 Contributing
Contributions are welcome! See the main repository's [Contributing Guidelines](https://github.com/kornia/kornia/blob/main/CONTRIBUTING.md).

## 📄 License
Apache-2.0
