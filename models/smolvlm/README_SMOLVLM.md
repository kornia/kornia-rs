# SmolVLM Implementation for Kornia-rs

This repository contains the SmolVLM (Small Visual Language Model) implementation for the Kornia-rs project, a computer vision library in Rust. SmolVLM enables efficient vision-language capabilities within Kornia, supporting multiple backends and model sizes.

## Features

- **Multiple Backends**: Supports both Candle and ONNX backends for neural network inference
- **Flexible Model Sizes**: Compatible with small, medium, and large model variants
- **Cross-Platform**: Works on both x86_64 and aarch64 architectures
- **Comprehensive API**: Simple Rust API for easy integration
- **Python Interoperability**: Python bindings and demo scripts included

## Requirements

- Rust 1.76+
- Python 3.8+ (for Python scripts)
- Python dependencies: PIL (Pillow), requests, numpy (optional)

## Installation

Clone the repository and build the project:

```bash
git clone https://github.com/kornia/kornia-rs.git
cd kornia-rs
cargo build --features="kornia-models/candle"  # For Candle backend
# or
cargo build --features="kornia-models/onnx"    # For ONNX backend
# or
cargo build --features="kornia-models/candle kornia-models/onnx"  # For both backends
```

## Usage

### Rust API

```rust
use kornia_models::smolvlm::{SmolVLM, ModelSize, Backend};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a SmolVLM instance with Candle backend
    let model = SmolVLM::new(ModelSize::Small, Backend::Candle)?;
    
    // Load an image
    let image = image::open("path/to/image.jpg")?;
    
    // Analyze the image with a prompt
    let prompt = "What objects are in this image?";
    let result = model.analyze(&image, prompt)?;
    
    println!("Analysis result: {}", result);
    
    Ok(())
}
```

### Command-Line Demo

```bash
# Run the SmolVLM demo with Candle backend
cargo run --release --example smolvlm_demo --features="kornia-models/candle" -- \
    --image path/to/image.jpg \
    --prompt "What objects are in this image?" \
    --model-size small \
    --backend candle

# Run the SmolVLM demo with ONNX backend
cargo run --release --example smolvlm_compare --features="kornia-models/onnx" -- \
    --image path/to/image.jpg \
    --prompt "What objects are in this image?" \
    --model-size small \
    --backend onnx
```

### Python Scripts

```bash
# Run the Python demo 
python smolvlm_demo.py -i path/to/image.jpg -p "What objects are in this image?" -s small

# Run benchmark comparing backends and model sizes
python benchmark.py -i path/to/image.jpg -b python candle onnx -s small medium -t objects scene
```

## Benchmarking

The `benchmark.py` script allows comparing performance across different backends and model sizes:

```bash
python benchmark.py -i path/to/image.jpg -b python candle onnx -s small medium -t objects scene -r 3 -o benchmark_results.json
```

Options:
- `-i/--image`: Path to test image
- `-b/--backends`: Backends to benchmark (python, candle, onnx)
- `-s/--sizes`: Model sizes to benchmark (small, medium, large)
- `-t/--tasks`: Tasks to benchmark (description, objects, colors, scene, etc.)
- `-r/--runs`: Number of benchmark runs for each configuration
- `-o/--output`: Output file for benchmark results (JSON)
- `--use-hf`: Use Hugging Face API for Python backend if token is available

## CI/CD Integration

This repository includes CI workflows for automated testing and validation. The workflows test:

1. Cross-platform compatibility (x86_64 and aarch64)
2. Python compatibility (3.8-3.13)
3. Formatting and linting (rustfmt)
4. Feature combinations (Candle, ONNX, both)

### CI Environment Notes

- CI environments use reduced timeouts and smaller image sizes
- Some tests are skipped in CI environments to avoid timeouts
- Platform detection is handled automatically

## Model Details

SmolVLM supports multiple model sizes:

- **Small**: Fastest, lowest memory usage
- **Medium**: Balanced performance and accuracy
- **Large**: Highest accuracy, slower inference

