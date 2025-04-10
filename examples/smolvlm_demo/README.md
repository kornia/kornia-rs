# SmolVLM Rust Demo

This is a demonstration of the SmolVLM (Small Vision Language Model) Rust API design and implementation structure. The demo showcases the planned architecture for integrating SmolVLM into the kornia-rs ecosystem.

## Features

- **Modular Architecture**: Separates concerns into processor, tokenizer, and backend components
- **Trait-Based Design**: Defines clear interfaces for each component
- **Extensible Backend System**: Support for multiple backends (currently demonstrating Candle)
- **Simple API**: Easy-to-use interface for image-to-text generation

## Components

1. **ImageProcessor**: Handles image loading, resizing, and preparation for the model
2. **Tokenizer**: Manages text tokenization for prompts and token decoding for outputs
3. **Backend**: Implements the actual model inference (with Candle or ONNX)
4. **SmolVLM**: The main model class that coordinates the components

## Usage

```rust
// Initialize components
let processor = smolvlm::BasicImageProcessor::new(224, 224);
let tokenizer = smolvlm::BasicTokenizer::new();
let backend = smolvlm::CandleBackend::new();

// Create model
let model = smolvlm::SmolVLM::new(processor, tokenizer, backend);

// Generate description from image
let result = model.generate("path/to/image.jpg", "Describe this image", 100)?;
println!("{}", result);
```

## Implementation Notes

This demo provides a structural template for the actual implementation. In the full version:

- **Image Processing**: Will use proper image loading and preprocessing with normalization
- **Tokenization**: Will implement proper tokenization using the SmolVLM tokenizer
- **Model Inference**: Will connect to actual model weights loaded in Candle or ONNX Runtime
- **Performance Optimization**: Will include memory and computational optimizations

## Future Enhancements

1. ONNX Runtime backend implementation
2. GPU acceleration support
3. Model quantization options
4. Streaming token generation
5. Additional model variants (small, medium, large)

## Building

Once the actual implementation is complete, the example will be buildable with:

```bash
cd kornia-rs
cargo build --example smolvlm_demo
```

## Running

```bash
cargo run --example smolvlm_demo -- test_image.jpg "Describe what you see in this image"
```