# Kornia: kornia-vlm

[![Crates.io](https://img.shields.io/crates/v/kornia-vlm.svg)](https://crates.io/crates/kornia-vlm)
[![Documentation](https://docs.rs/kornia-vlm/badge.svg)](https://docs.rs/kornia-vlm)
[![License](https://img.shields.io/crates/l/kornia-vlm.svg)](https://github.com/kornia/kornia/blob/main/LICENSE)

> **Vision Language Models (VLM) inference in Rust.**

## 🚀 Overview

`kornia-vlm` enables the use of state-of-the-art Vision Language Models directly in Rust. Built on top of the `candle` ML framework, it provides easy-to-use interfaces for models that can "see" and "understand" images, allowing for tasks like image captioning, visual question answering, and object detection.

## 🔑 Key Features

*   **Model Support:** Implementations for **PaliGemma**, **SmolVLM**, and **SmolVLM2**.
*   **Inference Pipeline:** Simplified API for loading models, processing images, and generating text.
*   **Efficient:** Leverages `candle` for hardware-accelerated inference (CUDA support via features).
*   **Video Understanding:** (Experimental) Support for processing video frames.

## 📦 Installation

Add the following to your `Cargo.toml`. Enable `cuda` for GPU support:

```toml
[dependencies]
kornia-vlm = { version = "0.1.0", features = ["cuda"] }
```

## 🛠️ Usage

### Image Captioning (Pseudo-code)

```rust
use kornia_vlm::paligemma::{Paligemma, PaligemmaConfig};
// use kornia_io::functional as F;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Load the model
    // let mut model = Paligemma::new(PaligemmaConfig::default())?;

    // 2. Load an image
    // let image = F::read_image_any_rgb8("image.jpg")?;

    // 3. Run inference with a prompt
    // let caption = model.inference(&image, "Describe this image.", 128, false)?;

    // println!("Caption: {}", caption);
    Ok(())
}
```

## 🧩 Modules

*   **`paligemma`**: Google's PaliGemma model implementation.
*   **`smolvlm`**: HuggingFace's SmolVLM implementation.
*   **`smolvlm2`**: SmolVLM2 implementation.
*   **`video`**: Utilities for video input processing for VLMs.

## 💡 Related Examples

You can find comprehensive examples in the `examples` folder of the repository:

*   [`paligemma`](../../examples/paligemma): Example using the PaliGemma model.
*   [`smol_vlm`](../../examples/smol_vlm): Example using the SmolVLM model.
*   [`smol_vlm2`](../../examples/smol_vlm2): Example using the SmolVLM2 model.
*   [`smol_vlm_convo`](../../examples/smol_vlm_convo): Conversational example with SmolVLM.
*   [`smol_vlm_video`](../../examples/smol_vlm_video): Video processing with SmolVLM.
*   [`smol_vlm2_video`](../../examples/smol_vlm2_video): Video processing with SmolVLM2.

## 🤝 Contributing

Contributions are welcome! This crate is part of the Kornia workspace. Please refer to the main repository for contribution guidelines.

## 📄 License

This crate is licensed under the Apache-2.0 License.
