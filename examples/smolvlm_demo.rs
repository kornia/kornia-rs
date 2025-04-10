//! SmolVLM Demo Example
//!
//! This example demonstrates how to use SmolVLM to analyze an image.
//!
//! Usage:
//! ```bash
//! cargo run --example smolvlm_demo --features="kornia-models/candle" -- \
//!     --image test_image.jpg \
//!     --prompt "What objects are in this image?" \
//!     --model-path models/candle/Small
//! ```

use std::path::PathBuf;
use std::time::Instant;

use argh::FromArgs;

#[cfg(feature = "kornia-models")]
use kornia_models::smolvlm::common::{ModelSize, SmolVLMConfig, SmolVLMError};
#[cfg(feature = "kornia-models")]
use kornia_models::smolvlm::processor::ImageProcessor;
#[cfg(all(feature = "kornia-models", feature = "candle"))]
use kornia_models::smolvlm::candle::CandleBackend;
#[cfg(all(feature = "kornia-models", feature = "onnx"))]
use kornia_models::smolvlm::onnx::OnnxBackend;

/// Command-line arguments
#[derive(FromArgs)]
struct Args {
    /// path to input image
    #[argh(option, short = 'i')]
    image: String,

    /// prompt to analyze the image
    #[argh(option, short = 'p', default = "String::from(\"What objects are in this image?\")")]
    prompt: String,

    /// backend to use (candle, onnx)
    #[argh(option, short = 'b', default = "String::from(\"candle\")")]
    backend: String,

    /// model size (small, medium, large)
    #[argh(option, short = 's', default = "String::from(\"small\")")]
    model_size: String,

    /// path to the model directory
    #[argh(option, short = 'm')]
    model_path: Option<String>,
    
    /// enable verbose output
    #[argh(switch, short = 'v')]
    verbose: bool,
}

#[cfg(all(feature = "kornia-models", any(feature = "candle", feature = "onnx")))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse command-line arguments
    let args: Args = argh::from_env();
    
    // Parse model size
    let model_size = match args.model_size.to_lowercase().as_str() {
        "small" => ModelSize::Small,
        "medium" => ModelSize::Medium,
        "large" => ModelSize::Large,
        _ => {
            println!("Unknown model size: {}. Using Small as default.", args.model_size);
            ModelSize::Small
        }
    };
    
    // Create SmolVLM config
    let config = SmolVLMConfig::new(model_size);
    
    // Check if image exists
    if !std::path::Path::new(&args.image).exists() {
        return Err(format!("Image not found: {}", args.image).into());
    }
    
    // Create image processor and process the image
    let processor = ImageProcessor::new(&config)?;
    
    if args.verbose {
        println!("Processing image: {}", args.image);
    }
    
    let start = Instant::now();
    let processed_image = processor.process_image_from_path(&args.image)?;
    let processing_time = start.elapsed();
    
    if args.verbose {
        println!("Image processing time: {:?}", processing_time);
    }
    
    // Run inference with the selected backend
    match args.backend.to_lowercase().as_str() {
        #[cfg(feature = "candle")]
        "candle" => {
            let model_path = args
                .model_path
                .clone()
                .unwrap_or_else(|| format!("models/candle/{:?}", model_size));
            
            if !std::path::Path::new(&model_path).exists() {
                return Err(format!(
                    "Model path does not exist: {}. Please download the model using download_models.sh",
                    model_path
                ).into());
            }
            
            if args.verbose {
                println!("Using Candle backend with model: {}", model_path);
            }
            
            let start = Instant::now();
            let backend = CandleBackend::new(&model_path, &config)?;
            let load_time = start.elapsed();
            
            if args.verbose {
                println!("Model load time: {:?}", load_time);
            }
            
            println!("Analyzing image with prompt: {}", args.prompt);
            
            let start = Instant::now();
            let result = backend.generate_caption_for_image(&processed_image, &args.prompt)?;
            let inference_time = start.elapsed();
            
            println!("\nResult:");
            println!("{}", result);
            
            if args.verbose {
                println!("\nInference time: {:?}", inference_time);
                println!("Total time: {:?}", start.elapsed() + load_time + processing_time);
            }
        }
        #[cfg(feature = "onnx")]
        "onnx" => {
            let model_path = args
                .model_path
                .clone()
                .unwrap_or_else(|| format!("models/onnx/{:?}", model_size));
            
            if !std::path::Path::new(&model_path).exists() {
                return Err(format!(
                    "Model path does not exist: {}. Please download the model using download_models.sh",
                    model_path
                ).into());
            }
            
            if args.verbose {
                println!("Using ONNX backend with model: {}", model_path);
            }
            
            let start = Instant::now();
            let backend = OnnxBackend::new(&model_path, &config)?;
            let load_time = start.elapsed();
            
            if args.verbose {
                println!("Model load time: {:?}", load_time);
            }
            
            println!("Analyzing image with prompt: {}", args.prompt);
            
            let start = Instant::now();
            let result = backend.generate_caption_for_image(&processed_image, &args.prompt)?;
            let inference_time = start.elapsed();
            
            println!("\nResult:");
            println!("{}", result);
            
            if args.verbose {
                println!("\nInference time: {:?}", inference_time);
                println!("Total time: {:?}", start.elapsed() + load_time + processing_time);
            }
        }
        _ => {
            #[cfg(feature = "candle")]
            {
                println!("Unknown backend: {}. Using Candle as default.", args.backend);
                
                let model_path = args
                    .model_path
                    .clone()
                    .unwrap_or_else(|| format!("models/candle/{:?}", model_size));
                
                if !std::path::Path::new(&model_path).exists() {
                    return Err(format!(
                        "Model path does not exist: {}. Please download the model using download_models.sh",
                        model_path
                    ).into());
                }
                
                if args.verbose {
                    println!("Using Candle backend with model: {}", model_path);
                }
                
                let start = Instant::now();
                let backend = CandleBackend::new(&model_path, &config)?;
                let load_time = start.elapsed();
                
                if args.verbose {
                    println!("Model load time: {:?}", load_time);
                }
                
                println!("Analyzing image with prompt: {}", args.prompt);
                
                let start = Instant::now();
                let result = backend.generate_caption_for_image(&processed_image, &args.prompt)?;
                let inference_time = start.elapsed();
                
                println!("\nResult:");
                println!("{}", result);
                
                if args.verbose {
                    println!("\nInference time: {:?}", inference_time);
                    println!("Total time: {:?}", start.elapsed() + load_time + processing_time);
                }
            }
            
            #[cfg(all(not(feature = "candle"), feature = "onnx"))]
            {
                println!("Unknown backend: {}. Using ONNX as default.", args.backend);
                
                let model_path = args
                    .model_path
                    .clone()
                    .unwrap_or_else(|| format!("models/onnx/{:?}", model_size));
                
                if !std::path::Path::new(&model_path).exists() {
                    return Err(format!(
                        "Model path does not exist: {}. Please download the model using download_models.sh",
                        model_path
                    ).into());
                }
                
                if args.verbose {
                    println!("Using ONNX backend with model: {}", model_path);
                }
                
                let start = Instant::now();
                let backend = OnnxBackend::new(&model_path, &config)?;
                let load_time = start.elapsed();
                
                if args.verbose {
                    println!("Model load time: {:?}", load_time);
                }
                
                println!("Analyzing image with prompt: {}", args.prompt);
                
                let start = Instant::now();
                let result = backend.generate_caption_for_image(&processed_image, &args.prompt)?;
                let inference_time = start.elapsed();
                
                println!("\nResult:");
                println!("{}", result);
                
                if args.verbose {
                    println!("\nInference time: {:?}", inference_time);
                    println!("Total time: {:?}", start.elapsed() + load_time + processing_time);
                }
            }
            
            #[cfg(not(any(feature = "candle", feature = "onnx")))]
            {
                return Err(format!(
                    "Unknown backend: {}. No supported backends available.",
                    args.backend
                ).into());
            }
        }
    }
    
    Ok(())
}

#[cfg(not(all(feature = "kornia-models", any(feature = "candle", feature = "onnx"))))]
fn main() {
    println!("This example requires the 'kornia-models' feature and at least one backend feature ('candle' or 'onnx').");
    println!("Please rebuild with one of the following commands:");
    println!("  cargo run --example smolvlm_demo --features=\"kornia-models/candle\"");
    println!("  cargo run --example smolvlm_demo --features=\"kornia-models/onnx\"");
    println!("  cargo run --example smolvlm_demo --features=\"kornia-models/candle kornia-models/onnx\"");
}