use kornia_models::smolvlm::{SmolVLM, SmolVLMConfig};
use std::env;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse command-line arguments
    let args: Vec<String> = env::args().collect();
    
    if args.len() < 3 {
        eprintln!("Usage: smolvlm_demo <image_path> <prompt>");
        std::process::exit(1);
    }
    
    let image_path = &args[1];
    let prompt = &args[2];
    
    // Configure the model
    let config = SmolVLMConfig::small();
    
    // Determine the backend based on environment or availability
    let use_onnx = env::var("USE_ONNX").map(|v| v == "1").unwrap_or(false);
    
    // Create the appropriate model instance
    let model_path = match use_onnx {
        true => "models/smolvlm/onnx/model-small.onnx",
        false => "models/smolvlm/candle/model-small",
    };
    
    // Check if the model file exists
    if !Path::new(model_path).exists() {
        println!("Warning: Model file {} not found. Using mock implementation.", model_path);
        // In a real scenario, we would download the model here
    }
    
    println!("SmolVLM Demo");
    println!("------------");
    println!("Image: {}", image_path);
    println!("Prompt: {}", prompt);
    println!("Backend: {}", if use_onnx { "ONNX Runtime" } else { "Candle" });
    
    let model = if use_onnx {
        SmolVLM::with_onnx(model_path, config)?
    } else {
        SmolVLM::with_candle(model_path, config)?
    };
    
    // Generate a description
    println!("\nGenerating description...");
    
    let description = model.generate(image_path, prompt)?;
    
    println!("\nGenerated Description:");
    println!("{}", description);
    
    Ok(())
}