//! SmolVLM Backend Comparison Example
//!
//! This example demonstrates how to compare different SmolVLM backends (Candle, ONNX)
//! and model sizes (Small, Medium, Large) for performance and accuracy.
//!
//! Usage:
//! ```bash
//! cargo run --example smolvlm_compare --features="kornia-models/candle kornia-models/onnx" -- \
//!     --image test_image.jpg \
//!     --prompt "What objects are in this image?" \
//!     --backends candle onnx \
//!     --model-sizes small medium
//! ```

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use argh::FromArgs;
use serde::{Deserialize, Serialize};

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

    /// backends to compare (candle, onnx)
    #[argh(option, short = 'b')]
    backends: Vec<String>,

    /// model sizes to compare (small, medium, large)
    #[argh(option, short = 's', default = "vec![String::from(\"small\")]")]
    model_sizes: Vec<String>,

    /// path to the model directory
    #[argh(option, short = 'm')]
    model_path: Option<String>,

    /// run benchmark
    #[argh(switch, short = 'n')]
    benchmark: bool,

    /// number of benchmark runs
    #[argh(option, short = 'r', default = "3")]
    runs: usize,

    /// number of warmup runs (not counted in benchmark)
    #[argh(option, short = 'w', default = "1")]
    warmup: usize,

    /// output file for benchmark results (JSON)
    #[argh(option, short = 'o')]
    output: Option<String>,
}

/// Benchmark result for a single configuration
#[derive(Serialize, Deserialize)]
struct BenchmarkResult {
    backend: String,
    model_size: String,
    durations_ms: Vec<f64>,
    avg_duration: f64,
    min_duration: f64,
    max_duration: f64,
    success_rate: f64,
    errors: Vec<String>,
}

/// Overall benchmark results
#[derive(Serialize, Deserialize)]
struct BenchmarkResults {
    image: String,
    prompt: String,
    timestamp: String,
    runs: usize,
    warmup: usize,
    results: Vec<BenchmarkResult>,
}

#[cfg(all(feature = "kornia-models", any(feature = "candle", feature = "onnx")))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse command-line arguments
    let args: Args = argh::from_env();
    
    // Parse backends
    let mut backends = Vec::new();
    if args.backends.is_empty() {
        // Default to all available backends
        #[cfg(feature = "candle")]
        backends.push("candle".to_string());
        #[cfg(feature = "onnx")]
        backends.push("onnx".to_string());
    } else {
        for backend in &args.backends {
            let backend_lowercase = backend.to_lowercase();
            match backend_lowercase.as_str() {
                "candle" => {
                    #[cfg(feature = "candle")]
                    backends.push("candle".to_string());
                    #[cfg(not(feature = "candle"))]
                    println!("Candle backend requested but not available (compile with --features=kornia-models/candle)");
                }
                "onnx" => {
                    #[cfg(feature = "onnx")]
                    backends.push("onnx".to_string());
                    #[cfg(not(feature = "onnx"))]
                    println!("ONNX backend requested but not available (compile with --features=kornia-models/onnx)");
                }
                _ => {
                    println!("Unknown backend: {}", backend);
                }
            }
        }
    }
    
    // Parse model sizes
    let model_sizes: Vec<ModelSize> = args
        .model_sizes
        .iter()
        .map(|s| match s.to_lowercase().as_str() {
            "small" => ModelSize::Small,
            "medium" => ModelSize::Medium,
            "large" => ModelSize::Large,
            _ => {
                println!("Unknown model size: {}. Using Small as default.", s);
                ModelSize::Small
            }
        })
        .collect();
    
    // Check if image exists
    if !std::path::Path::new(&args.image).exists() {
        return Err(format!("Image not found: {}", args.image).into());
    }
    
    // Check if at least one backend is available
    if backends.is_empty() {
        return Err("No supported backends available".into());
    }
    
    // Run comparison
    if args.benchmark {
        run_benchmark(&args, &backends, &model_sizes)?;
    } else {
        run_comparison(&args, &backends, &model_sizes)?;
    }
    
    Ok(())
}

#[cfg(all(feature = "kornia-models", any(feature = "candle", feature = "onnx")))]
fn run_comparison(
    args: &Args,
    backends: &[String],
    model_sizes: &[ModelSize],
) -> Result<(), Box<dyn std::error::Error>> {
    println!("===== SmolVLM Backend Comparison =====");
    println!("Image: {}", args.image);
    println!("Prompt: {}", args.prompt);
    println!("");
    
    // Process each backend and model size
    for &model_size in model_sizes {
        let config = SmolVLMConfig::new(model_size);
        let processor = ImageProcessor::new(&config)?;
        let processed_image = processor.process_image_from_path(&args.image)?;
        
        println!("===== Model Size: {:?} =====", model_size);
        
        for backend in backends {
            match backend.as_str() {
                #[cfg(feature = "candle")]
                "candle" => {
                    println!("Backend: Candle");
                    
                    let model_path = args
                        .model_path
                        .clone()
                        .unwrap_or_else(|| format!("models/candle/{:?}", model_size));
                    
                    if !std::path::Path::new(&model_path).exists() {
                        println!("Model path does not exist: {}", model_path);
                        println!("Please download the model using download_models.sh");
                        continue;
                    }
                    
                    let start = Instant::now();
                    let backend = CandleBackend::new(&model_path, &config)?;
                    let load_time = start.elapsed();
                    println!("  Model load time: {:?}", load_time);
                    
                    let start = Instant::now();
                    let result = backend.generate_caption_for_image(&processed_image, &args.prompt);
                    let inference_time = start.elapsed();
                    
                    match result {
                        Ok(caption) => {
                            println!("  Result: {}", caption);
                            println!("  Inference time: {:?}", inference_time);
                        }
                        Err(e) => {
                            println!("  Error: {}", e);
                        }
                    }
                }
                #[cfg(feature = "onnx")]
                "onnx" => {
                    println!("Backend: ONNX");
                    
                    let model_path = args
                        .model_path
                        .clone()
                        .unwrap_or_else(|| format!("models/onnx/{:?}", model_size));
                    
                    if !std::path::Path::new(&model_path).exists() {
                        println!("Model path does not exist: {}", model_path);
                        println!("Please download the model using download_models.sh");
                        continue;
                    }
                    
                    let start = Instant::now();
                    let backend = OnnxBackend::new(&model_path, &config)?;
                    let load_time = start.elapsed();
                    println!("  Model load time: {:?}", load_time);
                    
                    let start = Instant::now();
                    let result = backend.generate_caption_for_image(&processed_image, &args.prompt);
                    let inference_time = start.elapsed();
                    
                    match result {
                        Ok(caption) => {
                            println!("  Result: {}", caption);
                            println!("  Inference time: {:?}", inference_time);
                        }
                        Err(e) => {
                            println!("  Error: {}", e);
                        }
                    }
                }
                _ => {
                    println!("Backend: {} (not supported)", backend);
                }
            }
            
            println!("");
        }
    }
    
    Ok(())
}

#[cfg(all(feature = "kornia-models", any(feature = "candle", feature = "onnx")))]
fn run_benchmark(
    args: &Args,
    backends: &[String],
    model_sizes: &[ModelSize],
) -> Result<(), Box<dyn std::error::Error>> {
    println!("===== SmolVLM Benchmark =====");
    println!("Image: {}", args.image);
    println!("Prompt: {}", args.prompt);
    println!("Runs: {} (warmup: {})", args.runs, args.warmup);
    println!("");
    
    let timestamp = chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string();
    let mut benchmark_results = BenchmarkResults {
        image: args.image.clone(),
        prompt: args.prompt.clone(),
        timestamp,
        runs: args.runs,
        warmup: args.warmup,
        results: Vec::new(),
    };
    
    // Process each backend and model size
    for &model_size in model_sizes {
        let config = SmolVLMConfig::new(model_size);
        let processor = ImageProcessor::new(&config)?;
        let processed_image = processor.process_image_from_path(&args.image)?;
        
        println!("===== Model Size: {:?} =====", model_size);
        
        for backend in backends {
            println!("Backend: {}", backend);
            
            // Set up benchmark result
            let mut benchmark_result = BenchmarkResult {
                backend: backend.clone(),
                model_size: format!("{:?}", model_size).to_lowercase(),
                durations_ms: Vec::new(),
                avg_duration: 0.0,
                min_duration: 0.0,
                max_duration: 0.0,
                success_rate: 0.0,
                errors: Vec::new(),
            };
            
            // Choose backend
            match backend.as_str() {
                #[cfg(feature = "candle")]
                "candle" => {
                    let model_path = args
                        .model_path
                        .clone()
                        .unwrap_or_else(|| format!("models/candle/{:?}", model_size));
                    
                    if !std::path::Path::new(&model_path).exists() {
                        println!("Model path does not exist: {}", model_path);
                        println!("Please download the model using download_models.sh");
                        continue;
                    }
                    
                    println!("  Loading model from: {}", model_path);
                    let start = Instant::now();
                    let backend = match CandleBackend::new(&model_path, &config) {
                        Ok(backend) => backend,
                        Err(e) => {
                            println!("  Error loading model: {}", e);
                            continue;
                        }
                    };
                    let load_time = start.elapsed();
                    println!("  Model load time: {:?}", load_time);
                    
                    // Run warmup iterations
                    if args.warmup > 0 {
                        println!("  Running {} warmup iterations...", args.warmup);
                        for i in 0..args.warmup {
                            print!("    Warmup {}/{}...", i + 1, args.warmup);
                            std::io::stdout().flush()?;
                            let result = backend.generate_caption_for_image(&processed_image, &args.prompt);
                            match result {
                                Ok(_) => println!(" Done"),
                                Err(e) => println!(" Error: {}", e),
                            }
                        }
                    }
                    
                    // Run benchmark iterations
                    println!("  Running {} benchmark iterations...", args.runs);
                    let mut success_count = 0;
                    for i in 0..args.runs {
                        print!("    Run {}/{}...", i + 1, args.runs);
                        std::io::stdout().flush()?;
                        
                        let start = Instant::now();
                        let result = backend.generate_caption_for_image(&processed_image, &args.prompt);
                        let duration = start.elapsed();
                        let duration_ms = duration.as_secs_f64() * 1000.0;
                        
                        match result {
                            Ok(_) => {
                                println!(" Done in {:.2} ms", duration_ms);
                                benchmark_result.durations_ms.push(duration_ms);
                                success_count += 1;
                            }
                            Err(e) => {
                                println!(" Error: {}", e);
                                benchmark_result.errors.push(e.to_string());
                            }
                        }
                    }
                    
                    // Calculate statistics
                    if !benchmark_result.durations_ms.is_empty() {
                        benchmark_result.avg_duration = benchmark_result.durations_ms.iter().sum::<f64>()
                            / benchmark_result.durations_ms.len() as f64;
                        benchmark_result.min_duration = *benchmark_result.durations_ms.iter().min_by(|a, b| {
                            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                        }).unwrap();
                        benchmark_result.max_duration = *benchmark_result.durations_ms.iter().max_by(|a, b| {
                            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                        }).unwrap();
                    }
                    benchmark_result.success_rate = success_count as f64 / args.runs as f64 * 100.0;
                    
                    // Print statistics
                    println!("  Statistics:");
                    println!("    Avg Duration: {:.2} ms", benchmark_result.avg_duration);
                    println!("    Min Duration: {:.2} ms", benchmark_result.min_duration);
                    println!("    Max Duration: {:.2} ms", benchmark_result.max_duration);
                    println!("    Success Rate: {:.1}%", benchmark_result.success_rate);
                }
                #[cfg(feature = "onnx")]
                "onnx" => {
                    let model_path = args
                        .model_path
                        .clone()
                        .unwrap_or_else(|| format!("models/onnx/{:?}", model_size));
                    
                    if !std::path::Path::new(&model_path).exists() {
                        println!("Model path does not exist: {}", model_path);
                        println!("Please download the model using download_models.sh");
                        continue;
                    }
                    
                    println!("  Loading model from: {}", model_path);
                    let start = Instant::now();
                    let backend = match OnnxBackend::new(&model_path, &config) {
                        Ok(backend) => backend,
                        Err(e) => {
                            println!("  Error loading model: {}", e);
                            continue;
                        }
                    };
                    let load_time = start.elapsed();
                    println!("  Model load time: {:?}", load_time);
                    
                    // Run warmup iterations
                    if args.warmup > 0 {
                        println!("  Running {} warmup iterations...", args.warmup);
                        for i in 0..args.warmup {
                            print!("    Warmup {}/{}...", i + 1, args.warmup);
                            std::io::stdout().flush()?;
                            let result = backend.generate_caption_for_image(&processed_image, &args.prompt);
                            match result {
                                Ok(_) => println!(" Done"),
                                Err(e) => println!(" Error: {}", e),
                            }
                        }
                    }
                    
                    // Run benchmark iterations
                    println!("  Running {} benchmark iterations...", args.runs);
                    let mut success_count = 0;
                    for i in 0..args.runs {
                        print!("    Run {}/{}...", i + 1, args.runs);
                        std::io::stdout().flush()?;
                        
                        let start = Instant::now();
                        let result = backend.generate_caption_for_image(&processed_image, &args.prompt);
                        let duration = start.elapsed();
                        let duration_ms = duration.as_secs_f64() * 1000.0;
                        
                        match result {
                            Ok(_) => {
                                println!(" Done in {:.2} ms", duration_ms);
                                benchmark_result.durations_ms.push(duration_ms);
                                success_count += 1;
                            }
                            Err(e) => {
                                println!(" Error: {}", e);
                                benchmark_result.errors.push(e.to_string());
                            }
                        }
                    }
                    
                    // Calculate statistics
                    if !benchmark_result.durations_ms.is_empty() {
                        benchmark_result.avg_duration = benchmark_result.durations_ms.iter().sum::<f64>()
                            / benchmark_result.durations_ms.len() as f64;
                        benchmark_result.min_duration = *benchmark_result.durations_ms.iter().min_by(|a, b| {
                            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                        }).unwrap();
                        benchmark_result.max_duration = *benchmark_result.durations_ms.iter().max_by(|a, b| {
                            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                        }).unwrap();
                    }
                    benchmark_result.success_rate = success_count as f64 / args.runs as f64 * 100.0;
                    
                    // Print statistics
                    println!("  Statistics:");
                    println!("    Avg Duration: {:.2} ms", benchmark_result.avg_duration);
                    println!("    Min Duration: {:.2} ms", benchmark_result.min_duration);
                    println!("    Max Duration: {:.2} ms", benchmark_result.max_duration);
                    println!("    Success Rate: {:.1}%", benchmark_result.success_rate);
                }
                _ => {
                    println!("  Backend not supported");
                    continue;
                }
            }
            
            // Add result to benchmark results
            benchmark_results.results.push(benchmark_result);
            
            println!("");
        }
    }
    
    // Save benchmark results to file if requested
    if let Some(output_path) = &args.output {
        let file = File::create(output_path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, &benchmark_results)?;
        println!("Benchmark results saved to: {}", output_path);
    }
    
    Ok(())
}

#[cfg(not(all(feature = "kornia-models", any(feature = "candle", feature = "onnx"))))]
fn main() {
    println!("This example requires the 'kornia-models' feature and at least one backend feature ('candle' or 'onnx').");
    println!("Please rebuild with one of the following commands:");
    println!("  cargo run --example smolvlm_compare --features=\"kornia-models/candle\"");
    println!("  cargo run --example smolvlm_compare --features=\"kornia-models/onnx\"");
    println!("  cargo run --example smolvlm_compare --features=\"kornia-models/candle kornia-models/onnx\"");
}