//! Benchmarking utilities for SmolVLM

use std::path::Path;
use std::time::{Duration, Instant};

use crate::smolvlm::common::{ModelSize, SmolVLMConfig, SmolVLMError};
#[cfg(feature = "candle")]
use crate::smolvlm::candle::CandleBackend;
#[cfg(feature = "onnx")]
use crate::smolvlm::onnx::OnnxBackend;
use crate::smolvlm::processor::ImageProcessor;

/// Benchmark configuration
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Path to test image
    pub image_path: String,
    /// Prompt to use for benchmarking
    pub prompt: String,
    /// Model size to benchmark
    pub model_size: ModelSize,
    /// Number of warm-up runs
    pub warmup_runs: usize,
    /// Number of benchmark runs
    pub benchmark_runs: usize,
    /// Path to model directory
    pub model_path: String,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            image_path: "test_image.jpg".to_string(),
            prompt: "What objects are in this image?".to_string(),
            model_size: ModelSize::Small,
            warmup_runs: 1,
            benchmark_runs: 3,
            model_path: "models".to_string(),
        }
    }
}

/// Benchmark result
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Engine name
    pub engine: String,
    /// Model size
    pub model_size: ModelSize,
    /// Average duration
    pub avg_duration: Duration,
    /// Min duration
    pub min_duration: Duration,
    /// Max duration
    pub max_duration: Duration,
    /// Standard deviation of durations
    pub std_duration: Duration,
    /// Success rate (0.0 to 1.0)
    pub success_rate: f32,
    /// Average output length
    pub avg_output_length: usize,
}

/// Run benchmarks for all available backends
pub fn run_benchmarks(config: &BenchmarkConfig) -> Result<Vec<BenchmarkResult>, SmolVLMError> {
    let mut results = Vec::new();

    // Create SmolVLM config
    let smolvlm_config = SmolVLMConfig::new(config.model_size);

    // Process the image (done once to eliminate this from benchmark)
    let processor = ImageProcessor::new(&smolvlm_config)?;
    let image = processor.process_image_from_path(&config.image_path)?;

    // Log benchmark configuration
    log::info!(
        "Running SmolVLM benchmarks with image: {}, model size: {:?}, warmup: {}, runs: {}",
        config.image_path,
        config.model_size,
        config.warmup_runs,
        config.benchmark_runs
    );

    // Benchmark Candle backend if available
    #[cfg(feature = "candle")]
    {
        let candle_result = benchmark_candle(config, &smolvlm_config, &image)?;
        results.push(candle_result);
    }

    // Benchmark ONNX backend if available
    #[cfg(feature = "onnx")]
    {
        let onnx_result = benchmark_onnx(config, &smolvlm_config, &image)?;
        results.push(onnx_result);
    }

    // Log if no backends were benchmarked
    if results.is_empty() {
        log::warn!("No backends were benchmarked. Enable 'candle' or 'onnx' features.");
    }

    Ok(results)
}

/// Benchmark the Candle backend
#[cfg(feature = "candle")]
fn benchmark_candle(
    config: &BenchmarkConfig,
    smolvlm_config: &SmolVLMConfig,
    image: &crate::smolvlm::common::ProcessedImage,
) -> Result<BenchmarkResult, SmolVLMError> {
    let candle_model_path = format!("{}/candle/{:?}", config.model_path, config.model_size);
    
    if !Path::new(&candle_model_path).exists() {
        log::warn!("Candle model path does not exist: {}, skipping benchmark", candle_model_path);
        return Err(SmolVLMError::ModelLoadError(format!(
            "Model path does not exist: {}",
            candle_model_path
        )));
    }

    log::info!("Benchmarking Candle backend...");
    
    // Initialize backend
    let mut backend = CandleBackend::new(&candle_model_path, smolvlm_config)?;
    
    // Warm up
    for i in 0..config.warmup_runs {
        log::debug!("Candle warm-up run {}/{}", i + 1, config.warmup_runs);
        let _ = backend.generate_caption_for_image(image, &config.prompt);
    }
    
    // Run benchmark
    let mut durations = Vec::with_capacity(config.benchmark_runs);
    let mut outputs = Vec::with_capacity(config.benchmark_runs);
    let mut successes = 0;
    
    for i in 0..config.benchmark_runs {
        log::debug!("Candle benchmark run {}/{}", i + 1, config.benchmark_runs);
        
        let start = Instant::now();
        let result = backend.generate_caption_for_image(image, &config.prompt);
        let duration = start.elapsed();
        
        durations.push(duration);
        
        match result {
            Ok(output) => {
                successes += 1;
                outputs.push(output);
            }
            Err(e) => {
                log::error!("Candle benchmark run failed: {}", e);
            }
        }
    }
    
    // Calculate statistics
    let avg_duration = calculate_average_duration(&durations);
    let min_duration = *durations.iter().min().unwrap_or(&Duration::from_secs(0));
    let max_duration = *durations.iter().max().unwrap_or(&Duration::from_secs(0));
    let std_duration = calculate_std_duration(&durations, avg_duration);
    let success_rate = successes as f32 / config.benchmark_runs as f32;
    let avg_output_length = outputs.iter().map(|s| s.len()).sum::<usize>() / outputs.len().max(1);
    
    Ok(BenchmarkResult {
        engine: "candle".to_string(),
        model_size: config.model_size.clone(),
        avg_duration,
        min_duration,
        max_duration,
        std_duration,
        success_rate,
        avg_output_length,
    })
}

/// Benchmark the ONNX backend
#[cfg(feature = "onnx")]
fn benchmark_onnx(
    config: &BenchmarkConfig,
    smolvlm_config: &SmolVLMConfig,
    image: &crate::smolvlm::common::ProcessedImage,
) -> Result<BenchmarkResult, SmolVLMError> {
    let onnx_model_path = format!("{}/onnx/{:?}", config.model_path, config.model_size);
    
    if !Path::new(&onnx_model_path).exists() {
        log::warn!("ONNX model path does not exist: {}, skipping benchmark", onnx_model_path);
        return Err(SmolVLMError::ModelLoadError(format!(
            "Model path does not exist: {}",
            onnx_model_path
        )));
    }

    log::info!("Benchmarking ONNX backend...");
    
    // Initialize backend
    let mut backend = OnnxBackend::new(&onnx_model_path, smolvlm_config)?;
    
    // Warm up
    for i in 0..config.warmup_runs {
        log::debug!("ONNX warm-up run {}/{}", i + 1, config.warmup_runs);
        let _ = backend.generate_caption_for_image(image, &config.prompt);
    }
    
    // Run benchmark
    let mut durations = Vec::with_capacity(config.benchmark_runs);
    let mut outputs = Vec::with_capacity(config.benchmark_runs);
    let mut successes = 0;
    
    for i in 0..config.benchmark_runs {
        log::debug!("ONNX benchmark run {}/{}", i + 1, config.benchmark_runs);
        
        let start = Instant::now();
        let result = backend.generate_caption_for_image(image, &config.prompt);
        let duration = start.elapsed();
        
        durations.push(duration);
        
        match result {
            Ok(output) => {
                successes += 1;
                outputs.push(output);
            }
            Err(e) => {
                log::error!("ONNX benchmark run failed: {}", e);
            }
        }
    }
    
    // Calculate statistics
    let avg_duration = calculate_average_duration(&durations);
    let min_duration = *durations.iter().min().unwrap_or(&Duration::from_secs(0));
    let max_duration = *durations.iter().max().unwrap_or(&Duration::from_secs(0));
    let std_duration = calculate_std_duration(&durations, avg_duration);
    let success_rate = successes as f32 / config.benchmark_runs as f32;
    let avg_output_length = outputs.iter().map(|s| s.len()).sum::<usize>() / outputs.len().max(1);
    
    Ok(BenchmarkResult {
        engine: "onnx".to_string(),
        model_size: config.model_size.clone(),
        avg_duration,
        min_duration,
        max_duration,
        std_duration,
        success_rate,
        avg_output_length,
    })
}

/// Calculate average duration
fn calculate_average_duration(durations: &[Duration]) -> Duration {
    if durations.is_empty() {
        return Duration::from_secs(0);
    }
    
    let total_nanos: u128 = durations.iter().map(|d| d.as_nanos()).sum();
    Duration::from_nanos((total_nanos / durations.len() as u128) as u64)
}

/// Calculate standard deviation of durations
fn calculate_std_duration(durations: &[Duration], avg_duration: Duration) -> Duration {
    if durations.len() <= 1 {
        return Duration::from_secs(0);
    }
    
    let avg_nanos = avg_duration.as_nanos();
    let variance_sum: u128 = durations
        .iter()
        .map(|d| {
            let diff = if d.as_nanos() > avg_nanos {
                d.as_nanos() - avg_nanos
            } else {
                avg_nanos - d.as_nanos()
            };
            diff * diff
        })
        .sum();
    
    let variance = variance_sum / (durations.len() - 1) as u128;
    let std_dev = (variance as f64).sqrt();
    
    Duration::from_nanos(std_dev as u64)
}