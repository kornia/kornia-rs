//! SmolVLM Web API Example
//!
//! This example demonstrates how to create a simple web API for SmolVLM.
//!
//! Usage:
//! ```bash
//! cargo run --example web_api --features="kornia-models/candle" -- --port 8080 --model-path models/candle/Small
//! ```
//!
//! Then you can use the API:
//! ```bash
//! curl -X POST "http://localhost:8080/analyze" \
//!   -F "image=@test_image.jpg" \
//!   -F "prompt=What objects are in this image?"
//! ```

use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc;
use std::io::Read;

use argh::FromArgs;
use hyper::service::{make_service_fn, service_fn};
use hyper::{Body, Method, Request, Response, Server, StatusCode};
use multer::Multipart;
use futures::TryStreamExt;
use serde::{Deserialize, Serialize};
use tokio::runtime::Runtime;

#[cfg(feature = "kornia-models")]
use kornia_models::smolvlm::common::{ModelSize, SmolVLMConfig, SmolVLMError};
#[cfg(feature = "kornia-models")]
use kornia_models::smolvlm::processor::ImageProcessor;
#[cfg(all(feature = "kornia-models", feature = "candle"))]
use kornia_models::smolvlm::candle::CandleBackend;
#[cfg(all(feature = "kornia-models", feature = "onnx"))]
use kornia_models::smolvlm::onnx::OnnxBackend;

/// Web API configuration
#[derive(FromArgs)]
struct Args {
    /// HTTP port to listen on
    #[argh(option, default = "8080")]
    port: u16,

    /// Path to model directory
    #[argh(option, default = "String::from(\"models/candle/Small\")")]
    model_path: String,

    /// Model size (small, medium, large)
    #[argh(option, default = "String::from(\"small\")")]
    model_size: String,

    /// Backend to use (candle, onnx)
    #[argh(option, default = "String::from(\"candle\")")]
    backend: String,
}

/// API Response
#[derive(Serialize, Deserialize)]
struct ApiResponse {
    status: String,
    result: Option<String>,
    error: Option<String>,
    processing_time_ms: u64,
}

/// SmolVLM API Handler
#[cfg(feature = "kornia-models")]
struct SmolVLMHandler {
    config: SmolVLMConfig,
    processor: ImageProcessor,
    #[cfg(feature = "candle")]
    candle_backend: Option<Arc<CandleBackend>>,
    #[cfg(feature = "onnx")]
    onnx_backend: Option<Arc<OnnxBackend>>,
}

#[cfg(feature = "kornia-models")]
impl SmolVLMHandler {
    /// Create a new API handler
    fn new(args: &Args) -> Result<Self, SmolVLMError> {
        // Parse model size
        let model_size = match args.model_size.to_lowercase().as_str() {
            "small" => ModelSize::Small,
            "medium" => ModelSize::Medium,
            "large" => ModelSize::Large,
            _ => {
                eprintln!("Invalid model size: {}. Using small as default.", args.model_size);
                ModelSize::Small
            }
        };
        
        // Create SmolVLM config
        let config = SmolVLMConfig::new(model_size);
        
        // Create processor
        let processor = ImageProcessor::new(&config)?;
        
        // Create backend instances
        #[cfg(feature = "candle")]
        let candle_backend = if args.backend == "candle" {
            if !Path::new(&args.model_path).exists() {
                eprintln!("Model path does not exist: {}", args.model_path);
                return Err(SmolVLMError::ModelLoadError(format!(
                    "Model path does not exist: {}",
                    args.model_path
                )));
            }
            
            println!("Initializing Candle backend with model: {}", args.model_path);
            Some(Arc::new(CandleBackend::new(&args.model_path, &config)?))
        } else {
            None
        };
        
        #[cfg(feature = "onnx")]
        let onnx_backend = if args.backend == "onnx" {
            if !Path::new(&args.model_path).exists() {
                eprintln!("Model path does not exist: {}", args.model_path);
                return Err(SmolVLMError::ModelLoadError(format!(
                    "Model path does not exist: {}",
                    args.model_path
                )));
            }
            
            println!("Initializing ONNX backend with model: {}", args.model_path);
            Some(Arc::new(OnnxBackend::new(&args.model_path, &config)?))
        } else {
            None
        };
        
        #[cfg(not(feature = "candle"))]
        let candle_backend = None;
        
        #[cfg(not(feature = "onnx"))]
        let onnx_backend = None;
        
        Ok(Self {
            config,
            processor,
            #[cfg(feature = "candle")]
            candle_backend,
            #[cfg(feature = "onnx")]
            onnx_backend,
        })
    }
    
    /// Process an image with the appropriate backend
    async fn process_image(&self, image_data: &[u8], prompt: &str) -> Result<String, SmolVLMError> {
        // Process the image
        let processed_image = self.processor.process_image_from_bytes(image_data)?;
        
        // Run inference with the appropriate backend
        #[cfg(feature = "candle")]
        if let Some(backend) = &self.candle_backend {
            return backend.generate_caption_for_image(&processed_image, prompt);
        }
        
        #[cfg(feature = "onnx")]
        if let Some(backend) = &self.onnx_backend {
            return backend.generate_caption_for_image(&processed_image, prompt);
        }
        
        Err(SmolVLMError::ModelLoadError(
            "No backend available".to_string(),
        ))
    }
}

/// Handler for the root endpoint
async fn handle_root() -> Result<Response<Body>, hyper::Error> {
    let response = Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "text/html")
        .body(Body::from(
            r#"
            <!DOCTYPE html>
            <html>
            <head>
                <title>SmolVLM API</title>
                <style>
                    body {
                        font-family: Arial, sans-serif;
                        max-width: 800px;
                        margin: 0 auto;
                        padding: 20px;
                    }
                    h1 {
                        color: #333;
                    }
                    form {
                        background-color: #f5f5f5;
                        padding: 20px;
                        border-radius: 5px;
                    }
                    label {
                        display: block;
                        margin-bottom: 10px;
                        font-weight: bold;
                    }
                    input, textarea {
                        width: 100%;
                        padding: 8px;
                        margin-bottom: 15px;
                        border: 1px solid #ddd;
                        border-radius: 4px;
                    }
                    button {
                        background-color: #4CAF50;
                        color: white;
                        padding: 10px 15px;
                        border: none;
                        border-radius: 4px;
                        cursor: pointer;
                    }
                    pre {
                        background-color: #f0f0f0;
                        padding: 15px;
                        border-radius: 5px;
                        overflow-x: auto;
                    }
                </style>
            </head>
            <body>
                <h1>SmolVLM API</h1>
                <p>Upload an image and enter a prompt to analyze it with SmolVLM.</p>
                
                <form action="/analyze" method="post" enctype="multipart/form-data">
                    <label for="image">Image:</label>
                    <input type="file" id="image" name="image" accept="image/*" required>
                    
                    <label for="prompt">Prompt:</label>
                    <textarea id="prompt" name="prompt" rows="3" required>What objects are in this image?</textarea>
                    
                    <button type="submit">Analyze</button>
                </form>
                
                <h2>API Usage</h2>
                <pre>
curl -X POST "http://localhost:8080/analyze" \
  -F "image=@path/to/image.jpg" \
  -F "prompt=What objects are in this image?"
                </pre>
            </body>
            </html>
            "#,
        ))?;
    
    Ok(response)
}

/// Handler for the analyze endpoint
#[cfg(feature = "kornia-models")]
async fn handle_analyze(
    req: Request<Body>,
    handler: Arc<SmolVLMHandler>,
) -> Result<Response<Body>, hyper::Error> {
    // Only accept POST requests
    if req.method() != Method::POST {
        return Ok(Response::builder()
            .status(StatusCode::METHOD_NOT_ALLOWED)
            .body(Body::from("Method not allowed"))
            .unwrap());
    }
    
    // Parse the multipart form data
    let boundary = multer::parse_boundary(req.headers().get("content-type").and_then(|ct| ct.to_str().ok()).unwrap_or(""));
    if let Err(_) = boundary {
        return Ok(Response::builder()
            .status(StatusCode::BAD_REQUEST)
            .body(Body::from("Invalid content type"))
            .unwrap());
    }
    
    let mut multipart = Multipart::new(req.into_body(), boundary.unwrap());
    
    let mut image_data = Vec::new();
    let mut prompt = String::new();
    
    // Process form fields
    while let Some(mut field) = multipart.next_field().await.unwrap() {
        let name = field.name().unwrap_or("").to_string();
        
        if name == "image" {
            // Process image file
            while let Some(chunk) = field.chunk().await.unwrap() {
                image_data.extend_from_slice(&chunk);
            }
        } else if name == "prompt" {
            // Process prompt text
            while let Some(chunk) = field.chunk().await.unwrap() {
                prompt.push_str(std::str::from_utf8(&chunk).unwrap_or(""));
            }
        }
    }
    
    // Validate inputs
    if image_data.is_empty() {
        return Ok(Response::builder()
            .status(StatusCode::BAD_REQUEST)
            .body(Body::from("No image provided"))
            .unwrap());
    }
    
    if prompt.is_empty() {
        prompt = "What objects are in this image?".to_string();
    }
    
    // Process the image
    println!("Processing image ({} bytes) with prompt: {}", image_data.len(), prompt);
    let start_time = std::time::Instant::now();
    
    let result = handler.process_image(&image_data, &prompt).await;
    
    let elapsed = start_time.elapsed();
    let processing_time_ms = elapsed.as_millis() as u64;
    
    // Prepare response
    let response = match result {
        Ok(caption) => ApiResponse {
            status: "success".to_string(),
            result: Some(caption),
            error: None,
            processing_time_ms,
        },
        Err(e) => ApiResponse {
            status: "error".to_string(),
            result: None,
            error: Some(format!("{}", e)),
            processing_time_ms,
        },
    };
    
    // Serialize to JSON
    let json = serde_json::to_string(&response).unwrap();
    
    Ok(Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "application/json")
        .body(Body::from(json))
        .unwrap())
}

#[cfg(not(feature = "kornia-models"))]
async fn handle_analyze(
    _req: Request<Body>,
    _handler: (),
) -> Result<Response<Body>, hyper::Error> {
    Ok(Response::builder()
        .status(StatusCode::NOT_IMPLEMENTED)
        .body(Body::from("SmolVLM not available. Compile with --features=\"kornia-models/candle\""))
        .unwrap())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Setup logging
    env_logger::init();
    
    // Parse command-line arguments
    let args: Args = argh::from_env();
    
    // Create the API handler
    #[cfg(feature = "kornia-models")]
    let handler = Arc::new(SmolVLMHandler::new(&args)?);
    
    #[cfg(not(feature = "kornia-models"))]
    let handler = ();
    
    let addr = SocketAddr::from(([0, 0, 0, 0], args.port));
    println!("Starting SmolVLM API server on http://{}", addr);
    
    // Create the server service
    let make_svc = make_service_fn(move |_conn| {
        let handler = handler.clone();
        async move {
            Ok::<_, hyper::Error>(service_fn(move |req| {
                let handler = handler.clone();
                async move {
                    match (req.method(), req.uri().path()) {
                        (&Method::GET, "/") => handle_root().await,
                        (&Method::POST, "/analyze") => handle_analyze(req, handler).await,
                        _ => Ok(Response::builder()
                            .status(StatusCode::NOT_FOUND)
                            .body(Body::from("Not found"))
                            .unwrap()),
                    }
                }
            }))
        }
    });
    
    let server = Server::bind(&addr).serve(make_svc);
    println!("Server running at http://{}", addr);
    
    if let Err(e) = server.await {
        eprintln!("Server error: {}", e);
    }
    
    Ok(())
}