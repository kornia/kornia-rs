use kornia_vlm::smolvlm::{utils::SmolVlmConfig, SmolVlm};
use std::path::PathBuf;
use std::sync::{mpsc, Arc, Mutex};
use std::thread;

use kornia_image::Image;
use kornia_tensor::CpuAllocator;
pub enum ModelRequest {
    Inference {
        prompt: String,
        image: Option<Image<u8, 3, CpuAllocator>>,
        response_tx: mpsc::Sender<ModelResponse>,
    },
    Quit,
}

pub enum ModelResponse {
    StreamChunk(String), // partial response
    Done,
    Error(String),
}

pub struct ModelStateHandle {
    pub tx: mpsc::Sender<ModelRequest>,
}

impl ModelStateHandle {
    pub fn new() -> Self {
        let (tx, rx) = mpsc::channel::<ModelRequest>();
        thread::spawn(move || {
            let mut model = SmolVlm::new(SmolVlmConfig {
                temp: 1.0,
                ..Default::default()
            })
            .expect("Failed to init model");

            loop {
                match rx.recv() {
                    Ok(ModelRequest::Inference {
                        prompt,
                        image,
                        response_tx,
                    }) => {
                        // No streaming: send only the full response at once
                        let result = model.inference(&prompt, image, 200, CpuAllocator);
                        match result {
                            Ok(full) => {
                                let _ = response_tx.send(ModelResponse::StreamChunk(full));
                                let _ = response_tx.send(ModelResponse::Done);
                            }
                            Err(e) => {
                                let _ = response_tx
                                    .send(ModelResponse::Error(format!("Inference error: {e}")));
                            }
                        }
                    }
                    Ok(ModelRequest::Quit) | Err(_) => break,
                }
            }
        });
        Self { tx }
    }
}
