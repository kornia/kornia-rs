use kornia_vlm::smolvlm::{utils::SmolVlmConfig, SmolVlm};
use std::sync::mpsc;
use std::thread::{self, JoinHandle};

use kornia_image::Image;
use kornia_tensor::CpuAllocator;
pub enum ModelRequest {
    Inference {
        prompt: String,
        image: Option<Image<u8, 3, CpuAllocator>>,
        response_tx: mpsc::Sender<ModelResponse>,
    },
    SetTemperature(f64),
    SetTopP(f64),
    SetSampleLength(usize),
    SetDoSample(bool),
    ClearContext,
}

pub enum ModelResponse {
    StreamChunk(String), // partial response
    Done,
    Error(String),
}

pub struct ModelStateHandle {
    pub tx: mpsc::Sender<ModelRequest>,
    task_handle: Option<JoinHandle<()>>,
}

impl ModelStateHandle {
    pub fn new() -> Self {
        let (tx, rx) = mpsc::channel::<ModelRequest>();
        let task_handle = thread::spawn(move || {
            let mut config = SmolVlmConfig {
                temp: 1.0,
                ..Default::default()
            };
            let mut sample_len = 200;
            let mut model = SmolVlm::new(config).expect("Failed to init model");

            loop {
                match rx.recv() {
                    Ok(ModelRequest::Inference {
                        prompt,
                        image,
                        response_tx,
                    }) => {
                        // No streaming: send only the full response at once
                        let result = model.inference(&prompt, image, sample_len, CpuAllocator);
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
                    Ok(ModelRequest::SetTemperature(temp)) => {
                        config.temp = temp;
                        if let Err(e) = model.update_config(config) {
                            eprintln!("Failed to update config: {e}");
                        }
                    }
                    Ok(ModelRequest::SetTopP(top_p)) => {
                        config.top_p = top_p;
                        if let Err(e) = model.update_config(config) {
                            eprintln!("Failed to update config: {e}");
                        }
                    }
                    Ok(ModelRequest::SetSampleLength(len)) => {
                        sample_len = len;
                    }
                    Ok(ModelRequest::SetDoSample(do_sample)) => {
                        config.do_sample = do_sample;
                        if let Err(e) = model.update_config(config) {
                            eprintln!("Failed to update config: {e}");
                        }
                    }
                    Ok(ModelRequest::ClearContext) => {
                        // If the model has a method to clear context, call it here
                        model.clear_context();
                    }
                    Err(_) => break,
                }
            }
        });
        Self {
            tx,
            task_handle: Some(task_handle),
        }
    }
}

impl Drop for ModelStateHandle {
    fn drop(&mut self) {
        // Signal the thread to stop by dropping the sender
        // This will cause the receiver to return an error and break the loop
        drop(self.tx.clone());

        // Wait for the thread to finish
        if let Some(handle) = self.task_handle.take() {
            let _ = handle.join();
        }
    }
}
