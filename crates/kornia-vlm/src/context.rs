use std::collections::HashMap;

use candle_core::safetensors::save;
use candle_core::{Result, Tensor};

pub struct ActivationIntrospector {
    enabled: bool,
    activations: HashMap<String, Tensor>,
    counter_batch_pos: u32,
    counter_depth_arbitrary: u32, // useful arbitrary construct when debugging repeated layers found in any VLMs/LLMs
    // TODO: add counter_subdepth_arbitrary if for some reason a model is complex enough to have repeated sub-layers
    tracking_depth: bool, // useful for debugging repeated layers found in any VLMs/LLMs
}

impl ActivationIntrospector {
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            activations: HashMap::new(),
            counter_batch_pos: 0,
            counter_depth_arbitrary: 0,
            tracking_depth: false,
        }
    }

    pub fn save_as(self, fname: &str) -> Result<()> {
        if self.enabled {
            save(&self.activations, fname)?;
        }
        Ok(())
    }

    pub fn insert(&mut self, name: &str, activation: &Tensor) {
        if self.enabled {
            self.activations.insert(
                if self.tracking_depth {
                    format!(
                        "{}_d{}_i{}",
                        name, self.counter_depth_arbitrary, self.counter_batch_pos
                    )
                } else {
                    format!("{}_i{}", name, self.counter_batch_pos)
                },
                activation.clone(),
            );
        }
    }

    pub fn increment_batch_pos(&mut self) {
        self.counter_batch_pos += 1;
    }

    pub fn increment_depth(&mut self) {
        self.counter_depth_arbitrary += 1;
    }

    pub fn start_tracking_depth(&mut self) {
        if self.tracking_depth && self.enabled {
            panic!("Depth tracking has already started!");
        }

        self.counter_depth_arbitrary = 0;
        self.tracking_depth = true;
    }

    pub fn stop_tracking_depth(&mut self) {
        if !self.tracking_depth && self.enabled {
            panic!("Depth tracking has already stopped!");
        }

        self.counter_depth_arbitrary = 0;
        self.tracking_depth = false;
    }
}

pub struct InferenceContext {
    pub debug: bool,
    pub text_introspector: ActivationIntrospector,
    pub vis_introspector: ActivationIntrospector,
}

impl Default for InferenceContext {
    fn default() -> Self {
        Self::new(false, false)
    }
}

impl InferenceContext {
    pub fn new(enabled: bool, debug: bool) -> Self {
        Self {
            debug,
            text_introspector: ActivationIntrospector::new(enabled),
            vis_introspector: ActivationIntrospector::new(enabled),
        }
    }
}
