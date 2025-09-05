#[cfg(feature = "debug")]
use std::collections::HashMap;

#[cfg(feature = "debug")]
use candle_core::safetensors::save;
use candle_core::Tensor;

pub struct ActivationIntrospector {
    #[cfg(feature = "debug")]
    activations: HashMap<String, Tensor>,
    counter_batch_pos: u32,
    counter_depth_arbitrary: u32, // useful arbitrary construct when debugging repeated layers found in any VLMs/LLMs
    // TODO: add counter_subdepth_arbitrary if for some reason a model is complex enough to have repeated sub-layers
    tracking_depth: bool, // useful for debugging repeated layers found in any VLMs/LLMs
}

impl ActivationIntrospector {
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "debug")]
            activations: HashMap::new(),
            counter_batch_pos: 0,
            counter_depth_arbitrary: 0,
            tracking_depth: false,
        }
    }

    #[allow(unused_variables)]
    pub fn save_as(self, fname: &str) {
        #[cfg(feature = "debug")]
        save(&self.activations, fname).unwrap();
    }

    #[allow(unused_variables)]
    pub fn insert(&mut self, name: &str, activation: &Tensor) {
        #[cfg(feature = "debug")]
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

    pub fn increment_batch_pos(&mut self) {
        self.counter_batch_pos += 1;
    }

    pub fn increment_depth(&mut self) {
        self.counter_depth_arbitrary += 1;
    }

    pub fn start_tracking_depth(&mut self) {
        if self.tracking_depth {
            #[cfg(feature = "debug")]
            panic!("Depth tracking has already started!");
        }

        self.counter_depth_arbitrary = 0;
        self.tracking_depth = true;
    }

    pub fn stop_tracking_depth(&mut self) {
        if !self.tracking_depth {
            #[cfg(feature = "debug")]
            panic!("Depth tracking has already stopped!");
        }

        self.counter_depth_arbitrary = 0;
        self.tracking_depth = false;
    }
}

pub struct InferenceContext {
    pub text_introspector: ActivationIntrospector,
    pub vis_introspector: ActivationIntrospector,
}

impl InferenceContext {
    pub fn new() -> Self {
        Self {
            text_introspector: ActivationIntrospector::new(),
            vis_introspector: ActivationIntrospector::new(),
        }
    }
}
