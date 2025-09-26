/*
    To convert a user-friendly structured input into raw tokens for direct model usage.
    This version of text preprocessing uses a more widely used and flexible approach: Jinja2 templating.
*/

use std::{error::Error, fs};

use candle_core::{DType, IndexOp, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use hf_hub::api::sync::Api;
use minijinja::{context, AutoEscape, Environment, Template};
use serde::Serialize;
use tokenizers::Tokenizer;

use crate::smolvlm2::utils::{SmolVlm2Config, SmolVlm2Error};

#[derive(Serialize, Debug, Clone)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
    System,
}

#[derive(Serialize, Debug, Clone)]
#[serde(tag = "type")]
#[serde(rename_all = "lowercase")]
pub enum Line {
    Text { text: String },
    Image,
}

#[derive(Serialize, Debug, Clone)]
pub struct Message {
    role: Role,
    content: Vec<Line>,
}

struct TextProcessor<'a> {
    tokenizer: Tokenizer,

    env: Environment<'a>,
    message_history: Vec<Message>,
    rendered_history: String, // result of applying messages onto a template
    token_history: Vec<u32>,
    add_generation_prompt: bool,

    config: SmolVlm2Config,
    logits_processor: LogitsProcessor,
}

impl<'a> TextProcessor<'a> {
    pub fn new(identifier: String, config: SmolVlm2Config) -> Result<Self, Box<dyn Error>> {
        let logits_processor = if config.do_sample {
            LogitsProcessor::new(config.seed, Some(config.temp), Some(config.top_p))
        } else {
            LogitsProcessor::from_sampling(config.seed, Sampling::ArgMax)
        };

        let api = Api::new()?;
        let repo = api.model(identifier.clone());

        let path = repo.get("tokenizer_config.json")?;
        let data = fs::read_to_string(path)?;
        let tokenizer_config = serde_json::from_str::<serde_json::Value>(&data)?;
        let template_str = tokenizer_config["chat_template"]
            .as_str()
            .ok_or("template field not found in tokenizer_config.json")?
            .to_owned();

        let mut env = Environment::new();

        // disable auto-escaping (we're producing a plain text prompt, not HTML)
        env.set_auto_escape_callback(|_| AutoEscape::None);
        env.add_template_owned("chat", template_str)?;

        Ok(Self {
            tokenizer: Tokenizer::from_pretrained(identifier, None).unwrap(),

            env,
            message_history: Vec::new(),
            rendered_history: String::new(),
            token_history: Vec::new(),
            add_generation_prompt: true,

            config,
            logits_processor,
        })
    }

    /// Add a message to the history and render the full prompt.
    /// Returns the new tokens and the rendered prompt.
    /// If `delta` is true, only the new tokens from this message are returned.
    /// If `delta` is false, all tokens from the full history are returned.
    pub fn add_and_tokenize_message(
        &mut self,
        messages: Vec<Message>,
        delta: bool,
    ) -> Result<(Vec<u32>, String), SmolVlm2Error> {
        let template = self.env.get_template("chat")?;

        self.message_history.extend(messages);

        let rendered = template.render(context! {
            messages => &self.message_history,
            add_generation_prompt => self.add_generation_prompt,
        })?;

        let encoded = self.tokenizer.encode(rendered.clone(), false)?;
        let tokens = encoded.get_ids().to_vec();

        let rendered_old_end_ind = self.rendered_history.len();
        let token_old_end_ind = self.token_history.len();

        self.rendered_history.clear();
        self.rendered_history.push_str(&rendered);
        self.token_history.clear();
        self.token_history.extend(tokens.clone());

        if delta {
            let new_rendered = self.rendered_history[rendered_old_end_ind..].to_string();
            let new_tokens = self.token_history[token_old_end_ind..].to_vec();
            Ok((new_tokens, new_rendered))
        } else {
            Ok((self.token_history.clone(), self.rendered_history.clone()))
        }
    }

    pub fn clear_history(&mut self) {
        self.message_history.clear();
        self.rendered_history.clear();
        self.token_history.clear();
    }

    pub fn get_decoded_token(
        &mut self,
        raw_logits: &candle_core::Tensor,
        delta_token: &[u32],
    ) -> Result<String, SmolVlm2Error> {
        let (s, _embed_dim) = raw_logits.dims2()?;
        let last_logit = raw_logits.i((s - 1, ..))?;

        let output_logit = if self.config.do_sample {
            candle_transformers::utils::apply_repeat_penalty(
                &last_logit,
                self.config.repeat_penalty,
                &delta_token,
            )?
        } else {
            last_logit.clone()
        };

        if self.has_non_finite(&output_logit)? {
            Err(SmolVlm2Error::InvalidLogits(
                "Non-finite values (NaN or +/-Inf) found in logits".to_string(),
            ))
        } else {
            let out_token = self.logits_processor.sample(&output_logit)?;

            let token_output = self.tokenizer.decode(&[out_token], false)?;

            Ok(token_output)
        }
    }

    /// Return true if any element in `logits` is not finite (NaN or +/-Inf).
    fn has_non_finite(&self, logits: &Tensor) -> Result<bool, SmolVlm2Error> {
        let logits_vec = logits.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        Ok(logits_vec.iter().any(|&v| !v.is_finite()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_rendering() {
        let default_response: Vec<Message> = vec![Message {
            role: Role::Assistant,
            content: vec![Line::Text {
                text: "".to_string(),
            }],
        }];

        let mut preprocessor = TextProcessor::new(
            "HuggingFaceTB/SmolVLM2-2.2B-Instruct".to_string(),
            SmolVlm2Config {
                do_sample: false,
                seed: 42,
                temp: 1.0,
                top_p: 0.9,
                repeat_penalty: 1.0,
            },
        )
        .unwrap();

        let (_, rendered) = preprocessor
            .add_and_tokenize_message(
                vec![Message {
                    role: Role::User,
                    content: vec![Line::Text {
                        text: "What is life?".to_string(),
                    }],
                }],
                true,
            )
            .unwrap();
        let _ = preprocessor
            .add_and_tokenize_message(default_response.clone(), true)
            .unwrap();

        assert_eq!(
            rendered,
            "<|im_start|>User: What is life?<end_of_utterance>\nAssistant:"
        );

        let (_, rendered) = preprocessor
            .add_and_tokenize_message(
                vec![Message {
                    role: Role::User,
                    content: vec![
                        Line::Image,
                        Line::Text {
                            text: "Describe the image.".to_string(),
                        },
                    ],
                }],
                true,
            )
            .unwrap();
        let _ = preprocessor
            .add_and_tokenize_message(default_response.clone(), true)
            .unwrap();

        assert_eq!(
            rendered,
            "User:<image>Describe the image.<end_of_utterance>\nAssistant:"
        );

        let (_, rendered) = preprocessor
            .add_and_tokenize_message(
                vec![Message {
                    role: Role::User,
                    content: vec![
                        Line::Image,
                        Line::Image,
                        Line::Image,
                        Line::Text {
                            text: "Describe the images.".to_string(),
                        },
                    ],
                }],
                true,
            )
            .unwrap();
        let _ = preprocessor
            .add_and_tokenize_message(default_response.clone(), true)
            .unwrap();

        assert_eq!(
            rendered,
            "User:<image><image><image>Describe the images.<end_of_utterance>\nAssistant:"
        );

        let (_, rendered) = preprocessor
            .add_and_tokenize_message(
                vec![
                    Message {
                        role: Role::User,
                        content: vec![
                            Line::Image,
                            Line::Text {
                                text: "Describe the images.".to_string(),
                            },
                        ],
                    },
                    Message {
                        role: Role::Assistant,
                        content: vec![Line::Text {
                            text: "The image is beautiful!".to_string(),
                        }],
                    },
                    Message {
                        role: Role::User,
                        content: vec![Line::Text {
                            text: "How so?".to_string(),
                        }],
                    },
                ],
                true,
            )
            .unwrap();
        let _ = preprocessor
            .add_and_tokenize_message(default_response.clone(), true)
            .unwrap();

        assert_eq!(
            rendered,
            "User:<image>Describe the images.<end_of_utterance>\nAssistant: The image is beautiful!<end_of_utterance>\nUser: How so?<end_of_utterance>\nAssistant:"
        );

        preprocessor.clear_history();

        let (_, rendered) = preprocessor
            .add_and_tokenize_message(
                vec![Message {
                    role: Role::User,
                    content: vec![
                        Line::Image,
                        Line::Text {
                            text: "Misc".to_string(),
                        },
                    ],
                }],
                true,
            )
            .unwrap();

        assert_eq!(
            rendered,
            "<|im_start|>User:<image>Misc<end_of_utterance>\nAssistant:"
        );

        preprocessor.clear_history();

        let (_, rendered) = preprocessor
            .add_and_tokenize_message(
                vec![
                    Message {
                        role: Role::User,
                        content: vec![
                            Line::Image,
                            Line::Text {
                                text: "Misc".to_string(),
                            },
                        ],
                    },
                    Message {
                        role: Role::User,
                        content: vec![
                            Line::Image,
                            Line::Text {
                                text: "Misc again.".to_string(),
                            },
                        ],
                    },
                ],
                true,
            )
            .unwrap();

        assert_eq!(
            rendered,
            "<|im_start|>User:<image>Misc<end_of_utterance>\nUser:<image>Misc again.<end_of_utterance>\nAssistant:"
        );
    }

    #[test]
    fn test_add_generation_prompt_after_assistant() {
        let mut preprocessor =
            TextProcessor::new("HuggingFaceTB/SmolVLM2-2.2B-Instruct".to_string()).unwrap();

        let (_, rendered) = preprocessor
            .add_and_tokenize_message(
                vec![Message {
                    role: Role::Assistant,
                    content: vec![Line::Text {
                        text: "This is a wonderful world!".to_string(),
                    }],
                }],
                true,
            )
            .unwrap();

        assert_eq!(
            rendered,
            "<|im_start|>Assistant: This is a wonderful world!<end_of_utterance>\nAssistant:"
        );
    }
}
