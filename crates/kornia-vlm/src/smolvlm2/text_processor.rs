/*
    To convert a user-friendly structured input into raw tokens for direct model usage.
    This version of text preprocessing uses a more widely used and flexible approach: Jinja2 templating.
*/

use std::fs;

use candle_core::{DType, IndexOp, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use hf_hub::api::sync::Api;
use minijinja::{context, AutoEscape, Environment};
use serde::Serialize;
use tokenizers::Tokenizer;

use crate::smolvlm2::{SmolVlm2Config, SmolVlm2Error};

#[allow(dead_code)] // TODO: implement System role support
#[derive(Serialize, Debug, Clone)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,      // prompt
    Assistant, // the model's response
    System,
}

#[allow(dead_code)] // TODO: implement Video line type support
#[derive(Serialize, Debug, Clone)]
#[serde(tag = "type")]
#[serde(rename_all = "lowercase")]
pub enum Line {
    Text { text: String },
    Image,
    Video,
}

#[derive(Serialize, Debug, Clone)]
pub struct Message {
    pub role: Role,
    pub content: Vec<Line>,
}

pub struct TextProcessor {
    tokenizer: Option<Tokenizer>,

    env: Environment<'static>,
    message_history: Vec<Message>,
    formatted_history: String, // result of applying messages onto a template
    token_history: Vec<u32>,   // stores the history of generated tokens
    // note: refreshes at every add new message, and updates for every new logits sampled
    config: SmolVlm2Config,
    logits_processor: LogitsProcessor,
    previously_added_generation_prompt: bool, // assumes add generation prompt pre-emptively adds an assistant prompt

    eos_token: String,
}

impl TextProcessor {
    const EPSILON: f32 = 1e-5;

    pub fn new(identifier: String, config: SmolVlm2Config) -> Result<Self, SmolVlm2Error> {
        let logits_processor = if config.do_sample {
            LogitsProcessor::new(config.seed, Some(config.temp), Some(config.top_p))
        } else {
            LogitsProcessor::from_sampling(config.seed, Sampling::ArgMax)
        };

        let api = Api::new()?;
        let repo = api.model(identifier.clone());

        let path = repo.get("tokenizer_config.json")?;
        let data = fs::read_to_string(path.clone())?;
        let tokenizer_config = serde_json::from_str::<serde_json::Value>(&data)?;
        let template_str = tokenizer_config["chat_template"]
            .as_str()
            .ok_or(SmolVlm2Error::MissingChatTemplate(
                path.to_string_lossy().into_owned(),
            ))?
            .to_owned();

        let mut env = Environment::new();

        // disable auto-escaping (we're producing a plain text prompt, not HTML)
        env.set_auto_escape_callback(|_| AutoEscape::None);
        env.add_template_owned("chat", template_str)?;

        let eos_token = tokenizer_config["eos_token"]
            .as_str()
            .ok_or(SmolVlm2Error::EosTokenNotFound)?
            .to_string();

        Ok(Self {
            tokenizer: Some(Tokenizer::from_pretrained(identifier, None)?),

            env,
            message_history: Vec::new(),
            formatted_history: String::new(),
            token_history: Vec::new(),
            config,
            logits_processor,
            previously_added_generation_prompt: false,

            eos_token,
        })
    }

    pub fn with_template_string(
        mut self,
        new_chat_template: String,
    ) -> Result<Self, SmolVlm2Error> {
        self.env.add_template_owned("chat", new_chat_template)?;
        Ok(self)
    }

    pub fn is_eos(&self, token: &str) -> bool {
        token == self.eos_token
    }

    /// Add prompts (enable generation prompt) to the history, format them using the template,
    /// and return the formatted string.
    pub fn reformat_with_additional_prompts(
        &mut self,
        additional_messages: Vec<Message>,
        delta: bool,
    ) -> Result<String, SmolVlm2Error> {
        self.reformat_with_additional_raw_messages(additional_messages, true, delta)
    }

    /// Record the generated text into the message history from text-only assistant response.
    pub fn add_textual_response<S: Into<String>>(
        &mut self,
        textual_response: S,
    ) -> Result<(), SmolVlm2Error> {
        self.reformat_with_additional_raw_messages(
            vec![Message {
                role: Role::Assistant,
                content: vec![Line::Text {
                    text: textual_response.into(),
                }],
            }],
            false,
            false,
        )
        .map(|_| ())
    }

    /// Record the generated text a set of token by a set of token into the message
    /// history from text-only assistant response.
    pub fn update_last_textual_response<S: Into<String> + Clone>(
        &mut self,
        textual_response: S,
    ) -> Result<(), SmolVlm2Error> {
        if let Some(Message {
            role: Role::Assistant,
            content,
        }) = self.message_history.last_mut()
        {
            if let Some(Line::Text { text }) = content.last_mut() {
                text.push_str(&textual_response.clone().into());
            } else {
                content.push(Line::Text {
                    text: textual_response.clone().into(),
                });
            }

            let tokens = self.encode_all(&textual_response.clone().into())?;

            self.formatted_history.push_str(&textual_response.into());
            self.token_history.extend(tokens);
        } else {
            self.add_textual_response(textual_response)?;
        }

        self.previously_added_generation_prompt = false;

        Ok(())
    }

    /// Generalized method to add and return the formatted tokenized messages.
    /// # Arguments
    ///
    /// * `additional_messages` - The messages to add
    /// * `add_generation_prompt` - Whether to add the generation prompt at the end (note: )
    /// * `delta` - Whether to return only the newly added part of the formatted string
    pub fn reformat_with_additional_raw_messages(
        &mut self,
        additional_messages: Vec<Message>,
        add_generation_prompt: bool,
        delta: bool,
    ) -> Result<String, SmolVlm2Error> {
        if let Some(Message { role, content: _ }) = additional_messages.last() {
            if let Role::Assistant = role {
            } else if self.previously_added_generation_prompt {
                // if the last message is not from the assistant yet we supposedly added a generation prompt?
                return Err(SmolVlm2Error::MessageHistoryMismatch(
                    "Cannot add non-assistant message after adding a (generation) prompt"
                        .to_string(),
                ));
            }
        }

        let template = self.env.get_template("chat")?;

        self.message_history.extend(additional_messages);

        let formatted = template.render(context! {
            messages => &self.message_history,
            add_generation_prompt => add_generation_prompt,
        })?;
        self.previously_added_generation_prompt = add_generation_prompt;

        let tokens = self.encode_all(&formatted)?;

        let formatted_old_end_ind = self.formatted_history.len();

        self.formatted_history.clear();
        self.formatted_history.push_str(&formatted);
        self.token_history.clear();
        self.token_history.extend(tokens);

        if delta {
            let new_formatted = self.formatted_history[formatted_old_end_ind..].to_string();
            Ok(new_formatted)
        } else {
            Ok(self.formatted_history.clone())
        }
    }

    pub fn encode(&self, text: &str) -> Result<u32, SmolVlm2Error> {
        let encoding = self
            .tokenizer
            .as_ref()
            .ok_or(SmolVlm2Error::MissingTokenizer)?
            .encode(text, true)?;
        let encodings = encoding.get_ids();

        if encodings.len() != 1 {
            Err(SmolVlm2Error::InvalidEncoding(
                "Expected a single token".to_string(),
            ))
        } else {
            Ok(encodings[0])
        }
    }

    pub fn decode(&self, encoding: u32) -> Result<String, SmolVlm2Error> {
        self.decode_all(vec![encoding])
    }

    pub fn encode_all(&self, text: &str) -> Result<Vec<u32>, SmolVlm2Error> {
        let encoding = self
            .tokenizer
            .as_ref()
            .ok_or(SmolVlm2Error::MissingTokenizer)?
            .encode(text, true)?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode_all(&self, encodings: Vec<u32>) -> Result<String, SmolVlm2Error> {
        Ok(self
            .tokenizer
            .as_ref()
            .ok_or(SmolVlm2Error::MissingTokenizer)?
            .decode(&encodings, false)?)
    }

    pub fn clear_history(&mut self) {
        self.message_history.clear();
        self.formatted_history.clear();
    }

    pub fn sample_logits(&mut self, raw_logits: &Tensor) -> Result<u32, SmolVlm2Error> {
        let (s, _embed_dim) = raw_logits.dims2()?;
        let last_logit = raw_logits.i((s - 1, ..))?;

        let output_logit = if self.config.do_sample {
            let start_at = self
                .token_history
                .len()
                .saturating_sub(self.config.repeat_last_n);
            if 1. - Self::EPSILON < self.config.repeat_penalty
                && self.config.repeat_penalty < 1. + Self::EPSILON
            {
                last_logit
            } else {
                candle_transformers::utils::apply_repeat_penalty(
                    &last_logit,
                    self.config.repeat_penalty,
                    &self.token_history[start_at..],
                )?
            }
        } else {
            last_logit
        };

        let out_token = if self.config.do_sample {
            self.logits_processor.sample(&output_logit)?
        } else {
            output_logit
                .argmax(0)?
                .to_dtype(DType::U32)?
                .to_scalar::<u32>()?
        };

        Ok(out_token)
    }
}

impl Default for TextProcessor {
    fn default() -> Self {
        Self {
            tokenizer: None,
            env: Environment::new(),
            message_history: Vec::new(),
            formatted_history: String::new(),
            token_history: Vec::new(),
            config: SmolVlm2Config::default(),
            logits_processor: LogitsProcessor::from_sampling(42, Sampling::ArgMax),
            previously_added_generation_prompt: false,
            eos_token: String::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "Requires downloading config files from HuggingFace"]
    fn test_text_rendering() {
        let mut preprocessor = TextProcessor::new(
            "HuggingFaceTB/SmolVLM2-2.2B-Instruct".to_string(),
            SmolVlm2Config {
                do_sample: false,
                seed: 42,
                temp: 1.0,
                top_p: 0.9,
                repeat_penalty: 1.0,
                ..Default::default()
            },
        )
        .unwrap();

        let formatted = preprocessor
            .reformat_with_additional_prompts(
                vec![Message {
                    role: Role::User,
                    content: vec![Line::Text {
                        text: "What is life?".to_string(),
                    }],
                }],
                true,
            )
            .unwrap();
        preprocessor.add_textual_response("").unwrap();

        assert_eq!(
            formatted,
            "<|im_start|>User: What is life?<end_of_utterance>\nAssistant:"
        );

        let formatted = preprocessor
            .reformat_with_additional_prompts(
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
        preprocessor.add_textual_response("").unwrap();

        assert_eq!(
            formatted,
            "User:<image>Describe the image.<end_of_utterance>\nAssistant:"
        );

        let formatted = preprocessor
            .reformat_with_additional_prompts(
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
        preprocessor.add_textual_response("").unwrap();

        assert_eq!(
            formatted,
            "User:<image><image><image>Describe the images.<end_of_utterance>\nAssistant:"
        );

        let formatted = preprocessor
            .reformat_with_additional_prompts(
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
        preprocessor.add_textual_response("").unwrap();

        assert_eq!(
            formatted,
            "User:<image>Describe the images.<end_of_utterance>\nAssistant: The image is beautiful!<end_of_utterance>\nUser: How so?<end_of_utterance>\nAssistant:"
        );

        preprocessor.clear_history();

        let formatted = preprocessor
            .reformat_with_additional_prompts(
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
            formatted,
            "<|im_start|>User:<image>Misc<end_of_utterance>\nAssistant:"
        );

        preprocessor.clear_history();

        let formatted = preprocessor
            .reformat_with_additional_prompts(
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
            formatted,
            "<|im_start|>User:<image>Misc<end_of_utterance>\nUser:<image>Misc again.<end_of_utterance>\nAssistant:"
        );
    }

    #[test]
    #[ignore = "Requires downloading config files from HuggingFace"]
    fn test_add_generation_prompt_after_assistant() {
        let mut preprocessor = TextProcessor::new(
            "HuggingFaceTB/SmolVLM2-2.2B-Instruct".to_string(),
            SmolVlm2Config {
                seed: 42,
                temp: 1.0,
                top_p: 0.9,
                repeat_penalty: 1.0,
                do_sample: false,
                ..Default::default()
            },
        )
        .unwrap();

        let formatted = preprocessor
            .reformat_with_additional_prompts(
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
            formatted,
            "<|im_start|>Assistant: This is a wonderful world!<end_of_utterance>\nAssistant:"
        );
    }

    #[test]
    #[ignore = "Requires downloading config files from HuggingFace"]
    fn test_add_no_message() {
        let mut preprocessor = TextProcessor::new(
            "HuggingFaceTB/SmolVLM2-2.2B-Instruct".to_string(),
            SmolVlm2Config {
                seed: 42,
                temp: 1.0,
                top_p: 0.9,
                repeat_penalty: 1.0,
                do_sample: false,
                ..Default::default()
            },
        )
        .unwrap();

        let formatted = preprocessor
            .reformat_with_additional_prompts(vec![], true)
            .unwrap();

        assert_eq!(formatted, "<|im_start|>Assistant:");
    }

    #[test]
    #[ignore = "Requires downloading config files from HuggingFace"]
    fn test_add_non_assistant_after_generation_prompt() {
        let mut preprocessor = TextProcessor::new(
            "HuggingFaceTB/SmolVLM2-2.2B-Instruct".to_string(),
            SmolVlm2Config {
                seed: 42,
                temp: 1.0,
                top_p: 0.9,
                repeat_penalty: 1.0,
                do_sample: false,
                ..Default::default()
            },
        )
        .unwrap();

        let formatted = preprocessor
            .reformat_with_additional_prompts(
                vec![Message {
                    role: Role::User,
                    content: vec![Line::Text {
                        text: "What a beautiful day!".to_string(),
                    }],
                }],
                true,
            )
            .unwrap();

        assert_eq!(
            formatted,
            "<|im_start|>User: What a beautiful day!<end_of_utterance>\nAssistant:"
        );
        let err = preprocessor
            .reformat_with_additional_prompts(
                vec![Message {
                    role: Role::User,
                    content: vec![Line::Text {
                        text: "This should fail.".to_string(),
                    }],
                }],
                true,
            )
            .unwrap_err();

        assert!(matches!(err, SmolVlm2Error::MessageHistoryMismatch(_)));
    }
}
