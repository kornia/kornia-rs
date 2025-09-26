/*
    To convert a user-friendly structured input into raw tokens for direct model usage.
    This version of text preprocessing uses a more widely used and flexible approach: Jinja2 templating.
*/

use std::{error::Error, fs};

use hf_hub::api::sync::Api;
use minijinja::{context, AutoEscape, Environment, Template};
use serde::Serialize;
use tokenizers::Tokenizer;

use crate::smolvlm2::utils::SmolVlm2Error;

#[derive(Serialize, Debug)]
#[serde(rename_all = "lowercase")]
enum Role {
    User,
    Assistant,
    System,
}

#[derive(Serialize, Debug)]
#[serde(tag = "type")]
#[serde(rename_all = "lowercase")]
enum Line {
    Text { text: String },
    Image,
}

#[derive(Serialize, Debug)]
struct Message {
    role: Role,
    content: Vec<Line>,
}

struct TextPreprocessor<'a> {
    env: Environment<'a>,
    add_generation_prompt: bool,
    tokenizer: Tokenizer,
    token_history: Vec<u32>,
}

impl<'a> TextPreprocessor<'a> {
    pub fn new(identifier: String) -> Result<Self, Box<dyn Error>> {
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
            env,
            add_generation_prompt: true,
            token_history: Vec::new(),
            tokenizer: Tokenizer::from_pretrained(identifier, None).unwrap(),
        })
    }

    pub fn add_message(
        &mut self,
        messages: Vec<Message>,
    ) -> Result<(Vec<u32>, String), SmolVlm2Error> {
        let template = self.env.get_template("chat")?;

        let rendered = template.render(context! {
            messages => &messages,
            add_generation_prompt => self.add_generation_prompt
        })?;

        let encoded = self.tokenizer.encode(rendered.clone(), false)?;
        let tokens = encoded.get_ids().to_vec();

        self.token_history.extend(tokens.clone());

        Ok((tokens, rendered))
    }

    pub fn clear_history(&mut self) {
        self.token_history.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_text_rendering() {
        let mut preprocessor =
            TextPreprocessor::new("HuggingFaceTB/SmolVLM2-2.2B-Instruct".to_string()).unwrap();

        let messages = vec![Message {
            role: Role::User,
            content: vec![Line::Text {
                text: "What is life?".to_string(),
            }],
        }];

        let (_, rendered) = preprocessor.add_message(messages).unwrap();

        assert_eq!(
            rendered,
            "<|im_start|>User: What is life?<end_of_utterance>\nAssistant:"
        );

        let messages = vec![Message {
            role: Role::User,
            content: vec![
                Line::Image,
                Line::Text {
                    text: "Describe the image.".to_string(),
                },
            ],
        }];

        let (_, rendered) = preprocessor.add_message(messages).unwrap();

        assert_eq!(
            rendered,
            "<|im_start|>User:<image>Describe the image.<end_of_utterance>\nAssistant:"
        );

        let messages = vec![Message {
            role: Role::User,
            content: vec![
                Line::Image,
                Line::Image,
                Line::Image,
                Line::Text {
                    text: "Describe the images.".to_string(),
                },
            ],
        }];

        let (_, rendered) = preprocessor.add_message(messages).unwrap();

        assert_eq!(
            rendered,
            "<|im_start|>User:<image><image><image>Describe the images.<end_of_utterance>\nAssistant:"
        );
    }
}
