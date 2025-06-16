// TODO: port paligemma, smolvlm

mod text_model;
mod vision_model;

#[cfg(test)]
mod tests {
    use super::*;

    use std::io::{Error, Write};
    use std::io;

    use candle_core::{Device, IndexOp, Result, Shape, Tensor};
    use hf_hub::api::sync::Api;
    use tokenizers::{tokenizer::Tokenizer, PaddingDirection, PaddingParams, PaddingStrategy, TruncationDirection, TruncationParams, TruncationStrategy};
    use candle_nn::ops;
    use rand::rng;
    use rand::prelude::IndexedRandom;
    use vision_model::preprocess_image;
    use std::cmp::Ordering;
    use terminal_size::{terminal_size, Width};

    use crate::vision_model::{load_image_url, get_prompt_split_image};



    fn count_lines(text: &str) -> usize {
        if let Some((Width(w), _)) = terminal_size() {
            // Calculate the number of lines by dividing the text length by the terminal width
            (text.len() + w as usize + 1) / w as usize // Ceiling division
        } else {
            1 // Default to 1 if terminal size is not available
        }
    }

    fn clear_lines(n: usize) {
        for _ in 0..n {
            print!("\x1B[1A\x1B[2K"); // Move up and clear line
        }
        io::stdout().flush().unwrap();
    }

    fn read_input(cli_prompt: &str) -> String {
        let mut input = String::new();
        print!("{}", cli_prompt);
        io::stdout().flush().unwrap();
        io::stdin()
            .read_line(&mut input)
            .expect("Failed to read input");
        
        input.trim().to_owned()
    }


    fn main() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let mut tokenizer = Tokenizer::from_pretrained("HuggingFaceTB/SmolVLM-Instruct", None).unwrap();
        let api = Api::new().unwrap();
        let repo = api.model("HuggingFaceTB/SmolVLM-Instruct".to_string());
        let weights = repo.get("model.safetensors").unwrap();
        let weights = candle_core::safetensors::load(weights, &device)?;
        // tokenizer.with_padding(Some(
        //     PaddingParams { 
        //         strategy: PaddingStrategy::Fixed(1024),
        //         direction: PaddingDirection::Left,
        //         pad_to_multiple_of: None,
        //         pad_id: 2,
        //         pad_token: "<|im_end|>".to_string(),
        //         pad_type_id: 0,
        //     }
        // ));
        // tokenizer.with_truncation(Some(
        //     TruncationParams { 
        //         direction: TruncationDirection::Left, 
        //         max_length: 1024, // 8192
        //         strategy: TruncationStrategy::LongestFirst, 
        //         stride: 0,
        //     }
        // )).unwrap();

        let mut model = text_model::SmolVLM::load(&weights, &device)?;

        let image_token_enc = tokenizer.encode("<image>", false).unwrap();
        let image_token = image_token_enc.get_ids();

        /* CHAT TEMPLATE
        <|im_start|>
        {% for message in messages %}
            {{message['role'] | capitalize}}
            {% if message['content'][0]['type'] == 'image' %}
                {{':'}}
            {% else %}
                {{': '}}
            {% endif %}
            
            {% for line in message['content'] %}
                {% if line['type'] == 'text' %}
                    {{line['text']}}
                {% elif line['type'] == 'image' %}
                    {{ '<image>' }}
                {% endif %}
            {% endfor %}
            
            <end_of_utterance>\n
        {% endfor %}
        {% if add_generation_prompt %}
            {{ 'Assistant:' }}
        {% endif %}
        
        */

        // why are people making smaller inference LLM? is it the future?
        let mut message = String::from("<|im_start|>");
        let mut image: Vec<(Tensor, Tensor)> = Vec::new();
        let mut response = String::new();
        
        let mut output = String::new();
        let mut lines_printed = 0;
        for i in 0..10_000 {
            if i == 0 || output == "<end_of_utterance>" {
                // let img_url = String::from("https://res.cloudinary.com/enchanting/q_70,f_auto,w_5472,h_3078,c_fit/exodus-web/2023/05/mont-blanc.jpg");
                let img_url = read_input("img> ");
                let img = load_image_url(&img_url)
                    .and_then(
                        |v| if image.len() > 1 {
                            println!("One image max. Cannot add another image. (Restart)");
                            Err(Box::new(Error::new(io::ErrorKind::Other, "One image max")))
                        } else {Ok(v)}
                    )
                    .map(|img| preprocess_image(img, 1920, 384, &device));
                let mut txt_prompt = String::new();
                if let Ok((_,_,cols,rows)) = img {
                    let img_token = get_prompt_split_image(81, rows, cols);

                    txt_prompt += "\nUser:<image>";
                    txt_prompt = txt_prompt.replace("<image>", &img_token);
                } else {
                    println!("Invalid or empty URL (no image)");
                    println!("Error: {:?}", img);

                    txt_prompt += "\nUser: ";
                }

                txt_prompt += &read_input("txt> ");
                txt_prompt += "<end_of_utterance>\nAssistant:";

                response.clear();
                message += &txt_prompt;
                if let Ok((img_patches, mask_patches,_,_)) = img {
                    image.push((img_patches, mask_patches));
                }
            }

            // print!("#");
            // io::stdout().flush().unwrap();

            // println!("{:?}", message);
            let encoding = tokenizer.encode(message.clone(), false).unwrap();
            let tokens = encoding.get_ids();
            
            let input = Tensor::from_slice(tokens, Shape::from_dims(&[tokens.len()]), &device)?;
            let vision_data = if image.len() > 0 {
                let image_token_mask = Tensor::from_slice(image_token, &[1], &device)?;
                let image_token_mask = input.broadcast_eq(&image_token_mask)?;

                Some((image_token_mask, &image[0].0, &image[0].1))
            } else {
                None
            };
            let logits = model.forward(&input, i, vision_data, &device)?;
            
            let (s, _embed_dim) = logits.dims2()?;
            let last_logit = logits.i((s-1, ..))?;
            
            let out_token = {
                let temperature = Tensor::from_slice(&[
                    0.2f32
                ], (1,), &device)?;
                let k = 50;
        
                let scaled = last_logit.broadcast_div(&temperature)?;
                let probs = ops::softmax(&scaled, 0)?;
                let mut probs_vec: Vec<f32> = probs.to_vec1()?;
                let mut indices: Vec<usize> = (0..probs_vec.len()).collect(); 
                indices.sort_by(|&i, &j| probs_vec[j].partial_cmp(&probs_vec[i]).unwrap_or(Ordering::Equal));
                let top_k_indices = &indices[..k];
                let top_k_probs: Vec<f32> = top_k_indices.iter().map(|&i| probs_vec[i]).collect();
                let sum_probs: f32 = top_k_probs.iter().sum();
                let normalized_probs: Vec<f32> = top_k_probs.iter().map(|p| p / sum_probs).collect();
                let mut rng = rng();
                let sampled_index = top_k_indices
                    .choose_weighted(&mut rng, |&idx| normalized_probs[top_k_indices.iter().position(|&x| x == idx).unwrap()])
                    .expect("Sampling failed");

                [*sampled_index as u32]
            };
        
            output = tokenizer.decode(&out_token.as_slice(), false).unwrap();

            // println!("{:?}", output);
            // println!("{:?}", message);
            if !response.is_empty() {
                clear_lines(lines_printed);
            }
            println!("{:?}", response);
            io::stdout().flush().unwrap();
            lines_printed = count_lines(&response);
        
            message += &output;
            if output != "<end_of_utterance>" {
                response += &output;
            }
        }

        Ok(())
    }

    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
