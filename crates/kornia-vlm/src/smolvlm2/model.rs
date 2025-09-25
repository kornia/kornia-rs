use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::llama::{self, Cache};

pub struct Model {
    text_model: candle_transformers::models::llama::Llama,

    // TODO: move these caching into inference context
    cache: Cache,
}

impl Model {
    const HIDDEN_SIZE: usize = 2048;
    const VOCAB_SIZE: usize = 49280;
    const CONFIG: llama::Config = llama::Config {
        vocab_size: Self::VOCAB_SIZE,
        max_position_embeddings: 16384,
        num_attention_heads: 32,
        num_hidden_layers: 24,
        intermediate_size: 8192,
        hidden_size: Self::HIDDEN_SIZE,
        num_key_value_heads: 32,
        use_flash_attn: false,
        rms_norm_eps: 1e-5,
        rope_theta: 273768.0,
        bos_token_id: None,
        eos_token_id: None,
        rope_scaling: None,
        tie_word_embeddings: false,
    };

    pub fn load(vb: VarBuilder, dtype: DType, device: &Device) -> Result<Self> {
        let vb = vb.rename_f(|f: &str| {
            // If variables were namespaced as `model.text_model.model.*`, remove
            // the extra `.model` so we look up `model.text_model.*` which is
            // what the checkpoint uses in this repository.
            if let Some(rest) = f.strip_prefix("model.text_model.model") {
                // exact `model.text_model.model` -> `model.text_model`
                if rest.is_empty() {
                    return "model.text_model".to_string();
                }
                // keep the leading dot from rest (e.g. ".layers.0...")
                if rest.starts_with('.') {
                    return format!("model.text_model{}", rest);
                }
            }

            // Map bare lm_head key to the text model lm_head namespace.
            if f == "model.text_model.lm_head.weight" {
                return "lm_head.weight".to_string();
            }

            f.to_string()
        });

        Ok(Self {
            text_model: candle_transformers::models::llama::Llama::load(
                vb.pp("model.text_model"),
                &Self::CONFIG,
            )?,
            cache: Cache::new(true, dtype, &Self::CONFIG, device)?,
        })
    }

    pub fn forward(&mut self, xs: &Tensor, index_pos: usize) -> Result<Tensor> {
        let mut input_embeds = self.text_model.embed(xs)?;

        input_embeds = input_embeds.unsqueeze(0)?;

        let out = self
            .text_model
            .forward_input_embed(&input_embeds, index_pos, &mut self.cache)?;

        if out.dims().len() == 3 {
            Ok(out.squeeze(0)?)
        } else {
            Ok(out)
        }
    }
}
