Example showing how to use the HuggingFace SmolVLM model with Candle (conversation style).

## Usage

```bash
Usage: smol_vlm_convo [--sample-length <sample-length>] [--conversation-style]

Initialize a simple chatbot interface to interact with SmolVLM.

Options:
  --sample-length       the length of the generated text
  --help, help      display usage information
```

```bash
cargo run -p smol_vlm_convo --features cuda -- --sample-length 1000
```
remove `--features cuda` if you want to use CPU.

