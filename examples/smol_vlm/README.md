Example showing how to use the Google Paligemma model with Candle.

## Usage

```bash
Usage: paligemma -i <image-path> -p <text-prompt> [--sample-length <sample-length>] [--conversation-style]

Generate a description of an image using Google Paligemma

Options:
  -i, --image-path      path to an input image
  -p, --text-prompt     prompt to ask the model
  --sample-length       the length of the generated text
  --conversation_style  use it like a multimodal chatbot
  --help, help      display usage information
```

```bash
cargo run -p smol_vlm --features cuda -- -i ./.vscode/fuji-mountain-in-autumn.jpg -p "describe" --sample-length 100
```
remove `--features cuda` if you want to use CPU.

