Example showing how to use the HuggingFace SmolVLM model with Candle.

## Usage

```bash
<<<<<<< HEAD
Usage: smol_vlm -i <image-path> -p <text-prompt> [--sample-length <sample-length>]
=======
Usage: smol_vlm -i <image-path> -p <text-prompt> [--sample-length <sample-length>] [--conversation-style]
>>>>>>> main

Generate a description of an image using HuggingFace SmolVLM

Options:
  -i, --image-path      path to an input image
  -p, --text-prompt     prompt to ask the model
  --sample-length       the length of the generated text
<<<<<<< HEAD
=======
  --conversation_style  use it like a multimodal chatbot
>>>>>>> main
  --help, help      display usage information
```

```bash
<<<<<<< HEAD
cargo run -p smol_vlm --features cuda -- -i ./.vscode/fuji-mountain-in-autumn.jpg -p "Can you describe the image?" --sample-length 500
=======
cargo run -p smol_vlm --features cuda -- --sample-length 1000 --conversation-style

cargo run -p smol_vlm --features cuda -- -i ./.vscode/fuji-mountain-in-autumn.jpg -p "describe" --sample-length 100
>>>>>>> main
```
remove `--features cuda` if you want to use CPU.

