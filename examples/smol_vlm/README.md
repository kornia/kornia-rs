Example showing how to use the HuggingFace SmolVLM model with Candle.

## Usage

```bash
Usage: smol_vlm -i <image-path> -p <text-prompt> [--sample-length <sample-length>]

Generate a description of an image using HuggingFace SmolVLM

Options:
  -i, --image-path      path to an input image
  -p, --text-prompt     prompt to ask the model
  --sample-length       the length of the generated text
  --help, help      display usage information
```

```bash
cargo run -p smol_vlm --features cuda -- -i ./.vscode/fuji-mountain-in-autumn.jpg -p "describe" --sample-length 500
```
remove `--features cuda` if you want to use CPU.

