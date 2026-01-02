# Basic SmolVLM Example

## Usage
```bash
Usage: smol_vlm -i <image-path> -p <text-prompt> [--sample-length <sample-length>]

Options:
  -i, --image-path      path to an input image
  -p, --text-prompt     prompt to ask the model
  --sample-length       the length of the generated text
  --help, help      display usage information
```

## Command
```bash
cargo run -p smol_vlm --features cuda -- -i ./.vscode/fuji-mountain-in-autumn.jpg -p "Can you describe the image?" --sample-length 500
```
