# Simple Video Understanding Example

This is a simple, single-prompt video understanding example. Given a video file, the model will generate a description or answer based on the provided prompt.

## Usage


> **Note:** The `gstreamer` feature must be enabled for video support. Add `--features gstreamer` to your command if needed.

```bash
cargo run --bin smol_vlm2 --release --features "gstreamer flash-attn cuda" -- --video-path example_video.mp4 --sampling uniform --sample-frames 8 --max-tokens 128 --prompt "Describe the video."
```

### Arguments
- `--video-path <PATH>`: Path to the input video file
- `--sampling <METHOD>`: Sampling method (`uniform`, `fps`, `firstn`, `indices`)
- `--sample-frames <N>`: Number of frames to sample
- `--max-tokens <N>`: Maximum number of generated tokens
- `--prompt <PROMPT>`: Prompt for the model

## Example

```bash
cargo run --bin smol_vlm2 --release --features "gstreamer flash-attn cuda" -- --video-path example_video.mp4 --sampling uniform --sample-frames 8 --max-tokens 128 --prompt "Describe the video."
```
