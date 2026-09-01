Example showing how to use the Moondream vision-language model with Candle.

Moondream is a ~1.6B parameter VLM that answers questions about images. It is small
enough to run on a CPU, which makes it a reasonable starting point for edge
deployments.

## Setup

The weights are downloaded from Hugging Face on first run (~3.7 GB) and cached under
`~/.cache/huggingface`. The repository is public, so no token is required.

The checkpoint is pinned to a specific revision: the upstream `moondream2` repository
changed its weight layout, so the weights matching candle's architecture live under
`vikhyatk/moondream1` at a fixed commit.

## Usage

```bash
Usage: moondream -i <image-path> -p <text-prompt> [--sample-length <sample-length>] [--iterations <iterations>]

Answer a question about an image using Moondream

Options:
  -i, --image-path  path to an input image (jpeg or png)
  -p, --text-prompt prompt to ask the model
  --sample-length   the length of the generated text
  --iterations      how many times to run inference, for benchmarking
  --help, help      display usage information
```

```bash
cargo run --release -p moondream -- -i ./data/dog.jpeg -p "What animal is in this image?"
```

To run on a GPU, enable the backend for your platform:

```bash
# NVIDIA
cargo run --release -p moondream --features cuda -- -i ./data/dog.jpeg -p "Describe the scene."

# Apple Silicon
cargo run --release -p moondream --features metal -- -i ./data/dog.jpeg -p "Describe the scene."
```

Pass `--iterations N` to run the same prompt repeatedly and compare decode throughput
across backends.

## Measured

Apple M5 (4P + 6E), `tests/data/dog.jpeg`, prompt "What animal is in this image?",
12 tokens generated, mean of 3 iterations:

| backend | prefill | decode | throughput | end-to-end |
|---------|---------|--------|-----------|------------|
| CPU (F32)   | 9.77 s | 2.52 s | 4.4 token/s | 12.29 s |
| Metal (F16) | 1.97 s | 1.19 s | 9.4 token/s | 3.15 s  |

Metal is ~5.0x faster on prefill, ~2.1x on decode, ~3.9x end-to-end. Both backends
produce the same caption, so F16 on Metal costs no visible quality here.

Note that timings are taken after an explicit `Device::synchronize()`. Metal
enqueues work asynchronously, so without that barrier prefill appears to take
~2 ms and its real cost is misattributed to the decode loop.
