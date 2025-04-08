#!/bin/bash
# Script to download SmolVLM model weights

set -e

MODELS_DIR="$(pwd)"
VARIANT=${1:-small}  # Default to 'small' variant if not specified

echo "Downloading SmolVLM model weights (variant: $VARIANT)"
echo "Models will be saved to: $MODELS_DIR"

# Create URLs based on variant
MODEL_BASE_URL="https://huggingface.co/lmz/candle-smolvlm/resolve/main"
TOKENIZER_URL="$MODEL_BASE_URL/tokenizer-$VARIANT.json"
VISUAL_ENCODER_URL="$MODEL_BASE_URL/visual_encoder-$VARIANT.safetensors"
LANGUAGE_MODEL_URL="$MODEL_BASE_URL/language_model-$VARIANT.safetensors"

# Download tokenizer
echo "Downloading tokenizer..."
curl -L "$TOKENIZER_URL" -o "$MODELS_DIR/tokenizer-$VARIANT.json"

# Download visual encoder
echo "Downloading visual encoder..."
curl -L "$VISUAL_ENCODER_URL" -o "$MODELS_DIR/visual_encoder-$VARIANT.safetensors"

# Download language model
echo "Downloading language model..."
curl -L "$LANGUAGE_MODEL_URL" -o "$MODELS_DIR/language_model-$VARIANT.safetensors"

echo "Download complete!"
echo "Model files:"
ls -lh "$MODELS_DIR"
