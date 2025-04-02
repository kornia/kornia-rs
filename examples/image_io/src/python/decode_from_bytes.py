#!/usr/bin/env python3
# Example that demonstrates decoding images directly from raw bytes in Python
# using the kornia-rs library

import argparse
import kornia_rs as K
import numpy as np
import os


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Decode images from raw bytes")
    parser.add_argument(
        "-i", "--image-path", type=str, required=True, help="Path to the input image"
    )
    args = parser.parse_args()

    # Read the image file as raw bytes
    with open(args.image_path, "rb") as f:
        image_data = f.read()
    
    print(f"Image data size: {len(image_data)} bytes")

    # Extract format from file extension
    format_hint = os.path.splitext(args.image_path)[1][1:].lower()
    print(f"Format hint: {format_hint}")

    # Decode image in RGB format from bytes
    img_rgb = K.decode_image_bytes(image_data, format_hint)
    print(f"Decoded RGB image shape: {img_rgb.shape}")

    # Decode image in grayscale format from bytes
    img_gray = K.decode_image_bytes_gray(image_data, format_hint)
    print(f"Decoded grayscale image shape: {img_gray.shape}")

    # If it's a JPEG image, also try using the specialized JPEG decoder
    if format_hint in ["jpg", "jpeg"]:
        print("Using specialized JPEG decoder")
        
        img_jpeg_rgb = K.decode_jpeg_bytes(image_data)
        print(f"Decoded JPEG RGB image shape: {img_jpeg_rgb.shape}")
        
        img_jpeg_gray = K.decode_jpeg_bytes_gray(image_data)
        print(f"Decoded JPEG grayscale image shape: {img_jpeg_gray.shape}")

    print("All decoding methods successful!")


if __name__ == "__main__":
    main() 