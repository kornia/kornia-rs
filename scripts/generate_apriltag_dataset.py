import os
import json
import argparse
import random
from pathlib import Path

# Fix for deterministic behavior
import numpy as np
import torch
import cv2
import kornia
from kornia.utils import image_to_tensor
from torchvision.utils import save_image

# --- CONFIGURATION ---
DEFAULT_OUTPUT_DIR = "./tests/data/apriltag_dataset"
MANIFEST_FILENAME = "../manifest.json"
SEED = 2024

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def generate_negatives(output_dir, num_negatives=10):
    """Generates invalid tags (False Positive Bait)."""
    print(f"Generating {num_negatives} invalid tags...")
    negatives = []
    
    aug_pipeline = kornia.augmentation.AugmentationSequential(
        kornia.augmentation.RandomPerspective(distortion_scale=0.5, p=0.5),
        kornia.augmentation.RandomAffine(degrees=180, translate=(0.1, 0.1), scale=(0.5, 1.5), shear=15, p=0.8),
        kornia.augmentation.RandomPlasmaBrightness(roughness=(0.1, 0.5), p=0.5),
        kornia.augmentation.RandomGaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0), p=0.3),
        kornia.augmentation.RandomGaussianNoise(mean=0.0, std=0.05, p=0.5),
        same_on_batch=False
    )

    for i in range(num_negatives):
        grid_size = 10
        fake_tag = torch.ones((1, 1, grid_size, grid_size))
        
        # Valid Border
        fake_tag[:, :, 0, :] = 0 
        fake_tag[:, :, -1, :] = 0 
        fake_tag[:, :, :, 0] = 0 
        fake_tag[:, :, :, -1] = 0 
        
        # Invalid Random Code
        payload = torch.randint(0, 2, (1, 1, 8, 8)).float()
        fake_tag[:, :, 1:9, 1:9] = payload

        tag_img = torch.nn.functional.interpolate(fake_tag, size=(128, 128), mode='nearest')
        with torch.no_grad():
            final_img = aug_pipeline(tag_img)[0]

        filename = f"neg_invalid_code_{i}.jpg"
        path = os.path.join(output_dir, filename)
        save_image(final_img, path)

        negatives.append({
            "filename": filename,
            "family": "tag36h11",
            "type": "negative",
            "expect_ids": [],
            "description": "Synthetic invalid code (False Positive Check)"
        })
    return negatives

def generate_dataset(source_path_str, output_dir, num_variations=50):
    set_seed(SEED)
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, MANIFEST_FILENAME)
    manifest = []

    # 1. Generate Negatives first
    manifest.extend(generate_negatives(output_dir, num_negatives=10))

    # 2. Handle Source (Directory OR Single File)
    source_path = Path(source_path_str).resolve()
    source_files = []

    if source_path.is_file():
        print(f"📄 Detected single source file: {source_path.name}")
        source_files = [str(source_path)]
    elif source_path.is_dir():
        print(f"📂 Detected directory: {source_path.name}")
        source_files = list(source_path.rglob("*.png")) + list(source_path.rglob("*.jpg"))
        source_files = [str(p) for p in source_files]
    else:
        print(f"❌ Error: Source path {source_path} does not exist.")
        return

    print(f"Generating {num_variations} variations per source image...")

    aug_pipeline = kornia.augmentation.AugmentationSequential(
        # Geometry
        kornia.augmentation.RandomPerspective(distortion_scale=0.5, p=0.5),
        kornia.augmentation.RandomAffine(
            degrees=180,          
            translate=(0.1, 0.1), 
            scale=(0.5, 1.5),     
            shear=15,             
            p=0.8
        ),
        # Obstruction
        kornia.augmentation.RandomErasing(scale=(0.02, 0.15), ratio=(0.3, 3.3), p=0.5),
        # Lighting
        kornia.augmentation.RandomPlasmaBrightness(roughness=(0.1, 0.5), p=0.5),
        kornia.augmentation.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),
        # Sensor
        kornia.augmentation.RandomGaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0), p=0.3),
        kornia.augmentation.RandomMotionBlur(kernel_size=5, angle=35.0, direction=0.5, p=0.3),
        kornia.augmentation.RandomGaussianNoise(mean=0.0, std=0.05, p=0.5),
        same_on_batch=False
    )

    for file_path in source_files:
        filename = Path(file_path).name
        stem = Path(file_path).stem
        
        # Try to guess family from filename, default to tag36h11 if unknown
        family_str = "tag36h11" 
        if "16h5" in filename: family_str = "tag16h5"
        if "25h9" in filename: family_str = "tag25h9"

        # Determine Test Type
        # If we can't parse an ID (e.g. filename is just 'apriltags.jpg'), 
        # we switch to "Edge" mode (checking if *any* tags are found).
        test_type = "edge"
        expected_ids = []
        min_detections = 1
        
        # Check if filename has _idX format (e.g. tag36h11_id0.png)
        parts = stem.split('_')
        if len(parts) > 1 and parts[-1].isdigit():
            # It's a specific single tag! We can do a Golden test.
            test_type = "golden"
            expected_ids = [int(parts[-1])]
            family_str = "_".join(parts[:-1])

        img_cv = cv2.imread(file_path)
        if img_cv is None: continue
        
        if len(img_cv.shape) == 2:
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2RGB)
        else:
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        
        img_tensor = image_to_tensor(img_cv).float() / 255.0
        img_tensor = img_tensor[None]

        for i in range(num_variations):
            with torch.no_grad():
                out_tensor = aug_pipeline(img_tensor)
            
            out_filename = f"syn_{stem}_v{i}.jpg"
            out_path = os.path.join(output_dir, out_filename)
            save_image(out_tensor[0], out_path)
            
            # Construct Manifest Entry
            entry = {
                "filename": out_filename,
                "family": family_str,
                "type": test_type,
                "description": f"Synthetic variation v{i} of {filename}"
            }
            
            # Add specific fields based on type
            if test_type == "golden":
                entry["expect_ids"] = expected_ids
            elif test_type == "edge":
                entry["min_detections"] = min_detections

            manifest.append(entry)

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=4)

    print(f"✅ Generated dataset in {output_dir}")
    print(f"✅ Rust-compatible Manifest written to {manifest_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True, help="Path to input directory OR single image file")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--variations", type=int, default=50, help="Number of variations per image")
    args = parser.parse_args()

    generate_dataset(args.source, args.output, num_variations=args.variations)