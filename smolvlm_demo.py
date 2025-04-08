#!/usr/bin/env python3
"""
SmolVLM Demo - Enhanced Image Analysis Simulation

This script demonstrates functionality of SmolVLM with both simulated image analysis
and integration with Hugging Face API (when available) for real model inference.

Can be run directly or as part of a workflow in the Replit environment.
"""

import argparse
import os
import sys
import time
import json
import base64
import io
import platform
from datetime import datetime
# Make numpy optional for environments where it might not be available
numpy_available = False
try:
    import numpy as np
    numpy_available = True
except ImportError:
    # NumPy is optional for basic functionality
    np = None
from PIL import Image, ImageFilter, ImageStat, ImageEnhance
import requests

# Check if we have a Hugging Face token
HF_TOKEN = os.environ.get("HF_TOKEN")
USE_HUGGING_FACE = HF_TOKEN is not None and len(HF_TOKEN) > 0

# Log token availability without revealing the token
if USE_HUGGING_FACE and HF_TOKEN is not None:
    print(f"Hugging Face token is available (length: {len(HF_TOKEN)})")
else:
    print("No Hugging Face token found in environment")

# Default to current directory if run from workflow
if os.getcwd().endswith('kornia-rs'):
    DEFAULT_IMAGE_PATH = "../test_image.jpg"
else:
    DEFAULT_IMAGE_PATH = "test_image.jpg"

# Check if running in CI environment
IN_CI = os.environ.get("CI") == "true"

class SmolVLMAnalyzer:
    """SmolVLM Image Analysis Engine"""
    
    def __init__(self, model_size="small", use_hugging_face=False):
        """Initialize the SmolVLM analyzer"""
        self.model_size = model_size
        self.use_hugging_face = use_hugging_face and USE_HUGGING_FACE
        
        print("Initializing SmolVLM analyzer")
        if self.use_hugging_face:
            print(f"Using Hugging Face API with model size: {model_size}")
            self.model_loaded = True
        else:
            print("Using simulated analysis (HF API not available)")
            self.model_loaded = True
            
        print("Model loaded successfully")
        
    def analyze_image(self, image, prompt):
        """
        Analyze the image using SmolVLM (or simulation)
        """
        print(f"Analyzing image with prompt: '{prompt}'")
        
        if self.use_hugging_face:
            try:
                return self._analyze_with_hugging_face(image, prompt)
            except Exception as e:
                print(f"Error using Hugging Face API: {e}")
                print("Falling back to simulated analysis")
                return self._simulate_analysis(image, prompt)
        else:
            return self._simulate_analysis(image, prompt)
    
    def _analyze_with_hugging_face(self, image, prompt):
        """Use Hugging Face API to analyze the image"""
        # Convert image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Define API endpoint - this would be replaced with the actual SmolVLM API
        # For now, we're using a general image-to-text model as an example
        api_url = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"
        
        # Prepare payload
        payload = {
            "inputs": {
                "image": img_str,
                "text": prompt
            }
        }
        
        # Set up headers with token
        headers = {
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": "application/json"
        }
        
        # Call the API
        response = requests.post(api_url, headers=headers, json=payload)
        
        if response.status_code != 200:
            raise Exception(f"API error: {response.status_code} - {response.text}")
            
        # Parse the response
        result = response.json()
        
        # The format depends on the specific API we're using
        # This is a general format that works with several image-to-text models
        if isinstance(result, list) and len(result) > 0:
            return result[0].get("generated_text", "No description available")
        elif isinstance(result, dict):
            return result.get("generated_text", "No description available")
        else:
            return str(result)
    
    def _simulate_analysis(self, image, prompt):
        """Simulate SmolVLM analysis for demonstration purposes"""
        # Basic image description
        basic_description = self._get_basic_description(image)
        
        # Additional analysis based on prompt
        specific_analysis = self._analyze_based_on_prompt(image, prompt)
        
        # Combine the descriptions
        full_description = f"{basic_description}\n\n{specific_analysis}"
        
        return full_description
    
    def _get_basic_description(self, image):
        """Generate basic image description based on image properties"""
        # Get image dimensions
        width, height = image.size
        aspect_ratio = width / height
        
        # Analyze image color properties
        color_analysis = self._analyze_colors(image)
        
        # Analyze potential objects
        objects_analysis = self._analyze_potential_objects(image)
        
        # Analyze scene type
        scene_analysis = self._analyze_scene(image)
        
        # Analyze texture
        texture_analysis = self._analyze_texture(image)
        
        # Generate a description
        description = f"This image has dimensions of {width}x{height} pixels"
        if aspect_ratio > 1.2:
            description += " and has a landscape orientation."
        elif aspect_ratio < 0.8:
            description += " and has a portrait orientation."
        else:
            description += " and has a roughly square format."
            
        description += f" {color_analysis}"
        description += f" {scene_analysis}"
        description += f" {texture_analysis}"
        description += f" {objects_analysis}"
        
        return description
    
    def _analyze_based_on_prompt(self, image, prompt):
        """Generate additional analysis based on the prompt"""
        prompt_lower = prompt.lower()
        
        if "describe" in prompt_lower or "what" in prompt_lower or "see" in prompt_lower:
            # General description request
            return self._analyze_mood(image)
        elif "color" in prompt_lower or "colour" in prompt_lower:
            # Color-focused description
            return self._analyze_colors(image, detailed=True)
        elif "object" in prompt_lower or "thing" in prompt_lower:
            # Object-focused description
            return self._analyze_potential_objects(image, detailed=True)
        else:
            # Default to general description
            return "The image appears to be a photograph or digital creation capturing a specific moment or scene."
    
    def _analyze_colors(self, image, detailed=False):
        """Analyze color distribution in the image"""
        # Convert image to RGB if it's not
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        # Get image statistics
        stat = ImageStat.Stat(image)
        r, g, b = stat.mean
        
        # Determine dominant color
        max_channel = max(r, g, b)
        if max_channel == r and r > g * 1.2 and r > b * 1.2:
            dominant = "red"
        elif max_channel == g and g > r * 1.2 and g > b * 1.2:
            dominant = "green"
        elif max_channel == b and b > r * 1.2 and b > g * 1.2:
            dominant = "blue"
        elif r > 200 and g > 200 and b > 200:
            dominant = "white"
        elif r < 50 and g < 50 and b < 50:
            dominant = "black"
        elif abs(r - g) < 20 and abs(r - b) < 20 and abs(g - b) < 20:
            if r + g + b > 450:
                dominant = "light gray"
            else:
                dominant = "dark gray"
        else:
            dominant = "mixed"
            
        # Calculate color variance
        variance = self._get_color_variance(image)
            
        if detailed:
            if variance > 2000:
                color_desc = f"The image has a wide variety of colors, with {dominant} tones being prominent. "
                color_desc += "There's a rich palette with significant color variation throughout the image."
            elif variance > 1000:
                color_desc = f"The image has a moderate variety of colors, with {dominant} tones being most noticeable. "
                color_desc += "The color palette is varied but somewhat restrained."
            else:
                color_desc = f"The image has a limited color palette, primarily consisting of {dominant} tones. "
                color_desc += "The colors are quite uniform throughout the image."
        else:
            if variance > 2000:
                color_desc = f"The image is colorful with predominant {dominant} tones."
            elif variance > 1000:
                color_desc = f"The image has a moderate color palette with {dominant} being prominent."
            else:
                color_desc = f"The image has a limited color range, mostly in {dominant} tones."
                
        return color_desc
    
    def _analyze_potential_objects(self, image, detailed=False):
        """Estimate potential objects in the image based on edge detection"""
        # Convert to grayscale for edge detection
        gray = image.convert("L")
        
        # Apply edge detection
        edges = gray.filter(ImageFilter.FIND_EDGES)
        edge_stat = ImageStat.Stat(edges)
        edge_mean = edge_stat.mean[0]
        
        # Try to determine complexity based on edge detection
        if edge_mean > 30:
            complexity = "high"
            object_guess = "many distinct objects or details"
        elif edge_mean > 15:
            complexity = "moderate"
            object_guess = "several objects or elements"
        else:
            complexity = "low"
            object_guess = "few objects or mostly uniform areas"
            
        if detailed:
            if complexity == "high":
                return (f"The image appears to contain {object_guess}. There are many distinct edges and contours, "
                        "suggesting a complex scene with multiple subjects or a detailed central subject.")
            elif complexity == "moderate":
                return (f"The image appears to contain {object_guess}. The moderate level of detail suggests "
                        "a composition with clear subjects against a relatively simple background.")
            else:
                return (f"The image appears to contain {object_guess}. The minimal edge detail suggests "
                        "a simple composition, possibly with large uniform areas or soft gradients.")
        else:
            return f"The image has {complexity} complexity with {object_guess}."
    
    def _analyze_scene(self, image):
        """Analyze the overall scene type"""
        # Get image statistics
        width, height = image.size
        
        # Convert to RGB if not already
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        # Check color distribution
        r, g, b = ImageStat.Stat(image).mean
        
        # Very simple heuristics for scene type
        if g > max(r, b) * 1.2 and g > 100:
            # Green dominant might indicate nature/outdoors
            return "The image may depict an outdoor or natural scene with vegetation."
        elif b > max(r, g) * 1.2 and b > 100:
            # Blue dominant might indicate sky/water
            return "The image might contain sky or water elements, suggesting an outdoor scene."
        elif r > 150 and g > 150 and b > 150:
            # Bright image might be outdoor daylight
            return "The image appears to be bright, possibly an outdoor daylight scene."
        elif r < 60 and g < 60 and b < 60:
            # Dark image might be night or indoor
            return "The image appears to be dark, possibly a night scene or dimly lit indoor setting."
        else:
            # Mixed case
            return "The image could be an indoor scene or a mixed environment."
    
    def _analyze_texture(self, image):
        """Analyze texture patterns in the image"""
        # Convert to grayscale
        gray = image.convert("L")
        
        # Use edge detection to estimate texture
        edges = gray.filter(ImageFilter.FIND_EDGES)
        edge_stat = ImageStat.Stat(edges)
        edge_mean = edge_stat.mean[0]
        
        # Calculate standard deviation of the grayscale image as another texture metric
        gray_stat = ImageStat.Stat(gray)
        gray_std = gray_stat.stddev[0] if gray_stat.stddev else 0
        
        # Combined texture metric
        texture_score = edge_mean * 0.7 + gray_std * 0.3
        
        if texture_score > 25:
            return "The image contains fine detailed textures."
        elif texture_score > 15:
            return "The image has moderate textural details."
        else:
            return "The image has smooth areas with minimal texture."
    
    def _analyze_mood(self, image):
        """Estimate the mood conveyed by the image"""
        # Convert to RGB if not already
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        # Get color statistics
        stat = ImageStat.Stat(image)
        r, g, b = stat.mean
        brightness = (r + g + b) / 3
        
        # Calculate contrast
        contrast_img = ImageEnhance.Contrast(image)
        contrast_level = ImageStat.Stat(contrast_img.enhance(2.0)).var[0] if len(stat.var) > 0 else 50
        
        # Generate mood description
        if brightness > 180:
            if contrast_level > 50:
                return "The bright, high-contrast image evokes a sense of energy and clarity, possibly conveying happiness or excitement."
            else:
                return "The bright, low-contrast image creates a soft, ethereal atmosphere, possibly conveying serenity or dreaminess."
        elif brightness > 100:
            if contrast_level > 50:
                return "The moderately bright image with good contrast suggests a balanced, natural mood, possibly documentarian in nature."
            else:
                return "The moderately bright image with subtle contrast gives a calm, understated feeling, possibly conveying comfort or nostalgia."
        else:
            if contrast_level > 50:
                return "The darker image with strong contrast creates a dramatic or intense mood, possibly conveying mystery or tension."
            else:
                return "The darker image with minimal contrast suggests a somber or subdued mood, possibly conveying melancholy or reflection."
    
    def _get_color_variance(self, image):
        """Calculate variance in color distribution"""
        # Convert to RGB if not already
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        # Get color variance
        stat = ImageStat.Stat(image)
        if len(stat.var) >= 3:  # Make sure variance is available
            r_var, g_var, b_var = stat.var
            return r_var + g_var + b_var
        else:
            # Fallback if variance isn't available
            return 1000  # Moderate default

def get_platform_config():
    """Get platform-specific configuration and make adjustments"""
    system = platform.system().lower()
    config = {
        "max_image_size": 1024,  # Default max image dimension
        "use_simulation": False,  # Whether to force simulation mode
    }
    
    # CI environment detection
    if IN_CI:
        # In CI environments, use smaller images and avoid some operations
        config["max_image_size"] = 512
        if system == "linux":
            # Check for specific Ubuntu versions that might have issues
            if "ubuntu" in platform.version().lower():
                # Be more conservative with memory usage on Ubuntu CI
                config["max_image_size"] = 384
    
    return config

def process_image(image_path):
    """Load and process the image with platform-specific settings"""
    try:
        print(f"Loading image from: {image_path}")
        image = Image.open(image_path)
        print(f"Image loaded successfully: {image.format}, {image.size}x{image.mode}")
        
        # Get platform-specific configuration
        platform_config = get_platform_config()
        max_dim = platform_config["max_image_size"]
        
        # Resize image if needed
        if max(image.size) > max_dim:
            ratio = max_dim / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            print(f"Resizing image to {new_size}")
            try:
                # Try LANCZOS first (for newer PIL versions)
                try:
                    image = image.resize(new_size, Image.LANCZOS)
                except AttributeError:
                    # Fall back to BICUBIC for older PIL versions (Python 3.8 compatibility)
                    image = image.resize(new_size, Image.BICUBIC)
            except Exception as resize_error:
                # If any other issue, just use BICUBIC
                print(f"Warning: Using fallback resize method due to error: {resize_error}")
                image = image.resize(new_size, Image.BICUBIC)
        
        return image
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="SmolVLM Image Analysis Demo")
    parser.add_argument("-i", "--image", default=DEFAULT_IMAGE_PATH, help="Path to the input image")
    parser.add_argument("-p", "--prompt", default="What objects are in this image?", help="Text prompt for the image analysis")
    parser.add_argument("-s", "--size", choices=["small", "medium", "large"], default="small", help="Model size to use")
    parser.add_argument("--use-hf", action="store_true", help="Use Hugging Face API if available")
    parser.add_argument("-o", "--output", help="Save result to output file")
    args = parser.parse_args()
    
    print("=" * 50)
    print("SmolVLM Image Analysis Demo")
    print("=" * 50)
    print(f"Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Image: {args.image}")
    print(f"Prompt: {args.prompt}")
    print(f"Model Size: {args.size}")
    if args.use_hf:
        print(f"Using Hugging Face API: {'Yes (token available)' if USE_HUGGING_FACE else 'No (token not available)'}")
    print("-" * 50)
    
    # Process the image
    image = process_image(args.image)
    
    # Initialize the SmolVLM analyzer
    analyzer = SmolVLMAnalyzer(model_size=args.size, use_hugging_face=args.use_hf)
    
    # Analyze the image
    print("\nAnalyzing image...")
    start_time = time.time()
    
    description = analyzer.analyze_image(image, args.prompt)
    
    processing_time = time.time() - start_time
    print(f"Analysis completed in {processing_time:.2f} seconds")
    
    # Print the result
    print("\n" + "=" * 50)
    print("RESULT:")
    print("=" * 50)
    print(description)
    print("=" * 50)
    
    # Save to output file if requested
    if args.output:
        result = {
            "timestamp": datetime.now().isoformat(),
            "image": args.image,
            "prompt": args.prompt,
            "model_size": args.size,
            "processing_time_sec": processing_time,
            "result": description
        }
        
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()