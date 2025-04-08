#!/usr/bin/env python3
"""
SmolVLM Benchmarking Tool

This script benchmarks different backends (Python simulation, Rust Candle, Rust ONNX)
and measures performance across different model sizes and tasks.

Can be run directly or as part of a workflow in the Replit environment.
"""

import argparse
import os
import time
import json
import subprocess
import platform
from datetime import datetime
from PIL import Image

# Check if running in CI environment
IN_CI = os.environ.get("CI") == "true"

# Default to current directory if run from workflow
if os.getcwd().endswith('kornia-rs'):
    DEFAULT_IMAGE_PATH = "../test_image.jpg"
else:
    DEFAULT_IMAGE_PATH = "test_image.jpg"

# Available model sizes
MODEL_SIZES = ["small", "medium", "large"]

# Test prompts for different tasks
TEST_PROMPTS = {
    "description": "Describe what you see in this image in detail.",
    "objects": "What objects do you see in this image?",
    "colors": "What colors are prominent in this image?",
    "scene": "What type of scene is depicted in this image?",
    "action": "What action is happening in this image?",
    "emotions": "What emotions does this image convey?",
}

# Available backends
BACKENDS = ["python", "candle", "onnx"]

# CI-specific configurations
CI_CONFIG = {
    "timeout": 60,  # seconds
    "build_retries": 2,
    "reduced_iterations": True,
}

def get_system_info():
    """Get information about the system for benchmark results"""
    system_info = {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "platform_version": platform.version(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "timestamp": datetime.now().isoformat(),
    }
    
    # Try to get CPU info on Linux
    if platform.system() == "Linux":
        try:
            cpu_model = "Unknown"
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        cpu_model = line.split(":")[1].strip()
                        break
            system_info["cpu_model"] = cpu_model
        except (IOError, OSError, FileNotFoundError) as e:
            system_info["cpu_model"] = f"Could not determine: {str(e)}"
            
    # Try to get memory info
    try:
        # Try using psutil if available
        try:
            import psutil
            mem = psutil.virtual_memory()
            system_info["memory_total"] = str(mem.total)
            system_info["memory_available"] = str(mem.available)
        except ImportError:
            # Fallback to /proc/meminfo on Linux
            if platform.system() == "Linux":
                mem_total = "Unknown"
                mem_available = "Unknown"
                try:
                    with open("/proc/meminfo", "r") as f:
                        for line in f:
                            if "MemTotal" in line:
                                mem_total = line.split(":")[1].strip()
                            elif "MemAvailable" in line:
                                mem_available = line.split(":")[1].strip()
                    system_info["memory_total"] = mem_total
                    system_info["memory_available"] = mem_available
                except (IOError, OSError, FileNotFoundError) as e:
                    system_info["memory_total"] = f"Unknown: {str(e)}"
                    system_info["memory_available"] = f"Unknown: {str(e)}"
            else:
                system_info["memory_total"] = "Unknown (psutil not available)"
                system_info["memory_available"] = "Unknown (psutil not available)"
    except Exception as e:
        system_info["memory_total"] = f"Unknown: {str(e)}"
        system_info["memory_available"] = f"Unknown: {str(e)}"
        
    return system_info

def run_python_benchmark(image_path, prompt, model_size="small", use_hf=False):
    """Run benchmark using the Python implementation"""
    start_time = time.time()
    
    # Command to run the Python demo
    cmd = [
        "python3",
        "smolvlm_demo.py",
        "-i", image_path,
        "-p", prompt,
        "-s", model_size,
    ]
    
    if use_hf:
        cmd.append("--use-hf")
    
    # Run the command and capture output
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    # Extract the result from the output
    description = ""
    in_result = False
    for line in output.split("\n"):
        if "===================================================" in line and in_result:
            in_result = False
        elif in_result:
            description += line + "\n"
        elif "RESULT:" in line:
            in_result = True
            
    return {
        "elapsed_time": elapsed,
        "description": description.strip(),
        "exit_code": result.returncode,
        "success": result.returncode == 0,
    }

def run_rust_candle_benchmark(image_path, prompt, model_size="small"):
    """Run benchmark using the Rust Candle implementation"""
    # Path to the compiled Rust binary
    binary_path = "./target/release/examples/smolvlm_demo"
    
    # Check if binary exists
    if not os.path.exists(binary_path):
        print(f"Warning: Rust binary not found at {binary_path}. Attempting to build...")
        subprocess.run(["cargo", "build", "--release", "--example", "smolvlm_demo", "--features", "candle"], capture_output=True)
        if not os.path.exists(binary_path):
            return {
                "elapsed_time": 0,
                "description": "Failed to build Rust binary",
                "exit_code": 1,
                "success": False,
            }
            
    start_time = time.time()
    
    # Command to run the Rust demo with Candle backend
    cmd = [
        binary_path,
        "--image", image_path,
        "--prompt", prompt,
        "--model-size", model_size,
        "--backend", "candle"
    ]
    
    # Run the command and capture output
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    # Extract the result from the output
    description = ""
    in_result = False
    for line in output.split("\n"):
        if "======================================" in line and in_result:
            in_result = False
        elif in_result:
            description += line + "\n"
        elif "RESPONSE:" in line:
            in_result = True
            
    return {
        "elapsed_time": elapsed,
        "description": description.strip(),
        "exit_code": result.returncode,
        "success": result.returncode == 0,
    }

def run_rust_onnx_benchmark(image_path, prompt, model_size="small"):
    """Run benchmark using the Rust ONNX implementation"""
    # Path to the compiled Rust binary
    binary_path = "./target/release/examples/smolvlm_demo"
    
    # Check if binary exists
    if not os.path.exists(binary_path):
        print(f"Warning: Rust binary not found at {binary_path}. Attempting to build...")
        subprocess.run(["cargo", "build", "--release", "--example", "smolvlm_demo", "--features", "onnx"], capture_output=True)
        if not os.path.exists(binary_path):
            return {
                "elapsed_time": 0,
                "description": "Failed to build Rust binary",
                "exit_code": 1,
                "success": False,
            }
            
    start_time = time.time()
    
    # Command to run the Rust demo with ONNX backend
    cmd = [
        binary_path,
        "--image", image_path,
        "--prompt", prompt,
        "--model-size", model_size,
        "--backend", "onnx"
    ]
    
    # Run the command and capture output
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    # Extract the result from the output
    description = ""
    in_result = False
    for line in output.split("\n"):
        if "======================================" in line and in_result:
            in_result = False
        elif in_result:
            description += line + "\n"
        elif "RESPONSE:" in line:
            in_result = True
            
    return {
        "elapsed_time": elapsed,
        "description": description.strip(),
        "exit_code": result.returncode,
        "success": result.returncode == 0,
    }

def in_ci_environment():
    """Check if running in a CI environment"""
    return IN_CI

def run_benchmark(backend, image_path, prompt, model_size="small", use_hf=False):
    """Run benchmark with the specified backend, adapted for CI environments"""
    # In CI, use reduced settings for some backends
    if in_ci_environment() and backend in ["candle", "onnx"]:
        print("CI environment detected, using reduced settings")
        # Timeout settings depend on environment
        timeout = CI_CONFIG["timeout"]  # seconds for CI
        
        # For CI, we'll implement a simple timeout mechanism
        import threading
        import signal
        
        result = {"elapsed_time": 0, "description": "", "exit_code": -1, "success": False}
        
        def run_with_timeout():
            nonlocal result
            try:
                if backend == "python":
                    result = run_python_benchmark(image_path, prompt, model_size, use_hf)
                elif backend == "candle":
                    result = run_rust_candle_benchmark(image_path, prompt, model_size)
                elif backend == "onnx":
                    result = run_rust_onnx_benchmark(image_path, prompt, model_size)
            except Exception as e:
                result = {
                    "elapsed_time": 0,
                    "description": f"Error: {str(e)}",
                    "exit_code": 1,
                    "success": False,
                }
        
        thread = threading.Thread(target=run_with_timeout)
        thread.daemon = True
        thread.start()
        thread.join(timeout)
        
        if thread.is_alive():
            print(f"Timeout after {timeout} seconds")
            # Try to get partial results
            return {
                "elapsed_time": timeout,
                "description": "Timeout - operation took too long in CI environment",
                "exit_code": -1,
                "success": False,
            }
        
        return result
    else:
        # Regular execution for non-CI or Python backend
        if backend == "python":
            return run_python_benchmark(image_path, prompt, model_size, use_hf)
        elif backend == "candle":
            return run_rust_candle_benchmark(image_path, prompt, model_size)
        elif backend == "onnx":
            return run_rust_onnx_benchmark(image_path, prompt, model_size)
        else:
            raise ValueError(f"Unknown backend: {backend}")

def main():
    parser = argparse.ArgumentParser(description="SmolVLM Benchmarking Tool")
    parser.add_argument("-i", "--image", default=DEFAULT_IMAGE_PATH, help="Path to the test image")
    parser.add_argument("-b", "--backends", nargs="+", choices=BACKENDS, default=["python"], 
                       help="Backends to benchmark")
    parser.add_argument("-s", "--sizes", nargs="+", choices=MODEL_SIZES, default=["small"], 
                       help="Model sizes to benchmark")
    parser.add_argument("-t", "--tasks", nargs="+", choices=list(TEST_PROMPTS.keys()), default=["description"], 
                       help="Tasks to benchmark")
    parser.add_argument("-r", "--runs", type=int, default=1, help="Number of benchmark runs for each configuration")
    parser.add_argument("-o", "--output", help="Output file for benchmark results (JSON)")
    parser.add_argument("--use-hf", action="store_true", help="Use Hugging Face API for Python backend")
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        return 1
    
    print("=" * 60)
    print("SmolVLM Benchmarking Tool")
    print("=" * 60)
    print(f"Test image: {args.image}")
    print(f"Backends: {', '.join(args.backends)}")
    print(f"Model sizes: {', '.join(args.sizes)}")
    print(f"Tasks: {', '.join(args.tasks)}")
    print(f"Runs per configuration: {args.runs}")
    if "python" in args.backends:
        print(f"Use Hugging Face API: {args.use_hf}")
    print("-" * 60)
    
    # Collect system information
    system_info = get_system_info()
    print(f"Platform: {system_info['platform']} {system_info['platform_version']} ({system_info['architecture']})")
    print(f"Python version: {system_info['python_version']}")
    print("-" * 60)
    
    # Run benchmarks
    results = {
        "system_info": system_info,
        "config": {
            "image": args.image,
            "backends": args.backends,
            "model_sizes": args.sizes,
            "tasks": args.tasks,
            "runs": args.runs,
            "use_hf": args.use_hf,
        },
        "benchmarks": []
    }
    
    # For each configuration, run the benchmark
    total_benchmarks = len(args.backends) * len(args.sizes) * len(args.tasks) * args.runs
    completed = 0
    
    for backend in args.backends:
        for size in args.sizes:
            for task in args.tasks:
                prompt = TEST_PROMPTS[task]
                for run in range(args.runs):
                    print(f"Running benchmark: backend={backend}, size={size}, task={task}, run={run+1}/{args.runs}")
                    
                    # Run the benchmark
                    result = run_benchmark(backend, args.image, prompt, size, args.use_hf)
                    
                    # Store the results
                    benchmark_result = {
                        "backend": backend,
                        "model_size": size,
                        "task": task,
                        "prompt": prompt,
                        "run": run + 1,
                        "elapsed_time": result["elapsed_time"],
                        "success": result["success"],
                        "exit_code": result["exit_code"],
                    }
                    
                    if result["success"]:
                        benchmark_result["description"] = result["description"]
                        
                    results["benchmarks"].append(benchmark_result)
                    
                    # Update progress
                    completed += 1
                    print(f"Completed: {completed}/{total_benchmarks} ({completed/total_benchmarks*100:.1f}%)")
                    print(f"Elapsed time: {result['elapsed_time']:.2f} seconds")
                    print(f"Success: {result['success']}")
                    if result["success"]:
                        print(f"Description length: {len(result['description'])} characters")
                    print("-" * 60)
    
    # Save results if output file specified
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Benchmark results saved to {args.output}")
    
    # Display summary
    print("\nBenchmark Summary:")
    print("=" * 60)
    
    # Group by backend and model size
    summary = {}
    for benchmark in results["benchmarks"]:
        key = (benchmark["backend"], benchmark["model_size"])
        if key not in summary:
            summary[key] = {
                "total_time": 0,
                "count": 0,
                "success_count": 0,
                "tasks": {}
            }
        
        summary[key]["total_time"] += benchmark["elapsed_time"]
        summary[key]["count"] += 1
        if benchmark["success"]:
            summary[key]["success_count"] += 1
            
        # Add task-specific time
        task_key = benchmark["task"]
        if task_key not in summary[key]["tasks"]:
            summary[key]["tasks"][task_key] = {
                "total_time": 0,
                "count": 0
            }
        summary[key]["tasks"][task_key]["total_time"] += benchmark["elapsed_time"]
        summary[key]["tasks"][task_key]["count"] += 1
    
    # Print summary
    for (backend, model_size), data in sorted(summary.items()):
        avg_time = data["total_time"] / data["count"] if data["count"] > 0 else 0
        success_rate = data["success_count"] / data["count"] * 100 if data["count"] > 0 else 0
        
        print(f"Backend: {backend}, Model Size: {model_size}")
        print(f"  Average time: {avg_time:.2f} seconds")
        print(f"  Success rate: {success_rate:.1f}% ({data['success_count']}/{data['count']})")
        
        # Print task-specific averages
        if data["tasks"]:
            print("  Task averages:")
            for task, task_data in sorted(data["tasks"].items()):
                task_avg = task_data["total_time"] / task_data["count"] if task_data["count"] > 0 else 0
                print(f"    {task}: {task_avg:.2f} seconds")
                
        print("-" * 40)
        
    print("=" * 60)
    
    return 0

if __name__ == "__main__":
    main()