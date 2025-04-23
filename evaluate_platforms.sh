#!/bin/bash
# Platform Evaluation Script for SmolVLM

set -e

# Print colored status messages
print_status() {
    echo -e "\033[1;34m[INFO]\033[0m $1"
}

print_success() {
    echo -e "\033[1;32m[SUCCESS]\033[0m $1"
}

print_error() {
    echo -e "\033[1;31m[ERROR]\033[0m $1"
}

print_warning() {
    echo -e "\033[1;33m[WARNING]\033[0m $1"
}

print_header() {
    echo -e "\033[1;35m==================================================\033[0m"
    echo -e "\033[1;35m $1 \033[0m"
    echo -e "\033[1;35m==================================================\033[0m"
}

# Create results directory
RESULTS_DIR="./evaluation_results"
mkdir -p "$RESULTS_DIR"

# Check if test image exists
DEFAULT_TEST_IMAGE="./test_image.jpg"
if [ ! -f "$DEFAULT_TEST_IMAGE" ]; then
    print_warning "Default test image not found: $DEFAULT_TEST_IMAGE"
    print_status "You'll need to specify a test image path."
fi

# Check system information
check_system() {
    print_header "System Information"
    
    # OS info
    print_status "Operating System:"
    if [ -f /etc/os-release ]; then
        cat /etc/os-release | grep -E "^(NAME|VERSION)="
    else
        uname -a
    fi
    
    # CPU info
    print_status "CPU Information:"
    if [ -f /proc/cpuinfo ]; then
        echo "CPU Model: $(grep "model name" /proc/cpuinfo | head -n 1 | cut -d ":" -f 2 | xargs)"
        echo "CPU Cores: $(grep -c "processor" /proc/cpuinfo)"
    else
        sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "CPU info not available"
    fi
    
    # Memory info
    print_status "Memory Information:"
    if [ -f /proc/meminfo ]; then
        grep -E "MemTotal|MemFree|MemAvailable" /proc/meminfo
    else
        vm_stat 2>/dev/null || echo "Memory info not available"
    fi
    
    # GPU info if nvidia-smi is available
    if command -v nvidia-smi &> /dev/null; then
        print_status "GPU Information:"
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    fi
}

# Run Python benchmark for specified backends
run_python_benchmark() {
    local test_image=$1
    local tasks=$2
    local sizes=$3
    local backends=${4:-"python"}
    local runs=${5:-1}
    local output_file="$RESULTS_DIR/python_benchmark_$(date +%Y%m%d_%H%M%S).json"
    
    print_header "Running Python Benchmarks"
    print_status "Test image: $test_image"
    print_status "Tasks: $tasks"
    print_status "Model sizes: $sizes"
    print_status "Backends: $backends"
    print_status "Runs per config: $runs"
    
    # Run the benchmark
    python3 benchmark.py \
        -i "$test_image" \
        -t $tasks \
        -s $sizes \
        -b $backends \
        -r $runs \
        -o "$output_file"
    
    if [ $? -eq 0 ]; then
        print_success "Benchmark completed successfully!"
        print_success "Results saved to: $output_file"
    else
        print_error "Benchmark failed."
    fi
}

# Run Rust binary benchmark
run_rust_benchmark() {
    local test_image=$1
    local prompt=$2
    local model_size=${3:-"small"}
    local backend=${4:-"candle"}
    
    print_header "Running Rust Benchmark ($backend backend)"
    print_status "Test image: $test_image"
    print_status "Prompt: $prompt"
    print_status "Model size: $model_size"
    
    # Verify the binary exists or build it
    local binary_path="./target/release/examples/smolvlm_demo"
    if [ ! -f "$binary_path" ]; then
        print_status "Building Rust binary with $backend backend..."
        cargo build --release --example smolvlm_demo --features="kornia-models/$backend"
    fi
    
    # Run the benchmark
    print_status "Running benchmark..."
    time "$binary_path" \
        --image "$test_image" \
        --prompt "$prompt" \
        --model-size "$model_size" \
        --backend "$backend"
}

# Run comprehensive evaluation
run_evaluation() {
    local test_image=$1
    local output_dir="$RESULTS_DIR/evaluation_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$output_dir"
    
    print_header "SmolVLM Platform Evaluation"
    print_status "Test image: $test_image"
    print_status "Output directory: $output_dir"
    
    # Check if we have Hugging Face token
    if [ -n "$HF_TOKEN" ]; then
        print_status "Hugging Face token detected"
        HF_AVAILABLE=true
    else
        print_warning "No Hugging Face token found. Some tests will be limited."
        HF_AVAILABLE=false
    fi
    
    # Save system information
    check_system | tee "$output_dir/system_info.txt"
    
    # Run sequential tests
    print_header "Running Evaluation Tests"
    
    # 1. Basic Python simulation test
    print_status "1/5 Running basic Python simulation..."
    python3 smolvlm_demo.py -i "$test_image" -p "What objects are in this image?" \
        > "$output_dir/python_basic.txt" 2>&1
    
    # 2. Python benchmark for multiple tasks and sizes
    print_status "2/5 Running Python benchmarks..."
    python3 benchmark.py -i "$test_image" -b python -s small medium -t objects scene description \
        -r 1 -o "$output_dir/python_benchmark.json" > "$output_dir/python_benchmark.txt" 2>&1
    
    # 3. Rust Candle test if available
    print_status "3/5 Testing Rust Candle backend..."
    if cargo build --release --example smolvlm_demo --features="kornia-models/candle" 2>/dev/null; then
        ./target/release/examples/smolvlm_demo \
            --image "$test_image" \
            --prompt "What objects are in this image?" \
            --model-size small \
            --backend candle > "$output_dir/rust_candle.txt" 2>&1 || \
        print_warning "Candle backend test failed, see logs for details"
    else
        print_warning "Failed to build Candle backend, skipping test"
    fi
    
    # 4. Rust ONNX test if available
    print_status "4/5 Testing Rust ONNX backend..."
    if cargo build --release --example smolvlm_demo --features="kornia-models/onnx" 2>/dev/null; then
        ./target/release/examples/smolvlm_demo \
            --image "$test_image" \
            --prompt "What objects are in this image?" \
            --model-size small \
            --backend onnx > "$output_dir/rust_onnx.txt" 2>&1 || \
        print_warning "ONNX backend test failed, see logs for details"
    else
        print_warning "Failed to build ONNX backend, skipping test"
    fi
    
    # 5. Hugging Face API test if token available
    print_status "5/5 Testing Hugging Face API integration..."
    if [ "$HF_AVAILABLE" = true ]; then
        python3 smolvlm_demo.py -i "$test_image" -p "What objects are in this image?" --use-hf \
            > "$output_dir/hf_api.txt" 2>&1
    else
        print_warning "Skipping Hugging Face API test (no token available)"
    fi
    
    print_header "Evaluation Complete"
    print_success "All tests completed. Results saved to: $output_dir"
    print_status "To compare results, check the output files in this directory."
}

# Main function
main() {
    local test_image=${1:-$DEFAULT_TEST_IMAGE}
    
    # If no image specified and default doesn't exist, show error
    if [ -z "$test_image" ] || [ ! -f "$test_image" ]; then
        print_error "Test image not found. Please specify a valid image path."
        echo "Usage: $0 [test_image_path]"
        exit 1
    fi
    
    # Run the evaluation
    run_evaluation "$test_image"
}

# Check if executed directly (not sourced)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
