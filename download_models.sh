#!/bin/bash
# Model Downloader Script for SmolVLM

set -e

# Create models directory if it doesn't exist
MODELS_DIR="./models/smolvlm"
mkdir -p "$MODELS_DIR"

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

# Check dependencies
check_dependencies() {
    print_status "Checking dependencies..."
    
    # Check for curl or wget
    if command -v curl &> /dev/null; then
        DOWNLOAD_CMD="curl -L -o"
    elif command -v wget &> /dev/null; then
        DOWNLOAD_CMD="wget -O"
    else
        print_error "Neither curl nor wget is installed. Please install one of them."
        exit 1
    fi
    
    # Check for md5sum or md5
    if command -v md5sum &> /dev/null; then
        MD5_CMD="md5sum"
    elif command -v md5 &> /dev/null; then
        MD5_CMD="md5 -q"
    else
        print_warning "Neither md5sum nor md5 is installed. Checksum verification will be skipped."
        MD5_CMD="echo CHECKSUM_VERIFICATION_SKIPPED"
    fi
}

# Download a model with progress and checksum verification
download_model() {
    local url=$1
    local output_file=$2
    local expected_checksum=$3
    local size_desc=$4
    
    if [ -f "$output_file" ]; then
        print_status "File already exists: $output_file, checking integrity..."
        
        # Verify checksum if possible
        if [ -n "$expected_checksum" ] && [ "$MD5_CMD" != "echo CHECKSUM_VERIFICATION_SKIPPED" ]; then
            local actual_checksum
            if [ "$MD5_CMD" == "md5sum" ]; then
                actual_checksum=$(md5sum "$output_file" | cut -d ' ' -f 1)
            else
                actual_checksum=$(md5 -q "$output_file")
            fi
            
            if [ "$actual_checksum" == "$expected_checksum" ]; then
                print_success "Checksum verified for $output_file"
                return 0
            else
                print_warning "Checksum mismatch for $output_file. Re-downloading..."
                rm "$output_file"
            fi
        else
            print_warning "Skipping checksum verification for $output_file"
            return 0
        fi
    fi
    
    print_status "Downloading $size_desc model from $url..."
    
    # Download the file
    if [ "$DOWNLOAD_CMD" == "curl -L -o" ]; then
        curl -L --progress-bar -o "$output_file" "$url"
    else
        wget --progress=bar:force -O "$output_file" "$url"
    fi
    
    # Verify download succeeded
    if [ $? -eq 0 ]; then
        print_success "Downloaded $output_file successfully"
        
        # Verify checksum if possible
        if [ -n "$expected_checksum" ] && [ "$MD5_CMD" != "echo CHECKSUM_VERIFICATION_SKIPPED" ]; then
            local actual_checksum
            if [ "$MD5_CMD" == "md5sum" ]; then
                actual_checksum=$(md5sum "$output_file" | cut -d ' ' -f 1)
            else
                actual_checksum=$(md5 -q "$output_file")
            fi
            
            if [ "$actual_checksum" == "$expected_checksum" ]; then
                print_success "Checksum verified for $output_file"
            else
                print_error "Checksum mismatch for $output_file!"
                print_error "Expected: $expected_checksum"
                print_error "Got: $actual_checksum"
                exit 1
            fi
        fi
    else
        print_error "Failed to download $output_file"
        exit 1
    fi
}

# Download platform-specific models
download_platform_models() {
    local platform=$1
    
    print_status "Downloading models for platform: $platform"
    
    # Create platform directory
    local platform_dir="$MODELS_DIR/$platform"
    mkdir -p "$platform_dir"
    
    # Download small model
    download_model "https://huggingface.co/kornia/smolvlm-$platform/resolve/main/smolvlm-small.safetensors" \
                  "$platform_dir/smolvlm-small.safetensors" \
                  "" \
                  "small"
    
    # Download medium model (only available for some platforms)
    if [ "$platform" != "aarch64" ]; then
        download_model "https://huggingface.co/kornia/smolvlm-$platform/resolve/main/smolvlm-medium.safetensors" \
                      "$platform_dir/smolvlm-medium.safetensors" \
                      "" \
                      "medium"
    else
        print_warning "Medium model not available for $platform platform"
    fi
    
    # Download tokenizer regardless of platform
    download_model "https://huggingface.co/kornia/smolvlm-common/resolve/main/tokenizer.json" \
                  "$MODELS_DIR/tokenizer.json" \
                  "" \
                  "tokenizer"
}

# Main function
main() {
    print_status "SmolVLM Model Downloader"
    print_status "======================="
    
    # Check dependencies
    check_dependencies
    
    # Detect platform
    PLATFORM=$(uname -m)
    if [ "$PLATFORM" == "x86_64" ]; then
        PLATFORM_DIR="x86_64"
    elif [[ "$PLATFORM" == "arm64" || "$PLATFORM" == "aarch64" ]]; then
        PLATFORM_DIR="aarch64"
    else
        print_warning "Unknown platform: $PLATFORM. Defaulting to x86_64."
        PLATFORM_DIR="x86_64"
    fi
    
    # Ask which platform to download for
    echo "Available platforms:"
    echo "1) x86_64 (Intel/AMD 64-bit)"
    echo "2) aarch64 (ARM 64-bit, e.g., NVIDIA Jetson)"
    echo "3) Both platforms"
    echo "4) Current platform ($PLATFORM_DIR)"
    read -p "Select platform(s) to download models for [4]: " platform_choice
    
    # Default to current platform
    platform_choice=${platform_choice:-4}
    
    case $platform_choice in
        1) download_platform_models "x86_64" ;;
        2) download_platform_models "aarch64" ;;
        3) 
           download_platform_models "x86_64"
           download_platform_models "aarch64"
           ;;
        4) download_platform_models "$PLATFORM_DIR" ;;
        *) 
           print_error "Invalid choice. Exiting."
           exit 1
           ;;
    esac
    
    print_success "All models downloaded successfully!"
    print_status "Models are located in: $MODELS_DIR"
}

# Run main function
main "$@"
